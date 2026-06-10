import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from mrinufft.extras.cartesian import fft, ifft
from mrinufft.extras.smaps import _extract_kspace_center


def cartesian_kspace_centre(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    density: NDArray | None = None,
    max_iter: int = 10,
    calib_width: int | tuple[int, ...] = 12,
    plot: bool = False,
    plot_path: str | None = None,
) -> NDArray:
    """
    Trajectory between ±pi, rescaled to ±0.5 on call to _extract_kspace_center

    This function is copied from the mri-nufft espirit pipeline
    """
    # defer import to later to prevent circular import
    from mrinufft import get_operator

    # Normalise to [-0.5, 0.5] only for the center-selection step, so the
    # threshold (calib_width/shape) has a consistent meaning regardless of
    # the physical trajectory scale.
    traj_max   = np.max(np.abs(traj))
    traj_norm  = traj / (2 * traj_max)          # [-0.5, 0.5] for threshold check
    k_space, samples_norm, dc = _extract_kspace_center(
        kspace_data=kspace_data,
        kspace_loc=traj_norm,
        threshold=tuple(float(sh) for sh in calib_width / np.asarray(shape)),
        density=density,
        window_fun="rect",
    )
    # Scale samples back to the original trajectory range before passing to the
    # NUFFT operator — same convention as 02_b0_correction.py, which passes
    # the raw [-1.515, 1.515] trajectory directly to mrinufft.
    samples = samples_norm * (2 * traj_max)
    central_kspace_img = get_operator(backend)(
        samples,
        shape,
        n_coils=k_space.shape[-2],
        squeeze_dims=True,
    ).pinv_solver(k_space, max_iter=max_iter)

    if plot or plot_path:
        import matplotlib.pyplot as plt
        coil = 2
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].scatter(traj_norm[:, 0], traj_norm[:, 1], s=0.1, c=np.abs(kspace_data[coil, ...]))
        axes[0, 0].set_title(f"Full traj normalised (coil {coil})")
        axes[0, 1].scatter(samples_norm[:, 0], samples_norm[:, 1], s=1, c=np.abs(k_space[coil, ...]))
        axes[0, 1].set_title(f"Central region normalised (calib_width={calib_width})")
        axes[1, 0].imshow(np.abs(central_kspace_img[coil, :, :]), origin="lower", cmap="gray")
        axes[1, 0].set_title("pinv_solver recon (coil img)")
        axes[1, 1].imshow(np.abs(fft(central_kspace_img)[coil, :, :]), origin="lower", cmap="gray")
        axes[1, 1].set_title("FFT of recon (calib k-space)")
        plt.tight_layout()
        if plot_path:
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
        else:
            plt.show()

    return fft(central_kspace_img, dims=2)


def _morse_pi_vrc(
        sens: NDArray,
        centre: tuple[int, ...] | None = None):

    if centre is None:
        centre = tuple(int(x / 2) for x in sens.shape[1:-1])
    centre_coils = (0, ) + centre + (slice(None), )

    phi_coil_centre = np.angle(sens[centre_coils])

    vrc = np.sum(
        sens[0, ...] * np.exp(-1j * phi_coil_centre),
        axis=-1)

    return sens * np.exp(-1j * np.angle(vrc)[np.newaxis, ..., np.newaxis]), vrc


def morse_pi(
    data: NDArray,
    trajectory: NDArray,
    resolution: tuple[int, ...],
    backend: str,
    N_ref: int = 6,
    smoothing_sd: int = 3,
    max_iter: int = 10,
    calib_width: int = 12,
    calib_plot_path: str | None = None,
) -> NDArray:
    """Calculate sensitivity maps up to order N_ref using the 
    MORSE-PI method.

    See:
    MORSE-PI - Flexible and artefact-free image reconstruction for structural 
    and functional QSM and other phase-critical imaging applications.
    Barbara Dymerska et al.
    Proc. Intl. Soc. Mag. Reson. Med. 32 (2024)4265 DOI: https://doi.org/10.58530/2024/4265

    and

    MORSE: Multiple Orthogonal Reference Sensitivity Encoding
    Oliver Josephs et al
    https://doi.org/10.48550/arXiv.2510.09098

    :param data: Complex k-space data (NCoils x Npoints)
    :type data: NDArray
    :param trajectory: K-space trajectory scaled to ±π (NPoints x 3/2 spatial dimensions); divided by 2π internally to give ±0.5 for mrinufft
    :type trajectory: NDArray
    :param backend: MRI-NUFFT backend string, e.g. 'torchkbnufft-cpu'
    :type backend: str
    :param resolution: Three/two tuple of spatial resolution to reconstruct maps to.
    :type resolution: tuple[int]
    :param N_ref: Number of reference coils to construct and coil sensitivity order, defaults to 6
    :type N_ref: int, optional
    :return: Complex coil sensitivites (x, y, z, Ncoils, NRef)
    :rtype: NDArray
    """

    if len(resolution) != trajectory.shape[-1] \
            and len(resolution) not in (2, 3):
        raise ValueError(
            "The length of resolution must be 2 or three, "
            "and must match the number of spatial dimensions in trajectory.")

    c_kspace = cartesian_kspace_centre(
        trajectory,
        resolution,
        data,
        backend=backend,
        max_iter=max_iter,
        calib_width=calib_width,
        plot_path=calib_plot_path,
    )

    # Perform first SVD to form reference coils
    [U, S, Vh] = np.linalg.svd(
        c_kspace.reshape(c_kspace.shape[0], -1).T,
        full_matrices=False)

    smallref = U @ np.diag(S)
    smallref = smallref.T.reshape(c_kspace.shape)

    smallref_img = ifft(smallref, dims=2)

    # Form E, voxel-wise NCoil x NRef tensor
    E = np.einsum(
        "i...,j...->ij...",
        smallref_img.conj()[:N_ref, ...],
        smallref_img)

    # Smoothing step
    spatial_smoothing_kernel = (smoothing_sd, ) * len(resolution)
    E = gaussian_filter(
        E,
        (0, 0) + spatial_smoothing_kernel)

    # Second SVD
    [U2, S2, Vh2] = np.linalg.svd(
        E.T,
        full_matrices=False)

    sens = (Vh.T @ U2).T

    # From VRC
    sens_corrected, vrc = _morse_pi_vrc(
        np.moveaxis(sens, 1, -1))

    return np.ascontiguousarray(np.moveaxis(sens_corrected, 0, -1))
