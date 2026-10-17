"""
Concentric-ring (CRT) encoding for the learned-SPICE simulator.

The CSI grids here are small (8x8 to 12x12), so the encoding is an explicit
non-uniform DFT matrix instead of a NUFFT: exact, cheap, and differentiable.

Conventions
-----------
- Image vectors are row-major flattened (row, col) grids, as in
  calc_Bmatrix_simple / spice_reconstruct in scripts/05c_spice_recon_31p.py.
- k-space positions are in cycles per reconstruction voxel; traj[:, 0] = kx
  pairs with the column index, traj[:, 1] = ky with the row index.
- The Gram matrix is E^H diag(dcf) E, the same weighted normal operator that
  05c builds column by column from the mri-nufft operator with density=dcf.
"""

import numpy as np


def ring_radii(n_rings: int, n_acq: int) -> np.ndarray:
    """Ring radii in cycles per acquisition voxel: (i + 0.5) / n_acq."""
    return (np.arange(n_rings) + 0.5) / n_acq


def crt_trajectory(radii: np.ndarray, pts_per_ring: int = 32,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Points evenly spaced on each ring, with a random start angle per ring if rng is given."""
    traj = []
    for r in radii:
        phi0 = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
        ang = 2 * np.pi * np.arange(pts_per_ring) / pts_per_ring + phi0
        traj.append(np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1))
    return np.concatenate(traj).astype(np.float64)


def crt_dcf(traj: np.ndarray, hamming: bool = True) -> np.ndarray:
    """Ring-annulus DCF (x Hamming), following compute_crt_dcf_hamming in 05c."""
    r = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    unique_r = np.sort(np.unique(np.round(r, 6)))
    n_rings = len(unique_r)
    if n_rings < 2:
        return np.ones_like(r)

    r_norm = unique_r / (unique_r.max() + 1e-12)
    n_per_ring = np.array([(np.abs(r - ri) < 1e-5).sum() for ri in unique_r], dtype=np.float64)
    max_pts = n_per_ring.max()

    dcf_ring = np.zeros(n_rings)
    for i in range(n_rings):
        if i == 0:
            area = 0.25 * np.pi * (r_norm[0] + r_norm[1])**2
        elif i == n_rings - 1:
            r_out = 1.5 * r_norm[-1] - 0.5 * r_norm[-2]
            r_in = 0.5 * (r_norm[-1] + r_norm[-2])
            area = np.pi * (r_out**2 - r_in**2)
        else:
            upper = 0.5 * (r_norm[i] + r_norm[i + 1])
            lower = 0.5 * (r_norm[i - 1] + r_norm[i])
            area = np.pi * (upper**2 - lower**2)
        dcf_ring[i] = area * (max_pts / n_per_ring[i])

    dcf = np.zeros(len(r))
    for i, ri in enumerate(unique_r):
        dcf[np.abs(r - ri) < 1e-5] = dcf_ring[i]
    if hamming:
        dcf *= 0.54 + 0.46 * np.cos(2.0 * np.pi * r)
    return dcf


def voxel_centres(n: int, upsample: int = 1) -> np.ndarray:
    """Centres of an n x n grid (or of its upsample^2 sub-pixels), in reconstruction voxels.

    Voxel j is centred at j - n/2 (the NUFFT convention); sub-pixels tile that voxel.
    """
    sub = (np.arange(upsample) + 0.5) / upsample - 0.5
    c = (np.arange(n)[:, None] - n / 2 + sub[None, :]).ravel()
    return c


def encoding_matrix(traj: np.ndarray, n: int, upsample: int = 1) -> np.ndarray:
    """E[k, p] = exp(-2 pi i k . r_p) / upsample^2 on the (sub-pixel) grid, row-major.

    With upsample > 1, E integrates each voxel over upsample^2 sub-pixels, so a
    fine-grid object and its voxel average have consistent k-space.
    """
    c = voxel_centres(n, upsample)
    rows, cols = np.meshgrid(c, c, indexing="ij")
    phase = np.outer(traj[:, 0], cols.ravel()) + np.outer(traj[:, 1], rows.ravel())
    return np.exp(-2j * np.pi * phase) / upsample**2


def gram(E: np.ndarray, dcf: np.ndarray) -> np.ndarray:
    """Weighted normal operator E^H diag(dcf) E."""
    return (E.conj().T * dcf) @ E


def block_average(img: np.ndarray, upsample: int) -> np.ndarray:
    """Average (n*up, n*up, ...) sub-pixels into (n, n, ...) voxels."""
    n = img.shape[0] // upsample
    return img.reshape(n, upsample, n, upsample, *img.shape[2:]).mean(axis=(1, 3))
