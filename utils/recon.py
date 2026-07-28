"""
Reconstruction utilities: NUFFT operators, SPICE CG solver, B0 correction,
iterative reconstruction, phase correction, forward signal model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchkbnufft as tkbn
from typing import Union, Optional, Tuple
from scipy.sparse.linalg import LinearOperator as SciLin
from scipy.sparse.linalg import cg, bicgstab
from scipy.sparse.linalg import LinearOperator
from numpy.linalg import lstsq
from tqdm import tqdm

from fsl_mrs.utils.misc import FIDToSpec, SpecToFID

from .plotting import plot_voxel_spectrum_and_maps

D_TYPE       = np.complex64
FLOAT_D_TYPE = np.float32
T_D_TYPE     = torch.complex64
Trej_D_TYPE  = np.float32


# ── helpers: numpy <-> torch ──────────────────────────────────────────────────

def _to_torch(x: np.ndarray, device, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = T_D_TYPE
    t = torch.tensor(x, device=device, dtype=T_D_TYPE).unsqueeze(0).unsqueeze(0)
    return t


def _k_to_torch(x: np.ndarray, device, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = T_D_TYPE
    t = torch.tensor(x, device=device, dtype=T_D_TYPE).unsqueeze(0)
    return t


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    t = x.detach().cpu()
    shape = list(t.shape)
    while shape and shape[0] == 1:
        shape.pop(0)
        t = t.squeeze(dim=0)
    while shape and t.dim() > 0 and t.shape[0] == 1:
        t = t.squeeze(dim=0)
    return t.numpy()


def _fid_to_spec(fid):
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm="ortho"), axes=-1)


def _spec_to_fid(spec):
    return np.fft.ifft(np.fft.ifftshift(spec, axes=-1), axis=-1, norm="ortho")


# ── NUFFT linear operator (torchkbnufft) ─────────────────────────────────────

class NUFFTOp(nn.Module):
    """
    NumPy-friendly linear operator wrapper for torchkbnufft.

    I/O (NumPy):
      x_np: (..., C, *im_size) complex
      y_np: (..., C, Nsamp)   complex
    """
    def __init__(self,
                 im_size: Tuple[int, ...],
                 grid_size: Optional[Tuple[int, ...]] = None,
                 omega=None,
                 smaps=None,
                 dcf=None,
                 device=None,
                 norm: str = 'ortho',
                 nufft_ob: Optional[tkbn.KbNufft] = None,
                 adjnufft_ob: Optional[tkbn.KbNufftAdjoint] = None):
        super().__init__()
        self.im_size   = tuple(im_size)
        self.grid_size = tuple(grid_size) if grid_size is not None else None
        self.norm      = norm
        self.device    = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if omega is None:
            raise ValueError("omega (trajectory) must be provided.")
        if isinstance(omega, np.ndarray):
            omega = torch.tensor(omega, device=self.device)
        else:
            omega = omega.to(self.device)
        self.register_buffer('omega', omega)

        if smaps is not None:
            if isinstance(smaps, np.ndarray):
                smaps_t = torch.tensor(smaps).to(device=self.device)
            else:
                smaps.to(device=self.device)
                smaps_t = smaps
            self.smaps = smaps_t
        else:
            self.smaps = None

        if dcf is not None:
            dcf_t = _to_torch(dcf, self.device) if isinstance(dcf, np.ndarray) else dcf.to(self.device)
            self.register_buffer('dcf', dcf_t.real)
        else:
            self.dcf = None

        if nufft_ob is None:
            self.A = tkbn.KbNufft(im_size=self.im_size, grid_size=self.grid_size).to(self.device)
        else:
            self.A = nufft_ob.to(self.device)

        if adjnufft_ob is None:
            self.AH = tkbn.KbNufftAdjoint(im_size=self.im_size, grid_size=self.grid_size).to(self.device)
        else:
            self.AH = adjnufft_ob.to(self.device)

    def _A_torch(self, x_t: torch.Tensor) -> torch.Tensor:
        y = self.A(x_t, self.omega, smaps=self.smaps, norm=self.norm)
        return y

    def _AH_torch(self, y_t: torch.Tensor) -> torch.Tensor:
        x = self.AH(y_t, self.omega, smaps=self.smaps, norm=self.norm)
        return x

    def A_np(self, x_np: np.ndarray) -> np.ndarray:
        if isinstance(x_np, np.ndarray):
            x_t = _to_torch(x_np, self.device)
        y_t = self._A_torch(x_t)
        return _to_numpy(y_t)

    def AH_np(self, y_np: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(y_np, np.ndarray):
            if self.smaps is None:
                y_t = _to_torch(y_np, self.device)
            else:
                y_t = _k_to_torch(y_np, self.device)
        x_t = self._AH_torch(y_t)
        return _to_numpy(x_t)


# ── NUFFT linear operator (mrinufft / finufft backend) ───────────────────────

class NUFFTLinearOperator:
    """
    Wrap a mrinufft operator into a scipy-compatible LinearOperator.

    Forward  : FID image → k-space   (applies _fid_to_spec before NUFFT)
    Adjoint  : k-space → FID image   (applies _spec_to_fid after NUFFT adjoint)
    """

    def __init__(self, nufft_op, img_shape, n_samples: int, n_coils: int = 1,
                 dtype=np.complex64):
        self.nufft_op  = nufft_op
        self.img_shape = tuple(int(x) for x in img_shape)
        self.n_samples = int(n_samples)
        self.n_coils   = int(n_coils)
        self.dtype     = np.dtype(dtype)
        self.nvox      = int(np.prod(self.img_shape))
        self.range_len = n_coils * n_samples

    def op(self, x_img: np.ndarray) -> np.ndarray:
        x_spec = _fid_to_spec(x_img.reshape(self.img_shape))
        return np.asarray(self.nufft_op.op(x_spec), dtype=self.dtype)

    def adj_op(self, y_ksp: np.ndarray) -> np.ndarray:
        x_spec = self.nufft_op.adj_op(y_ksp)
        return np.asarray(_spec_to_fid(x_spec.reshape(self.img_shape)), dtype=self.dtype)

    def matvec(self, x_vec: np.ndarray) -> np.ndarray:
        return self.op(np.asarray(x_vec, dtype=self.dtype).reshape(self.img_shape)).ravel()

    def rmatvec(self, y_vec: np.ndarray) -> np.ndarray:
        y_2d = np.asarray(y_vec, dtype=self.dtype).reshape(self.n_coils, self.n_samples)
        return self.adj_op(y_2d).ravel()

    def to_scipy(self) -> SciLin:
        """Return scipy.sparse.linalg.LinearOperator (range × domain)."""
        return SciLin(shape=(self.range_len, self.nvox),
                      matvec=self.matvec, rmatvec=self.rmatvec,
                      dtype=self.dtype)

    def __repr__(self):
        return (f"NUFFTLinearOperator(img_shape={self.img_shape}, "
                f"n_coils={self.n_coils}, n_samples={self.n_samples})")


# ── NUFFT backend factory ─────────────────────────────────────────────────────

def build_nufft_ops(
    backend: str,
    trej: np.ndarray,           # (K_POINTS*N_shots, 3) float32, mrsi_ksp_scaled.T
    im_size: tuple,             # (Ny, Nx, N_SEQ)
    coil_smap_raw: np.ndarray,  # (n_coils, Ny, Nx)
    n_coils: int,
    D_TYPE=np.complex64,
    osamp: float = 2.0,
    ost:   float = 2.0,
    device: str  = None,
) -> tuple:
    """
    Returns (F_OP, Gram_OP, F1D, device_str).  Both backends share the same
    scipy LinearOperator interface and FID-domain convention.

    "torchnufft": torchkbnufft + Toeplitz Gram + FFT F1D (existing logic).
    "finufft"   : mrinufft/finufft, exact Gram = lambda x: F.rmatvec(F.matvec(x)),
                  F1D = identity (spectral FT is inside NUFFTLinearOperator).
    """
    Ny, Nx, N_SEQ = im_size
    N_VOXEL = Ny * Nx
    K_total = int(np.prod(trej.shape[:-1]))  # total k-space samples (handles 2D and 3D trej)

    if device is None:
        try:
            import torch as _torch
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    from scipy.sparse.linalg import LinearOperator as _LO

    if backend == "torchnufft":
        import torch
        import torchkbnufft as tkbn
        T_D_TYPE = torch.complex64
        dev = torch.device(device)
        print(f"[nufft/torchnufft] Building Toeplitz NUFFT  device={device} …")

        grid_size = (int(np.ceil(osamp * Ny)), int(np.ceil(osamp * Nx)), int(np.ceil(ost * N_SEQ)))
        coil_smap = np.repeat(coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1).astype(D_TYPE)
        ktraj     = torch.from_numpy(trej).permute(2, 0, 1).reshape(3, -1).to(dev)
        smap_t    = torch.from_numpy(coil_smap).to(dev, dtype=T_D_TYPE)

        tnufft_ob    = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(dev)
        tadjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(dev)
        F_tkbn = NUFFTOp(im_size=im_size, grid_size=grid_size, omega=ktraj,
                          smaps=coil_smap, norm="ortho", device=dev,
                          nufft_ob=tnufft_ob, adjnufft_ob=tadjnufft_ob)

        toep_ob = tkbn.ToepNufft()
        kernel  = tkbn.calc_toeplitz_kernel(ktraj, im_size, grid_size=grid_size, norm="ortho")

        def _toep_mv(x_np):
            x_t = torch.from_numpy(x_np.astype(D_TYPE)).reshape(1, 1, *im_size).to(dev, dtype=T_D_TYPE)
            return toep_ob(x_t, kernel, smaps=smap_t, norm="ortho").squeeze().cpu().numpy().astype(D_TYPE).ravel()

        Gram_OP = _LO((N_VOXEL * N_SEQ, N_VOXEL * N_SEQ), matvec=_toep_mv, dtype=D_TYPE)

        def _mv(x):  return F_tkbn.A_np(x.astype(D_TYPE).reshape(Ny, Nx, N_SEQ)).ravel()
        def _rmv(y): return F_tkbn.AH_np(y.astype(D_TYPE).reshape(n_coils, -1)).ravel()
        F_OP = _LO((n_coils * K_total, N_VOXEL * N_SEQ), matvec=_mv, rmatvec=_rmv, dtype=D_TYPE)

        def _fid2spec(x):
            xr = np.asarray(x).reshape(Ny, Nx, N_SEQ)
            return np.fft.fftshift(np.fft.fft(xr, axis=-1, norm='ortho'), axes=-1).ravel().astype(D_TYPE, copy=False)
        def _spec2fid(x):
            xr = np.asarray(x).reshape(Ny, Nx, N_SEQ)
            return np.fft.ifft(np.fft.ifftshift(xr, axes=-1), axis=-1, norm='ortho').ravel().astype(D_TYPE, copy=False)
        F1D = _LO((N_VOXEL * N_SEQ, N_VOXEL * N_SEQ), matvec=_fid2spec, rmatvec=_spec2fid, dtype=D_TYPE)

        return F_OP, Gram_OP, F1D, device

    elif backend == "finufft":
        import mrinufft
        print(f"[nufft/finufft] Building finufft NUFFT operator …")

        coil_smap = np.repeat(coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1).astype(D_TYPE)
        smap_time = coil_smap.squeeze(0)   # (n_coils, Ny, Nx, N_SEQ)

        NufftOpCls = mrinufft.get_operator("finufft")
        nufft_raw  = NufftOpCls(trej, shape=im_size, n_coils=n_coils, n_batchs=1,
                                 squeeze_dims=True, smaps=smap_time)
        fop        = NUFFTLinearOperator(nufft_raw, img_shape=im_size,
                                          n_samples=K_total, n_coils=n_coils, dtype=D_TYPE)
        F_OP_sci   = fop.to_scipy()

        Gram_OP = _LO(
            (N_VOXEL * N_SEQ, N_VOXEL * N_SEQ),
            matvec=lambda x: F_OP_sci.rmatvec(F_OP_sci.matvec(x.astype(D_TYPE))),
            dtype=D_TYPE,
        )
        _eye = lambda x: np.asarray(x, dtype=D_TYPE)
        F1D  = _LO((N_VOXEL * N_SEQ, N_VOXEL * N_SEQ), matvec=_eye, rmatvec=_eye, dtype=D_TYPE)

        return F_OP_sci, Gram_OP, F1D, "cpu"

    else:
        raise ValueError(f"Unknown NUFFT backend '{backend}'. Use 'torchnufft' or 'finufft'.")


def build_gram_for_worker(
    backend: str,
    im_size: tuple,
    D_TYPE=np.complex64,
    # toeplitz params:
    ktraj_np=None, grid_size=None, kernel_np=None, device_str="cpu",
    # finufft params:
    trej_np=None, coil_smap_raw_np=None, n_coils=None,
    osamp: float = 2.0, ost: float = 2.0,
):
    """
    Build (Gram_OP, F1D) inside a worker process.  Signature accepts both backends;
    only the relevant params are used.
    """
    Ny, Nx, N_SEQ = im_size
    N_VOXEL = Ny * Nx
    from scipy.sparse.linalg import LinearOperator as _LO

    if backend == "torchnufft":
        import torch
        import torchkbnufft as tkbn
        T_D_TYPE = torch.complex64

        toep_ob  = tkbn.ToepNufft().to(device_str)
        kernel_t = torch.tensor(kernel_np).to(device_str)

        def _toep_mv(x_np):
            x_t = torch.from_numpy(x_np.astype(D_TYPE)).reshape(1, 1, *im_size).to(device_str)
            return toep_ob(x_t, kernel_t, smaps=None, norm="ortho").squeeze().cpu().numpy().astype(D_TYPE).ravel()

        def _fid2spec(x):
            xr = np.asarray(x).reshape(im_size)
            return np.fft.fftshift(np.fft.fft(xr, axis=-1, norm='ortho'), axes=-1).ravel().astype(D_TYPE, copy=False)
        def _spec2fid(x):
            xr = np.asarray(x).reshape(im_size)
            return np.fft.ifft(np.fft.ifftshift(xr, axes=-1), axis=-1, norm='ortho').ravel().astype(D_TYPE, copy=False)

        n = N_VOXEL * N_SEQ
        Gram_OP = _LO((n, n), matvec=_toep_mv, rmatvec=_toep_mv, dtype=D_TYPE)
        F1D     = _LO((n, n), matvec=_fid2spec, rmatvec=_spec2fid, dtype=D_TYPE)
        return Gram_OP, F1D, None

    else:  # finufft
        import mrinufft
        coil_smap = np.repeat(coil_smap_raw_np[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1).astype(D_TYPE)
        smap_time = coil_smap.squeeze(0)
        NufftOpCls = mrinufft.get_operator("finufft")
        nufft_raw  = NufftOpCls(trej_np, shape=im_size, n_coils=n_coils, n_batchs=1,
                                 squeeze_dims=True, smaps=smap_time)
        fop   = NUFFTLinearOperator(nufft_raw, img_shape=im_size,
                                     n_samples=int(np.prod(trej_np.shape[:-1])), n_coils=n_coils, dtype=D_TYPE)
        F_loc = fop.to_scipy()
        return None, None, F_loc  # caller uses F_loc.rmatvec/matvec directly; no Gram needed


# ── B0 modulation matrices ────────────────────────────────────────────────────

def Calc_B0_matrix(df_high, taxis):
    """
    Vectorized B0 modulation matrix.

    df_high : (H,W) high-res Δf map [Hz]
    taxis   : (T,) time points [s]

    Returns
    -------
    B : (H*W, T) complex ndarray, each row = exp(-i 2π Δf[r] t)
    """
    H, W = df_high.shape
    df_flat = df_high.reshape(-1, 1)
    t = taxis.reshape(1, -1)
    B = np.exp(-1j * 2 * np.pi * df_flat * t)
    return B.astype(D_TYPE)


def Calc_B0_matrix_mx(df_high, taxis):
    """Alias for Calc_B0_matrix (identical implementation)."""
    H, W = df_high.shape
    df_flat = df_high.reshape(-1, 1)
    t = taxis.reshape(1, -1)
    B = np.exp(-1j * 2 * np.pi * df_flat * t)
    return B.astype(D_TYPE)


# ── B0 polynomial interpolation ───────────────────────────────────────────────

def fit_polynomial_patch(df_low, high_shape, order=2):
    """
    Fit local polynomial around each voxel from low-res Δf map.

    Parameters
    ----------
    df_low : (Nx_low, Ny_low) ndarray, low-res Δf map [Hz]
    high_shape : (Nx_high, Ny_high) desired high-res output shape
    order : polynomial order (1=linear, 2=quadratic)

    Returns
    -------
    coeffs : (Nx_low, Ny_low, n_terms) ndarray
    terms : list of (i,j) monomial exponents
    patch_size : int
    """
    Nx_low, Ny_low = df_low.shape
    Nx_high, Ny_high = high_shape

    srf_x = Nx_high // Nx_low
    srf_y = Ny_high // Ny_low
    srf = max(srf_x, srf_y)

    patch_size = 2 * srf + 1
    half = patch_size // 2

    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append((i, j))
    n_terms = len(terms)

    coeffs = np.zeros((Nx_low, Ny_low, n_terms))

    for ix in range(Nx_low):
        for iy in range(Ny_low):
            x0, y0 = ix, iy
            xs = np.arange(ix - half, ix + half + 1)
            ys = np.arange(iy - half, iy + half + 1)
            xs = xs[(xs >= 0) & (xs < Nx_low)]
            ys = ys[(ys >= 0) & (ys < Ny_low)]
            X, Y = np.meshgrid(xs, ys, indexing='ij')
            vals = df_low[X, Y].ravel()

            dx = (X.ravel() - x0)
            dy = (Y.ravel() - y0)

            A = np.stack([(dx ** i) * (dy ** j) for (i, j) in terms], axis=1)
            c, _, _, _ = lstsq(A, vals, rcond=None)
            coeffs[ix, iy] = c

    return coeffs, terms, patch_size


def eval_polynomial_to_highres_b0(coeffs, terms, low_shape, high_shape):
    """
    Evaluate fitted local polynomial (Eq. S15) on a high-res grid.

    Parameters
    ----------
    coeffs : (Nx_low, Ny_low, n_terms) ndarray
    terms : list of (i,j) monomial exponents
    low_shape : (Nx_low, Ny_low)
    high_shape : (Nx_high, Ny_high)

    Returns
    -------
    df_high_img : (Nx_high, Ny_high) ndarray
    """
    import matplotlib.pyplot as plt

    Nx_low, Ny_low = low_shape
    Nx_high, Ny_high = high_shape
    assert Nx_high % Nx_low == 0 and Ny_high % Ny_low == 0, \
        "high_shape must be integer multiples of low_shape"

    srf_x = Nx_high // Nx_low
    srf_y = Ny_high // Ny_low

    ax_x = np.linspace(-0.5, 0.5, srf_x, endpoint=False) + 0.5 / srf_x
    ax_y = np.linspace(-0.5, 0.5, srf_y, endpoint=False) + 0.5 / srf_y
    dx_grid, dy_grid = np.meshgrid(ax_x, ax_y, indexing='ij')

    df_high_img = np.zeros((Nx_high, Ny_high), dtype=np.float32)

    for ix in range(Nx_low):
        for iy in range(Ny_low):
            c = coeffs[ix, iy]
            block = sum(c[k] * (dx_grid ** terms[k][0]) * (dy_grid ** terms[k][1])
                        for k in range(len(terms)))
            df_high_img[ix * srf_x:(ix + 1) * srf_x, iy * srf_y:(iy + 1) * srf_y] = block

    plt.figure(figsize=(6, 5))
    plt.imshow(df_high_img, cmap='jet', origin='lower')
    plt.colorbar(label='Δf (Hz)')
    plt.title('estimated high resolution B0 Map')
    plt.xlabel('x (voxels)')
    plt.ylabel('y (voxels)')
    plt.show()

    return df_high_img


# ── Forward signal model ──────────────────────────────────────────────────────

def Sig_func_Multi_Peak_2D(bm_list,
                           lw_list,
                           Cm,
                           time_axis,
                           n_voxels,
                           N_SEQ_POINTS,
                           freq_shift: Optional[Union[list, np.ndarray, int]] = None,
                           whole_shift: Optional[Union[list, np.ndarray, int]] = None,
                           phi0_shift: Optional[Union[list, np.ndarray, int]] = None,
                           phi1_shift: Optional[Union[list, np.ndarray, int]] = None,
                           freq_axis: Optional[Union[list, np.ndarray, int]] = None):
    """Multi-peak 2D MRSI forward signal model."""
    taxis = np.array(time_axis).reshape(N_SEQ_POINTS, 1)
    min_exp = -50
    max_exp = 50

    n_peaks = len(bm_list)
    fids = []

    if isinstance(lw_list, np.ndarray):
        lw_list = [lw_list[i, :] if lw_list.ndim == 2 else lw_list[i] for i in range(n_peaks)]
    if isinstance(Cm, np.ndarray):
        Cm = [Cm[i, :] if Cm.ndim == 2 else Cm[i] for i in range(n_peaks)]
    if isinstance(freq_shift, np.ndarray):
        freq_shift = [freq_shift[i, :] if freq_shift.ndim == 2 else freq_shift[i] for i in range(n_peaks)]
    if isinstance(phi1_shift, np.ndarray):
        phi1_shift = [phi1_shift[i, :] if phi1_shift.ndim == 2 else phi1_shift[i] for i in range(n_peaks)]

    for i in range(n_peaks):
        this_lw = lw_list[i]
        this_cm = Cm[i]

        if np.isscalar(this_lw):
            this_lw = np.ones(n_voxels) * this_lw
        else:
            this_lw = np.asarray(this_lw)
            assert this_lw.shape == (n_voxels,), f"lw[{i}] shape {this_lw.shape} != ({n_voxels},)"

        if np.isscalar(this_cm):
            this_cm = np.ones(n_voxels) * this_cm
        else:
            this_cm = np.asarray(this_cm)
            assert this_cm.shape == (n_voxels,), f"Cm[{i}] shape {this_cm.shape} != ({n_voxels},)"

        if freq_shift is not None:
            this_freq_shift = freq_shift[i]
            if np.isscalar(this_freq_shift):
                this_freq_shift = np.ones(n_voxels) * this_freq_shift
            else:
                this_freq_shift = np.asarray(this_freq_shift)
                assert this_freq_shift.shape == (n_voxels,), f"freq_shift[{i}] shape {this_freq_shift.shape} != ({n_voxels},)"

        if phi1_shift is not None:
            this_phi1_shift = phi1_shift[i]
            if np.isscalar(this_phi1_shift):
                this_phi1_shift = np.ones(n_voxels) * this_phi1_shift
            else:
                this_phi1_shift = np.asarray(this_phi1_shift)
                assert this_phi1_shift.shape == (n_voxels,), f"ph1_shift[{i}] shape {this_phi1_shift.shape} != ({n_voxels},)"

        if not np.all((min_exp <= this_lw) & (this_lw <= max_exp)):
            raise ValueError(f"lw must be in range [{min_exp}, {max_exp}], got {this_lw}.")

        lw_array = this_lw[np.newaxis, :]
        broadening = np.exp(-lw_array * 2 * np.pi * taxis)

        fid = bm_list[i].reshape(N_SEQ_POINTS, 1) * broadening
        fid = fid * this_cm

        if freq_shift is not None:
            freq_shift_array = this_freq_shift[np.newaxis, :]
            taxis_array = time_axis[:, np.newaxis]
            fid *= np.exp(-1j * freq_shift_array * taxis_array * np.pi * 2)

        if phi1_shift is not None:
            spec = FIDToSpec(fid, axis=0)
            phi1_shift_array = this_phi1_shift[np.newaxis, :]
            freq_axis_array = freq_axis[:, np.newaxis]
            spec *= np.exp(-1j * phi1_shift_array * freq_axis_array * np.pi * 2)
            fid = SpecToFID(spec, axis=0)

        fids.append(fid)

    total = np.sum(fids, axis=0).T

    if whole_shift is not None:
        if np.isscalar(whole_shift):
            whole_shift = np.ones(n_voxels) * whole_shift
        else:
            whole_shift = np.asarray(whole_shift)
            assert whole_shift.shape == (n_voxels,), f"whole_shift shape {whole_shift.shape} != ({n_voxels},)"
        taxis_array = time_axis[:, np.newaxis]
        whole_shift_array = whole_shift[:, np.newaxis]
        total *= np.exp(-1j * whole_shift_array * taxis_array.T * np.pi * 2)

    if phi0_shift is not None:
        if np.isscalar(phi0_shift):
            phi0_shift = np.ones(n_voxels) * phi0_shift
        else:
            phi0_shift = np.asarray(phi0_shift)
            assert phi0_shift.shape == (n_voxels,), f"phi0_shift shape {phi0_shift.shape} != ({n_voxels},)"
        phi0_shift_array = phi0_shift[:, np.newaxis]
        spec_total = FIDToSpec(total, axis=0)
        spec_total *= np.exp(-1j * np.pi * phi0_shift_array)
        total = SpecToFID(spec_total, axis=0)

    return total


# ── Phase correction ──────────────────────────────────────────────────────────

def phase_corr(
    mrsi_fid: np.ndarray,
    mag_map_2d: np.ndarray,
    brain_mask: np.ndarray,
    TS: float,
    img_shape: tuple,
    out_dir: str,
    ppmlim: tuple = (0, 5),
    ref_img=None,
    out_fname: str = "phase_corr_tmp",
    method: str = 'phasta',
    affine=None,
) -> np.ndarray:
    """
    Apply FSL-MRS 0th-order phase correction to MRSI FID data.

    Parameters
    ----------
    mrsi_fid   : FID ndarray, shape (Ny, Nx, npts) or (Ny*Nx, npts)
    mag_map_2d : magnitude map, shape (Ny, Nx)
    brain_mask : boolean brain mask, shape (Ny, Nx)
    TS         : dwell time in seconds
    img_shape  : (Ny, Nx) spatial dimensions
    out_dir    : directory for intermediate NIfTI files
    ppmlim     : ppm window for phase alignment (default (0,5))
    ref_img    : fsl.data.image.Image with .voxToWorldMat (optional)
    out_fname  : stem of the temporary output .nii.gz
    method     : phase correction method

    Returns
    -------
    mrsi_phcorr_f : phase-corrected FID, shape (Ny, Nx, npts)
    """
    from fsl_mrs.utils.preproc.mrsi import mrsi_phase_corr as _mrsi_phase_corr
    import fsl_mrs.utils.preproc.mrsi as _mrsi_mod
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    from fsl.data.image import Image

    # fsl_mrs phase_corr_max_real expects a 2-D array (N_fids, Nt) but
    # iterate_over_spatial() passes a 1-D FID (Nt,) for each voxel, which
    # causes a ValueError when the returned phase array is assigned back.
    # Patch the module-level function so it reshapes 1-D input to (1, Nt)
    # and unwraps the result, keeping phase correction valid per voxel.
    _orig_pcmr = _mrsi_mod.phase_corr_max_real

    def _patched_pcmr(fids, dwelltime, limits=None, apodization=0):
        fids = np.asarray(fids)
        if fids.ndim == 1:
            phased, phases = _orig_pcmr(
                fids[np.newaxis, :], dwelltime,
                limits=limits, apodization=apodization)
            return phased[0], phases[0]
        return _orig_pcmr(fids, dwelltime, limits=limits, apodization=apodization)

    _mrsi_mod.phase_corr_max_real = _patched_pcmr

    Ny, Nx = img_shape[0], img_shape[1]

    brain_mask_img = Image(np.ascontiguousarray((mag_map_2d * brain_mask).T[::-1, :]))

    mrsi_f = mrsi_fid.reshape(Ny, Nx, -1)
    mrsi_f_4 = np.ascontiguousarray(mrsi_f.transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :]

    _affine = ref_img.voxToWorldMat if ref_img is not None else affine
    if _affine is None:
        nifti_in = gen_nifti_mrs(mrsi_f_4, dwelltime=TS, spec_freq=297.219)
    else:
        nifti_in = gen_nifti_mrs(mrsi_f_4, dwelltime=TS, spec_freq=297.219, affine=_affine)

    out_path = os.path.join(out_dir, out_fname + ".nii.gz")
    kwargs = dict(mask=brain_mask_img, ppmlim=list(ppmlim), method=method)
    if method is not None:
        kwargs["method"] = method
    phcorr_nifti, _ = _mrsi_phase_corr(nifti_in, **kwargs)
    phcorr_nifti.save(out_path)

    phcorr_img = Image(out_path)
    mrsi_phcorr_f = np.ascontiguousarray(np.transpose(np.array(phcorr_img[:, :, 0, :]), (1, 0, 2)))[:, ::-1, :]

    return mrsi_phcorr_f.conj()


# ── Iterative NUFFT reconstruction ───────────────────────────────────────────

def iterative_nufft_recon(
    kspace: np.ndarray,
    image_shape: Tuple[int, int, int],
    B0_mat: Union[np.ndarray, SciLin],
    F_OP: Union[np.ndarray, SciLin],
    Gram_OP: Union[np.ndarray, SciLin],
    F1D_OP: Union[np.ndarray, SciLin],
    n_coils: int = 1,
    smaps: Optional[np.ndarray] = None,
    density_method: Optional[str] = None,
    traj: Optional[np.ndarray] = None,
    lam: float = 0.0,
    maxiter: int = 30,
    rtol: float = 1e-3,
    solver: str = "cg",
) -> Tuple[np.ndarray, dict]:
    """
    Iterative NUFFT reconstruction solving (A^H A + lam I) x = A^H y.

    Returns (image_recon, diagnostics, b_img)
    """
    from mrinufft.density import get_density

    if isinstance(density_method, str):
        dfunc = get_density(density_method)
        try:
            density = dfunc(traj, shape=image_shape)
        except Exception:
            density = dfunc(traj)
        density = np.asarray(density, dtype=Trej_D_TYPE)
        density = density * (density.size / np.sum(density))
    elif density_method is None:
        density = None
    else:
        density = np.asarray(density_method)

    img_npix = image_shape[0] * image_shape[1] * image_shape[2]

    def matvec_norm(x_vec):
        x_img = x_vec.reshape((-1, image_shape[-1]))
        AA = B0_mat * x_img
        _tmp = F1D_OP.matvec(AA.ravel())
        _tmp = Gram_OP.matvec(_tmp)
        _tmp = F1D_OP.rmatvec(_tmp)
        BB = B0_mat.conj() * _tmp.reshape((-1, image_shape[-1]))
        return np.asarray(BB).ravel().astype(D_TYPE)

    yk = kspace.copy().astype(D_TYPE)
    if density is not None:
        yk = yk * density.reshape(1, -1)

    _adj_y = F1D_OP.rmatvec(F_OP.rmatvec(yk.ravel()))
    b_img  = (B0_mat.conj() * _adj_y.reshape((-1, image_shape[-1]))).ravel().astype(D_TYPE)

    LinOp = LinearOperator((img_npix, img_npix), matvec=matvec_norm, dtype=D_TYPE)

    x0_img = F1D_OP.rmatvec(F_OP.rmatvec(kspace.ravel()))
    x0 = x0_img.reshape(-1).astype(D_TYPE)

    pbar = tqdm(total=maxiter)

    def _cb(xk):
        pbar.update(1)

    if solver.lower() == "cg":
        x_flat, info = cg(LinOp, b_img, x0=x0, maxiter=maxiter, rtol=rtol, callback=_cb)
    else:
        x_flat, info = bicgstab(LinOp, b_img, x0=x0, maxiter=maxiter, rtol=rtol, callback=_cb)

    pbar.close()

    x_rec = x_flat.reshape(image_shape)

    diagnostics = {
        "info": info,
        "density_used": density is not None,
        "lambda": lam,
        "n_iter": maxiter if info > 0 else "converged_or_done",
    }
    print("cg_info:", info)

    return x_rec, diagnostics, b_img.reshape((-1, image_shape[-1]))


# ── SPICE solvers ─────────────────────────────────────────────────────────────

def SPICEWithSpatialConstrain_cg_finufft(
    noisy_kt_spaces: np.ndarray,
    img_shape: tuple,
    F: SciLin,
    B0_mat: np.ndarray,
    V: np.ndarray,
    N_Vox: int,
    NUM_SPICE_RANK: int,
    WW: np.ndarray,
    Solver: str = "cg",
    lamda_1: float = 1e-4,
    maxiter: int = 120,
    rtol: float = 1e-3,
    save_iter_every: int = 0,
    save_folder: str = "./spice_iters",
    x0: Optional[np.ndarray] = None,
) -> tuple:
    """
    SPICE CG solver using finufft backend (no Toeplitz — uses F.H @ F directly).

    Returns (spice_est, U, info_dict).
    """
    if save_iter_every > 0:
        os.makedirs(save_folder, exist_ok=True)
        np.save(os.path.join(save_folder, "Basis_V.npy"), V)

    FH = F.H

    D = FIDToSpec(
        B0_mat.conj() * SpecToFID(
            (FH @ noisy_kt_spaces.ravel()).reshape(-1, V.shape[0]),
            axis=-1,
        ),
        axis=-1,
    )
    b_flat = (D @ V).ravel().astype(D_TYPE)

    if x0 is None:
        x0 = (FH @ noisy_kt_spaces.ravel()).reshape(-1, V.shape[0])
        x0 = (x0 @ np.linalg.pinv(V.conj().T)).ravel().astype(D_TYPE)
    else:
        x0 = np.asarray(x0, dtype=D_TYPE).ravel()

    def mv(x_vec: np.ndarray) -> np.ndarray:
        X  = x_vec.reshape(N_Vox, NUM_SPICE_RANK)
        AA = FIDToSpec(B0_mat * SpecToFID((X @ V.conj().T), axis=-1), axis=-1)
        BB = FIDToSpec(
            B0_mat.conj() * SpecToFID(
                (FH @ (F @ AA.ravel())).reshape(-1, V.shape[0]),
                axis=-1,
            ),
            axis=-1,
        )
        CC = (BB.reshape(-1, V.shape[0]) @ V).ravel()
        DD = lamda_1 * (WW @ X).ravel()
        return (CC + DD).astype(D_TYPE)

    A = SciLin(shape=(N_Vox * NUM_SPICE_RANK, N_Vox * NUM_SPICE_RANK),
               matvec=mv, dtype=D_TYPE)

    norm_b        = np.linalg.norm(b_flat) + 1e-16
    best_x        = x0.copy()
    best_rel_res  = np.linalg.norm(A.matvec(x0) - b_flat) / norm_b
    x_prev        = x0.copy()
    iter_count    = [0]
    no_improve    = [0]
    small_dx      = [0]

    pbar = tqdm(total=maxiter, desc="SPICE-finufft CG", unit="iter")

    def _cb(xk):
        iter_count[0] += 1
        pbar.update(1)
        nonlocal best_x, best_rel_res
        Ax      = A.matvec(xk)
        rel_res = np.linalg.norm(Ax - b_flat) / norm_b
        rel_dx  = np.linalg.norm(xk - x_prev) / (np.linalg.norm(xk) + 1e-16)
        if best_rel_res - rel_res > max(5e-2 * best_rel_res, 1e-4):
            best_rel_res = rel_res
            best_x       = xk.copy()
            no_improve[0] = 0
        else:
            no_improve[0] += 1
        small_dx[0] = small_dx[0] + 1 if rel_dx < 1e-6 else 0
        pbar.set_postfix(rel_res=f"{rel_res:.3e}", best=f"{best_rel_res:.3e}")
        if save_iter_every and (iter_count[0] % save_iter_every == 0):
            np.save(os.path.join(save_folder, f"U_iter_{iter_count[0]:06d}.npy"), xk)
        if iter_count[0] >= 2 and (no_improve[0] >= 4 or small_dx[0] >= 3):
            raise StopIteration("early stop")
        x_prev[:] = xk

    try:
        X_flat, info = cg(A, b_flat, x0=x0, rtol=rtol, maxiter=maxiter, callback=_cb)
    except StopIteration:
        X_flat = best_x
        info   = 0
        pbar.write(f"[early-stop] best residual at iter {iter_count[0]}: {best_rel_res:.3e}")
    finally:
        pbar.close()

    U         = X_flat.reshape(N_Vox, NUM_SPICE_RANK)
    spice_est = U @ V.conj().T

    info_dict = {
        "iterations": iter_count[0], "cg_info": info,
        "best_rel_res": best_rel_res,
        "early_stopped": (no_improve[0] >= 4 or small_dx[0] >= 3),
    }
    print(f"[SPICE-finufft] Done  iters={iter_count[0]}  best_rel_res={best_rel_res:.3e}")
    return spice_est, U, info_dict


def SPICEWithSpatialConstrain_cg_nufft(
    noisy_kt_spaces: np.ndarray,
    img_shape: Optional[tuple],
    F: Union[np.ndarray, SciLin],
    Gram_OP: Union[np.ndarray, SciLin],
    F1D_OP: Union[np.ndarray, SciLin],
    B0_mat: Union[np.ndarray, SciLin],
    V: np.ndarray,
    N_Vox: int,
    NUM_SPICE_RANK: int,
    WW: np.ndarray,
    Solver: str = "cg",
    lamda_1: float = 15,
    maxiter: Optional[int] = 120,
    save_iter_every: int = 10,
    save_folder: str = "./saved_iters",
    rtol: float = 1e-3,
    x0: Optional[np.ndarray] = None,
    brain_mask_inner: Optional[np.ndarray] = None,
    PPM_AXIS: Optional[np.ndarray] = None,
    dx_tol: float = 1e-6,
    patience: int = 4,
    patience_dx: int = 3,
) -> tuple:
    """
    SPICE CG solver with Toeplitz NUFFT and internal early stopping.

    Returns (spice_est, U, info_dict).
    """
    os.makedirs(save_folder, exist_ok=True)
    lamda_1 = float(lamda_1)

    np.save(os.path.join(save_folder, 'Basis_V.npy'), V)

    if Solver.lower() not in ("cg", "conjugate gradient", "conjugate grandient"):
        raise ValueError("Only CG solver is currently supported in this wrapper.")

    if x0 is None:
        print('[SPICE] Running iterative NUFFT for initial guess (5 iters)…')
        recon_nufft, _, b_init = iterative_nufft_recon(
            kspace=noisy_kt_spaces,
            B0_mat=B0_mat,
            F_OP=F,
            Gram_OP=Gram_OP,
            F1D_OP=F1D_OP,
            image_shape=img_shape,
            n_coils=32,
            smaps=None,
            density_method=None,
            maxiter=5,
            solver="cg",
        )
        U_init = recon_nufft.reshape(img_shape) @ np.linalg.pinv(V.conj().T)
        x0     = U_init.ravel().astype(D_TYPE)
        print('[SPICE] Iterative NUFFT init done.')
        _, fig_init, _ = plot_voxel_spectrum_and_maps(
            FIDToSpec(U_init @ V.conj().T, axis=-1), img_shape,
            32, 32, 0, 0, brain_mask_inner, PPM_AXIS, show=False,
        )
        fig_init.savefig(os.path.join(save_folder, "init_guess.png"), dpi=120)
        import matplotlib.pyplot as _plt
        _plt.close(fig_init)
    else:
        adjoint_y = F1D_OP.rmatvec(F.rmatvec(noisy_kt_spaces.ravel())).reshape(-1, img_shape[-1])
        b_init    = (B0_mat.conj() * adjoint_y).astype(D_TYPE)
        x0        = np.asarray(x0, dtype=D_TYPE).ravel()
        print('[SPICE] Using provided x0.')

    b_flat = (b_init @ V).ravel().astype(D_TYPE)

    def mv(x_vec: np.ndarray) -> np.ndarray:
        X  = x_vec.reshape(N_Vox, NUM_SPICE_RANK)
        AA = B0_mat * (X @ V.conj().T)
        BB = (B0_mat.conj() * (Gram_OP @ AA.ravel()).reshape(-1, V.shape[0]))
        CC = (BB.reshape(-1, V.shape[0]) @ V).ravel()
        DD = lamda_1 * (WW @ X).ravel()
        return (CC + DD).astype(D_TYPE)

    A = SciLin(shape=(N_Vox * NUM_SPICE_RANK, N_Vox * NUM_SPICE_RANK), matvec=mv, dtype=D_TYPE)

    iter_count = 0
    pbar = tqdm(total=maxiter, desc="CG iters", unit="iter")

    best_x = x0.copy()
    Ax0    = A.matvec(x0)
    r0     = Ax0 - b_flat
    norm_b = np.linalg.norm(b_flat) + 1e-16
    best_rel_res = np.linalg.norm(r0) / norm_b
    best_iter    = 0

    x_prev          = x0.copy()
    count_no_improve = 0
    count_small_dx   = 0
    iter_count       = 0

    def cg_callback_internal(xk):
        nonlocal iter_count, best_x, best_rel_res, best_iter, x_prev
        nonlocal count_no_improve, count_small_dx

        iter_count += 1
        pbar.update(1)

        Ax     = A.matvec(xk)
        r      = Ax - b_flat
        rel_res = np.linalg.norm(r) / norm_b
        rel_dx  = np.linalg.norm(xk - x_prev) / (1e-16 + np.linalg.norm(xk))

        if save_iter_every and (iter_count % save_iter_every == 0):
            np.save(os.path.join(save_folder, f"U_iter_{iter_count:06d}.npy"), xk)

        rel_tol = 5e-2
        abs_tol = 5e-4

        delta = best_rel_res - rel_res
        required_decrease = max(rel_tol * best_rel_res, abs_tol)

        if delta > required_decrease:
            best_rel_res    = rel_res
            best_x          = xk.copy()
            best_iter       = iter_count
            count_no_improve = 0
        else:
            count_no_improve += 1

        if rel_dx < dx_tol:
            count_small_dx += 1
        else:
            count_small_dx = 0

        pbar.set_postfix({"rel_res": f"{rel_res:.3e}", "rel_dx": f"{rel_dx:.3e}",
                          "best_res": f"{best_rel_res:.3e}", "cnt_no_improve": f"{count_no_improve},"})

        if (iter_count >= 2) and (count_no_improve >= patience or count_small_dx >= patience_dx):
            pbar.write(f"[earlystop_internal] iter {iter_count}: count_no_improve={count_no_improve}, "
                       f"count_small_dx={count_small_dx}, best_rel_res={best_rel_res:.3e}")
            raise StopIteration("early stop (internal metrics)")

        x_prev[:] = xk

    try:
        X_flat, info = cg(A, b_flat, x0=x0, rtol=rtol, maxiter=maxiter, callback=cg_callback_internal)
    except StopIteration:
        X_flat = best_x
        info   = 0
        pbar.write(f"CG stopped early; using best iterate from iter {best_iter}")
    except Exception as e:
        pbar.write(f"CG raised exception: {e}")
        pbar.close()
        raise
    finally:
        pbar.close()

    U         = X_flat.reshape(N_Vox, NUM_SPICE_RANK)
    spice_est = U @ V.conj().T

    print('[SPICE] Done, saving result plot...')
    _, fig_result, _ = plot_voxel_spectrum_and_maps(
        FIDToSpec(spice_est, axis=-1), img_shape,
        32, 32, 0, 0, brain_mask_inner, PPM_AXIS, show=False,
    )
    fig_result.savefig(os.path.join(save_folder, "spice_result.png"), dpi=120)
    import matplotlib.pyplot as _plt
    _plt.close(fig_result)

    np.save(os.path.join(save_folder, "U_final.npy"), U)

    info_dict = {
        "iterations": iter_count,
        "cg_info": info,
        "earlystop_used": (count_no_improve >= patience or count_small_dx >= patience_dx)
    }

    print("===== CG Solver Report =====")
    print("iterations:", iter_count)
    print("cg_info:", info)
    print("earlystop_used:", info_dict["earlystop_used"])

    return spice_est, U, info_dict


def SPICEWithSpatialConstrain_cg_nufft_joint(
    noisy_kt_spaces: np.ndarray,
    img_shape: Optional[tuple],
    F: Union[np.ndarray, SciLin],
    Gram_OP: Union[np.ndarray, SciLin],
    F1D_OP: Union[np.ndarray, SciLin],
    B0_mat: Union[np.ndarray, SciLin],
    V: np.ndarray,
    N_Vox: int,
    rank_blocks: list,
    WW_blocks: list,
    lamda_blocks: list,
    maxiter: Optional[int] = 120,
    save_folder: str = "./saved_iters",
    rtol: float = 1e-3,
    x0: Optional[np.ndarray] = None,
    brain_mask_inner: Optional[np.ndarray] = None,
    PPM_AXIS: Optional[np.ndarray] = None,
    patience: int = 4,
    patience_dx: int = 3,
    dx_tol: float = 1e-6,
) -> tuple:
    """
    SPICE CG solver supporting multiple rank blocks with per-block spatial regularization.

    rank_blocks, WW_blocks, lamda_blocks are parallel lists — one entry per component
    (e.g. [R_lipid, R_metab], [WW_lip, WW_met], [lam_lip, lam_met]).

    Returns (spice_est, U, info_dict).
    """
    os.makedirs(save_folder, exist_ok=True)
    NUM_RANK = sum(rank_blocks)

    np.save(os.path.join(save_folder, 'Basis_V.npy'), V)

    if x0 is None:
        print('[SPICE-joint] Running iterative NUFFT for initial guess (5 iters)…')
        recon_nufft, _, b_init = iterative_nufft_recon(
            kspace=noisy_kt_spaces,
            B0_mat=B0_mat,
            F_OP=F,
            Gram_OP=Gram_OP,
            F1D_OP=F1D_OP,
            image_shape=img_shape,
            n_coils=32,
            smaps=None,
            density_method=None,
            maxiter=5,
            solver="cg",
        )
        U_init = recon_nufft.reshape(img_shape) @ np.linalg.pinv(V.conj().T)
        x0     = U_init.ravel().astype(D_TYPE)
        print('[SPICE-joint] Iterative NUFFT init done.')
    else:
        adjoint_y = F1D_OP.rmatvec(F.rmatvec(noisy_kt_spaces.ravel())).reshape(-1, img_shape[-1])
        b_init    = (B0_mat.conj() * adjoint_y).astype(D_TYPE)
        x0        = np.asarray(x0, dtype=D_TYPE).ravel()
        print('[SPICE-joint] Using provided x0.')

    b_flat = (b_init @ V).ravel().astype(D_TYPE)

    def mv(x_vec: np.ndarray) -> np.ndarray:
        X  = x_vec.reshape(N_Vox, NUM_RANK)
        AA = B0_mat * (X @ V.conj().T)
        BB = (B0_mat.conj() * (Gram_OP @ AA.ravel()).reshape(-1, V.shape[0]))
        CC = (BB.reshape(-1, V.shape[0]) @ V).ravel()
        DD = np.zeros((N_Vox, NUM_RANK), dtype=D_TYPE)
        offset = 0
        for rank, WW_b, lam in zip(rank_blocks, WW_blocks, lamda_blocks):
            DD[:, offset:offset + rank] = lam * (WW_b @ X[:, offset:offset + rank])
            offset += rank
        return (CC + DD.ravel()).astype(D_TYPE)

    A = SciLin(shape=(N_Vox * NUM_RANK, N_Vox * NUM_RANK), matvec=mv, dtype=D_TYPE)

    iter_count       = 0
    pbar             = tqdm(total=maxiter, desc="CG iters", unit="iter")

    best_x           = x0.copy()
    Ax0              = A.matvec(x0)
    r0               = Ax0 - b_flat
    norm_b           = np.linalg.norm(b_flat) + 1e-16
    best_rel_res     = np.linalg.norm(r0) / norm_b
    best_iter        = 0
    x_prev           = x0.copy()
    count_no_improve = 0
    count_small_dx   = 0

    def cg_callback_internal(xk):
        nonlocal iter_count, best_x, best_rel_res, best_iter, x_prev
        nonlocal count_no_improve, count_small_dx

        iter_count += 1
        pbar.update(1)

        Ax      = A.matvec(xk)
        r       = Ax - b_flat
        rel_res = np.linalg.norm(r) / norm_b
        rel_dx  = np.linalg.norm(xk - x_prev) / (1e-16 + np.linalg.norm(xk))

        rel_tol           = 5e-2
        abs_tol           = 5e-4
        delta             = best_rel_res - rel_res
        required_decrease = max(rel_tol * best_rel_res, abs_tol)

        if delta > required_decrease:
            best_rel_res     = rel_res
            best_x           = xk.copy()
            best_iter        = iter_count
            count_no_improve = 0
        else:
            count_no_improve += 1

        if rel_dx < dx_tol:
            count_small_dx += 1
        else:
            count_small_dx = 0

        pbar.set_postfix({"rel_res": f"{rel_res:.3e}", "rel_dx": f"{rel_dx:.3e}",
                          "best_res": f"{best_rel_res:.3e}", "cnt_no_improve": f"{count_no_improve},"})

        if (iter_count >= 2) and (count_no_improve >= patience or count_small_dx >= patience_dx):
            pbar.write(f"[earlystop] iter {iter_count}: count_no_improve={count_no_improve}, "
                       f"count_small_dx={count_small_dx}, best_rel_res={best_rel_res:.3e}")
            raise StopIteration("early stop (internal metrics)")

        x_prev[:] = xk

    try:
        X_flat, info = cg(A, b_flat, x0=x0, rtol=rtol, maxiter=maxiter, callback=cg_callback_internal)
    except StopIteration:
        X_flat = best_x
        info   = 0
        pbar.write(f"CG stopped early; using best iterate from iter {best_iter}")
    except Exception as e:
        pbar.write(f"CG raised exception: {e}")
        pbar.close()
        raise
    finally:
        pbar.close()

    U         = X_flat.reshape(N_Vox, NUM_RANK)
    spice_est = U @ V.conj().T

    np.save(os.path.join(save_folder, "U_final.npy"), U)

    info_dict = {
        "iterations":     iter_count,
        "cg_info":        info,
        "earlystop_used": (count_no_improve >= patience or count_small_dx >= patience_dx),
    }

    print("===== CG Solver Report (joint) =====")
    print("iterations:", iter_count)
    print("cg_info:", info)
    print("earlystop_used:", info_dict["earlystop_used"])

    return spice_est, U, info_dict
