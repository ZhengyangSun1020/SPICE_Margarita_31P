#!/usr/bin/env python3
"""
31P CONCEPT CRT SPICE reconstruction with spatial regularization.

Adapted from 04_run_spice.py (1H MRSI) for 31P CONCEPT CRT data.

Key differences from 1H pipeline:
  - Single coil (no sensitivity maps needed)
  - 2D CRT trajectory (NUFFT per spectral point, not 3D)
  - No B0 correction (set to identity)
  - No lipid removal
  - 31P metabolite basis (PCr, Pi, ATP)
  - Spatial prior from adjoint recon magnitude map (no water reference)

Usage:
    python 05c_spice_recon_31p.py --dat "path/to/meas_*.dat" --out-dir ./output
    python 05c_spice_recon_31p.py --dat "path/to/meas_*.dat" --rank 8 --lambda1 1e-2
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mrinufft
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

import importlib.util as _ilu, pathlib as _pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_concept_twix_31p import concept_phase_correction

# Import loader and trajectory builder from 06c (correct multi-ADC handling)
_spec = _ilu.spec_from_file_location(
    "_06c", _pl.Path(__file__).parent / "06c_iterative_recon_31p_3d.py")
_06c = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_06c)
load_concept_twix_3d = _06c.load_concept_twix_3d


# ── FFT helpers (replacing fsl_mrs) ─────────────────────────────────────

def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


# ── Trajectory & DCF ─────────────────────────────────────────────────────

def build_trajectory_2d(first_circ, n_rings, traj_pts):
    """Build 2D CRT trajectory max-normalized to [-0.5, 0.5] (mri-nufft convention)."""
    radii = np.sqrt(first_circ[0]**2 + first_circ[1]**2)
    phi0 = np.arctan2(first_circ[1], first_circ[0])
    all_kx, all_ky = [], []
    for r in range(n_rings):
        angles = 2 * np.pi * np.arange(traj_pts) / traj_pts + phi0[r]
        all_kx.append(radii[r] * np.cos(angles))
        all_ky.append(radii[r] * np.sin(angles))
    kx_raw = np.concatenate(all_kx)
    ky_raw = np.concatenate(all_ky)
    max_radius = np.sqrt(kx_raw**2 + ky_raw**2).max()
    kx = kx_raw * 0.5 / max_radius
    ky = ky_raw * 0.5 / max_radius
    return np.stack([kx, ky], axis=-1).astype(np.float32)


def compute_crt_dcf_hamming(traj, hamming_radius=None):
    """CRT DCF × Hamming matching MATLAB op_CalcAndApplyDensComp.m.

    Hamming is evaluated on the physical in-plane ring radius, not only on
    max-normalized NUFFT radius.
    """
    r = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    r_ham = np.asarray(hamming_radius, dtype=np.float64) if hamming_radius is not None else r
    if r_ham.shape != r.shape:
        raise ValueError("hamming_radius must match trajectory length")

    unique_r = np.sort(np.unique(np.round(r, 6)))
    n_rings = len(unique_r)
    if n_rings < 2:
        return np.ones_like(r, dtype=np.float32)

    r_norm = unique_r / (float(unique_r.max()) + 1e-12)
    n_per_ring = np.array([(np.abs(r - ri) < 1e-5).sum() for ri in unique_r], dtype=np.float64)
    max_pts = float(n_per_ring.max())

    dcf_ring = np.zeros(n_rings, dtype=np.float64)
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

    dcf = np.zeros(len(r), dtype=np.float64)
    for i, ri in enumerate(unique_r):
        dcf[np.abs(r - ri) < 1e-5] = dcf_ring[i]

    ham = 0.54 + 0.46 * np.cos(2.0 * np.pi * r_ham)
    return (dcf * ham).astype(np.float32)


def _fGSLClassOri(dGs):
    s, c, t = abs(dGs[0]), abs(dGs[1]), abs(dGs[2])
    eq_sc, eq_st, eq_ct = abs(s-c)<=1e-6, abs(s-t)<=1e-6, abs(c-t)<=1e-6
    if ((eq_sc and eq_st) or (eq_sc and s<t) or (eq_st and s>c) or
            (eq_ct and c>s) or (s>c and s<t) or (s<c and c<t) or
            (s<t and t>c) or (c<t and t>s)):
        return 2
    elif ((eq_sc and s>t) or (eq_st and s<c) or (s<c and c>t) or
            (s>t and s<c) or (s<t and t<c)):
        return 1
    elif ((eq_ct and c<s) or (s>c and s>t) or (c>t and c<s) or (c<t and t<s)):
        return 0
    raise ValueError("Invalid slice orientation")


def _fGSLCalcPRS(dGs, dPhi):
    ori = _fGSLClassOri(dGs)
    if ori == 2:
        d = np.sqrt(dGs[1]**2 + dGs[2]**2)
        dGp = np.array([0.0, dGs[2]/d, -dGs[1]/d])
    elif ori == 1:
        d = np.sqrt(dGs[0]**2 + dGs[1]**2)
        dGp = np.array([dGs[1]/d, -dGs[0]/d, 0.0])
    else:
        d = np.sqrt(dGs[0]**2 + dGs[1]**2)
        dGp = np.array([-dGs[1]/d, dGs[0]/d, 0.0])
    dGr = np.cross(dGs, dGp)
    if abs(dPhi) > 1e-10:
        co, si = np.cos(dPhi), np.sin(dPhi)
        dGp = co * dGp - si * dGr
        dGr = np.cross(dGs, dGp)
    return dGp, dGr


def compute_fov_shift(info):
    """Return (d_pe, d_ro) in voxels for 2D FOV shift correction."""
    pos = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]])
    snv = np.array([info["snv_x"], info["snv_y"], info["snv_z"]])
    snv = snv / (np.linalg.norm(snv) + 1e-30)
    phase_vec, read_vec = _fGSLCalcPRS(snv, info["in_plane_rotation"])
    rot_mat = np.column_stack([phase_vec, read_vec, snv])
    base_pos = rot_mat.T @ pos
    vox_size = info["fov_mm"] / info["matrix_size"]
    return base_pos[0] / vox_size, base_pos[1] / vox_size   # (d_pe, d_ro)


# ── Spatial regularization (from utils/graph.py) ────────────────────────

def calc_Bmatrix_simple(ref_img, wmax=5e4, adj=8, mask=None, out_edge_scale=10.0,
                        mask_dilation=1, hard_boundary=False, return_graph=False):
    """Graph Laplacian regularization from reference image.

    If mask (bool array [xi,yi] from roi_selector) is given, edges that cross
    the ROI boundary are attenuated. By default this uses a soft boundary
    (optionally with dilation). With hard_boundary=True, any edge not fully
    inside ROI is set to zero so ROI support is exact.
    """
    from scipy.ndimage import binary_dilation

    radius = 1.001 if adj == 4 else 1.415
    xy = np.indices(ref_img.shape).reshape((2, -1)).T
    Nb = KDTree(xy).query_pairs(p=2, r=radius, output_type='ndarray')

    ref_vec = ref_img.flatten()
    n_vox = ref_vec.size
    C = (np.diff(ref_vec[Nb], axis=1)**2).flatten()
    old_err = np.seterr(divide='ignore')
    W = np.minimum(1.0 / C, wmax)
    np.seterr(**old_err)

    if mask is not None:
        # roi_selector stores mask[xi, yi]; numpy [row=yi, col=xi] order needs .T
        if int(mask_dilation) > 0:
            mask_img = binary_dilation(mask.T, iterations=int(mask_dilation))
        else:
            mask_img = mask.T.astype(bool)
        mask_flat = mask_img.flatten()
        out_of_mask = ~(mask_flat[Nb[:, 0]] & mask_flat[Nb[:, 1]])
        W = W.copy()
        if hard_boundary:
            W[out_of_mask] = 0.0
        else:
            W[out_of_mask] = wmax / float(max(out_edge_scale, 1e-6))

    A = csr_matrix((W, (Nb[:, 0], Nb[:, 1])), shape=(n_vox, n_vox)).toarray()
    A = 0.5 * (A + A.T)
    L = np.diag(np.sum(A, axis=0)) + np.diag(np.sum(A, axis=1)) - 2.0 * A

    D, V = np.linalg.eigh(L)
    B = np.diag(np.sqrt(np.abs(D))) @ V.T
    if return_graph:
        return B.conj().T @ B, B, A
    return B.conj().T @ B, B


# ── 31P training data generation ────────────────────────────────────────

# 31P skeletal muscle metabolites at 3T (literature values)
# Refs: Valkovic MRM 2014, Meyerspeer NMR Biomed 2020, Krssak NMR Biomed 2004
P31_METABOLITES = {
    "PCr":  {"shift_ppm":  0.0,    "lw_range": (5, 15),  "amp_range": (3.5, 4.5)},
    "Pi":   {"shift_ppm":  4.85,   "lw_range": (5, 20),  "amp_range": (0.0, 0.8)},
    "Pi2":  {"shift_ppm":  4.25,   "lw_range": (5, 20),  "amp_range": (0.0, 0.5)},
    "PDE":  {"shift_ppm":  3.07,   "lw_range": (10, 35), "amp_range": (0.0, 0.3)},
    "gATP": {"shift_ppm": -2.53,   "lw_range": (10, 30), "amp_range": (0.8, 1.2)},
    "aATP": {"shift_ppm": -7.58,   "lw_range": (10, 30), "amp_range": (0.8, 1.2)},
    "bATP": {"shift_ppm": -15.86,  "lw_range": (12, 35), "amp_range": (0.8, 1.2)},
}


def load_localizer_registered(dcm_path, N, fov_mm, pos_sag, pos_cor, pos_tra=0.0):
    """Load a DICOM localizer and sample it at MRSI voxel positions.

    Handles single .IMA files or directories (picks the slice closest to pos_tra).
    Returns an (N, N) float32 image with:
      - row 0  = bottom of display = posterior (+P)
      - col 0  = left  of display = patient's right (−L)
    Pass directly to imshow(origin='lower', extent=[0, N, 0, N]).

    Registration uses the DICOM ImagePositionPatient and PixelSpacing to map
    each MRSI voxel center (in LPS mm) to the localizer pixel grid, so the
    MRSI FOV and position are correctly overlaid regardless of localizer FOV.
    Convention (matches overlay_31p_localizer.py):
      L(xi) = pos_sag + (xi − N//2) * vox_size   (col index; xi=0 = patient right)
      P(yi) = pos_cor − (yi − N//2) * vox_size   (row index; yi=0 = posterior = bottom)
    """
    import pydicom
    from scipy.ndimage import map_coordinates

    def _load_slice(path):
        """Load one DICOM slice closest to pos_tra, return (img, ds)."""
        if os.path.isfile(path):
            ds = pydicom.dcmread(path)
            return ds.pixel_array.astype(np.float32), ds

        # Directory: find the transverse slice closest to pos_tra (LPS Z)
        import glob
        candidates = sorted(glob.glob(os.path.join(path, '*.IMA')) +
                            glob.glob(os.path.join(path, '*.dcm')) +
                            glob.glob(os.path.join(path, '*.ima')))
        if not candidates:
            candidates = sorted(os.path.join(path, f)
                                for f in os.listdir(path) if not f.startswith('.'))
        best, best_d = None, float('inf')
        for f in candidates:
            try:
                _ds = pydicom.dcmread(f, stop_before_pixels=True)
                z = float(_ds.ImagePositionPatient[2])
                if abs(z - pos_tra) < best_d:
                    best_d = abs(z - pos_tra)
                    best = f
            except Exception:
                continue
        if best is None:
            best = candidates[len(candidates) // 2]
        ds = pydicom.dcmread(best)
        print(f"[spice] Localizer: using slice z={float(ds.ImagePositionPatient[2]):.1f} mm "
              f"(target pos_tra={pos_tra:.1f} mm)")
        return ds.pixel_array.astype(np.float32), ds

    img, ds = _load_slice(dcm_path)
    ipp = [float(x) for x in ds.ImagePositionPatient]   # [L, P, S] of first DICOM pixel
    ps  = float(ds.PixelSpacing[0])                      # mm / pixel (assume square)
    L0, P0 = ipp[0], ipp[1]                              # LPS origin of pixel (0, 0)
    print(f"[spice] Localizer DICOM: {img.shape}  PixelSpacing={ps:.4f} mm"
          f"  FOV={img.shape[1]*ps:.0f}x{img.shape[0]*ps:.0f} mm"
          f"  origin=({L0:.1f}, {P0:.1f}) LPS")

    vox_size = fov_mm / N
    ctr = N // 2
    xis = np.arange(N)
    yis = np.arange(N)
    xi_2d, yi_2d = np.meshgrid(xis, yis)   # xi_2d[r, c] = c,  yi_2d[r, c] = r

    # LPS coordinates of each output pixel's MRSI voxel center
    L_2d = pos_sag + (xi_2d - ctr) * vox_size
    P_2d = pos_cor - (yi_2d - ctr) * vox_size

    # Map to DICOM pixel coords (DICOM: row 0 = top = P0, row↑ = P↑)
    col_loc = (L_2d - L0) / ps   # float localizer column
    row_loc = (P_2d - P0) / ps   # float localizer row

    coords  = np.array([row_loc.ravel(), col_loc.ravel()])
    sampled = map_coordinates(img, coords, order=1, mode='nearest')
    out_img = sampled.reshape(N, N).astype(np.float32)
    out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min() + 1e-12)

    print(f"[spice] Localizer registered: MRSI grid {N}×{N}"
          f"  vox={vox_size:.1f} mm  center=({pos_sag:.1f}, {pos_cor:.1f}) LPS")
    return out_img


def segment_localizer_prior(localizer_img, threshold=0.12, erosion=1, cleanup=True):
    """Segment localizer prior to suppress phantom/background regions.

    Returns
    -------
    prior_masked : (N, N) float32
        Localizer prior masked to the segmented tissue region.
    mask_rowcol : (N, N) bool
        Segmentation mask in numpy row/col convention (yi, xi).
    mask_inner_rowcol : (N, N) bool
        Eroded interior mask for conservative regularization.
    """
    from scipy.ndimage import binary_erosion, binary_fill_holes, label

    img = np.asarray(localizer_img, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)

    mask = img > float(threshold)
    if cleanup:
        labeled, n_comp = label(mask)
        if n_comp > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            mask = labeled == sizes.argmax()
        mask = binary_fill_holes(mask)

    if erosion > 0:
        mask_inner = binary_erosion(mask, iterations=int(erosion))
        if mask_inner.sum() > 0:
            mask = mask_inner
    mask_inner = binary_erosion(mask, iterations=1) if mask.sum() > 0 else mask

    prior_masked = (img * mask.astype(np.float32)).astype(np.float32)
    return prior_masked, mask.astype(bool), mask_inner.astype(bool)


def load_csi_prior(dcm_path, crt_n_spec, crt_dwell_s):
    """Load 8x8 CSI DICOM and resample FIDs to the CRT time grid."""
    import pydicom
    from scipy.interpolate import interp1d

    ds = pydicom.dcmread(dcm_path)
    data_raw = np.frombuffer(ds[0x7FE1, 0x1010].value, dtype=np.float32)
    csi_fid = (data_raw[0::2] + 1j * data_raw[1::2]).reshape(64, -1)
    csi_n_spec = csi_fid.shape[1]

    csa_data = ds[0x0029, 0x1120].value.decode('latin-1', errors='ignore')
    idx = csa_data.find('alDwellTime[0]')
    dwell_str = csa_data[idx:idx+80].split('=')[1].strip().split('\n')[0].strip()
    csi_dwell_s = float(dwell_str) * 1e-9

    csi_time = np.arange(csi_n_spec) * csi_dwell_s
    crt_time = np.arange(crt_n_spec) * crt_dwell_s

    resampled = np.zeros((64, crt_n_spec), dtype=np.complex64)
    for v in range(64):
        interp_r = interp1d(csi_time, csi_fid[v].real, kind='linear',
                            fill_value=0, bounds_error=False)
        interp_i = interp1d(csi_time, csi_fid[v].imag, kind='linear',
                            fill_value=0, bounds_error=False)
        resampled[v] = interp_r(crt_time) + 1j * interp_i(crt_time)

    print(f"[spice] Loaded CSI prior: {dcm_path}")
    print(f"[spice]   CSI: {csi_n_spec} pts, dwell={csi_dwell_s*1e6:.1f} us")
    print(f"[spice]   Resampled to {crt_n_spec} pts, dwell={crt_dwell_s*1e6:.1f} us")
    return resampled


def generate_training_data(n_spec, dwell_s, larmor_mhz, n_training=5000):
    """Simulate random 31P muscle FIDs using literature amplitude ratios."""
    rng = np.random.default_rng(42)
    time_axis = np.arange(n_spec) * dwell_s
    metabs = list(P31_METABOLITES.items())

    training = np.zeros((n_training, n_spec), dtype=np.complex64)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        for name, props in metabs:
            cm = rng.uniform(*props["amp_range"])
            lw = rng.uniform(*props["lw_range"])
            freq_hz = props["shift_ppm"] * larmor_mhz
            sig += cm * np.exp(-lw * np.pi * time_axis) * np.exp(+2j * np.pi * freq_hz * time_axis)
        whole_shift = rng.uniform(-0.05, 0.05) * larmor_mhz
        sig *= np.exp(+2j * np.pi * whole_shift * time_axis)
        training[i] = sig.astype(np.complex64)
    return training


# PK_3T_Skeletal_2Pi multiplet definitions (from PK_3T_Skeletal_2Pi.m)
# Each entry: (name, center_ppm, lw_init_hz, amp_range, components)
# components: list of (delta_ppm_from_center, relative_weight)
# Doublet convention: peaks at center ± J/2 (separation = J)
# Triplet 1:2:1 convention: peaks at center - J, center, center + J
_DELTA_BETA  = 15.0 / 51.0   # β-ATP triplet spacing  (ppm)
_DELTA_ALPHA = 16.0 / 51.0   # α-ATP doublet spacing  (ppm)
_DELTA_GAMMA = 15.0 / 51.0   # γ-ATP doublet spacing  (ppm)

PK2PI_METABOLITES = [
    # singlets
    ("PCr",  0.0,    5.0,  (1.5, 2.5),  [(0.0, 1.0)]),
    ("Pi",   4.85,   5.0,  (0.0, 0.8),  [(0.0, 1.0)]),
    ("Pi2",  4.25,  10.0,  (0.0, 0.5),  [(0.0, 1.0)]),
    ("PDE",  3.07,  30.0,  (0.0, 0.15), [(0.0, 1.0)]),
    # γ-ATP doublet: peaks at center ± delta/2
    ("gATP", -2.53,  10.0, (0.5, 1.5),
     [(-_DELTA_GAMMA / 2, 1.0), (_DELTA_GAMMA / 2, 1.0)]),
    # α-ATP doublet: peaks at center ± delta/2
    ("aATP", -7.58,  10.0, (0.5, 1.5),
     [(-_DELTA_ALPHA / 2, 1.0), (_DELTA_ALPHA / 2, 1.0)]),
    # β-ATP triplet 1:2:1: peaks at center ± delta and center
    ("bATP", -15.86, 10.0, (0.5, 1.5),
     [(-_DELTA_BETA, 0.5), (0.0, 1.0), (_DELTA_BETA, 0.5)]),
]


def generate_training_data_pk2pi(n_spec, dwell_s, larmor_mhz, n_training=5000):
    """Synthetic 31P FIDs using PK_3T_Skeletal_2Pi multiplet structure (J-coupling)."""
    rng = np.random.default_rng(42)
    t = np.arange(n_spec) * dwell_s
    training = np.zeros((n_training, n_spec), dtype=np.complex64)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        for name, center_ppm, lw_init, amp_range, components in PK2PI_METABOLITES:
            amp = rng.uniform(*amp_range)
            lw  = rng.uniform(lw_init * 0.5, min(50.0, lw_init * 3.0))
            phase = rng.uniform(0, 2 * np.pi)
            for delta_ppm, weight in components:
                freq_hz = (center_ppm + delta_ppm) * larmor_mhz
                sig += (amp * weight * np.exp(1j * phase)
                        * np.exp(-lw * np.pi * t)
                        * np.exp(2j * np.pi * freq_hz * t))
        whole_shift = rng.uniform(-0.1, 0.1) * larmor_mhz
        sig *= np.exp(2j * np.pi * whole_shift * t)
        training[i] = sig.astype(np.complex64)
    return training


# ── SPICE solver ────────────────────────────────────────────────────────

def spice_reconstruct(adj_fid_flat, nufft_op, V, WW, N, n_spec,
                      rank, lamda, n_iter=0, maxiter=120,
                      out_roi_mask=None, lambda_out=0.0):
    """
    SPICE reconstruction for 2D CRT MRSI.

    Parameters
    ----------
    adj_fid_flat : (N_Vox, n_spec) pre-computed adjoint FIDs
    n_iter       : 0 = direct solve, >0 = CG with n_iter iterations
    out_roi_mask : (N_Vox,) bool — True for voxels OUTSIDE the ROI
    lambda_out   : Tikhonov weight on out-of-ROI voxels; forces their
                   amplitude toward zero.  Should be >> lambda1 (e.g. 1.0)
    """
    N_Vox = N * N
    D_TYPE = np.complex64

    print("[spice] Precomputing Gram matrix ...")
    Gram = np.zeros((N_Vox, N_Vox), dtype=D_TYPE)
    for v in tqdm(range(N_Vox), desc="Gram columns"):
        delta = np.zeros((N, N), dtype=D_TYPE)
        delta.flat[v] = 1.0
        Gram[:, v] = nufft_op.adj_op(nufft_op.op(delta)).reshape(N_Vox)

    b_mat = adj_fid_flat @ V  # (N_Vox, rank)

    Gram_sys = Gram.copy()
    if lamda > 0:
        Gram_sys += lamda * WW.astype(D_TYPE)
    # Diagonal Tikhonov: explicitly suppress outside-ROI voxel amplitudes
    if lambda_out > 0 and out_roi_mask is not None:
        diag_penalty = np.zeros(N_Vox, dtype=np.float32)
        diag_penalty[out_roi_mask] = lambda_out
        Gram_sys += np.diag(diag_penalty).astype(D_TYPE)
        n_out = int(out_roi_mask.sum())
        print(f"[spice] Out-of-ROI Tikhonov: {n_out} voxels suppressed, lambda_out={lambda_out}")

    if n_iter > 0:
        print(f"[spice] Iterative CG solve ({n_iter} iters, rank={rank}, lambda={lamda}) ...")
        U = np.zeros((N_Vox, rank), dtype=D_TYPE)
        for k in range(rank):
            def mv(x, G=Gram_sys):
                return (G @ x.reshape(-1, 1)).ravel().astype(D_TYPE)
            A_op = LinearOperator((N_Vox, N_Vox), matvec=mv, dtype=D_TYPE)
            u_k, info = cg(A_op, b_mat[:, k], maxiter=n_iter)
            U[:, k] = u_k
        print(f"[spice] CG done ({n_iter} iters)")
    else:
        print(f"[spice] Direct solve ({N_Vox}x{N_Vox}, rank={rank}, lambda={lamda}) ...")
        U = np.linalg.solve(Gram_sys, b_mat)

    spice_fid = (U @ V.conj().T).reshape(N, N, n_spec)
    return spice_fid, U


# ── Plotting ────────────────────────────────────────────────────────────

def find_best_snr(image_spec, roi=None):
    Ny, Nx, n_spec = image_spec.shape
    snr_map = np.zeros((Ny, Nx))
    for y in range(Ny):
        for x in range(Nx):
            s = np.abs(image_spec[y, x, :])
            snr_map[y, x] = s.max() / (np.std(s[int(0.8*n_spec):]) + 1e-20)
    if roi is not None:
        # only pick best voxel within ROI (roi is [xi,yi]; snr_map is [yi,xi])
        snr_masked = snr_map * roi.T
        vy, vx = np.unravel_index(snr_masked.argmax(), snr_map.shape)
    else:
        vy, vx = np.unravel_index(snr_map.argmax(), snr_map.shape)
    return vy, vx, snr_map


def plot_spectra_on_anatomy(spice_spec, PPM, ppmlim, anatomy, fig_dir, roi=None):
    """Per-voxel spectra overlaid on anatomy image (localizer or adjoint mag)."""
    Ny, Nx, _ = spice_spec.shape
    ppm_mask = (PPM >= ppmlim[0]) & (PPM <= ppmlim[1])
    PPM_c = PPM[ppm_mask]

    fig = plt.figure(figsize=(10, 10), facecolor='k')

    # Full-figure background axes
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.imshow(anatomy, cmap='gray', origin='lower',
                 extent=[0, Nx, 0, Ny], aspect='auto',
                 vmin=0, vmax=anatomy.max())
    ax_bg.set_xlim(0, Nx); ax_bg.set_ylim(0, Ny)
    ax_bg.axis('off')

    pad = 0.03
    vw, vh = 1.0 / Nx, 1.0 / Ny
    for yi in range(Ny):
        for xi in range(Nx):
            in_roi = (roi is None) or bool(roi[xi, yi])
            sp = np.abs(spice_spec[yi, xi, ppm_mask])  # spec[row,col]: yi=row, xi=col
            sp = sp / (sp.max() + 1e-20)
            ax = fig.add_axes([xi*vw + pad*vw, yi*vh + pad*vh,
                                vw*(1-2*pad), vh*(1-2*pad)])
            color = 'w' if in_roi else '#555555'
            ax.plot(PPM_c, sp, color=color, lw=0.8)
            ax.set_xlim(ppmlim[1], ppmlim[0])   # high ppm left
            ax.set_ylim(-0.05, 1.15)
            ax.patch.set_alpha(0.0)
            ax.axis('off')

    fig.savefig(os.path.join(fig_dir, "spice_on_anatomy.png"),
                dpi=150, bbox_inches='tight', facecolor='k')
    plt.close(fig)


def plot_results(adj_spec, spice_spec, PPM, ppmlim, fig_dir, anatomy=None, roi=None):
    Ny, Nx, n_spec = spice_spec.shape
    vy, vx, snr_map = find_best_snr(spice_spec, roi=roi)
    print(f"[spice] Best SNR voxel: [{vy},{vx}] SNR={snr_map[vy, vx]:.1f}")

    peaks = [(0, 'PCr'), (5, 'Pi'), (-2.5, 'γ-ATP'), (-7.5, 'α-ATP'), (-16, 'β-ATP')]

    # ── PCr/Pi spatial maps (replaces old generic spatial map panel) ──
    pcr_win = (PPM >= -1.5) & (PPM <= 1.5)
    pi_win = (PPM >= 4.2) & (PPM <= 5.2)
    if not np.any(pcr_win) or not np.any(pi_win):
        raise RuntimeError("PCr/Pi ppm windows are empty. Check PPM axis and dwell/larmor settings.")

    pcr_adj = np.max(np.abs(adj_spec[:, :, pcr_win]), axis=-1)
    pcr_sp = np.max(np.abs(spice_spec[:, :, pcr_win]), axis=-1)
    pi_sp = np.max(np.abs(spice_spec[:, :, pi_win]), axis=-1)

    # Keep spice_vs_adjoint filename but change to PCr/Pi-focused map comparison.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im1 = axes[0].imshow(pcr_adj, origin="lower", cmap="magma")
    axes[0].set_title("Adjoint PCr map (-1.5 to 1.5 ppm)")
    axes[0].set_xlabel("x voxel"); axes[0].set_ylabel("y voxel")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(pcr_sp, origin="lower", cmap="magma")
    axes[1].set_title("SPICE PCr map (-1.5 to 1.5 ppm)")
    axes[1].set_xlabel("x voxel"); axes[1].set_ylabel("y voxel")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(pi_sp, origin="lower", cmap="viridis")
    axes[2].set_title("SPICE Pi map (4.2 to 5.2 ppm)")
    axes[2].set_xlabel("x voxel"); axes[2].set_ylabel("y voxel")
    plt.colorbar(im3, ax=axes[2])

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spice_vs_adjoint.png"), dpi=150)
    plt.close(fig)

    # Save dedicated standalone maps matching your requested plotting style.
    fig_pcr, ax_pcr = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    im_pcr = ax_pcr.imshow(pcr_sp, origin='lower', cmap='magma')
    ax_pcr.set_title('SPICE PCr map (-1.5 to 1.5 ppm)')
    ax_pcr.set_xlabel('x voxel'); ax_pcr.set_ylabel('y voxel')
    fig_pcr.colorbar(im_pcr, ax=ax_pcr, label='max |spectrum| (a.u.)')
    fig_pcr.savefig(os.path.join(fig_dir, 'spice_pcr_spatial_map.png'), dpi=170, bbox_inches='tight')
    plt.close(fig_pcr)

    fig_pi, ax_pi = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    im_pi = ax_pi.imshow(pi_sp, origin='lower', cmap='viridis')
    ax_pi.set_title('SPICE Pi map (4.2 to 5.2 ppm)')
    ax_pi.set_xlabel('x voxel'); ax_pi.set_ylabel('y voxel')
    fig_pi.colorbar(im_pi, ax=ax_pi, label='max |spectrum| (a.u.)')
    fig_pi.savefig(os.path.join(fig_dir, 'spice_pi_spatial_map.png'), dpi=170, bbox_inches='tight')
    plt.close(fig_pi)

    fig_combo, axes_combo = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    imc1 = axes_combo[0].imshow(pcr_sp, origin='lower', cmap='magma')
    axes_combo[0].set_title('SPICE PCr map (-1.5 to 1.5 ppm)')
    axes_combo[0].set_xlabel('x voxel'); axes_combo[0].set_ylabel('y voxel')
    fig_combo.colorbar(imc1, ax=axes_combo[0], label='max |spectrum|')
    imc2 = axes_combo[1].imshow(pi_sp, origin='lower', cmap='viridis')
    axes_combo[1].set_title('SPICE Pi map (4.2 to 5.2 ppm)')
    axes_combo[1].set_xlabel('x voxel'); axes_combo[1].set_ylabel('y voxel')
    fig_combo.colorbar(imc2, ax=axes_combo[1], label='max |spectrum|')
    fig_combo.savefig(os.path.join(fig_dir, 'spice_pcr_pi_spatial_maps.png'), dpi=170, bbox_inches='tight')
    plt.close(fig_combo)

    # ── Best SNR spectrum ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    sp = np.abs(spice_spec[vy, vx, :])
    ax.plot(PPM, sp, 'k-', lw=0.8)
    ax.fill_between(PPM, 0, sp, alpha=0.15, color='blue')
    ax.set_xlim(ppmlim[1], ppmlim[0]); ax.set_ylim(bottom=0)
    ax.set_xlabel("ppm"); ax.set_ylabel("|spectrum|")
    ax.set_title(f"SPICE — Voxel [{vy},{vx}] (best SNR={snr_map[vy,vx]:.1f})")
    for pos, name in peaks:
        ax.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.text(pos, ax.get_ylim()[1]*0.92, name, fontsize=8, ha='center', color='red')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spice_spectrum_best_snr.png"), dpi=150)
    plt.close(fig)

    # ── All spectra grid — per-voxel normalised, PPM axis flipped ──
    fig, axes = plt.subplots(Nx, Ny, figsize=(Ny * 2.5, Nx * 2))
    axes = np.array(axes).reshape(Nx, Ny)
    # Shared y-scale across all voxels so magnitude distribution is visible.
    spec_abs_all = np.nan_to_num(np.abs(spice_spec))
    global_ymax = float(spec_abs_all.max())
    if global_ymax <= 0:
        global_ymax = 1.0
    for yi in range(Ny):
        for xi in range(Nx):
            ax = axes[Nx - 1 - xi, yi]
            sp_v = np.nan_to_num(np.abs(spice_spec[xi, yi, :]))
            ax.plot(PPM, sp_v, 'k', lw=0.4)
            ax.set_xlim(ppmlim[1], ppmlim[0])
            ax.set_ylim(0, global_ymax * 1.02)
            ax.axis("off")
    fig.suptitle("SPICE — All voxel spectra (shared y-scale)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spice_all_spectra.png"), dpi=150)
    plt.close(fig)

    if anatomy is not None:
        plot_spectra_on_anatomy(spice_spec, PPM, ppmlim, anatomy, fig_dir, roi=roi)


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="31P CONCEPT CRT SPICE reconstruction")
    p.add_argument("--dat", required=True, help="Siemens Twix .dat file")
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--extra-shift-ro", type=float, default=0.0,
                   help="Additional NUFFT readout-axis shift (voxels), added to header shift")
    p.add_argument("--extra-shift-pe", type=float, default=0.0,
                   help="Additional NUFFT phase-encode-axis shift (voxels), added to header shift")
    p.add_argument("--apodization", type=float, default=0.0,
                   help="Gaussian apodization FWHM in Hz (0 = none, e.g. 20)")
    p.add_argument("--rank", type=int, nargs="+", default=[15],
                   help="SPICE subspace rank(s) — pass multiple to compare")
    p.add_argument("--lambda1", type=float, default=0.0, help="Spatial regularization weight")
    p.add_argument("--n-iter", type=int, default=0,
                   help="CG iterations (0 = direct solve, >0 = iterative)")
    p.add_argument("--maxiter", type=int, default=120, help="CG max iterations (large grids only)")
    p.add_argument("--training-size", type=int, default=5000)
    p.add_argument("--subspace-src", default="adjoint",
                   choices=["adjoint", "csi", "synthetic", "pk2pi", "basis-dat"],
                   help="Subspace source: adjoint, csi, synthetic, pk2pi (PK_3T_Skeletal_2Pi multiplets), or basis-dat")
    p.add_argument("--basis-dat", default=None,
                   help="Reference .dat file for subspace (--subspace-src basis-dat)")
    p.add_argument("--csi-prior", default=None,
                   help="Path to CSI DICOM (required when --subspace-src csi)")
    p.add_argument("--localizer", default=None,
                   help="Path to DICOM localizer dir for spatial prior")
    p.add_argument("--localizer-threshold", type=float, default=0.12,
                   help="Threshold for localizer segmentation after [0,1] normalization")
    p.add_argument("--localizer-erosion", type=int, default=1,
                   help="Erosion iterations for localizer segmentation mask")
    p.add_argument("--localizer-mask-cleanup", action="store_true",
                   help="Keep largest connected component + fill holes in localizer segmentation")
    p.add_argument("--localizer-auto-seg", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable automatic localizer segmentation for spatial prior (default: on)")
    p.add_argument("--roi-mask", default=None,
                   help="Path to .npy boolean mask (NxN) from roi_selector_31p.py")
    p.add_argument("--wmax", type=float, default=5e4, help="Max edge weight for spatial prior")
    p.add_argument("--roi-edge-scale", type=float, default=10.0,
                   help="Cross/outside ROI edge attenuation factor in graph prior. "
                        "Larger value means weaker coupling outside ROI (e.g. 100-1000).")
    p.add_argument("--roi-dilation", type=int, default=1,
                   help="Mask dilation (in voxels) before graph construction. "
                        "Set 0 to use exact drawn ROI boundary.")
    p.add_argument("--roi-hard-boundary", action="store_true",
                   help="Set any edge touching outside ROI to zero weight (strict ROI graph boundary).")
    p.add_argument("--prior-gamma", type=float, default=1.0,
                   help="Contrast shaping for spatial prior map: prior <- prior**gamma. "
                        "gamma>1 increases contrast, gamma<1 flattens.")
    p.add_argument("--prior-scale", type=float, default=1.0,
                   help="Multiplicative scale for displayed/saved spatial prior map only. "
                        "Does not need to match graph scale.")
    p.add_argument("--prior-scale-graph", type=float, default=1.0,
                   help="Multiplicative scale for graph construction prior. "
                        "Keep 1.0 to preserve graph-weight comparability.")
    p.add_argument("--roi-renorm", action="store_true",
                   help="After applying ROI mask, renormalize ROI intensities to [0,1] before graph prior.")
    p.add_argument("--lambda-out", type=float, default=0.0,
                   help="Tikhonov penalty on out-of-ROI voxels (e.g. 1.0). "
                        "Drives outside-ROI amplitudes to zero. Requires --roi-mask.")
    p.add_argument("--rep", type=int, default=None,
                   help="Rep index to reconstruct (0-based); default = all reps merged")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--adjoint-only", action="store_true",
                   help="Stop after adjoint recon — skip SPICE, plot spectra on localizer")
    return p.parse_args()


def main():
    args = parse_args()

    rep_tag  = f"_rep{args.rep:02d}" if args.rep is not None else ""
    data_dir = os.path.join(args.out_dir, "data")
    fig_dir  = os.path.join(args.out_dir, f"figures{rep_tag}")
    for d in [data_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load raw data ────────────────────────────────────────────
    info = load_concept_twix_3d(args.dat, rep_sel=args.rep)
    kdata   = info["kdata"]          # (pts_per_ring, n_spec, 1, n_rings)
    N       = info["matrix_size"]
    n_spec  = info["n_spec"]
    n_rings = info["n_rings"]
    traj_pts = int(info["traj_pts"].max())
    dwell_s  = info["dwell_s"] * 2   # spectral dwell (ADC dwell × 2 after OS removal)
    larmor_mhz       = info["larmor_freq_mhz"]
    gamma_mhz_per_mt = info["gamma_2pi"] / 1e9

    print(f"\n[spice] Matrix={N}x{N}, vecSize={n_spec}, rings={n_rings}, coils={kdata.shape[2]}")

    # ── 2. Phase correction ─────────────────────────────────────────
    if not args.no_phase_corr:
        kdata = concept_phase_correction(kdata, info["_ti_per_ring"])

    # ── 3. Trajectory (max-normalised to [-0.5, 0.5]) ───────────────
    traj = build_trajectory_2d(info["first_circle"], n_rings, traj_pts)
    print(f"[spice] traj: {len(traj)} k-pts  kx=[{traj[:,0].min():.3f},{traj[:,0].max():.3f}]")

    # ── 4. FOV shift ────────────────────────────────────────────────
    kdata_flat = kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
    if not args.no_fov_shift:
        d_pe, d_ro = compute_fov_shift(info)
        d_ro += args.extra_shift_ro
        d_pe += args.extra_shift_pe
        shift_phase = np.exp(-2j * np.pi * (traj[:, 0] * d_pe + traj[:, 1] * d_ro))
        kdata_flat = kdata_flat * shift_phase[:, np.newaxis]
        print(f"[spice] FOV shift: pe={d_pe:.4f}, ro={d_ro:.4f} vox")

    # ── 5. DCF (Hamming CRT, matches op_CalcAndApplyDensComp.m) ────
    # Match 06c: evaluate Hamming on physical ring radius (not max-normalized 0.5 radius).
    r_xy = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    ham_radius = r_xy * (float(info["k_norm_max_phys"]) / 0.5)
    dcf = compute_crt_dcf_hamming(traj, hamming_radius=ham_radius)
    print(f"[spice] DCF weights: min={dcf.min():.4f}  max={dcf.max():.4f}")

    # ── 6. NUFFT operator ───────────────────────────────────────────
    nufft_op = mrinufft.get_operator(
        "finufft", samples=traj, shape=(N, N),
        n_coils=1, density=dcf)

    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    PPM = -FREQ / larmor_mhz + args.ppm_center

    # ── 7. Full adjoint recon (needed for subspace & spatial prior) ──
    print("[spice] Computing full adjoint reconstruction ...")
    adj_fid_flat = np.zeros((N * N, n_spec), dtype=np.complex64)
    for t in tqdm(range(n_spec), desc="Adjoint", miniters=100):
        adj_fid_flat[:, t] = nufft_op.adj_op(kdata_flat[:, t]).reshape(N * N)
    adj_fid_img = adj_fid_flat.reshape(N, N, n_spec)
    mag_map = np.mean(np.abs(FIDToSpec(adj_fid_img, axis=-1)), axis=-1)
    mag_norm = mag_map / (mag_map.max() + 1e-12)

    # ── 8. Subspace estimation ──────────────────────────────────────
    if args.subspace_src == "basis-dat":
        if not args.basis_dat:
            sys.exit("--basis-dat path required when --subspace-src basis-dat")
        print(f"[spice] Loading reference .dat for subspace: {args.basis_dat}")
        ref_info = load_concept_twix_3d(args.basis_dat)
        ref_kdata = ref_info["kdata"]
        if not args.no_phase_corr:
            ref_kdata = concept_phase_correction(ref_kdata, ref_info["_ti_per_ring"])
        ref_N = ref_info["matrix_size"]
        ref_n_spec = ref_info["n_spec"]
        ref_traj_pts = int(ref_info["traj_pts"].max())
        ref_traj = build_trajectory_2d(
            ref_info["first_circle"], ref_info["n_rings"], ref_traj_pts)
        ref_r_xy = np.sqrt(ref_traj[:, 0]**2 + ref_traj[:, 1]**2)
        ref_ham_radius = ref_r_xy * (float(ref_info["k_norm_max_phys"]) / 0.5)
        ref_dcf = compute_crt_dcf_hamming(ref_traj, hamming_radius=ref_ham_radius)
        ref_nufft = mrinufft.get_operator(
            "finufft", samples=ref_traj, shape=(ref_N, ref_N),
            n_coils=1, density=ref_dcf)
        ref_flat = ref_kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, ref_n_spec)
        if not args.no_fov_shift:
            rd_pe, rd_ro = compute_fov_shift(ref_info)
            rd_ro += args.extra_shift_ro
            rd_pe += args.extra_shift_pe
            ref_flat = ref_flat * np.exp(
                -2j * np.pi * (ref_traj[:, 0] * rd_pe + ref_traj[:, 1] * rd_ro))[:, np.newaxis]
        print("[spice] Computing adjoint of reference data ...")
        ref_adj = np.zeros((ref_N * ref_N, ref_n_spec), dtype=np.complex64)
        for t in tqdm(range(ref_n_spec), desc="Ref adjoint", miniters=100):
            ref_adj[:, t] = ref_nufft.adj_op(ref_flat[:, t]).reshape(ref_N * ref_N)
        training = ref_adj
    elif args.subspace_src == "adjoint":
        print("[spice] Deriving subspace from adjoint reconstruction ...")
        training = adj_fid_flat
    elif args.subspace_src == "csi":
        if not args.csi_prior:
            sys.exit("--csi-prior path required when --subspace-src csi")
        print("[spice] Using CSI phantom prior for subspace ...")
        training = load_csi_prior(args.csi_prior, n_spec, dwell_s)
    else:  # "synthetic" or "pk2pi"
        cache_name = ("training_data_pk2pi.npy" if args.subspace_src == "pk2pi"
                      else "training_data.npy")
        training_cache = os.path.join(data_dir, cache_name)
        if os.path.exists(training_cache):
            print(f"[spice] Loading cached training data: {training_cache}")
            training = np.load(training_cache)
        else:
            if args.subspace_src == "pk2pi":
                print(f"[spice] Generating {args.training_size} PK2Pi multiplet training FIDs ...")
                training = generate_training_data_pk2pi(
                    n_spec, dwell_s, larmor_mhz, args.training_size)
            else:
                print(f"[spice] Generating {args.training_size} synthetic 31P training FIDs ...")
                training = generate_training_data(
                    n_spec, dwell_s, larmor_mhz, args.training_size)
            np.save(training_cache, training)
            print(f"[spice] Saved training data to {training_cache}")

    _, s_vals, Vh = np.linalg.svd(training, full_matrices=False)
    max_rank = max(args.rank)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(s_vals[:min(30, len(s_vals))], "x-"); ax1.set_title("Singular values")
    for k in range(min(6, max_rank)):
        ax2.plot(PPM, np.abs(FIDToSpec(Vh[k, :].conj())), label=f"V{k}")
    ax2.set_xlim(20, -20); ax2.set_xlabel("ppm")
    ax2.set_title("Top subspace vectors (spectrum domain)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "spice_subspace.png"), dpi=120)
    plt.close(fig)

    # Spatial prior for Laplacian: use registered localizer when provided
    # (better anatomical contrast than 31P adjoint at low resolution),
    # otherwise fall back to adjoint magnitude.
    auto_roi = None
    if args.localizer:
        localizer_raw = load_localizer_registered(
            args.localizer, N,
            fov_mm   = info["fov_mm"],
            pos_sag  = info["pos_sag"],
            pos_cor  = info["pos_cor"],
            pos_tra  = info["pos_tra"])

        if args.localizer_auto_seg:
            localizer_prior, localizer_mask_rc, localizer_inner_rc = segment_localizer_prior(
                localizer_raw,
                threshold=args.localizer_threshold,
                erosion=args.localizer_erosion,
                cleanup=args.localizer_mask_cleanup,
            )
            if localizer_mask_rc.sum() == 0:
                print("[spice] Warning: localizer segmentation mask is empty; using unmasked localizer prior")
                anatomy_bg = localizer_raw
                spatial_prior = localizer_raw
                _prior_label = "registered localizer (unmasked)"
                auto_roi = None
            else:
                anatomy_bg = localizer_raw
                spatial_prior = localizer_prior
                _prior_label = "registered localizer (segmented)"
                # roi convention in this script is [xi, yi]. localizer mask is [yi, xi].
                auto_roi = localizer_mask_rc.T
                print(f"[spice] Localizer segmentation: {int(localizer_mask_rc.sum())}/{N*N} voxels kept "
                      f"(thr={args.localizer_threshold}, erosion={args.localizer_erosion}, "
                      f"cleanup={args.localizer_mask_cleanup})")

                fig_loc, axs_loc = plt.subplots(1, 3, figsize=(12, 4))
                axs_loc[0].imshow(localizer_raw, origin='lower', cmap='gray')
                axs_loc[0].set_title("Localizer (registered)")
                axs_loc[1].imshow(localizer_mask_rc, origin='lower', cmap='gray')
                axs_loc[1].set_title("Segmented mask")
                axs_loc[2].imshow(localizer_prior, origin='lower', cmap='gray')
                axs_loc[2].set_title("Masked localizer prior")
                for _ax in axs_loc:
                    _ax.axis('off')
                fig_loc.tight_layout()
                fig_loc.savefig(os.path.join(fig_dir, "localizer_segmentation.png"), dpi=150)
                plt.close(fig_loc)
        else:
            print("[spice] Localizer auto-segmentation disabled; using full registered localizer prior")
            anatomy_bg = localizer_raw
            spatial_prior = localizer_raw
            _prior_label = "registered localizer (manual-mask expected)"
            auto_roi = None
    else:
        anatomy_bg    = mag_norm
        spatial_prior = mag_norm
        _prior_label  = "31P adjoint mag"

    if args.roi_mask:
        roi = np.load(args.roi_mask).astype(bool)
        if roi.shape != (N, N):
            sys.exit(f"[spice] ROI mask shape {roi.shape} != ({N},{N})")
        if auto_roi is not None:
            roi = roi & auto_roi
            print(f"[spice] ROI mask ∩ localizer segmentation: {int(roi.sum())}/{N*N} voxels")
        else:
            print(f"[spice] ROI mask: {int(roi.sum())}/{N*N} voxels — decoupling edges at boundary")
    else:
        roi = auto_roi
        if roi is not None:
            print(f"[spice] Using localizer-segmentation ROI for spatial regularization: {int(roi.sum())}/{N*N}")

    # Optional contrast shaping before ROI masking.
    if args.prior_gamma != 1.0:
        spatial_prior = np.power(np.clip(spatial_prior, 0.0, 1.0), args.prior_gamma).astype(np.float32)
        _prior_label = f"{_prior_label}, gamma={args.prior_gamma:g}"

    # Apply ROI directly to the spatial prior intensity map so the saved prior
    # and the graph regularization input both follow the same mask.
    if roi is not None:
        spatial_prior = spatial_prior * roi.T.astype(np.float32)
        _prior_label = f"{_prior_label} + ROI masked"
        print(f"[spice] Applied ROI mask to spatial prior map: {int(roi.sum())}/{N*N} voxels kept")

    # Optional ROI-only renormalization to [0,1] after masking.
    # Keeps outside ROI at zero while stretching in-ROI contrast.
    if roi is not None and args.roi_renorm:
        roi_rc = roi.T.astype(bool)
        vals = spatial_prior[roi_rc]
        if vals.size > 0:
            vmin = float(vals.min())
            vmax = float(vals.max())
            denom = vmax - vmin
            if denom > 1e-12:
                spatial_prior = np.zeros_like(spatial_prior, dtype=np.float32)
                spatial_prior[roi_rc] = ((vals - vmin) / denom).astype(np.float32)
                _prior_label = f"{_prior_label}, roi_renorm"
                print(f"[spice] ROI renorm: in-ROI prior normalized to [0,1] (min={vmin:.4g}, max={vmax:.4g})")
            else:
                print("[spice] ROI renorm skipped: in-ROI prior is constant")

    # Decouple prior scale for visualization vs graph construction.
    spatial_prior_graph = spatial_prior.astype(np.float32)
    spatial_prior_plot = spatial_prior.astype(np.float32)

    if args.prior_scale != 1.0:
        spatial_prior_plot = (spatial_prior_plot * float(args.prior_scale)).astype(np.float32)
        _prior_label = f"{_prior_label}, plot_scale={args.prior_scale:g}"
        print(f"[spice] Applied plot prior scale factor: {args.prior_scale:g}")

    if args.prior_scale_graph != 1.0:
        spatial_prior_graph = (spatial_prior_graph * float(args.prior_scale_graph)).astype(np.float32)
    print(f"[spice] Graph prior scale factor: {args.prior_scale_graph:g}")

    # ── Save spatial prior map (matches 1H wref "Voxel-level sum prior map") ───
    _fig_sp, _ax_sp = plt.subplots(figsize=(6, 5))
    _im_sp = _ax_sp.imshow(spatial_prior_plot, origin='lower', cmap='Greens', aspect='equal')
    plt.colorbar(_im_sp, ax=_ax_sp)
    _ax_sp.set_title(f"Voxel-level sum prior map ({_prior_label})", fontsize=11)
    _ax_sp.set_xlabel("x (voxel)"); _ax_sp.set_ylabel("y (voxel)")
    _fig_sp.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    _fig_sp.savefig(os.path.join(fig_dir, "spatial_prior_map.png"), dpi=150, bbox_inches='tight')
    plt.close(_fig_sp)
    print(f"[spice] Saved spatial prior map -> {fig_dir}/spatial_prior_map.png")

    # Pass roi as mask: cuts cross-boundary edges so inside/outside are decoupled.
    # Outside-ROI voxels get no regularization (free to take noise/leakage values).
    WW, B, A_graph = calc_Bmatrix_simple(
        spatial_prior_graph, wmax=args.wmax, adj=8, mask=roi,
        out_edge_scale=args.roi_edge_scale,
        mask_dilation=args.roi_dilation,
        hard_boundary=args.roi_hard_boundary,
        return_graph=True)

    if roi is not None:
        if args.roi_hard_boundary:
            print(f"[spice] ROI graph boundary: strict (dilation={args.roi_dilation}, cross-boundary edges=0)")
        else:
            print(f"[spice] ROI graph boundary: soft (dilation={args.roi_dilation}, edge_scale={args.roi_edge_scale:g})")

    # 1H-like diagnostic: per-voxel sum of graph edge weights.
    # This map is on the same scale family as the 1H "voxel-level sum prior map"
    # and can naturally reach values near O(wmax * neighbor_count).
    voxel_sum_map = np.sum(A_graph, axis=0).reshape(N, N)
    _fig_vs, _ax_vs = plt.subplots(figsize=(6, 5))
    _im_vs = _ax_vs.imshow(voxel_sum_map, origin='lower', cmap='Greens', aspect='equal')
    plt.colorbar(_im_vs, ax=_ax_vs)
    _ax_vs.set_title("Voxel-level sum prior map (graph weights)", fontsize=11)
    _ax_vs.set_xlabel("x (voxel)")
    _ax_vs.set_ylabel("y (voxel)")
    _fig_vs.tight_layout()
    _fig_vs.savefig(os.path.join(fig_dir, "spatial_prior_voxel_sum_map.png"), dpi=150, bbox_inches='tight')
    plt.close(_fig_vs)
    print(f"[spice] Saved graph weight-sum prior map -> {fig_dir}/spatial_prior_voxel_sum_map.png")
    adj_spec = FIDToSpec(adj_fid_img.conj(), axis=-1)

    # ── 8b. Adjoint-only mode: plot spectra on anatomy and exit ────
    if args.adjoint_only:
        np.save(os.path.join(data_dir, "adj_fid.npy"),  adj_fid_img.astype(np.complex64))
        np.save(os.path.join(data_dir, "adj_spec.npy"), adj_spec.astype(np.complex64))
        print("[spice] Saved adjoint FID and spectrum")

        # PCr-window magnitude map
        pcr_win = (PPM >= -1) & (PPM <= 1)
        mag_adj = np.max(np.abs(adj_spec[:, :, pcr_win]), axis=-1)
        fig_m, ax_m = plt.subplots(figsize=(5, 4))
        im_m = ax_m.imshow(mag_adj, origin='lower', cmap='viridis', aspect='equal')
        plt.colorbar(im_m, ax=ax_m)
        ax_m.set_title("Adjoint — max |spectrum| (−1 to 1 ppm / PCr)")
        fig_m.tight_layout()
        fig_m.savefig(os.path.join(fig_dir, "adj_magnitude_map.png"), dpi=150)
        plt.close(fig_m)

        # Best-SNR spectrum
        vy, vx, _ = find_best_snr(adj_spec, roi=roi)
        peaks = [(0,'PCr'),(4.85,'Pi'),(-2.53,'γATP'),(-7.58,'αATP'),(-15.86,'βATP')]
        fig_s, ax_s = plt.subplots(figsize=(10, 4))
        ax_s.plot(PPM, np.abs(adj_spec[vy, vx, :]), 'k-', lw=0.9)
        ax_s.set_xlim(args.ppmlim[1], args.ppmlim[0])
        ax_s.set_xlabel("ppm"); ax_s.set_title(f"Adjoint rep0 best voxel [{vy},{vx}]")
        for pos, name in peaks:
            ax_s.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.5)
            ax_s.text(pos, ax_s.get_ylim()[1]*0.9, name, fontsize=7, ha='center', color='red')
        fig_s.tight_layout()
        fig_s.savefig(os.path.join(fig_dir, "adj_best_voxel.png"), dpi=150)
        plt.close(fig_s)

        # Spectra on anatomy
        plot_spectra_on_anatomy(adj_spec, PPM, args.ppmlim, anatomy_bg, fig_dir, roi=roi)
        # rename to adj_ prefix so it's distinct from SPICE version
        src = os.path.join(fig_dir, "spice_on_anatomy.png")
        dst = os.path.join(fig_dir, "adj_on_anatomy.png")
        if os.path.exists(src):
            os.rename(src, dst)
        print(f"[spice] Adjoint-only mode done. Outputs in {fig_dir}")
        return

    # Build out-of-ROI mask (flat, row-major) for Tikhonov suppression
    # roi[xi, yi] is col-first; flatten row-major → need roi.T.flatten()
    out_roi_flat = None
    if roi is not None and args.lambda_out > 0:
        out_roi_flat = ~roi.T.flatten()   # True where OUTSIDE roi

    # ── 9. SPICE reconstruction (loop over ranks) ───────────────────
    for rank in args.rank:
        V = Vh[:rank, :].conj().T
        energy = np.linalg.norm(adj_fid_flat @ V @ V.conj().T) / (np.linalg.norm(adj_fid_flat) + 1e-16)
        print(f"\n[spice] === Rank {rank}, energy captured={energy*100:.1f}% ===")

        t0 = time.time()
        spice_fid, U = spice_reconstruct(
            adj_fid_flat, nufft_op, V, WW, N, n_spec,
            rank=rank, lamda=args.lambda1, n_iter=args.n_iter,
            out_roi_mask=out_roi_flat, lambda_out=args.lambda_out)
        print(f"[spice] Done in {time.time()-t0:.1f}s")

        # Gaussian apodization: exp(-π²·lb²·t²/(4·ln2)), lb=FWHM in Hz
        if args.apodization > 0:
            t_apo = np.arange(n_spec) * dwell_s
            apo = np.exp(-np.pi**2 * args.apodization**2 * t_apo**2 / (4 * np.log(2)))
            spice_fid = spice_fid * apo
            print(f"[spice] Applied {args.apodization:.0f} Hz Gaussian apodization")

        # Conjugate FID before FFT to match scanner sign convention
        spice_spec = FIDToSpec(spice_fid.conj(), axis=-1)

        tag = f"rank{rank}"
        np.save(os.path.join(data_dir, f"spice_fid_{tag}.npy"), spice_fid)
        np.save(os.path.join(data_dir, f"spice_spec_{tag}.npy"), spice_spec)
        np.save(os.path.join(data_dir, f"U_est_{tag}.npy"), U)
        np.save(os.path.join(data_dir, f"V_sub_{tag}.npy"), V)

        rank_fig_dir = os.path.join(fig_dir, tag)
        os.makedirs(rank_fig_dir, exist_ok=True)
        plot_results(adj_spec, spice_spec, PPM, args.ppmlim, rank_fig_dir,
                     anatomy=anatomy_bg, roi=roi)
        print(f"[spice] Figures saved to {rank_fig_dir}")

    print("\n[spice] All done.")


if __name__ == "__main__":
    main()
