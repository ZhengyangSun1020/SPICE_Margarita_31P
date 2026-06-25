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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_concept_twix_31p import load_concept_twix, concept_phase_correction


# ── FFT helpers (replacing fsl_mrs) ─────────────────────────────────────

def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


# ── Trajectory & FOV (from 05b) ─────────────────────────────────────────

def build_trajectory(first_circ, n_rings, traj_pts, gamma_mhz_per_mt, fov_m, N):
    radii = np.sqrt(first_circ[0]**2 + first_circ[1]**2)
    phi0 = np.arctan2(first_circ[1], first_circ[0])
    all_kx, all_ky = [], []
    for r in range(n_rings):
        angles = 2 * np.pi * np.arange(traj_pts) / traj_pts + phi0[r]
        all_kx.append(radii[r] * np.cos(angles))
        all_ky.append(radii[r] * np.sin(angles))
    kx_raw = np.concatenate(all_kx)
    ky_raw = np.concatenate(all_ky)
    kx_cpp = kx_raw * gamma_mhz_per_mt * (fov_m / N)
    ky_cpp = ky_raw * gamma_mhz_per_mt * (fov_m / N)
    return kx_raw, ky_raw, kx_cpp, ky_cpp


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
    pos = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]])
    snv = np.array([info["snv_x"], info["snv_y"], info["snv_z"]])
    phase_vec, read_vec = _fGSLCalcPRS(snv, info["in_plane_rotation"])
    rot_mat = np.column_stack([phase_vec, read_vec, snv])
    base_pos = rot_mat.T @ pos
    vox_size = info["fov_mm"] / info["matrix_size"]
    return base_pos[0] / vox_size, base_pos[1] / vox_size


def compute_dcf(kx_cpp, ky_cpp, kx_raw, ky_raw, use_hamming=True):
    r_per_pt = np.sqrt(kx_cpp**2 + ky_cpp**2)
    weights = np.maximum(r_per_pt / (r_per_pt.max() + 1e-12), 0.01)
    if use_hamming:
        kx_norm = np.pi * kx_raw / (np.abs(kx_raw).max() + 1e-30)
        ky_norm = np.pi * ky_raw / (np.abs(ky_raw).max() + 1e-30)
        r_norm = np.sqrt(kx_norm**2 + ky_norm**2)
        weights = weights * (0.54 + 0.46 * np.cos(r_norm))
    return weights.astype(np.float32)


# ── Spatial regularization (from utils/graph.py) ────────────────────────

def calc_Bmatrix_simple(ref_img, wmax=5e3, adj=8):
    """Graph Laplacian regularization from reference image."""
    radius = 1.001 if adj == 4 else 1.415
    xy = np.indices(ref_img.shape).reshape((2, -1)).T
    Nb = KDTree(xy).query_pairs(p=2, r=radius, output_type='ndarray')

    ref_vec = ref_img.flatten()
    n_vox = ref_vec.size
    C = (np.diff(ref_vec[Nb], axis=1)**2).flatten()
    old_err = np.seterr(divide='ignore')
    W = np.minimum(1.0 / C, wmax)
    np.seterr(**old_err)

    A = csr_matrix((W, (Nb[:, 0], Nb[:, 1])), shape=(n_vox, n_vox)).toarray()
    A = 0.5 * (A + A.T)
    L = np.diag(np.sum(A, axis=0)) + np.diag(np.sum(A, axis=1)) - 2.0 * A

    D, V = np.linalg.eigh(L)
    B = np.diag(np.sqrt(np.abs(D))) @ V.T
    return B.conj().T @ B, B


# ── 31P training data generation ────────────────────────────────────────

# 31P skeletal muscle metabolites at 3T (literature values)
# Refs: Valkovic MRM 2014, Meyerspeer NMR Biomed 2020, Krssak NMR Biomed 2004
P31_METABOLITES = {
    "PCr":  {"shift_ppm": 0.0,    "lw_range": (5, 12),  "amp_range": (3.5, 4.5)},
    "Pi":   {"shift_ppm": 4.9,    "lw_range": (6, 18),  "amp_range": (0.3, 0.7)},
    "gATP": {"shift_ppm": -2.48,  "lw_range": (12, 30), "amp_range": (0.8, 1.2)},
    "aATP": {"shift_ppm": -7.52,  "lw_range": (12, 30), "amp_range": (0.8, 1.2)},
    "bATP": {"shift_ppm": -16.26, "lw_range": (15, 35), "amp_range": (0.8, 1.2)},
    "PDE":  {"shift_ppm": 2.8,    "lw_range": (10, 25), "amp_range": (0.3, 1.0)},
    "PME":  {"shift_ppm": 6.3,    "lw_range": (10, 25), "amp_range": (0.0, 0.3)},
}


def load_localizer(dcm_dir, target_shape):
    """Load DICOM localizer and downsample to target spatial grid."""
    import pydicom
    from scipy.ndimage import zoom

    dcm_files = sorted([os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir)
                        if f.endswith('.dcm')])
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        slices.append(ds.pixel_array.astype(np.float32))

    vol = np.stack(slices, axis=-1)
    mid_slice = vol[:, :, vol.shape[2] // 2]

    Ny, Nx = target_shape
    zoom_y = Ny / mid_slice.shape[0]
    zoom_x = Nx / mid_slice.shape[1]
    downsampled = zoom(mid_slice, (zoom_y, zoom_x), order=1)

    norm = (downsampled - downsampled.min()) / (downsampled.max() - downsampled.min() + 1e-12)
    print(f"[spice] Loaded localizer: {mid_slice.shape} -> {norm.shape}, "
          f"{len(dcm_files)} slices (used middle)")
    return norm


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
    n_metab = len(metabs)

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


# ── SPICE solver ────────────────────────────────────────────────────────

def spice_reconstruct(adj_fid_flat, nufft_op, V, WW, N, n_spec,
                      rank, lamda, n_iter=0, maxiter=120):
    """
    SPICE reconstruction for 2D CRT MRSI.

    Parameters
    ----------
    adj_fid_flat : (N_Vox, n_spec) pre-computed adjoint FIDs
    n_iter       : 0 = direct solve, >0 = CG with n_iter iterations
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

    if lamda > 0:
        Gram_sys = Gram + lamda * WW.astype(D_TYPE)
    else:
        Gram_sys = Gram

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

def find_best_snr(image_spec):
    Ny, Nx, n_spec = image_spec.shape
    snr_map = np.zeros((Ny, Nx))
    for y in range(Ny):
        for x in range(Nx):
            s = np.abs(image_spec[y, x, :])
            snr_map[y, x] = s.max() / (np.std(s[int(0.8*n_spec):]) + 1e-20)
    vy, vx = np.unravel_index(snr_map.argmax(), snr_map.shape)
    return vy, vx, snr_map


def plot_results(adj_spec, spice_spec, PPM, ppmlim, fig_dir):
    Ny, Nx, n_spec = spice_spec.shape
    vy, vx, snr_map = find_best_snr(spice_spec)
    print(f"[spice] Best SNR voxel: [{vy},{vx}] SNR={snr_map[vy, vx]:.1f}")

    peaks = [(0, 'PCr'), (5, 'Pi'), (-2.5, 'γ-ATP'), (-7.5, 'α-ATP'), (-16, 'β-ATP')]

    # ── Magnitude maps comparison ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mag_adj = np.mean(np.abs(adj_spec), axis=-1)
    mag_sp = np.mean(np.abs(spice_spec), axis=-1)
    im1 = axes[0].imshow(mag_adj, origin="lower", cmap="viridis")
    axes[0].set_title("Adjoint — mean |spectrum|")
    plt.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(mag_sp, origin="lower", cmap="viridis")
    axes[1].set_title("SPICE — mean |spectrum|")
    plt.colorbar(im2, ax=axes[1])
    axes[2].plot(PPM, np.abs(adj_spec[vy, vx, :]), 'b-', alpha=0.5, lw=0.8, label='Adjoint')
    axes[2].plot(PPM, np.abs(spice_spec[vy, vx, :]), 'r-', lw=1.0, label='SPICE')
    axes[2].set_xlim(ppmlim[1], ppmlim[0])
    axes[2].set_xlabel("ppm"); axes[2].set_ylabel("|spectrum|")
    axes[2].set_title(f"Voxel [{vy},{vx}]")
    axes[2].legend(fontsize=8)
    for pos, name in peaks:
        axes[2].axvline(pos, color='gray', ls='--', lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spice_vs_adjoint.png"), dpi=150)
    plt.close(fig)

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

    # ── All spectra grid ────────────────────────────────────────────
    fig, axes = plt.subplots(Ny, Nx, figsize=(Nx*2.5, Ny*2), sharex=True, sharey=True)
    if Ny == 1: axes = axes[np.newaxis, :]
    if Nx == 1: axes = axes[:, np.newaxis]
    for y in range(Ny):
        for x in range(Nx):
            ax = axes[Ny-1-y, x]
            sp = np.abs(spice_spec[y, x, :])
            ax.plot(PPM, sp, 'k-', lw=0.4)
            ax.fill_between(PPM, 0, sp, alpha=0.08, color='blue')
            ax.set_xlim(ppmlim[1], ppmlim[0])
            ax.set_title(f"[{y},{x}]", fontsize=7, pad=1)
            ax.tick_params(labelsize=4)
    fig.suptitle("SPICE — All voxel spectra", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(fig_dir, "spice_all_spectra.png"), dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="31P CONCEPT CRT SPICE reconstruction")
    p.add_argument("--dat", required=True, help="Siemens Twix .dat file")
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--no-dcf", action="store_true")
    p.add_argument("--no-hamming", action="store_true")
    p.add_argument("--rank", type=int, nargs="+", default=[15],
                   help="SPICE subspace rank(s) — pass multiple to compare")
    p.add_argument("--lambda1", type=float, default=0.0, help="Spatial regularization weight")
    p.add_argument("--n-iter", type=int, default=0,
                   help="CG iterations (0 = direct solve, >0 = iterative)")
    p.add_argument("--maxiter", type=int, default=120, help="CG max iterations (large grids only)")
    p.add_argument("--training-size", type=int, default=5000)
    p.add_argument("--subspace-src", default="adjoint",
                   choices=["adjoint", "csi", "synthetic", "basis-dat"],
                   help="Subspace source: adjoint, csi, synthetic, or basis-dat")
    p.add_argument("--basis-dat", default=None,
                   help="Reference .dat file for subspace (--subspace-src basis-dat)")
    p.add_argument("--csi-prior", default=None,
                   help="Path to CSI DICOM (required when --subspace-src csi)")
    p.add_argument("--localizer", default=None,
                   help="Path to DICOM localizer dir for spatial prior")
    p.add_argument("--wmax", type=float, default=5e3, help="Max edge weight for spatial prior")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    fig_dir = os.path.join(args.out_dir, "figures")
    for d in [data_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load raw data ────────────────────────────────────────────
    info = load_concept_twix(args.dat)
    kdata = info["kdata"]
    N = info["matrix_size"]
    n_spec = info["n_spec"]
    n_rings = info["n_rings"]
    traj_pts = int(info["traj_pts"][0])
    dwell_s = info["dwell_s"] * 2
    larmor_mhz = info["larmor_freq_mhz"]
    gamma_mhz_per_mt = info["gamma_2pi"] / 1e9

    print(f"\n[spice] Matrix={N}x{N}, vecSize={n_spec}, rings={n_rings}")

    # ── 2. Phase correction ─────────────────────────────────────────
    if not args.no_phase_corr:
        kdata = concept_phase_correction(kdata, info["_ti_per_ring"])

    # ── 3. Trajectory ───────────────────────────────────────────────
    kx_raw, ky_raw, kx_cpp, ky_cpp = build_trajectory(
        info["first_circle"], n_rings, traj_pts,
        gamma_mhz_per_mt, info["fov_m"], N)
    traj = np.stack([kx_cpp, ky_cpp], axis=-1).astype(np.float32)

    # ── 4. FOV shift ────────────────────────────────────────────────
    kdata_flat = kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
    if not args.no_fov_shift:
        d1, d2 = compute_fov_shift(info)
        shift_phase = np.exp(-2j * np.pi * (kx_cpp * d1 + ky_cpp * d2))
        kdata_flat = kdata_flat * shift_phase[:, np.newaxis]
        print(f"[spice] FOV shift: {d1:.4f}, {d2:.4f} vox")

    # ── 5. DCF ──────────────────────────────────────────────────────
    dcf = None
    if not args.no_dcf:
        dcf = compute_dcf(kx_cpp, ky_cpp, kx_raw, ky_raw,
                          use_hamming=not args.no_hamming)
        print(f"[spice] DCF + Hamming applied")

    # ── 6. NUFFT operator ───────────────────────────────────────────
    nufft_op = mrinufft.get_operator(
        "finufft", samples=traj, shape=(N, N),
        n_coils=1, density=dcf if dcf is not None else False)

    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    PPM = FREQ / larmor_mhz + args.ppm_center

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
        ref_info = load_concept_twix(args.basis_dat)
        ref_kdata = ref_info["kdata"]
        if not args.no_phase_corr:
            ref_kdata = concept_phase_correction(ref_kdata, ref_info["_ti_per_ring"])
        ref_N = ref_info["matrix_size"]
        ref_n_spec = ref_info["n_spec"]
        ref_traj_pts = int(ref_info["traj_pts"][0])
        ref_gamma = ref_info["gamma_2pi"] / 1e9
        _, _, ref_kx, ref_ky = build_trajectory(
            ref_info["first_circle"], ref_info["n_rings"], ref_traj_pts,
            ref_gamma, ref_info["fov_m"], ref_N)
        ref_traj = np.stack([ref_kx, ref_ky], axis=-1).astype(np.float32)
        ref_dcf = compute_dcf(ref_kx, ref_ky,
            *build_trajectory(ref_info["first_circle"], ref_info["n_rings"],
                              ref_traj_pts, ref_gamma, ref_info["fov_m"], ref_N)[:2])
        ref_nufft = mrinufft.get_operator(
            "finufft", samples=ref_traj, shape=(ref_N, ref_N),
            n_coils=1, density=ref_dcf)
        ref_flat = ref_kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, ref_n_spec)
        if not args.no_fov_shift:
            rd1, rd2 = compute_fov_shift(ref_info)
            ref_flat = ref_flat * np.exp(-2j * np.pi * (ref_kx * rd1 + ref_ky * rd2))[:, np.newaxis]
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
    else:
        print(f"[spice] Generating {args.training_size} synthetic 31P training FIDs ...")
        training = generate_training_data(n_spec, dwell_s, larmor_mhz, args.training_size)

    _, s_vals, Vh = np.linalg.svd(training, full_matrices=False)
    max_rank = max(args.rank)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(s_vals[:min(30, len(s_vals))], "x-"); ax1.set_title("Singular values")
    for k in range(min(6, max_rank)):
        ax2.plot(PPM, np.abs(FIDToSpec(Vh[k, :])), label=f"V{k}")
    ax2.set_xlim(20, -20); ax2.set_xlabel("ppm")
    ax2.set_title("Top subspace vectors (spectrum domain)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spice_subspace.png"), dpi=120)
    plt.close(fig)

    if args.localizer:
        spatial_prior = load_localizer(args.localizer, (N, N))
        print(f"[spice] Using localizer as spatial prior")
    else:
        spatial_prior = mag_norm
    WW, B = calc_Bmatrix_simple(spatial_prior, wmax=args.wmax, adj=8)
    adj_spec = FIDToSpec(adj_fid_img, axis=-1)

    # ── 9. SPICE reconstruction (loop over ranks) ───────────────────
    for rank in args.rank:
        V = Vh[:rank, :].conj().T
        energy = np.linalg.norm(adj_fid_flat @ V @ V.conj().T) / (np.linalg.norm(adj_fid_flat) + 1e-16)
        print(f"\n[spice] === Rank {rank}, energy captured={energy*100:.1f}% ===")

        t0 = time.time()
        spice_fid, U = spice_reconstruct(
            adj_fid_flat, nufft_op, V, WW, N, n_spec,
            rank=rank, lamda=args.lambda1, n_iter=args.n_iter)
        print(f"[spice] Done in {time.time()-t0:.1f}s")

        spice_spec = FIDToSpec(spice_fid, axis=-1)

        tag = f"rank{rank}"
        np.save(os.path.join(data_dir, f"spice_fid_{tag}.npy"), spice_fid)
        np.save(os.path.join(data_dir, f"spice_spec_{tag}.npy"), spice_spec)
        np.save(os.path.join(data_dir, f"U_est_{tag}.npy"), U)
        np.save(os.path.join(data_dir, f"V_sub_{tag}.npy"), V)

        rank_fig_dir = os.path.join(fig_dir, tag)
        os.makedirs(rank_fig_dir, exist_ok=True)
        plot_results(adj_spec, spice_spec, PPM, args.ppmlim, rank_fig_dir)
        print(f"[spice] Figures saved to {rank_fig_dir}")

    print("\n[spice] All done.")


if __name__ == "__main__":
    main()
