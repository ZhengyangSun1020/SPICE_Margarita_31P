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


def compute_crt_dcf_hamming(traj):
    """CRT Hamming DCF matching MATLAB op_CalcAndApplyDensComp.m."""
    r = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    unique_r = np.unique(np.round(r, 6))
    n_rings = len(unique_r)
    r_norm = unique_r / unique_r.max()
    N_per_ring = np.array([int((np.abs(r - ri) < 1e-5).sum()) for ri in unique_r])
    max_pts = int(N_per_ring.max())
    dcf_ring = np.zeros(n_rings)
    for i in range(n_rings):
        if i == 0:
            r_out = (r_norm[0] + r_norm[1]) / 2.0
            area = np.pi * r_out**2
        elif i == n_rings - 1:
            r_in = (r_norm[-2] + r_norm[-1]) / 2.0
            r_out = 1.5 * r_norm[-1] - 0.5 * r_norm[-2]
            area = np.pi * (r_out**2 - r_in**2)
        else:
            r_in = (r_norm[i - 1] + r_norm[i]) / 2.0
            r_out = (r_norm[i] + r_norm[i + 1]) / 2.0
            area = np.pi * (r_out**2 - r_in**2)
        dcf_ring[i] = area * (max_pts / N_per_ring[i])
    dcf = np.zeros(len(r), dtype=np.float32)
    for i, ri in enumerate(unique_r):
        dcf[np.abs(r - ri) < 1e-5] = dcf_ring[i]
    dcf *= (0.54 + 0.46 * np.cos(2.0 * np.pi * r)).astype(np.float32)
    return dcf


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
    "PCr":  {"shift_ppm":  0.0,    "lw_range": (5, 15),  "amp_range": (3.5, 4.5)},
    "Pi":   {"shift_ppm":  4.85,   "lw_range": (5, 20),  "amp_range": (0.0, 0.8)},
    "Pi2":  {"shift_ppm":  4.25,   "lw_range": (5, 20),  "amp_range": (0.0, 0.5)},
    "PDE":  {"shift_ppm":  3.07,   "lw_range": (10, 35), "amp_range": (0.0, 0.3)},
    "gATP": {"shift_ppm": -2.53,   "lw_range": (10, 30), "amp_range": (0.8, 1.2)},
    "aATP": {"shift_ppm": -7.58,   "lw_range": (10, 30), "amp_range": (0.8, 1.2)},
    "bATP": {"shift_ppm": -15.86,  "lw_range": (12, 35), "amp_range": (0.8, 1.2)},
}


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
            sig += cm * np.exp(-lw * np.pi * time_axis) * np.exp(-2j * np.pi * freq_hz * time_axis)
        whole_shift = rng.uniform(-0.05, 0.05) * larmor_mhz
        sig *= np.exp(-2j * np.pi * whole_shift * time_axis)
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

    # ── All spectra grid — per-voxel normalised, axis off ──────────
    fig, axes = plt.subplots(Nx, Ny, figsize=(Ny * 2.5, Nx * 2))
    axes = np.array(axes).reshape(Nx, Ny)
    for yi in range(Ny):
        for xi in range(Nx):
            ax = axes[Nx - 1 - xi, yi]
            sp_v = np.nan_to_num(np.abs(spice_spec[xi, yi, :]))
            sp_v = sp_v / (sp_v.max() + 1e-20)
            ax.plot(sp_v, 'k', lw=0.4)
            ax.set_xlim(0, len(sp_v) - 1)
            ax.set_ylim(0, 1.05)
            ax.axis("off")
    fig.suptitle("SPICE — All voxel spectra", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spice_all_spectra.png"), dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="31P CONCEPT CRT SPICE reconstruction")
    p.add_argument("--dat", required=True, help="Siemens Twix .dat file")
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
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
                   choices=["adjoint", "csi", "synthetic", "basis-dat"],
                   help="Subspace source: adjoint, csi, synthetic, or basis-dat")
    p.add_argument("--basis-dat", default=None,
                   help="Reference .dat file for subspace (--subspace-src basis-dat)")
    p.add_argument("--csi-prior", default=None,
                   help="Path to CSI DICOM (required when --subspace-src csi)")
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
    info = load_concept_twix_3d(args.dat)
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
        shift_phase = np.exp(-2j * np.pi * (traj[:, 0] * d_pe + traj[:, 1] * d_ro))
        kdata_flat = kdata_flat * shift_phase[:, np.newaxis]
        print(f"[spice] FOV shift: pe={d_pe:.4f}, ro={d_ro:.4f} vox")

    # ── 5. DCF (Hamming CRT, matches op_CalcAndApplyDensComp.m) ────
    dcf = compute_crt_dcf_hamming(traj)
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
        ref_dcf = compute_crt_dcf_hamming(ref_traj)
        ref_nufft = mrinufft.get_operator(
            "finufft", samples=ref_traj, shape=(ref_N, ref_N),
            n_coils=1, density=ref_dcf)
        ref_flat = ref_kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, ref_n_spec)
        if not args.no_fov_shift:
            rd_pe, rd_ro = compute_fov_shift(ref_info)
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
    else:
        print(f"[spice] Generating {args.training_size} synthetic 31P training FIDs ...")
        training = generate_training_data(n_spec, dwell_s, larmor_mhz, args.training_size)

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
    fig.savefig(os.path.join(fig_dir, "spice_subspace.png"), dpi=120)
    plt.close(fig)

    WW, B = calc_Bmatrix_simple(mag_norm, wmax=args.wmax, adj=8)
    adj_spec = FIDToSpec(adj_fid_img.conj(), axis=-1)

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
        plot_results(adj_spec, spice_spec, PPM, args.ppmlim, rank_fig_dir)
        print(f"[spice] Figures saved to {rank_fig_dir}")

    print("\n[spice] All done.")


if __name__ == "__main__":
    main()
