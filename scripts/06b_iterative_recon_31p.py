#!/usr/bin/env python3
"""
31P CONCEPT CRT iterative NUFFT reconstruction (CG solver).

Adapted from 06_iterative_nufft_recon.py (1H) for 31P CONCEPT CRT data.

Solves per spectral point:  (F^H F + lam I) x_t = F^H y_t
where F is the 2D NUFFT (same for all spectral points).

Key differences from 1H pipeline:
  - Single coil (no sensitivity maps)
  - 2D CRT trajectory (same Gram for all spectral points)
  - No B0 correction, no lipid removal
  - Precomputed Gram matrix for small grids

Usage:
    python 06b_iterative_recon_31p.py --dat "path/to/meas_*.dat" --out-dir ./output
    python 06b_iterative_recon_31p.py --dat "path/to/meas_*.dat" --maxiter 30 --lam 1e-4
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
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_concept_twix_31p import load_concept_twix, concept_phase_correction


# ── FFT helper ──────────────────────────────────────────────────────────

def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


# ── Trajectory & FOV (shared with 05b/05c) ──────────────────────────────

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


# ── Iterative reconstruction ────────────────────────────────────────────

def iterative_recon_2d(kdata_flat, nufft_op, N, n_spec, maxiter=30,
                       lam=0.0, solver="cg", rtol=1e-3):
    """
    Iterative CG reconstruction for 2D CRT MRSI.

    Since the 2D Gram F^H F is the same for all spectral points,
    precompute it once and solve all spectral points in one CG call
    on the full (N_Vox * n_spec) vector.

    Parameters
    ----------
    kdata_flat : (n_kpts, n_spec) k-space data
    nufft_op   : mrinufft 2D NUFFT operator (with DCF)
    N          : matrix size (NxN grid)
    n_spec     : number of spectral points
    maxiter    : CG iterations
    lam        : Tikhonov regularization (0 = none)
    """
    N_Vox = N * N
    D_TYPE = np.complex64

    # ── Precompute Gram matrix ──────────────────────────────────────
    print("[iter] Precomputing Gram matrix ...")
    Gram = np.zeros((N_Vox, N_Vox), dtype=D_TYPE)
    for v in tqdm(range(N_Vox), desc="Gram columns"):
        delta = np.zeros((N, N), dtype=D_TYPE)
        delta.flat[v] = 1.0
        Gram[:, v] = nufft_op.adj_op(nufft_op.op(delta)).reshape(N_Vox)

    if lam > 0:
        Gram += lam * np.eye(N_Vox, dtype=D_TYPE)

    # ── RHS: F^H y for each spectral point ──────────────────────────
    print("[iter] Computing adjoint (F^H y) ...")
    b = np.zeros((N_Vox, n_spec), dtype=D_TYPE)
    for t in tqdm(range(n_spec), desc="Adjoint", miniters=100):
        b[:, t] = nufft_op.adj_op(kdata_flat[:, t]).reshape(N_Vox)

    # ── Solve ───────────────────────────────────────────────────────
    # Gram is (N_Vox, N_Vox), same for all spectral points.
    # Solve Gram @ X = B where X, B are (N_Vox, n_spec).
    #
    # For small grids: direct solve is fast.
    # For iterative: CG on the full (N_Vox * n_spec) vector,
    # where the matvec applies Gram to each spectral column.

    if maxiter == 0:
        print(f"[iter] Direct solve ({N_Vox}x{N_Vox}, {n_spec} spectral pts) ...")
        X = np.linalg.solve(Gram, b)
    else:
        print(f"[iter] CG solve ({maxiter} iters, lam={lam}) ...")
        total_size = N_Vox * n_spec

        def mv(x_vec):
            X = x_vec.reshape(N_Vox, n_spec)
            return (Gram @ X).ravel().astype(D_TYPE)

        A_op = LinearOperator((total_size, total_size), matvec=mv, dtype=D_TYPE)

        x0 = b.ravel().astype(D_TYPE)
        b_flat = b.ravel().astype(D_TYPE)

        pbar = tqdm(total=maxiter, desc="CG iters")
        def _cb(xk):
            pbar.update(1)

        x_flat, info = cg(A_op, b_flat, x0=x0, maxiter=maxiter, rtol=rtol, callback=_cb)
        pbar.close()
        X = x_flat.reshape(N_Vox, n_spec)
        print(f"[iter] CG info={info}")

    recon_fid = X.reshape(N, N, n_spec)
    adj_fid = b.reshape(N, N, n_spec)
    return recon_fid, adj_fid


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


def plot_results(adj_spec, iter_spec, PPM, ppmlim, fig_dir):
    Ny, Nx, n_spec = iter_spec.shape
    vy, vx, snr_map = find_best_snr(iter_spec)
    print(f"[iter] Best SNR voxel: [{vy},{vx}] SNR={snr_map[vy, vx]:.1f}")

    peaks = [(0, 'PCr'), (5, 'Pi'), (-2.5, 'g-ATP'), (-7.5, 'a-ATP'), (-16, 'b-ATP')]

    # ── Comparison: adjoint vs iterative ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mag_adj = np.mean(np.abs(adj_spec), axis=-1)
    mag_it = np.mean(np.abs(iter_spec), axis=-1)
    im1 = axes[0].imshow(mag_adj, origin="lower", cmap="viridis")
    axes[0].set_title("Adjoint"); plt.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(mag_it, origin="lower", cmap="viridis")
    axes[1].set_title("Iterative CG"); plt.colorbar(im2, ax=axes[1])
    axes[2].plot(PPM, np.abs(adj_spec[vy, vx, :]), 'b-', alpha=0.5, lw=0.8, label='Adjoint')
    axes[2].plot(PPM, np.abs(iter_spec[vy, vx, :]), 'r-', lw=1.0, label='Iterative')
    axes[2].set_xlim(ppmlim[1], ppmlim[0])
    axes[2].set_xlabel("ppm"); axes[2].set_title(f"Voxel [{vy},{vx}]")
    axes[2].legend(fontsize=8)
    for pos, _ in peaks:
        axes[2].axvline(pos, color='gray', ls='--', lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "iter_vs_adjoint.png"), dpi=150)
    plt.close(fig)

    # ── Best SNR spectrum ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    sp = np.abs(iter_spec[vy, vx, :])
    ax.plot(PPM, sp, 'k-', lw=0.8)
    ax.fill_between(PPM, 0, sp, alpha=0.15, color='blue')
    ax.set_xlim(ppmlim[1], ppmlim[0]); ax.set_ylim(bottom=0)
    ax.set_xlabel("ppm"); ax.set_ylabel("|spectrum|")
    ax.set_title(f"Iterative CG -- Voxel [{vy},{vx}] (SNR={snr_map[vy,vx]:.1f})")
    for pos, name in peaks:
        ax.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.text(pos, ax.get_ylim()[1]*0.92, name, fontsize=8, ha='center', color='red')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "iter_spectrum_best_snr.png"), dpi=150)
    plt.close(fig)

    # ── All spectra grid ────────────────────────────────────────────
    fig, axes = plt.subplots(Ny, Nx, figsize=(Nx*2.5, Ny*2), sharex=True, sharey=True)
    if Ny == 1: axes = axes[np.newaxis, :]
    if Nx == 1: axes = axes[:, np.newaxis]
    for y in range(Ny):
        for x in range(Nx):
            ax = axes[Ny-1-y, x]
            sp = np.abs(iter_spec[y, x, :])
            ax.plot(PPM, sp, 'k-', lw=0.4)
            ax.fill_between(PPM, 0, sp, alpha=0.08, color='blue')
            ax.set_xlim(ppmlim[1], ppmlim[0])
            ax.set_title(f"[{y},{x}]", fontsize=7, pad=1)
            ax.tick_params(labelsize=4)
    fig.suptitle("Iterative CG -- All voxel spectra", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(fig_dir, "iter_all_spectra.png"), dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="31P CONCEPT CRT iterative NUFFT reconstruction")
    p.add_argument("--dat", required=True, help="Siemens Twix .dat file")
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--no-dcf", action="store_true")
    p.add_argument("--no-hamming", action="store_true")
    p.add_argument("--maxiter", type=int, default=150,
                   help="CG iterations (0 = direct solve)")
    p.add_argument("--lam", type=float, default=0.0,
                   help="Tikhonov regularization weight")
    p.add_argument("--solver", default="cg", choices=["cg"])
    p.add_argument("--rtol", type=float, default=1e-3)
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

    print(f"\n[iter] Matrix={N}x{N}, vecSize={n_spec}, rings={n_rings}")

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
        print(f"[iter] FOV shift: {d1:.4f}, {d2:.4f} vox")

    # ── 5. DCF ──────────────────────────────────────────────────────
    dcf = None
    if not args.no_dcf:
        dcf = compute_dcf(kx_cpp, ky_cpp, kx_raw, ky_raw,
                          use_hamming=not args.no_hamming)
        print("[iter] DCF + Hamming applied")

    # ── 6. NUFFT operator ───────────────────────────────────────────
    # Like 06 (1H): iterative solver uses pure F^H F (no DCF).
    # DCF is only used for the adjoint (05b) as a quick approximation.
    nufft_op = mrinufft.get_operator(
        "finufft", samples=traj, shape=(N, N),
        n_coils=1, density=False)

    # ── 7. Iterative reconstruction ─────────────────────────────────
    t0 = time.time()
    recon_fid, adj_fid = iterative_recon_2d(
        kdata_flat, nufft_op, N, n_spec,
        maxiter=args.maxiter, lam=args.lam,
        solver=args.solver, rtol=args.rtol)
    elapsed = time.time() - t0
    print(f"[iter] Reconstruction done in {elapsed:.1f}s")

    recon_spec = FIDToSpec(recon_fid, axis=-1)
    adj_spec = FIDToSpec(adj_fid, axis=-1)

    # ── 8. Save ─────────────────────────────────────────────────────
    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    PPM = FREQ / larmor_mhz + args.ppm_center

    np.save(os.path.join(data_dir, "iter_recon_fid.npy"), recon_fid)
    np.save(os.path.join(data_dir, "iter_recon_spec.npy"), recon_spec)
    np.save(os.path.join(data_dir, "adjoint_fid.npy"), adj_fid)
    print(f"[iter] Saved to {data_dir}")

    # ── 9. Figures ──────────────────────────────────────────────────
    plot_results(adj_spec, recon_spec, PPM, args.ppmlim, fig_dir)
    print(f"[iter] Figures saved to {fig_dir}")
    print("[iter] Done.")


if __name__ == "__main__":
    main()
