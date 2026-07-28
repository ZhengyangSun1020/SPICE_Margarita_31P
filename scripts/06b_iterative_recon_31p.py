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


# ── wSVD coil combination ───────────────────────────────────────────────

def wsvd_combine(fid_mc, noise_cov=None):
    """Per-voxel weighted SVD coil combination in frequency domain.

    Parameters
    ----------
    fid_mc    : (n_spec, n_ch, Nx, Ny[, Nz])  per-coil FID images (complex128)
    noise_cov : (n_ch, n_ch) noise covariance matrix, or None

    Returns
    -------
    combined_fid : (n_spec, Nx, Ny[, Nz])
    quality      : (Nx, Ny[, Nz])
    """
    n_spec, n_ch = fid_mc.shape[:2]
    spatial_shape = fid_mc.shape[2:]
    n_vox = int(np.prod(spatial_shape))

    spec_mc   = np.fft.fftshift(np.fft.fft(fid_mc, axis=0), axes=0)
    spec_flat = spec_mc.reshape(n_spec, n_ch, n_vox)

    if noise_cov is None or n_ch == 1:
        scale_mat     = np.eye(n_ch, dtype=np.complex128) * np.sqrt(0.5)
        inv_scale_mat = np.eye(n_ch, dtype=np.complex128) / np.sqrt(0.5)
    else:
        noiseCov = np.asarray(noise_cov, dtype=np.complex128)
        noiseVal, noiseVec = np.linalg.eigh(noiseCov)
        noiseVal      = np.maximum(np.real(noiseVal), 1e-30)
        scale_mat     = noiseVec @ np.diag(np.sqrt(0.5) / np.sqrt(noiseVal))
        inv_scale_mat = np.linalg.inv(scale_mat)

    combined_fd = np.zeros((n_spec, n_vox), dtype=np.complex128)
    quality     = np.zeros(n_vox)
    sqrt_nch    = np.sqrt(n_ch)

    for i in range(n_vox):
        raw    = spec_flat[:, :, i]
        scaled = raw @ scale_mat
        u, s, vh = np.linalg.svd(scaled, full_matrices=False)
        s_norm     = np.linalg.norm(s) + 1e-30
        quality[i] = float((s[0] / s_norm * sqrt_nch - 1) / (sqrt_nch - 1 + 1e-30))
        v1 = vh[0, :].conj()
        svdCoilAmplitudes = v1 @ inv_scale_mat
        ref        = svdCoilAmplitudes[0]
        svdRescale = np.linalg.norm(svdCoilAmplitudes) * ref / (abs(ref) + 1e-30)
        combined_fd[:, i] = u[:, 0] * s[0] * svdRescale

    combined_fd  = combined_fd.reshape((n_spec,) + spatial_shape)
    quality      = quality.reshape(spatial_shape)
    combined_fid = np.fft.ifft(np.fft.ifftshift(combined_fd, axes=0), axis=0)
    return combined_fid, quality


# ── Iterative reconstruction ────────────────────────────────────────────

def iterative_recon_2d(kdata_mc, nufft_op, N, n_spec, n_ch=1,
                        noise_cov=None, maxiter=30, lam=0.0, solver="cg", rtol=1e-3):
    """Iterative CG reconstruction for 2D CRT MRSI.

    For multi-coil (n_ch > 1): Gram is computed once, each coil solved with
    the same Gram and its own RHS, then wSVD combines the per-coil images.
    For n_ch=1: single-coil path with direct or CG solve.

    kdata_mc : (n_kpts, n_spec, n_ch)
    Returns (recon_fid, adj_fid) each (N, N, n_spec).
    """
    N_Vox  = N * N
    D_TYPE = np.complex64
    G_TYPE = np.complex128  # λ_min ≈ 9e-7 → must use float64

    # Precompute Gram matrix once (same for all coils and spectral points)
    print("[iter] Precomputing Gram matrix (float64) ...")
    Gram = np.zeros((N_Vox, N_Vox), dtype=G_TYPE)
    for v in tqdm(range(N_Vox), desc="Gram columns"):
        delta = np.zeros((N, N), dtype=D_TYPE)
        delta.flat[v] = 1.0
        Gram[:, v] = nufft_op.adj_op(nufft_op.op(delta)).reshape(N_Vox)

    if lam > 0:
        lam_max = np.real(np.linalg.eigvalsh(Gram)).max()
        lam_eff = lam * lam_max
        print(f"[iter] lam_max(Gram)={lam_max:.4g}, lam_rel={lam:.4g} -> lam_eff={lam_eff:.4g}")
        Gram += lam_eff * np.eye(N_Vox, dtype=G_TYPE)

    total_size = N_Vox * n_spec

    def mv(x_vec):
        return (Gram @ x_vec.reshape(N_Vox, n_spec)).ravel()

    A_op = LinearOperator((total_size, total_size), matvec=mv, dtype=G_TYPE)

    # Per-coil images: (n_spec, n_ch, N, N)
    X_mc   = np.zeros((n_spec, n_ch, N, N), dtype=G_TYPE)
    adj_mc = np.zeros((n_spec, n_ch, N, N), dtype=G_TYPE)

    for ch in range(n_ch):
        label = f"Coil {ch+1}/{n_ch}" if n_ch > 1 else "Single coil"
        print(f"[iter] {label} — adjoint + solve ...")
        kdata_ch = kdata_mc[:, :, ch]   # (n_kpts, n_spec)

        b_ch = np.zeros((N_Vox, n_spec), dtype=G_TYPE)
        for t in tqdm(range(n_spec), desc="Adjoint", miniters=100):
            b_ch[:, t] = nufft_op.adj_op(kdata_ch[:, t]).reshape(N_Vox)

        adj_mc[:, ch] = b_ch.T.reshape(n_spec, N, N)

        if maxiter == 0:
            X_ch = np.linalg.solve(Gram, b_ch)       # (N_Vox, n_spec)
        else:
            pbar = tqdm(total=maxiter, desc=f"CG ch{ch}", leave=(n_ch == 1))
            def _cb(xk): pbar.update(1)
            x_flat, info_code = cg(A_op, b_ch.ravel(),
                                   x0=np.zeros(total_size, dtype=G_TYPE),
                                   maxiter=maxiter, rtol=rtol, callback=_cb)
            pbar.close()
            print(f"[iter]   CG info={info_code}")
            X_ch = x_flat.reshape(N_Vox, n_spec)

        X_mc[:, ch] = X_ch.T.reshape(n_spec, N, N)

    # wSVD combination: (n_spec, n_ch, N, N) → (n_spec, N, N)
    print("[iter] Applying wSVD coil combination ...")
    recon_fid, quality = wsvd_combine(X_mc,   noise_cov=noise_cov)
    adj_fid,   _       = wsvd_combine(adj_mc, noise_cov=noise_cov)

    if n_ch > 1:
        print(f"[iter] wSVD quality  mean={float(np.mean(quality)):.3f}")

    # (n_spec, N, N) → (N, N, n_spec)
    return (recon_fid.transpose(1, 2, 0).astype(D_TYPE),
            adj_fid.transpose(1, 2, 0).astype(D_TYPE))


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

    # ── 4. Flatten kdata: (n_kpts, n_spec, n_ch) ────────────────────
    n_ch = kdata.shape[2]
    kdata_flat_mc = kdata.transpose(3, 0, 1, 2).reshape(-1, n_spec, n_ch)
    if not args.no_fov_shift:
        d1, d2 = compute_fov_shift(info)
        shift_phase = np.exp(-2j * np.pi * (kx_cpp * d1 + ky_cpp * d2))
        kdata_flat_mc = kdata_flat_mc * shift_phase[:, np.newaxis, np.newaxis]
        print(f"[iter] FOV shift: {d1:.4f}, {d2:.4f} vox")
    print(f"[iter] kdata_flat_mc: {kdata_flat_mc.shape}  (n_kpts, n_spec, n_ch={n_ch})")

    # ── 5. NUFFT operator ───────────────────────────────────────────
    nufft_op = mrinufft.get_operator(
        "finufft", samples=traj, shape=(N, N),
        n_coils=1, density=False)

    # ── 6. Per-coil iterative reconstruction + wSVD ─────────────────
    t0 = time.time()
    recon_fid, adj_fid = iterative_recon_2d(
        kdata_flat_mc, nufft_op, N, n_spec, n_ch=n_ch,
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
