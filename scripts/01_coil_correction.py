#!/usr/bin/env python3
"""
Step 1 (alt) — MORSE-PI coil sensitivity estimation from water-reference k-space.

Alternative to 01_coilmap.py, which starts from a pre-computed ESPIRiT ecalib.npy.
This script computes coil sensitivities directly from raw k-space using the
MORSE-PI method implemented in coil_sens.py.

Reads  : <data_dir>/wref_data.npy  (water-ref k-space)
         <data_dir>/wref_ksp.npy   (k-space trajectory)
         <data_dir>/wref_o.npy     (magnitude image for brain mask)
Writes : <out_dir>/coilmap/ecalib_pp.npy   (same format as 01_coilmap.py output)
         <out_dir>/coilmap/fig_01c_*.png   (when --save-plots)

Usage:
    python scripts/01_coil_correction.py \\
        --data-dir ./invivo_260305/cr/ \\
        [--out-dir ./output] [--backend finufft] [--n-ref 6] [--dim 64 64]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from warnings import filterwarnings
filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.coil_sens import morse_pi


def parse_args():
    p = argparse.ArgumentParser(description="MORSE-PI coil sensitivity — step 1 alt")
    p.add_argument("--data-dir",        required=True,
                   help="Raw data directory (wref_data.npy, wref_ksp.npy, wref_o.npy)")
    p.add_argument("--out-dir",         default="./output")
    p.add_argument("--dim",             type=int, nargs=2, default=[64, 64],
                   metavar=("NY", "NX"), help="Reconstructed image size (default: 64 64)")
    p.add_argument("--n-coils",         type=int, default=32)
    # MORSE-PI options
    p.add_argument("--backend",         default="finufft",
                   help="mrinufft backend (default: finufft)")
    p.add_argument("--n-ref",           type=int, default=6,
                   help="MORSE-PI N_ref (number of reference coils, default: 6)")
    p.add_argument("--smoothing-sd",    type=int, default=3,
                   help="Gaussian smoothing SD for MORSE-PI E matrix (default: 3)")
    p.add_argument("--calib-width",     type=int, default=12,
                   help="Calibration region half-width in k-space (default: 12)")
    p.add_argument("--max-iter",        type=int, default=30,
                   help="CG iterations for pinv_solver in MORSE-PI (default: 30)")
    # Trajectory — same scaling as 02_b0_correction.py uses for wref_ksp
    p.add_argument("--ksp-scale",       type=float, default=None,
                   help="Scale for wref_ksp kx/ky axes (default: 15.7139.../16, same as step 02)")
    # Post-processing (same as 01_coilmap.py)
    p.add_argument("--brain-threshold", type=float, default=5e-6)
    p.add_argument("--median-filter",   type=int,   default=3,
                   help="Median filter window after coil map (default: 3, 0=off)")
    p.add_argument("--save-plots",      action="store_true")
    return p.parse_args()


def _load_kspace_and_traj(data_dir, n_coils, ksp_scale):
    """
    Load wref k-space and trajectory, mirroring 02_b0_correction.py exactly.

    Returns:
        data : (N_coils, K_points * N_seq)  complex64   — all temporal steps
        traj : (K_points * N_seq, 2)        float32      — (kx, ky) in [-pi, pi]
    """
    # ── trajectory: same scaling as 02 ───────────────────────────────────────
    wref_ksp = np.load(data_dir + "wref_ksp.npy").copy()   # (3, K, N_seq)
    print(f"[morse-pi] wref_ksp  shape: {wref_ksp.shape}  dtype: {wref_ksp.dtype}")
    scale = ksp_scale if ksp_scale is not None else (15.713940692571413 / 16.0)
    wref_ksp[:2, ...] *= scale
    # temporal axis flip not needed for spatial coil map
    print(f"[morse-pi] wref_ksp[:2] after scale  "
          f"kx range=[{wref_ksp[0].min():.4f}, {wref_ksp[0].max():.4f}]  "
          f"ky range=[{wref_ksp[1].min():.4f}, {wref_ksp[1].max():.4f}]")

    # ── k-space data ─────────────────────────────────────────────────────────
    wref_raw = np.load(data_dir + "wref_data.npy", mmap_mode="r")
    print(f"[morse-pi] wref_data shape: {wref_raw.shape}  dtype: {wref_raw.dtype}")
    # wref_raw: (1, K, N_seq, N_coils)
    N_seq = wref_raw.shape[2]

    # Normalize trajectory to [-π, π] using global max across all temporal steps.
    # After 02-style scale, wref_ksp[:2] spans [-1.515, 1.515] (not [-0.5, 0.5]).
    # Multiplying by π/global_max maps full k-space to [-π, π], so that inside
    # morse_pi: kspace_loc = traj/(2π) ≈ [-0.5, 0.5] for finufft. ✓
    global_kmax = np.max(np.abs(wref_ksp[:2]))
    print(f"[morse-pi] devided global max to pi: {global_kmax}")

    # Stack data and trajectory from all temporal steps (N_seq * K samples)
    # Ordering must match 02's convention: N_seq outer, K inner (index = t*K + k)
    K = wref_raw.shape[1]
    data    = np.asarray(wref_raw[0, :, :, :], dtype=np.complex64)  # (K, N_seq, C)
    data    = data.transpose(2, 1, 0).reshape(n_coils, -1)           # (C, N_seq*K)

    traj_raw = wref_ksp[:2, :, :]                                          # (2, K, N_seq)
    traj_xy  = traj_raw.transpose(2, 1, 0).reshape(-1, 2).astype(np.float32)  # (N_seq*K, 2) in [-1.515,1.515]

    print(f"[morse-pi] k-space: {data.shape}   trajectory: {traj_xy.shape}  "
          f"kx=[{traj_xy[:,0].min():.4f}, {traj_xy[:,0].max():.4f}]  "
          f"ky=[{traj_xy[:,1].min():.4f}, {traj_xy[:,1].max():.4f}]")

    return data, traj_xy


def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    out_dir  = os.path.join(args.out_dir, "coilmap")
    os.makedirs(out_dir, exist_ok=True)

    Ny, Nx   = args.dim[0], args.dim[1]
    N_COILS  = args.n_coils

    # ── Load k-space and trajectory ───────────────────────────────────────────
    data, traj = _load_kspace_and_traj(
        data_dir, N_COILS, args.ksp_scale)

    # ── Brain mask ────────────────────────────────────────────────────────────
    wref_img = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    img      = np.abs(wref_img[:, :, 0])
    brain_mask = img > args.brain_threshold

    # ── MORSE-PI ──────────────────────────────────────────────────────────────
    print(f"[morse-pi] Running MORSE-PI  N_ref={args.n_ref}  "
          f"smoothing_sd={args.smoothing_sd}  max_iter={args.max_iter}  "
          f"calib_width={args.calib_width}  backend={args.backend} …")
    sens_out = morse_pi(
        data          = data,
        trajectory    = traj,
        resolution    = (Ny, Nx),
        backend       = args.backend,
        N_ref         = args.n_ref,
        smoothing_sd  = args.smoothing_sd,
        max_iter      = args.max_iter,
        calib_width   = args.calib_width,
        calib_plot_path = os.path.join(out_dir, "fig_01e_calib_diagnostic.png") if args.save_plots else None,
    )
    # sens_out : (Ny, Nx, NCoils, NRef)
    print(f"[morse-pi] MORSE-PI output shape: {sens_out.shape}")

    # ── Take primary sensitivity set (first reference) ────────────────────────
    # shape: (Ny, Nx, NCoils, NRef) → (NCoils, Ny, Nx)
    smap = np.moveaxis(sens_out[:, :, :, 0], -1, 0).astype(np.complex64)
    print(f"[morse-pi] Primary sens shape: {smap.shape}")

    # ── RSS normalise ─────────────────────────────────────────────────────────
    rss_before = np.sqrt(np.sum(np.abs(smap) ** 2, axis=0))   # save for diagnostic plot
    rss = rss_before[np.newaxis]
    rss = np.where(rss < 1e-10, 1.0, rss)
    smap = smap / rss

    # # ── Optional median filter ────────────────────────────────────────────────
    # if args.median_filter > 1:
    #     from scipy.ndimage import median_filter
    #     smap_smooth = np.zeros_like(smap)
    #     for c in range(N_COILS):
    #         smap_smooth[c] = (median_filter(smap[c].real, size=args.median_filter)
    #                           + 1j * median_filter(smap[c].imag, size=args.median_filter))
    #     rss2 = np.sqrt(np.sum(np.abs(smap_smooth) ** 2, axis=0, keepdims=True))
    #     rss2 = np.where(rss2 < 1e-10, 1.0, rss2)
    #     smap = (smap_smooth / rss2).astype(np.complex64)

    # ── Apply brain mask ──────────────────────────────────────────────────────
    smap *= brain_mask[np.newaxis]
    coil_smap = smap.astype(np.complex64)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "ecalib_pp.npy")
    np.save(out_path, coil_smap)
    print(f"[morse-pi] Saved → {out_path}  shape={coil_smap.shape}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.save_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].imshow(rss_before * brain_mask, origin="lower", cmap="gray")
        axes[0].set_title("RSS before normalisation")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        axes[1].imshow(img * brain_mask, origin="lower", cmap="gray")
        axes[1].set_title(f"wref magnitude (thr={args.brain_threshold:.2e})")
        rss_after = np.sqrt(np.sum(np.abs(coil_smap) ** 2, axis=0))
        axes[2].imshow(rss_after * brain_mask, origin="lower", cmap="gray", vmin=0, vmax=1.05)
        axes[2].set_title("RSS after normalisation (should be ≈1)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_01c_coilmap_morse_rss.png"), dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        for c, ax in enumerate(axes.flat):
            if c < N_COILS:
                ax.imshow(np.abs(coil_smap[c]), origin="lower", cmap="magma")
                ax.set_title(f"C{c}", fontsize=7)
            ax.axis("off")
        plt.suptitle("MORSE-PI coil sensitivities |magnitude|")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_01d_coilmap_morse_all.png"), dpi=120)
        plt.close(fig)
        print("[morse-pi] Saved diagnostic plots.")

    print("[morse-pi] Done.")


if __name__ == "__main__":
    main()
