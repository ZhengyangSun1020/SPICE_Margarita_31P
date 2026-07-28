#!/usr/bin/env python3
"""
Step 1 — Coil sensitivity estimation.

Two methods are available via --method (default: morse-pi):

  morse-pi  (default)
      Estimates coil maps directly from non-Cartesian water-reference k-space
      using MORSE-PI (Multi-coil Optimal Reference Selection for Phase-Insensitive
      coil sensitivity estimation).
      Ref: Dymerska et al., ISMRM 2024, Abstract #4265.
           https://archive.ismrm.org/2024/4265.html
      Reads : <data_dir>/wref_data.npy, wref_ksp.npy, wref_o.npy

  rni
      Post-processes a pre-computed ESPIRiT ecalib.npy using phase-pole correction
      based on Regularized Nonlinear Inversion (RNI).
      Ref: Blumenthal & Uecker, Magnetic Resonance in Medicine, 2026.
           https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.70333
      WARNING: This method is designed for 2D inputs only. Do not use with 3D data.
      Reads : <data_dir>/ecalib.npy, wref_o.npy

Both methods write: <out_dir>/coilmap/ecalib_pp.npy  (n_coils × Ny × Nx, complex64)

Usage:
    # MORSE-PI (default)
    python scripts/data_preproc/data_proc_02_coil_correction.py \
        --data-dir data/processed/invivo_250305_01 \
        --n-ref 6 --max-iter 50 --calib-width 16 --save-plots \
        [--out-dir output/invivo_250305_01] [--dim 64 64]

    # RNI (2D only)
    python scripts/data_preproc/data_proc_02_coil_correction.py \
        --data-dir data/processed/invivo_250305_01 --method rni --save-plots \
        [--out-dir output/invivo_250305_01]
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from warnings import filterwarnings
filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.scan_params import load_scan_params
from utils.coil_sens import morse_pi
from utils.pipeline_utils import try_symlink_shared_output, patch_spicefit_tree


# ── RNI: phase-pole correction for ESPIRiT maps ───────────────────────────────
# Blumenthal & Uecker, MRM 2026 — 1-pixel winding number, no morphological closing.

def _winding_number_2x2(smap_coil):
    p  = np.angle(smap_coil)
    d1 = np.angle(np.exp(1j * (p[:-1, 1:]  - p[:-1, :-1])))
    d2 = np.angle(np.exp(1j * (p[1:,  1:]  - p[:-1, 1:])))
    d3 = np.angle(np.exp(1j * (p[1:, :-1]  - p[1:,  1:])))
    d4 = np.angle(np.exp(1j * (p[:-1, :-1] - p[1:, :-1])))
    S      = (d1 + d2 + d3 + d4) / (2 * np.pi)
    S_full = np.zeros_like(p)
    S_full[:-1, :-1] = S
    return S_full


def _correct_espirit_poles(ecalib, mask=None, threshold=0.5):
    """Phase pole detection + correction for ESPIRiT coil maps (RNI method).
    ecalib : (n_coils, Ny, Nx) complex
    Returns: (corrected_ecalib, pole_centers, W_avg_map)
    """
    n_coils, Ny, Nx = ecalib.shape
    windings = np.stack([_winding_number_2x2(ecalib[c]) for c in range(n_coils)])
    weights  = np.abs(ecalib) ** 2
    if mask is not None:
        weights *= mask[np.newaxis]
    W_avg    = (np.sum(weights * windings, axis=0)
                / (np.sum(weights, axis=0) + 1e-10))
    pole_mask = np.abs(W_avg) > threshold
    labeled, n_found = ndimage.label(pole_mask)
    if n_found == 0:
        print("[rni] No phase poles detected.")
        return ecalib.copy(), [], W_avg
    centers = ndimage.center_of_mass(pole_mask, labeled, range(1, n_found + 1))
    signs   = [np.sign(W_avg[int(round(c[0])), int(round(c[1]))]) for c in centers]
    print(f"[rni] Detected {n_found} pole(s) at: "
          f"{[(round(c[0],1), round(c[1],1)) for c in centers]}")
    yy, xx = np.mgrid[0:Ny, 0:Nx].astype(float)
    correction = np.ones((Ny, Nx), dtype=complex)
    for center, sign in zip(centers, signs):
        dy = yy - center[0]
        dx = xx - center[1]
        r  = np.sqrt(dx**2 + dy**2)
        r  = np.where(r < 0.5, 1.0, r)
        vortex = (dx + 1j * dy) / r
        correction *= np.conj(vortex) if sign > 0 else vortex
    virtual = np.sum(ecalib, axis=0)
    inner   = np.vdot(virtual.ravel(), (virtual * correction).ravel())
    correction *= np.exp(-1j * np.angle(inner))
    return ecalib * correction[np.newaxis], centers, W_avg


# ── MORSE-PI k-space loader ───────────────────────────────────────────────────

def _load_kspace_and_traj(data_dir, n_coils, ksp_scale):
    """Load wref k-space and trajectory (same scaling convention as 02_b0_correction.py)."""
    wref_ksp = np.load(data_dir + "wref_ksp.npy").copy()   # (3, K, N_seq)
    print(f"[morse-pi] wref_ksp shape: {wref_ksp.shape}  dtype: {wref_ksp.dtype}")
    scale = ksp_scale if ksp_scale is not None else (15.713940692571413 / 16.0)
    wref_ksp[:2, ...] *= scale
    print(f"[morse-pi] wref_ksp[:2] after scale  "
          f"kx=[{wref_ksp[0].min():.4f}, {wref_ksp[0].max():.4f}]  "
          f"ky=[{wref_ksp[1].min():.4f}, {wref_ksp[1].max():.4f}]")
    wref_raw    = np.load(data_dir + "wref_data.npy", mmap_mode="r")
    print(f"[morse-pi] wref_data shape: {wref_raw.shape}  dtype: {wref_raw.dtype}")
    global_kmax = np.max(np.abs(wref_ksp[:2]))
    print(f"[morse-pi] global kmax (before pi-norm): {global_kmax:.4f}")
    K        = wref_raw.shape[1]
    data     = np.asarray(wref_raw[0, :, :, :], dtype=np.complex64)   # (K, N_seq, C)
    data     = data.transpose(2, 1, 0).reshape(n_coils, -1)            # (C, N_seq*K)
    traj_raw = wref_ksp[:2, :, :]                                       # (2, K, N_seq)
    traj_xy  = traj_raw.transpose(2, 1, 0).reshape(-1, 2).astype(np.float32)
    print(f"[morse-pi] k-space: {data.shape}   traj: {traj_xy.shape}  "
          f"kx=[{traj_xy[:,0].min():.4f}, {traj_xy[:,0].max():.4f}]  "
          f"ky=[{traj_xy[:,1].min():.4f}, {traj_xy[:,1].max():.4f}]")
    return data, traj_xy


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Coil sensitivity estimation — step 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",  required=True,
                   help="Raw data directory")
    p.add_argument("--out-dir",   default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--method",    default="morse-pi", choices=["morse-pi", "rni"],
                   help="Coil sensitivity method: morse-pi (default) or rni. "
                        "WARNING: rni only supports 2D inputs.")

    # Shared
    p.add_argument("--dim",             type=int, nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    p.add_argument("--n-coils",         type=int, default=None)
    p.add_argument("--brain-threshold", type=float, default=5e-6)
    p.add_argument("--save-plots",      action="store_true")

    # MORSE-PI options
    g = p.add_argument_group("MORSE-PI options (--method morse-pi)")
    g.add_argument("--backend",         default="finufft",
                   help="mrinufft backend")
    g.add_argument("--n-ref",           type=int, default=6,
                   help="Number of reference coils")
    g.add_argument("--smoothing-sd",    type=int, default=3,
                   help="Gaussian smoothing SD for MORSE-PI E matrix")
    g.add_argument("--calib-width",     type=int, default=16,
                   help="Calibration region half-width in k-space")
    g.add_argument("--max-iter",        type=int, default=50,
                   help="CG iterations in MORSE-PI")
    g.add_argument("--ksp-scale",       type=float, default=None,
                   help="Scale for wref_ksp kx/ky (default: 15.7139.../16)")

    # RNI options
    r = p.add_argument_group("RNI options (--method rni)")
    r.add_argument("--pole-threshold",  type=float, default=0.5,
                   help="Winding-number threshold for pole detection")
    r.add_argument("--median-filter",   type=int,   default=3,
                   help="Median filter window after pole correction (0=off)")

    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir)

    if try_symlink_shared_output(data_dir, args.out_dir, "coilmap"):
        return

    out_dir  = os.path.join(args.out_dir, "coilmap")
    os.makedirs(out_dir, exist_ok=True)
    Ny, Nx  = args.dim[0], args.dim[1]
    N_COILS = args.n_coils

    if args.method == "rni":
        warnings.warn(
            "RNI method is designed for 2D inputs only. "
            "Results may be incorrect for 3D data.",
            UserWarning, stacklevel=2,
        )

    # ── Brain mask ────────────────────────────────────────────────────────────
    wref_img   = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    img        = np.abs(wref_img[:, :, 0])
    brain_mask = img > args.brain_threshold

    if args.save_plots:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(img, origin="lower", cmap="gray")
        ax.imshow(brain_mask, origin="lower", cmap="Reds", alpha=0.35)
        ax.set_title(f"wref_o |magnitude| + brain mask (thr={args.brain_threshold:.2e})")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_01f_brain_mask.png"), dpi=120)
        plt.close(fig)
        print("[step-01] Saved fig_01f_brain_mask.png")

    # ── Method dispatch ───────────────────────────────────────────────────────
    if args.method == "morse-pi":
        print(f"[step-01] Method: MORSE-PI  "
              f"(Dymerska et al., ISMRM 2024, #4265)")
        data, traj = _load_kspace_and_traj(data_dir, N_COILS, args.ksp_scale)
        print(f"[morse-pi] N_ref={args.n_ref}  smoothing_sd={args.smoothing_sd}  "
              f"max_iter={args.max_iter}  calib_width={args.calib_width}  "
              f"backend={args.backend} …")
        sens_out = morse_pi(
            data            = data,
            trajectory      = traj,
            resolution      = (Ny, Nx),
            backend         = args.backend,
            N_ref           = args.n_ref,
            smoothing_sd    = args.smoothing_sd,
            max_iter        = args.max_iter,
            calib_width     = args.calib_width,
            calib_plot_path = (os.path.join(out_dir, "fig_01e_calib_diagnostic.png")
                               if args.save_plots else None),
        )
        # (Ny, Nx, NCoils, NRef) → (NCoils, Ny, Nx)
        smap       = np.moveaxis(sens_out[:, :, :, 0], -1, 0).astype(np.complex64)
        rss_before = np.sqrt(np.sum(np.abs(smap) ** 2, axis=0))
        rss        = np.where(rss_before[np.newaxis] < 1e-10, 1.0, rss_before[np.newaxis])
        smap       = smap / rss
        smap      *= brain_mask[np.newaxis]
        coil_smap  = smap.astype(np.complex64)

        if args.save_plots:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            im0 = axes[0].imshow(rss_before * brain_mask, origin="lower", cmap="gray")
            axes[0].set_title("RSS before normalisation")
            plt.colorbar(im0, ax=axes[0], fraction=0.046)
            axes[1].imshow(img * brain_mask, origin="lower", cmap="gray")
            axes[1].set_title(f"wref magnitude (thr={args.brain_threshold:.2e})")
            rss_after = np.sqrt(np.sum(np.abs(coil_smap) ** 2, axis=0))
            axes[2].imshow(rss_after * brain_mask, origin="lower",
                           cmap="gray", vmin=0, vmax=1.05)
            axes[2].set_title("RSS after normalisation (≈1)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_01a_morse_rss.png"), dpi=120)
            plt.close(fig)

            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            for c, ax in enumerate(axes.flat):
                if c < N_COILS:
                    ax.imshow(np.abs(coil_smap[c]), origin="lower", cmap="magma")
                    ax.set_title(f"C{c}", fontsize=7)
                ax.axis("off")
            plt.suptitle("MORSE-PI coil sensitivities |magnitude|")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_01b_morse_all.png"), dpi=120)
            plt.close(fig)

    else:  # rni
        print(f"[step-01] Method: RNI (phase-pole correction)  "
              f"(Blumenthal & Uecker, MRM 2026)")
        print("[rni] Loading ecalib.npy …")
        ecalib_raw = np.load(data_dir + "ecalib.npy").squeeze()  # (Ny, Nx, n_coils)
        ecalib     = np.moveaxis(ecalib_raw, -1, 0)               # (n_coils, Ny, Nx)
        n_coils    = ecalib.shape[0]
        print(f"[rni] ecalib shape: {ecalib.shape}")
        print("[rni] Running ESPIRiT phase-pole correction …")
        ecalib_corrected, pole_locs, W_avg = _correct_espirit_poles(
            ecalib, mask=brain_mask, threshold=args.pole_threshold,
        )
        smap = ecalib_corrected.copy()
        rss  = np.sqrt(np.sum(np.abs(smap) ** 2, axis=0, keepdims=True))
        rss  = np.where(rss < 1e-10, 1.0, rss)
        smap = smap / rss
        if args.median_filter > 1:
            from scipy.ndimage import median_filter
            smap_smooth = np.zeros_like(smap)
            for c in range(n_coils):
                smap_smooth[c] = (median_filter(smap[c].real, size=args.median_filter)
                                  + 1j * median_filter(smap[c].imag, size=args.median_filter))
            rss2 = np.sqrt(np.sum(np.abs(smap_smooth) ** 2, axis=0, keepdims=True))
            rss2 = np.where(rss2 < 1e-10, 1.0, rss2)
            smap = smap_smooth / rss2
        smap     *= brain_mask[np.newaxis]
        coil_smap = smap.astype(np.complex64)

        if args.save_plots:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            im = axes[0].imshow(W_avg, origin="lower", cmap="RdBu", vmin=-1, vmax=1)
            axes[0].set_title("Weighted winding number")
            plt.colorbar(im, ax=axes[0])
            for loc in pole_locs:
                axes[0].plot(loc[1], loc[0], "ko", ms=8)
            rss_b = np.sqrt(np.sum(np.abs(ecalib) ** 2, axis=0))
            rss_a = np.sqrt(np.sum(np.abs(coil_smap) ** 2, axis=0))
            axes[1].imshow(rss_b * brain_mask, origin="lower", cmap="gray")
            axes[1].set_title("RSS before correction")
            axes[2].imshow(rss_a * brain_mask, origin="lower", cmap="gray")
            axes[2].set_title("RSS after correction")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_01a_rni_poles.png"), dpi=120)
            plt.close(fig)

            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            for c, ax in enumerate(axes.flat):
                if c < n_coils:
                    ax.imshow(np.abs(coil_smap[c]), origin="lower", cmap="magma")
                    ax.set_title(f"C{c}", fontsize=7)
                ax.axis("off")
            plt.suptitle("RNI coil sensitivities |magnitude| (pole-corrected)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_01b_rni_all.png"), dpi=120)
            plt.close(fig)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "ecalib_pp.npy")
    np.save(out_path, coil_smap)
    print(f"[step-01] Saved → {out_path}  shape={coil_smap.shape}")

    _affine_path = data_dir + "affine.npy"
    _nii_affine  = np.load(_affine_path) if os.path.exists(_affine_path) else np.eye(4)
    # coil_smap: (N_coils, Ny, Nx) → (Nx_flip, Ny, 1, N_coils) for correct FSLeyes display
    _smap_nii = np.ascontiguousarray(
        coil_smap.transpose(2, 1, 0)[::-1, :, np.newaxis, :]
    ).astype(np.complex64)
    nib.save(
        nib.Nifti1Image(_smap_nii, _nii_affine),
        os.path.join(out_dir, "ecalib_pp.nii.gz"),
    )
    print(f"[step-01] Saved ecalib_pp.nii.gz  shape={_smap_nii.shape}")
    patch_spicefit_tree(args.out_dir, [
        ("coilmap.nii.gz", os.path.join(out_dir, "ecalib_pp.nii.gz"), "data-coilmap"),
    ])
    print("[step-01] Done.")


if __name__ == "__main__":
    main()
