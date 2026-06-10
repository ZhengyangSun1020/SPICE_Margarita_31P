#!/usr/bin/env python3
"""
Step 1 — ESPIRiT coil map post-processing with phase pole correction.

Reads  : <data_dir>/ecalib.npy   (ESPIRiT output, shape Ny×Nx×n_coils)
         <data_dir>/wref_o.npy   (for brain mask)
Writes : <out_dir>/coilmap/ecalib_pp.npy

Implements Blumenthal & Uecker MRM 2026 phase-pole correction
(1-pixel winding number, no morphological closing).

Usage:
    python scripts/01_coilmap.py \\
        --data-dir ./invivo_260305/cr/ \\
        [--out-dir ./output] \\
        [--brain-threshold 5e-6] [--pole-threshold 0.5] \\
        [--median-filter 3] [--save-plots]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from warnings import filterwarnings
filterwarnings("ignore")


# ── Phase pole correction ──────────────────────────────────────────────────────

def winding_number_2x2(smap_coil):
    """Discrete winding number via 2×2 square loop (1-pixel circle for ESPIRiT)."""
    p  = np.angle(smap_coil)
    d1 = np.angle(np.exp(1j * (p[:-1, 1:]  - p[:-1, :-1])))   # right
    d2 = np.angle(np.exp(1j * (p[1:,  1:]  - p[:-1, 1:])))    # down
    d3 = np.angle(np.exp(1j * (p[1:, :-1]  - p[1:,  1:])))    # left
    d4 = np.angle(np.exp(1j * (p[:-1, :-1] - p[1:, :-1])))    # up
    S        = (d1 + d2 + d3 + d4) / (2 * np.pi)
    S_full   = np.zeros_like(p)
    S_full[:-1, :-1] = S
    return S_full


def correct_espirit_poles(ecalib, mask=None, threshold=0.5):
    """
    Phase pole detection + correction for ESPIRiT coil maps.
    Blumenthal & Uecker, MRM 2026 — algorithm from Section 2.3/2.4.

    ecalib    : (n_coils, Ny, Nx) complex
    mask      : (Ny, Nx) bool, optional
    threshold : winding-number threshold (default 0.5)

    Returns: (corrected_ecalib, pole_centers, W_avg_map)
    """
    n_coils, Ny, Nx = ecalib.shape

    windings = np.stack([winding_number_2x2(ecalib[c]) for c in range(n_coils)])

    weights = np.abs(ecalib) ** 2
    if mask is not None:
        weights *= mask[np.newaxis]

    W_avg = (np.sum(weights * windings, axis=0)
             / (np.sum(weights, axis=0) + 1e-10))

    pole_mask = np.abs(W_avg) > threshold

    labeled, n_found = ndimage.label(pole_mask)
    if n_found == 0:
        print("[coilmap] No phase poles detected.")
        return ecalib.copy(), [], W_avg

    centers = ndimage.center_of_mass(pole_mask, labeled, range(1, n_found + 1))
    signs   = [np.sign(W_avg[int(round(c[0])), int(round(c[1]))]) for c in centers]
    print(f"[coilmap] Detected {n_found} pole(s) at: "
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


# ── argparse ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ESPIRiT coil map post-processing — step 1")
    p.add_argument("--data-dir",        required=True,
                   help="Raw data directory (read-only)")
    p.add_argument("--out-dir",         default="./output",
                   help="Root output directory (default: ./output)")
    p.add_argument("--brain-threshold", type=float, default=5e-6,
                   help="Brain mask threshold on wref_o.npy magnitude (default: 5e-6)")
    p.add_argument("--pole-threshold",  type=float, default=0.5,
                   help="Winding-number threshold for pole detection (default: 0.5)")
    p.add_argument("--median-filter",   type=int,   default=3,
                   help="Median filter window after pole correction (default: 3, 0=off)")
    p.add_argument("--save-plots",      action="store_true")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    out_dir  = os.path.join(args.out_dir, "coilmap")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load ESPIRiT maps ──────────────────────────────────────────────────────
    print("[coilmap] Loading ecalib.npy …")
    ecalib_raw = np.load(data_dir + "ecalib.npy").squeeze()   # (Ny, Nx, n_coils)
    ecalib     = np.moveaxis(ecalib_raw, -1, 0)               # (n_coils, Ny, Nx)
    n_coils    = ecalib.shape[0]
    print(f"[coilmap] ecalib shape: {ecalib.shape}")

    # ── Brain mask from wref_o ─────────────────────────────────────────────────
    wref_img         = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    img              = np.abs(wref_img[:, :, 0])
    brain_nolip_mask = img > args.brain_threshold

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(img, origin="lower", cmap="gray")
        ax.imshow(brain_nolip_mask, origin="lower", cmap="Reds", alpha=0.35)
        ax.set_title(f"Brain mask  thr={args.brain_threshold:.2e}")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_01a_brain_mask.png"), dpi=120)
        plt.close(fig)

    # ── Phase pole correction ──────────────────────────────────────────────────
    print("[coilmap] Running ESPIRiT phase pole correction …")
    ecalib_corrected, pole_locs, W_avg = correct_espirit_poles(
        ecalib, mask=brain_nolip_mask, threshold=args.pole_threshold
    )

    if args.save_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im = axes[0].imshow(W_avg, origin="lower", cmap="RdBu", vmin=-1, vmax=1)
        axes[0].set_title("Weighted winding number")
        plt.colorbar(im, ax=axes[0])
        for loc in pole_locs:
            axes[0].plot(loc[1], loc[0], "ko", ms=8,
                         label=f"pole ({loc[1]:.0f},{loc[0]:.0f})")
        if pole_locs:
            axes[0].legend(fontsize=7)
        rss_before = np.sqrt(np.sum(np.abs(ecalib) ** 2, axis=0))
        rss_after  = np.sqrt(np.sum(np.abs(ecalib_corrected) ** 2, axis=0))
        axes[1].imshow(rss_before * brain_nolip_mask, origin="lower", cmap="gray")
        axes[1].set_title("RSS before correction")
        axes[2].imshow(rss_after  * brain_nolip_mask, origin="lower", cmap="gray")
        axes[2].set_title("RSS after correction")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_01b_pole_correction.png"), dpi=120)
        plt.close(fig)

    # ── RSS normalise ─────────────────────────────────────────────────────────
    smap = ecalib_corrected.copy()
    rss  = np.sqrt(np.sum(np.abs(smap) ** 2, axis=0, keepdims=True))
    rss  = np.where(rss < 1e-10, 1.0, rss)
    smap = smap / rss

    # ── Optional median filter ────────────────────────────────────────────────
    if args.median_filter > 1:
        from scipy.ndimage import median_filter
        smap_smooth = np.zeros_like(smap)
        for c in range(n_coils):
            smap_smooth[c] = (median_filter(smap[c].real, size=args.median_filter)
                              + 1j * median_filter(smap[c].imag, size=args.median_filter))
        rss2 = np.sqrt(np.sum(np.abs(smap_smooth) ** 2, axis=0, keepdims=True))
        rss2 = np.where(rss2 < 1e-10, 1.0, rss2)
        smap = smap_smooth / rss2

    # ── Apply brain mask & cast ────────────────────────────────────────────────
    smap *= brain_nolip_mask[np.newaxis]
    coil_smap_corrected = smap.astype(np.complex64)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "ecalib_pp.npy")
    np.save(out_path, coil_smap_corrected)
    print(f"[coilmap] Saved → {out_path}  shape={coil_smap_corrected.shape}")

    if args.save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        rss_check = np.sqrt(np.sum(np.abs(coil_smap_corrected) ** 2, axis=0))
        axes[0].imshow(rss_check * brain_nolip_mask, origin="lower", cmap="gray")
        axes[0].set_title("RSS after pole correction")
        axes[1].imshow(np.abs(ecalib[0]) * brain_nolip_mask, origin="lower", cmap="gray")
        axes[1].set_title("Original coil 0 (reference)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_01c_coilmap_final.png"), dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        for c, ax in enumerate(axes.flat):
            ax.imshow(np.abs(coil_smap_corrected[c]), origin="lower", cmap="magma")
            ax.set_title(f"C{c}", fontsize=7)
            ax.axis("off")
        plt.suptitle("ESPIRiT coil sensitivities |magnitude| (pole-corrected)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_01d_coilmap_all.png"), dpi=120)
        plt.close(fig)
        print("[coilmap] Saved diagnostic plots.")

    print("[coilmap] Done.")


if __name__ == "__main__":
    main()
