#!/usr/bin/env python3
"""
Automatic whole-tissue ROI mask for 05d SPICE (interleaved 31P CRT) — no drawing.

Mirrors recon_method/recon_01_run_spice.py's make_brain_mask (normalize the
anatomical image -> threshold -> keep the largest connected component -> fill
holes -> optional erosion), but applied to the *high-resolution* localizer
sampled over the exact MRSI FOV (same sampler as the manual drawer), then binned
down to the (Nx, Ny, Nz) voxel grid a voxel is kept if its in-ROI fraction
>= --voxel-frac. Output is a bool mask 05d takes via --roi-mask.

Runs headless (matplotlib Agg) — no display needed.

Usage:
  python scripts/05d_auto_tissue_mask_3d.py \
      --dat "/mnt/f/.../meas_MID00047_...dat" --rep 2 --cache-dir _tmp_recon/cache \
      --localizer "/mnt/f/.../0626_invivo/locolizer" \
      --supersample 16 --voxel-frac 0.5 [--threshold auto|0.12] [--erosion 0] \
      --out-mask roi_mask_cardiac_auto.npy
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_fill_holes, label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_drawer():
    import importlib.util as ilu, pathlib as pl
    spec = ilu.spec_from_file_location(
        "_drawer", pl.Path(__file__).parent / "05d_draw_manual_roi_mask_3d.py")
    m = ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _otsu(vals):
    """Otsu threshold on values in [0,1] (numpy-only; skimage not required)."""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.5
    hist, edges = np.histogram(vals, bins=256, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    wb = np.cumsum(hist).astype(float)
    wf = wb[-1] - wb
    mb = np.cumsum(hist * centers)
    mtot = mb[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_b = mb / wb
        mu_f = (mtot - mb) / wf
        sigma = wb * wf * (mu_b - mu_f) ** 2
    k = int(np.nanargmax(sigma))
    return float(centers[k])


def _seg_slice(fine2d, thr, erosion, cleanup, min_size):
    """Threshold + cleanup on one fine slice -> bool (H,W).

    Like make_brain_mask, but for multi-region chest anatomy we keep *all*
    connected components above `min_size` (heart, chest wall, muscle) instead of
    only the largest — keeping only the largest would drop the heart when it is a
    separate blob from the chest wall. Then fill enclosed holes.
    """
    m = fine2d > thr
    if cleanup:
        lab, n = label(m)
        if n > 0:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            keep = np.nonzero(sizes >= min_size)[0]
            m = np.isin(lab, keep)
        m = binary_fill_holes(m)
    if erosion > 0:
        m = binary_erosion(m, iterations=erosion)
    return m


def main():
    p = argparse.ArgumentParser(description="Auto whole-tissue ROI mask for 05d")
    p.add_argument("--dat", required=True)
    p.add_argument("--localizer", required=True)
    p.add_argument("--rep", type=int, default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--supersample", type=int, default=16)
    p.add_argument("--voxel-frac", type=float, default=0.5,
                   help="keep voxel if in-ROI fine fraction >= this")
    p.add_argument("--threshold", default="auto",
                   help="'auto' (Otsu on the whole volume) or a float in [0,1]")
    p.add_argument("--erosion", type=int, default=0, help="fine-pixel erosion iters")
    p.add_argument("--no-cleanup", action="store_true",
                   help="skip largest-component + fill-holes cleanup")
    p.add_argument("--out-mask", required=True)
    args = p.parse_args()

    d = _load_drawer()

    class _A:  # reuse the drawer's high-res loader
        pass
    a = _A()
    a.dat, a.localizer, a.rep = args.dat, args.localizer, args.rep
    a.cache_dir, a.supersample = args.cache_dir, args.supersample
    a.voxel_frac, a.prior = args.voxel_frac, None
    fine, S, (Nx, Ny, Nz) = d._load_background(a)

    p_norm = (fine - fine.min()) / (fine.max() - fine.min() + 1e-12)
    if str(args.threshold).lower() == "auto":
        thr = _otsu(p_norm[p_norm > 0])
        print(f"[auto-mask] Otsu threshold = {thr:.3f}")
    else:
        thr = float(args.threshold)
        print(f"[auto-mask] fixed threshold = {thr:.3f}")

    cleanup = not args.no_cleanup
    min_size = (0.5 * S) ** 2                    # drop specks < ~1/4 voxel
    fmask = np.zeros_like(p_norm, dtype=bool)
    for z in range(Nz):
        fmask[:, :, z] = _seg_slice(p_norm[:, :, z], thr, args.erosion, cleanup, min_size)

    # bin fine -> voxel grid
    vox = np.zeros((Nx, Ny, Nz), dtype=bool)
    for z in range(Nz):
        blk = fmask[:, :, z].reshape(Nx, S, Ny, S).mean(axis=(1, 3))
        vox[:, :, z] = blk >= args.voxel_frac
    np.save(args.out_mask, vox)

    # preview montage (upright, same convention as the drawer)
    vmax = float(np.percentile(fine, 99.5))
    cols = min(Nz, 5); rows = (Nz + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), squeeze=False)
    for z in range(Nz):
        ax = axes[z // cols][z % cols]
        ax.imshow(fine[:, :, z], cmap="gray", origin="upper", vmax=vmax)
        up = np.repeat(np.repeat(vox[:, :, z].astype(float), S, 0), S, 1)
        ax.contour(up, levels=[0.5], colors="lime", linewidths=1.1)
        ax.set_title(f"z={z+1} ({int(vox[:,:,z].sum())})", fontsize=7)
        ax.axis("off")
    for j in range(Nz, rows * cols):
        axes[j // cols][j % cols].axis("off")
    png = os.path.splitext(args.out_mask)[0] + "_preview.png"
    fig.suptitle(f"Auto tissue mask (thr={thr:.3f}, frac>={args.voxel_frac:g})", fontsize=11)
    fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)

    print(f"[auto-mask] saved {args.out_mask}  shape={vox.shape}  "
          f"({int(vox.sum())}/{vox.size} voxels)  + {png}")
    print("[auto-mask] per-slice voxel counts:",
          [int(vox[:, :, z].sum()) for z in range(Nz)])


if __name__ == "__main__":
    main()
