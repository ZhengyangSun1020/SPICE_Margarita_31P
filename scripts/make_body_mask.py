#!/usr/bin/env python3
"""Threshold a completed 05d run's localizer prior into a body-support mask
for --roi-mask.

The mask is derived from ``data/localizer_spatial_prior_native_3d.npy`` — the
*native-frame* prior that 05d multiplies the ROI against (05d saves it at
localizer_spatial_prior_native_3d.npy and applies the ROI to that same array),
so a mask built from it is guaranteed to be in the correct (nx, ny, nz) frame.
Do NOT build a mask from data/adj_spec.npy — that file is saved in the *output*
(rotated/flipped) frame and would land misaligned.

Workflow:
    1. run 05d once WITH --localizer               (produces the native prior)
    2. python scripts/make_body_mask.py --out-dir <that run's out-dir> ...
    3. re-run 05d with  --roi-mask <out-dir>/body_mask.npy --roi-strength strong

Tune --threshold using the preview PNG it writes; raise it to shrink the mask,
lower it to grow it.
"""
import argparse
import os

import numpy as np
from scipy.ndimage import (binary_fill_holes, binary_dilation,
                           binary_erosion, label)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Body-support mask from a 05d localizer prior")
    ap.add_argument("--out-dir", required=True,
                    help="A completed 05d run dir containing "
                         "data/localizer_spatial_prior_native_3d.npy")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="Keep voxels >= this fraction of the prior max (0-1). Default 0.15")
    ap.add_argument("--keep-largest", action="store_true",
                    help="Keep only the largest connected component (drops stray blobs)")
    ap.add_argument("--fill-holes", action="store_true",
                    help="Fill enclosed holes (in-plane per slice, then 3D)")
    ap.add_argument("--dilate", type=int, default=0, help="Dilate the mask by N voxels")
    ap.add_argument("--erode", type=int, default=0, help="Erode the mask by N voxels")
    ap.add_argument("--out", default=None,
                    help="Output mask .npy (default: <out-dir>/body_mask.npy)")
    args = ap.parse_args()

    prior_path = os.path.join(args.out_dir, "data",
                              "localizer_spatial_prior_native_3d.npy")
    if not os.path.exists(prior_path):
        raise SystemExit(
            f"[mask] Not found: {prior_path}\n"
            f"        Run 05d once WITH --localizer first, then point --out-dir here.")

    prior = np.load(prior_path).astype(np.float32)          # (nx, ny, nz), native frame
    p = prior / (prior.max() + 1e-12)
    mask = p >= args.threshold

    if args.fill_holes:
        for z in range(mask.shape[2]):
            mask[:, :, z] = binary_fill_holes(mask[:, :, z])
        mask = binary_fill_holes(mask)
    if args.keep_largest:
        lab, n = label(mask)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0                                    # ignore background label
            mask = lab == int(sizes.argmax())
    if args.dilate > 0:
        mask = binary_dilation(mask, iterations=args.dilate)
    if args.erode > 0:
        mask = binary_erosion(mask, iterations=args.erode)

    out = args.out or os.path.join(args.out_dir, "body_mask.npy")
    np.save(out, mask.astype(bool))
    cov = 100.0 * mask.sum() / mask.size
    print(f"[mask] prior shape {prior.shape}  threshold {args.threshold}  "
          f"coverage {cov:.1f}%  ({int(mask.sum())}/{mask.size} vox)")
    print(f"[mask] saved -> {out}")
    print(f"[mask] use:  --roi-mask {out} --roi-strength strong")

    # ── preview: mask outline over the prior, per slice ──────────────────────
    nz = prior.shape[2]
    cols = min(nz, 6)
    rows = int(np.ceil(nz / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < nz:
            ax.imshow(p[:, :, i].T, origin="lower", cmap="gray")
            if mask[:, :, i].any():
                ax.contour(mask[:, :, i].T, levels=[0.5], colors="r", linewidths=0.8)
            ax.set_title(f"z{i}", fontsize=8)
    fig.suptitle(f"body mask (thr={args.threshold}, cov={cov:.0f}%) — red = kept")
    fig.tight_layout()
    png = os.path.splitext(out)[0] + "_preview.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[mask] preview -> {png}")


if __name__ == "__main__":
    main()
