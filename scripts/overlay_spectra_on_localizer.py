#!/usr/bin/env python3
"""Overlay per-voxel SPICE spectra on the REAL localizer image, registered.

Reuses the exact registration from roi_selector_31p.py (the tool that made
roi_grid_240.png): the localizer is shown in LPS world-mm via its DICOM
geometry, and each MRSI voxel is placed at voxel_to_lps(xi, yi). The spice_spec
array is indexed spice_spec[xi, yi] to match — the same (xi, yi) convention the
ROI masks from roi_selector use, so the spectra land in the same cells as the
grid/numbers in roi_grid_240.png.

Example:
    python scripts/overlay_spectra_on_localizer.py \
        --localizer "/path/to/localizer.IMA_or_dicom_dir" \
        --dat "/path/to/scan.dat" \
        --spec "output_..._rerun/rep0/data/spice_spec_rank12.npy" \
        --out spectra_on_localizer_B_rep0.png
"""
import argparse
import importlib.util
import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import roi_selector_31p (same-dir, not a package) to reuse its registration
_p = pathlib.Path(__file__).parent / "roi_selector_31p.py"
_spec = importlib.util.spec_from_file_location("roi_selector_31p", _p)
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--localizer", required=True, help="DICOM localizer (file or dir), same as roi_grid")
    ap.add_argument("--dat", default=None, help="MRSI .dat for geometry (FOV/position)")
    ap.add_argument("--spec", required=True, help="spice_spec npy, shape (Nx, Ny, nspec)")
    ap.add_argument("--out", default="spectra_on_localizer.png")
    ap.add_argument("--fov-mm", type=float, default=None, help="override FOV (mm)")
    ap.add_argument("--pos-sag", type=float, default=None, help="override L center (mm)")
    ap.add_argument("--pos-cor", type=float, default=None, help="override P center (mm)")
    ap.add_argument("--color", default="#39ff14", help="spectra color")
    ap.add_argument("--lw", type=float, default=0.6)
    ap.add_argument("--win", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="spectral index window; default = auto around the signal")
    args = ap.parse_args()

    spec = np.abs(np.load(args.spec)).astype(np.float64)   # (Nx, Ny, nspec)
    Nx, Ny, nspec = spec.shape
    N = Nx

    img, extent, _ds = _rs.load_localizer_with_geometry(args.localizer)

    if args.dat is not None:
        _Nd, pos_sag, pos_cor, fov_mm = _rs.load_mrsi_geometry(args.dat)
    else:
        fov_mm, pos_sag, pos_cor = 200.0, 0.0, 0.0
    if args.fov_mm is not None:  fov_mm = args.fov_mm
    if args.pos_sag is not None: pos_sag = args.pos_sag
    if args.pos_cor is not None: pos_cor = args.pos_cor
    vox = fov_mm / N
    print(f"[overlay] N={N}, FOV={fov_mm:.0f} mm, vox={vox:.2f} mm, "
          f"center L={pos_sag:.1f} P={pos_cor:.1f}")

    # signal window + shared y-scale (one scale for all voxels)
    if args.win is None:
        env = spec.max(axis=(0, 1))
        idx = np.where(env > 0.03 * env.max())[0]
        lo, hi = max(0, idx.min() - 10), min(nspec, idx.max() + 10)
    else:
        lo, hi = args.win
    g = spec[..., lo:hi].max() + 1e-12
    k = np.arange(lo, hi)
    span = max(hi - lo - 1, 1)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(img, cmap="gray", origin="lower", extent=extent, aspect="equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    for xi in range(N):
        for yi in range(N):
            L, P = _rs.voxel_to_lps(xi, yi, N, pos_sag, pos_cor, vox)
            s = spec[xi, yi, lo:hi] / g                     # 0..1, shared scale
            # map spectral index across the cell's L-extent, amplitude up the P-extent
            xr = (L - vox / 2.0) + (k - lo) / span * vox
            yr = (P - vox / 2.0) + s * (0.9 * vox)
            ax.plot(xr, yr, color=args.color, lw=args.lw)
            ax.add_patch(plt.Rectangle((L - vox / 2, P - vox / 2), vox, vox,
                                       fill=False, ec="cyan", lw=0.3, alpha=0.4))

    ax.set_title(f"SPICE spectra on registered localizer   FOV={fov_mm:.0f}mm  N={N}  "
                 f"vox={vox:.1f}mm\n{os.path.basename(args.spec)}", fontsize=10, color="w")
    ax.tick_params(colors="w")
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="k")
    plt.close(fig)
    print(f"[overlay] saved -> {args.out}")


if __name__ == "__main__":
    main()
