#!/usr/bin/env python3
"""Overlay two SPICE all-voxel spectra grids for direct comparison.

Replots the `spice_all_spectra.png` layout from saved `spice_spec_*.npy`
arrays, drawing two datasets on the same axes in different colours. Unlike
the per-run figure (which normalises to its own global max), the shared
y-scale here spans BOTH datasets, so amplitude differences are real.

Example
-------
python scripts/plot_spectra_overlay.py \
  --spec-a out_rep23/data/spice_spec_rank8.npy --label-a "rep23" \
  --spec-b out_rep40/data/spice_spec_rank8.npy --label-b "rep40" \
  --out spectra_overlay_rep23_vs_rep40.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Acquisition constants for the 0724 dynamic CRT 10x10 5-ring series.
DEFAULT_DWELL_S = 3.6e-4      # spectral dwell (ADC dwell x 2 after OS removal)
DEFAULT_LARMOR_MHZ = 49.894558


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec-a", required=True, help="First spice_spec_*.npy (Nx, Ny, n_spec)")
    p.add_argument("--spec-b", required=True, help="Second spice_spec_*.npy")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--color-a", default="#1f77b4")
    p.add_argument("--color-b", default="#d62728")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--dwell-s", type=float, default=DEFAULT_DWELL_S)
    p.add_argument("--larmor-mhz", type=float, default=DEFAULT_LARMOR_MHZ)
    p.add_argument("--per-voxel-norm", action="store_true",
                   help="Normalise each voxel to its own max (compares lineshape only, "
                        "not amplitude). Default is a single shared scale across both sets.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    spec_a = np.nan_to_num(np.abs(np.load(args.spec_a)))
    spec_b = np.nan_to_num(np.abs(np.load(args.spec_b)))
    if spec_a.shape != spec_b.shape:
        raise ValueError(f"Shape mismatch: {spec_a.shape} vs {spec_b.shape}")
    Nx, Ny, n_spec = spec_a.shape

    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=args.dwell_s))
    PPM = -FREQ / args.larmor_mhz + args.ppm_center

    global_ymax = max(float(spec_a.max()), float(spec_b.max())) or 1.0

    fig, axes = plt.subplots(Nx, Ny, figsize=(Ny * 2.5, Nx * 2))
    axes = np.array(axes).reshape(Nx, Ny)
    for yi in range(Ny):
        for xi in range(Nx):
            ax = axes[Nx - 1 - xi, yi]
            va = spec_a[xi, yi, :]
            vb = spec_b[xi, yi, :]
            if args.per_voxel_norm:
                va = va / (va.max() + 1e-12)
                vb = vb / (vb.max() + 1e-12)
                ymax = 1.0
            else:
                ymax = global_ymax
            ax.plot(PPM, va, color=args.color_a, lw=0.5, label=args.label_a)
            ax.plot(PPM, vb, color=args.color_b, lw=0.5, alpha=0.8, label=args.label_b)
            ax.set_xlim(args.ppmlim[1], args.ppmlim[0])
            ax.set_ylim(0, ymax * 1.02)
            ax.axis("off")

    scale_note = ("per-voxel normalised" if args.per_voxel_norm
                  else "shared y-scale across both datasets")
    fig.suptitle(f"SPICE — All voxel spectra: {args.label_a} vs {args.label_b} ({scale_note})",
                 fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[overlay] wrote {args.out}  (grid {Nx}x{Ny}, {n_spec} pts, ymax={global_ymax:.3e})")


if __name__ == "__main__":
    main()
