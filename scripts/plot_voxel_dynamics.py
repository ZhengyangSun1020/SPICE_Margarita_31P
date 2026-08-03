#!/usr/bin/env python3
"""
Post-hoc per-voxel PCr and Pi dynamics plot from saved spec_all.

Usage:
    python scripts/plot_voxel_dynamics.py \
        --spec-all output_dynamic_3T_spice/data/spice_spec_all_rank8.npy \
        --dat "/path/to/meas_*.dat" \
        --voxels 6 4 6 5 6 6 \
        --out-dir output_dynamic_3T_spice
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


def extract_amplitude(spec, PPM, ppm_lo, ppm_hi):
    mask = (PPM >= ppm_lo) & (PPM <= ppm_hi)
    return np.abs(spec[:, :, mask]).max(axis=-1)


def build_ppm(dat_path):
    """Derive PPM axis from a .dat file (same logic as 05e)."""
    import pathlib, importlib.util
    here = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_05d", here / "05d_dynamic_adjoint_31p.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, info = mod.load_dynamic_concept_twix(dat_path)
    n_spec     = info["n_spec"]
    dwell_s    = info["dwell_s"] * 2
    larmor_mhz = info["larmor_freq_mhz"]
    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    return -FREQ / larmor_mhz  # ppm_center = 0


def plot_voxel_dynamics(spec_all, PPM, voxels, out_dir, tr_s=None):
    n_reps = spec_all.shape[0]
    t_axis = np.arange(n_reps) * tr_s if tr_s is not None else np.arange(n_reps)
    xlabel = "Time (s)" if tr_s is not None else "Rep index"

    n_vox = len(voxels)
    fig, axes = plt.subplots(2, n_vox, figsize=(5 * n_vox, 8), squeeze=False)

    for col, (xi, yi) in enumerate(voxels):
        pcr = np.array([extract_amplitude(spec_all[r], PPM, -1.5, 1.5)[xi, yi]
                        for r in range(n_reps)])
        pi  = np.array([extract_amplitude(spec_all[r], PPM,  4.2, 5.2)[xi, yi]
                        for r in range(n_reps)])

        axes[0, col].plot(t_axis, pcr, '-o', ms=4, lw=1.5, color='steelblue')
        axes[0, col].set_xlabel(xlabel)
        axes[0, col].set_ylabel("|PCr| (a.u.)")
        axes[0, col].set_title(f"PCr  voxel ({xi},{yi})")

        axes[1, col].plot(t_axis, pi, '-o', ms=4, lw=1.5, color='firebrick')
        axes[1, col].set_xlabel(xlabel)
        axes[1, col].set_ylabel("|Pi| (a.u.)")
        axes[1, col].set_title(f"Pi  voxel ({xi},{yi})")

    fig.suptitle("Per-voxel PCr and Pi dynamics", fontsize=12, y=1.01)
    fig.tight_layout()
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "voxel_pcr_pi_dynamics.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec-all", required=True, help="Path to spice_spec_all_rankN.npy")
    p.add_argument("--dat",      required=True, help="Original .dat file (for PPM axis)")
    p.add_argument("--voxels",   type=int, nargs="+", required=True,
                   help="Pairs xi yi, e.g. --voxels 6 4 6 5 6 6")
    p.add_argument("--out-dir",  default=".", help="Output directory")
    p.add_argument("--tr",       type=float, default=None, help="TR in seconds")
    p.add_argument("--ppm-center", type=float, default=0.0)
    args = p.parse_args()

    if len(args.voxels) % 2 != 0:
        sys.exit("--voxels must be pairs: xi yi xi yi ...")

    vox_pairs = [(args.voxels[i], args.voxels[i+1])
                 for i in range(0, len(args.voxels), 2)]

    print(f"Loading {args.spec_all} ...")
    spec_all = np.load(args.spec_all)
    print(f"  shape: {spec_all.shape}")

    print(f"Deriving PPM axis from {args.dat} ...")
    PPM = build_ppm(args.dat)
    PPM += args.ppm_center

    plot_voxel_dynamics(spec_all, PPM, vox_pairs, args.out_dir, tr_s=args.tr)


if __name__ == "__main__":
    main()
