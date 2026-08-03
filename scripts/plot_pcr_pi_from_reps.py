#!/usr/bin/env python3
"""
Plot per-voxel PCr and Pi dynamics from rep-wise 05c outputs.

Example:
  python scripts/plot_pcr_pi_from_reps.py \
    --rep-dir-pattern "output_dynamic_..._rep{rep}" \
    --dat "/path/to/meas.dat" \
    --rank 10 \
    --voxels 35 45 55 65 \
    --rep-start 0 --rep-end 59 \
    --out-dir output_dynamic_..._voxel_dynamics
"""

import argparse
import csv
import importlib.util
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_ppm_from_dat(dat_path):
    """Build PPM axis using the same loader metadata used in 05c."""
    here = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location("_06c", here / "06c_iterative_recon_31p_3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    info = mod.load_concept_twix_3d(dat_path, rep_sel=0)
    n_spec = int(info["n_spec"])
    dwell_s = float(info["dwell_s"]) * 2.0
    larmor_mhz = float(info["larmor_freq_mhz"])
    freq = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    return -freq / larmor_mhz


def voxel_id_to_xy(voxel_id, nxy):
    yi = int(voxel_id) // int(nxy)
    xi = int(voxel_id) % int(nxy)
    return xi, yi


def extract_peak_amp(spec_yx, ppm, lo, hi):
    mask = (ppm >= lo) & (ppm <= hi)
    if not np.any(mask):
        return np.nan
    return float(np.max(np.abs(spec_yx[mask])))


def main():
    p = argparse.ArgumentParser(description="Plot PCr/Pi dynamics from rep-wise 05c outputs")
    p.add_argument("--rep-dir-pattern", required=True,
                   help="Pattern for rep output dirs, must contain {rep}, e.g. output_xxx_rep{rep}")
    p.add_argument("--dat", required=True, help="Original .dat file for PPM axis")
    p.add_argument("--rank", type=int, default=10)
    p.add_argument("--voxels", type=int, nargs="+", required=True,
                   help="Voxel IDs in 0..N*N-1, e.g. 35 45 55 65")
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--rep-end", type=int, default=59)
    p.add_argument("--grid-size", type=int, default=10, help="Spatial grid size N (N x N)")
    p.add_argument("--tr", type=float, default=None, help="Optional TR in seconds")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ppm = build_ppm_from_dat(args.dat)
    pcr_win = (-1.5, 1.5)
    pi_win = (4.2, 5.2)

    rep_list = list(range(args.rep_start, args.rep_end + 1))
    time_axis = np.array(rep_list, dtype=float)
    if args.tr is not None:
        time_axis = time_axis * float(args.tr)
    x_label = "Time (s)" if args.tr is not None else "Rep index"

    # Store per-voxel traces for combined metabolite-specific plots.
    combined = {}

    all_missing = []
    for voxel_id in args.voxels:
        xi, yi = voxel_id_to_xy(voxel_id, args.grid_size)
        pcr_vals = []
        pi_vals = []
        kept_reps = []

        for rep in rep_list:
            rep_dir = args.rep_dir_pattern.format(rep=rep)
            spec_path = os.path.join(rep_dir, "data", f"spice_spec_rank{args.rank}.npy")
            if not os.path.exists(spec_path):
                all_missing.append(spec_path)
                continue

            spec = np.load(spec_path)
            if spec.ndim != 3:
                all_missing.append(spec_path)
                continue
            ny, nx, ns = spec.shape
            if yi < 0 or yi >= ny or xi < 0 or xi >= nx:
                raise ValueError(f"Voxel {voxel_id} -> (x={xi}, y={yi}) out of bounds for shape {spec.shape}")
            if ns != ppm.shape[0]:
                raise ValueError(
                    f"PPM length ({ppm.shape[0]}) mismatch with spectrum length ({ns}) in {spec_path}")

            s = spec[yi, xi, :]
            pcr_vals.append(extract_peak_amp(s, ppm, pcr_win[0], pcr_win[1]))
            pi_vals.append(extract_peak_amp(s, ppm, pi_win[0], pi_win[1]))
            kept_reps.append(rep)

        if len(kept_reps) == 0:
            print(f"[warn] no valid reps found for voxel {voxel_id}")
            continue

        t = np.array(kept_reps, dtype=float)
        if args.tr is not None:
            t = t * float(args.tr)
        pcr_vals = np.array(pcr_vals, dtype=float)
        pi_vals = np.array(pi_vals, dtype=float)

        combined[int(voxel_id)] = {
            "x": int(xi),
            "y": int(yi),
            "t": t.copy(),
            "pcr": pcr_vals.copy(),
            "pi": pi_vals.copy(),
        }

        csv_path = os.path.join(args.out_dir, f"voxel_{voxel_id:03d}_pcr_pi.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rep", "time", "x", "y", "pcr_amp", "pi_amp"])
            for k, rep in enumerate(kept_reps):
                w.writerow([rep, t[k], xi, yi, pcr_vals[k], pi_vals[k]])

        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        axes[0].plot(t, pcr_vals, "-o", ms=4, lw=1.5, color="steelblue")
        axes[0].set_ylabel("|PCr| (a.u.)")
        axes[0].set_title(f"Voxel {voxel_id} (x={xi}, y={yi}) - PCr")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, pi_vals, "-o", ms=4, lw=1.5, color="firebrick")
        axes[1].set_ylabel("|Pi| (a.u.)")
        axes[1].set_xlabel(x_label)
        axes[1].set_title(f"Voxel {voxel_id} (x={xi}, y={yi}) - Pi")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        png_path = os.path.join(args.out_dir, f"voxel_{voxel_id:03d}_pcr_pi.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {png_path}")
        print(f"Saved {csv_path}")

    if all_missing:
        miss_txt = os.path.join(args.out_dir, "missing_specs.txt")
        with open(miss_txt, "w") as f:
            for m in all_missing:
                f.write(m + "\n")
        print(f"[warn] Missing/invalid spectra in {len(all_missing)} reps/files, list saved to {miss_txt}")

    # Separate summary plots: one figure for PCr change, one for Pi change.
    if combined:
        # PCr-only plot across requested voxels.
        fig_pcr, ax_pcr = plt.subplots(figsize=(8, 4.8))
        for voxel_id in args.voxels:
            if int(voxel_id) not in combined:
                continue
            c = combined[int(voxel_id)]
            ax_pcr.plot(c["t"], c["pcr"], "-o", ms=3.5, lw=1.3,
                        label=f"V{int(voxel_id)} (x={c['x']},y={c['y']})")
        ax_pcr.set_xlabel(x_label)
        ax_pcr.set_ylabel("|PCr| (a.u.)")
        ax_pcr.set_title("PCr change by voxel")
        ax_pcr.grid(True, alpha=0.3)
        ax_pcr.legend(fontsize=8, ncol=2)
        fig_pcr.tight_layout()
        out_pcr = os.path.join(args.out_dir, "pcr_change_voxels.png")
        fig_pcr.savefig(out_pcr, dpi=150, bbox_inches="tight")
        plt.close(fig_pcr)
        print(f"Saved {out_pcr}")

        # Pi-only plot across requested voxels.
        fig_pi, ax_pi = plt.subplots(figsize=(8, 4.8))
        for voxel_id in args.voxels:
            if int(voxel_id) not in combined:
                continue
            c = combined[int(voxel_id)]
            ax_pi.plot(c["t"], c["pi"], "-o", ms=3.5, lw=1.3,
                       label=f"V{int(voxel_id)} (x={c['x']},y={c['y']})")
        ax_pi.set_xlabel(x_label)
        ax_pi.set_ylabel("|Pi| (a.u.)")
        ax_pi.set_title("Pi change by voxel")
        ax_pi.grid(True, alpha=0.3)
        ax_pi.legend(fontsize=8, ncol=2)
        fig_pi.tight_layout()
        out_pi = os.path.join(args.out_dir, "pi_change_voxels.png")
        fig_pi.savefig(out_pi, dpi=150, bbox_inches="tight")
        plt.close(fig_pi)
        print(f"Saved {out_pi}")

    # Spatial PCr map with requested voxels labeled.
    pcr_mask = (ppm >= pcr_win[0]) & (ppm <= pcr_win[1])
    pcr_sum = None
    pcr_count = 0
    for rep in rep_list:
        rep_dir = args.rep_dir_pattern.format(rep=rep)
        spec_path = os.path.join(rep_dir, "data", f"spice_spec_rank{args.rank}.npy")
        if not os.path.exists(spec_path):
            continue
        spec = np.load(spec_path)
        if spec.ndim != 3 or spec.shape[-1] != ppm.shape[0]:
            continue
        pcr_map_rep = np.max(np.abs(spec[..., pcr_mask]), axis=-1)
        if pcr_sum is None:
            pcr_sum = np.zeros_like(pcr_map_rep, dtype=float)
        pcr_sum += pcr_map_rep
        pcr_count += 1

    if pcr_sum is not None and pcr_count > 0:
        pcr_mean = pcr_sum / float(pcr_count)
        fig_map, ax_map = plt.subplots(figsize=(6.2, 5.4))
        im = ax_map.imshow(pcr_mean, origin="upper", cmap="hot", aspect="equal")
        plt.colorbar(im, ax=ax_map, label="Mean |PCr| (a.u.)")

        ny, nx = pcr_mean.shape
        for voxel_id in args.voxels:
            xi, yi = voxel_id_to_xy(voxel_id, args.grid_size)
            if yi < 0 or yi >= ny or xi < 0 or xi >= nx:
                continue
            ax_map.plot(xi, yi, marker="o", ms=7, mfc="none", mec="cyan", mew=1.6)
            ax_map.text(xi + 0.2, yi - 0.2, str(int(voxel_id)),
                        color="cyan", fontsize=9, weight="bold")

        ax_map.set_title("Mean PCr spatial map with labeled voxels")
        ax_map.set_xlabel("x voxel")
        ax_map.set_ylabel("y voxel")
        ax_map.set_xlim(-0.5, nx - 0.5)
        ax_map.set_ylim(ny - 0.5, -0.5)
        ax_map.set_xticks(np.arange(0, nx, 1), minor=True)
        ax_map.set_yticks(np.arange(0, ny, 1), minor=True)
        ax_map.grid(which="minor", color="w", alpha=0.12, lw=0.5)
        fig_map.tight_layout()
        out_map = os.path.join(args.out_dir, "pcr_spatial_map_labeled_voxels.png")
        fig_map.savefig(out_map, dpi=160, bbox_inches="tight")
        plt.close(fig_map)
        print(f"Saved {out_map}")


if __name__ == "__main__":
    main()
