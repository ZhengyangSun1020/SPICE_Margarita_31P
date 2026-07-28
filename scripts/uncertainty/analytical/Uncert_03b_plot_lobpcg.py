#!/usr/bin/env python3
"""
Step 10b — Re-plot LOBPCG pre-fitting uncertainty figures (no recomputation).

Reads  : <out_dir>/lobpcg_<run_tag>_k<k_eig>/posterior_std.npy   (Ny, Nx, N_seq)
         <out_dir>/lobpcg_<run_tag>_k<k_eig>/mean_spec.npy        (Ny, Nx, N_seq)  [optional]
         <data_dir>/wref_o.npy                                     (for brain mask)
Writes : <out_dir>/lobpcg_<run_tag>_k<k_eig>/fig_10_uncert_map.png
         <out_dir>/lobpcg_<run_tag>_k<k_eig>/fig_10_spice_mean.png  (if mean_spec.npy exists)

Usage:
    python scripts/uncertainty/analytical/Uncert_03b_plot_lobpcg.py \
        --data-dir data/processed/invivo_260623_01 \
        --run-tag  w5000_l0.0001 \
        --k-eig 500\
        --brain-threshold 0.16
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.scan_params import load_scan_params


def plot_average_variation(spice_test, img_shape, voxel_x, voxel_y,
                            brain_mask=None, PPM_AXIS=None,
                            threshold=None, dark_mode=True, cmap="Reds"):
    nx, ny, nt = img_shape
    img = np.asarray(spice_test).reshape(nx, ny, nt)
    mag = np.mean(np.abs(img), axis=-1)
    mag_masked = np.where(brain_mask, mag, np.nan) if brain_mask is not None else mag
    spec = img[voxel_y, voxel_x, :].astype(np.complex128)
    x_ax = PPM_AXIS if PPM_AXIS is not None else np.arange(nt)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    if dark_mode:
        fig.patch.set_facecolor("black")
        for ax in axs:
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            for sp in ax.spines.values():
                sp.set_color("white")

    im0 = axs[0].imshow(mag_masked, cmap="viridis", origin="lower")
    axs[0].set_title("Avg magnitude")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    axs[0].add_patch(Rectangle((voxel_x - .5, voxel_y - .5), 1, 1,
                                linewidth=2, edgecolor="green", facecolor="none"))
    im1 = axs[1].imshow(np.abs(mag_masked), cmap=cmap, origin="lower",
                         vmin=0, vmax=threshold)
    axs[1].set_title("Uncertainty (brain mask)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)
    c = "white" if dark_mode else "C0"
    axs[2].plot(x_ax, np.real(spec), color=c, label="Real")
    axs[2].plot(x_ax, np.abs(spec), alpha=.7, label="|S|")
    axs[2].set_title(f"Spectrum voxel ({voxel_y},{voxel_x})")
    if PPM_AXIS is not None:
        axs[2].invert_xaxis()
    axs[2].legend(facecolor="black" if dark_mode else "white",
                  labelcolor="white" if dark_mode else "black")
    axs[2].grid(alpha=.3, color="gray")
    plt.tight_layout()
    return fig


def parse_args():
    p = argparse.ArgumentParser(description="Re-plot LOBPCG uncertainty figures")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--out-dir",         default=None)
    p.add_argument("--run-tag",         default="")
    p.add_argument("--k-eig",           type=int,   default=5000)
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int,   default=None)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64])
    p.add_argument("--brain-threshold", type=float, default=0.16)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--voxel-x",         type=int,   default=38)
    p.add_argument("--voxel-y",         type=int,   default=20)
    p.add_argument("--threshold",       type=float, default=5e-5)
    p.add_argument("--dark-mode",       action="store_true", default=True)
    p.add_argument("--no-dark-mode",    dest="dark_mode", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")

    _tg     = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    out_dir = os.path.join(args.out_dir, f"{_tg('lobpcg')}_k{args.k_eig}")

    Ny, Nx = args.dim
    N_SEQ  = args.n_seq_points
    TS     = (args.k_points / N_SEQ) * args.dwelltime
    sweepwidth = 1.0 / TS
    FREQ_AXIS  = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS   = FREQ_AXIS / args.center_freq + args.ppm_center
    im_size    = (Ny, Nx, N_SEQ)

    # brain mask
    wref_img  = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # posterior std
    std_path = os.path.join(out_dir, "posterior_std.npy")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"posterior_std.npy not found: {std_path}\nRun Uncert_03 first.")
    posterior_std = np.load(std_path)
    print(f"[03b] Loaded posterior_std {posterior_std.shape}")

    fig = plot_average_variation(
        spice_test = posterior_std,
        img_shape  = im_size,
        voxel_x    = args.voxel_x,
        voxel_y    = args.voxel_y,
        brain_mask = brain_mask_inner,
        PPM_AXIS   = PPM_AXIS,
        threshold  = args.threshold,
        dark_mode  = args.dark_mode,
    )
    path = os.path.join(out_dir, "fig_10_uncert_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[03b] Saved {path}")

    # mean spec (optional)
    mean_path = os.path.join(out_dir, "mean_spec.npy")
    if os.path.exists(mean_path):
        mean_spec = np.load(mean_path)
        print(f"[03b] Loaded mean_spec {mean_spec.shape}")
        fig2 = plot_average_variation(
            spice_test = mean_spec,
            img_shape  = im_size,
            voxel_x    = args.voxel_x,
            voxel_y    = args.voxel_y,
            brain_mask = brain_mask,
            PPM_AXIS   = PPM_AXIS,
            threshold  = None,
            dark_mode  = args.dark_mode,
            cmap       = "viridis",
        )
        path2 = os.path.join(out_dir, "fig_10_spice_mean.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f"[03b] Saved {path2}")
    else:
        print("[03b] mean_spec.npy not found — skipping mean spectrum plot (re-run Uncert_03 to generate it)")

    print("[03b] Done.")


if __name__ == "__main__":
    main()
