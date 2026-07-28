#!/usr/bin/env python3
"""
Step 16 — Compare LOBPCG pre-fitting uncertainty (Uncert_03) with
           group/MC pre-fitting std (Uncert_07), voxel-wise.

Loads:
    LOBPCG:   <out-root>/<subject>/lobpcg_<run-tag>_k<k-eig>/posterior_std.npy
              (Uncert_03 output)

    group MC: <group-dir>/prefitting_std.npy
              (Uncert_07 output; group-dir auto-derived as
               <out-root>/group_<series>_<run-tag> when not given explicitly,
               e.g. output/group_invivo_260623_w5000_l0.0001)

Outputs (saved to <out-dir>/):
    fig_16_ratio_map.png         — voxel-wise ratio (MC / LOBPCG), mean over ppm
    fig_16_ratio_histogram.png   — ratio histogram
    fig_16_scatter.png           — LOBPCG std vs MC std scatter with Pearson r
    fig_16_voxel_spectra.png     — spectral comparison at a selected voxel
    compare_stats.txt            — summary statistics

Usage:
    python scripts/uncertainty/MC/Uncert_10_compare_mc_lobpcg.py \
        --subject       invivo_260623_01 \
        --run-tag       w5000_l0.0001 \
        --k-eig         5000 \
        --data-dir      data/processed/invivo_260623_01 \
        --dim 64 64 \
        --brain-threshold 0.16 --brain-erosion 1

    # explicit paths:
    python scripts/uncertainty/MC/Uncert_10_compare_mc_lobpcg.py \
        --lobpcg-std output/invivo_260623_01/lobpcg_w5000_l0.0001_k5000/posterior_std.npy \
        --group-std  output/group_invivo_260623_w5000_l0.0001/prefitting_std.npy \
        --data-dir   data/processed/invivo_260623_01 \
        --dim 64 64
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr, spearmanr

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)
from utils.pipeline_utils import make_brain_mask
from utils.scan_params import load_scan_params


# ── helpers ───────────────────────────────────────────────────────────────────

def _dark_ax(ax, bg="black", fg="white"):
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for sp in ax.spines.values():
        sp.set_color(fg)


def _masked_imshow(ax, data_2d, brain_mask, cmap="Reds", vmin=None, vmax=None,
                   wref_norm=None, origin="lower"):
    masked = np.where(brain_mask, data_2d, np.nan)
    cmap_obj = copy.copy(plt.cm.get_cmap(cmap))
    cmap_obj.set_bad(color="black", alpha=1.0)
    if wref_norm is not None:
        wref_brain = np.where(brain_mask, wref_norm, np.nan)
        wref_cmap = copy.copy(plt.cm.get_cmap("gray"))
        wref_cmap.set_bad(color="black", alpha=1.0)
        ax.imshow(wref_brain, origin=origin, cmap=wref_cmap, alpha=0.5, zorder=0)
    im = ax.imshow(masked, origin=origin, cmap=cmap_obj,
                   vmin=vmin, vmax=vmax, alpha=0.9, zorder=1)
    ax.contour(brain_mask, levels=[0.5], colors="white", linewidths=0.5, zorder=2)
    return im


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_ratio_map(ratio_map, brain_mask, wref_norm, out_path,
                   clip_pct=97.5, title="MC / LOBPCG ratio"):
    """Ratio map with 1.0 = white (TwoSlopeNorm), extreme values clipped at clip_pct."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    _dark_ax(ax)

    brain_vals = ratio_map[brain_mask]
    vmax = float(np.nanpercentile(brain_vals, clip_pct))
    vmin = float(np.nanpercentile(brain_vals, 100.0 - clip_pct))
    vmin = min(vmin, 0.9)   # ensure vcenter=1 stays strictly between vmin and vmax
    vmax = max(vmax, 1.1)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    display = np.where(brain_mask, np.clip(ratio_map, vmin, vmax), np.nan)

    cmap_obj = plt.cm.get_cmap("RdBu_r").copy()
    cmap_obj.set_bad(color="black", alpha=1.0)

    if wref_norm is not None:
        wref_brain = np.where(brain_mask, wref_norm, np.nan)
        wref_cmap = plt.cm.get_cmap("gray").copy()
        wref_cmap.set_bad(color="black", alpha=1.0)
        ax.imshow(wref_brain, origin="lower", cmap=wref_cmap, alpha=0.5, zorder=0)
    im = ax.imshow(display, origin="lower", cmap=cmap_obj, norm=norm, alpha=0.9, zorder=1)
    ax.contour(brain_mask, levels=[0.5], colors="white", linewidths=0.5, zorder=2)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ratio  (clipped)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    ax.set_title(f"{title}\n(display clipped to [{vmin:.2f}, {vmax:.2f}], white=1.0)",
                 color="white", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)
    print(f"[compare] Saved {out_path}")


def plot_ratio_histogram(ratios_flat, out_path, bins=80):
    median_r = float(np.nanmedian(ratios_flat))
    iqr_r    = float(np.nanpercentile(ratios_flat, 75) - np.nanpercentile(ratios_flat, 25))
    cv       = iqr_r / (median_r + 1e-30)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="black")
    _dark_ax(ax)
    ax.hist(ratios_flat[np.isfinite(ratios_flat)], bins=bins,
            color="steelblue", edgecolor="none", alpha=0.85, density=True)
    ax.axvline(median_r, color="tomato", linewidth=1.5, label=f"Median={median_r:.3f}")
    ax.axvline(1.0,      color="lime",   linewidth=1.0, linestyle="--", label="Ratio = 1")
    ax.set_xlabel("Ratio  (MC std / LOBPCG std)")
    ax.set_ylabel("Density")
    diag = ("Likely SYSTEMATIC bias" if cv < 0.20
            else "Significant SPATIAL variation present")
    ax.set_title(f"Ratio histogram  |  Median={median_r:.3f}  IQR/Median={cv:.3f}\n"
                 f"→ {diag}", color="white", fontsize=9)
    ax.legend(labelcolor="white", facecolor="black", fontsize=8)
    ax.grid(True, alpha=0.3, color="gray")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)
    print(f"[compare] Saved {out_path}")
    return median_r, iqr_r, cv


def plot_scatter(lobpcg_flat, mc_flat, out_path, max_pts=5000):
    valid = np.isfinite(lobpcg_flat) & np.isfinite(mc_flat) & (lobpcg_flat > 0) & (mc_flat > 0)
    x = lobpcg_flat[valid]
    y = mc_flat[valid]

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    if len(x) > max_pts:
        idx = np.random.default_rng(0).choice(len(x), max_pts, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    _dark_ax(ax)
    ax.scatter(xs, ys, s=4, alpha=0.4, color="steelblue", rasterized=True)
    lim = max(float(np.nanpercentile(x, 99)), float(np.nanpercentile(y, 99)))
    ax.plot([0, lim], [0, lim], color="lime", linewidth=1.0, linestyle="--", label="y=x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("LOBPCG std  (Uncert_03)")
    ax.set_ylabel("MC/Group std  (Uncert_07)")
    ax.set_title(f"Pearson r={pr:.4f}  Spearman r={sr:.4f}\n(n={valid.sum()} brain voxels × ppm)",
                 color="white", fontsize=9)
    ax.legend(labelcolor="white", facecolor="black", fontsize=8)
    ax.grid(True, alpha=0.3, color="gray")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)
    print(f"[compare] Saved {out_path}")
    return pr, sr


def plot_voxel_spectra(lobpcg_std, mc_std, PPM_AXIS, vy, vx, out_path):
    l_spec = np.abs(lobpcg_std[vy, vx, :])
    m_spec = np.abs(mc_std[vy, vx, :])
    ratio  = m_spec / (l_spec + 1e-30)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4), facecolor="black")
    for ax in axs:
        _dark_ax(ax)

    axs[0].plot(PPM_AXIS, l_spec, color="dodgerblue", label="LOBPCG (Uncert_03)", linewidth=1.2)
    axs[0].plot(PPM_AXIS, m_spec, color="tomato",     label="MC/Group (Uncert_07)", linewidth=1.2)
    axs[0].set_xlabel("ppm")
    axs[0].set_ylabel("Pre-fitting std |σ(ν)|")
    axs[0].set_title(f"Spectral std at voxel ({vy},{vx})", color="white")
    axs[0].invert_xaxis()
    axs[0].legend(labelcolor="white", facecolor="black", fontsize=8)
    axs[0].grid(True, alpha=0.3, color="gray")

    axs[1].plot(PPM_AXIS, ratio, color="gold", linewidth=1.2)
    axs[1].axhline(1.0, color="lime", linewidth=0.8, linestyle="--")
    axs[1].axhline(float(np.median(ratio)), color="tomato", linewidth=0.8,
                   linestyle=":", label=f"Median={float(np.median(ratio)):.3f}")
    axs[1].set_xlabel("ppm")
    axs[1].set_ylabel("Ratio  (MC / LOBPCG)")
    axs[1].set_title(f"Ratio vs ppm at voxel ({vy},{vx})", color="white")
    axs[1].invert_xaxis()
    axs[1].legend(labelcolor="white", facecolor="black", fontsize=8)
    axs[1].grid(True, alpha=0.3, color="gray")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)
    print(f"[compare] Saved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare LOBPCG vs MC/group pre-fitting std — step 16")

    # path resolution (auto or explicit)
    p.add_argument("--subject",         default=None,
                   help="Subject ID, used to auto-derive LOBPCG std path")
    p.add_argument("--out-root",        default="./output",
                   help="Root containing per-subject output dirs (default: ./output)")
    p.add_argument("--run-tag",         default="",
                   help="Run tag used in all pipeline dirs (e.g. w5000_l0.0001)")
    p.add_argument("--k-eig",           type=int, default=5000,
                   help="Number of eigenpairs used in Uncert_03 (default: 5000); "
                        "used to locate lobpcg_<run-tag>_k<k-eig>/ directory")
    p.add_argument("--group-dir",       default=None,
                   help="Path to Uncert_07 group output dir; auto-derived as "
                        "<out-root>/group_<series>_<run-tag> when omitted "
                        "(e.g. output/group_invivo_260623_w5000_l0.0001)")

    # explicit overrides (skip auto-derive)
    p.add_argument("--lobpcg-std",      default=None,
                   help="Direct path to posterior_std.npy from Uncert_03 "
                        "(overrides --subject / --k-eig auto-derive)")
    p.add_argument("--group-std",       default=None,
                   help="Direct path to prefitting_std.npy from Uncert_07 "
                        "(overrides --group-dir auto-derive)")

    # brain mask
    p.add_argument("--data-dir",        required=True,
                   help="Per-subject data directory containing wref_o.npy")
    p.add_argument("--dim",             type=int, nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    p.add_argument("--brain-threshold", type=float, default=0.16)
    p.add_argument("--brain-erosion",   type=int,   default=2)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component and fill enclosed holes. Default: off.")

    # spectrum axis
    p.add_argument("--n-seq-points",    type=int, default=300)
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int,   default=None)
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)

    # output
    p.add_argument("--out-dir",         default=None,
                   help="Where to save figures and stats "
                        "(default: <out-root>/<subject>/compare_mc_lobpcg_<run-tag>_k<k-eig>)")
    p.add_argument("--voxel-x",         type=int, default=38)
    p.add_argument("--voxel-y",         type=int, default=20)
    p.add_argument("--ppm-range",       type=float, nargs=2, default=None,
                   metavar=("PPM_LO", "PPM_HI"),
                   help="Restrict ratio/correlation analysis to this ppm window "
                        "(e.g. 0.5 4.2)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    Ny, Nx   = args.dim

    _tg = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b

    # ── Resolve LOBPCG std path ──────────────────────────────────────────────
    if args.lobpcg_std:
        lobpcg_path = args.lobpcg_std
    elif args.subject:
        subdir      = f"lobpcg_{args.run_tag}_k{args.k_eig}" if args.run_tag else f"lobpcg_k{args.k_eig}"
        lobpcg_path = os.path.join(args.out_root, args.subject, subdir, "posterior_std.npy")
    else:
        raise ValueError("Provide either --lobpcg-std or (--subject + --out-root + --run-tag + --k-eig).")

    # ── Resolve group std path ───────────────────────────────────────────────
    if args.group_std:
        group_path = args.group_std
    elif args.group_dir:
        group_path = os.path.join(args.group_dir, "prefitting_std.npy")
    elif args.subject and args.run_tag:
        series     = "_".join(args.subject.split("_")[:-1])   # e.g. invivo_260623
        group_path = os.path.join(args.out_root, _tg(f"group_{series}"), "prefitting_std.npy")
    else:
        raise ValueError("Provide --group-std, --group-dir, or (--subject + --run-tag) for auto-derive.")

    # ── Output directory ─────────────────────────────────────────────────────
    if args.out_dir:
        out_dir = args.out_dir
    elif args.subject:
        tag     = f"compare_mc_lobpcg_{args.run_tag}_k{args.k_eig}" if args.run_tag \
                  else f"compare_mc_lobpcg_k{args.k_eig}"
        out_dir = os.path.join(args.out_root, args.subject, tag)
    else:
        out_dir = os.path.join(os.path.dirname(group_path), f"compare_mc_lobpcg_k{args.k_eig}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[compare] LOBPCG std  : {lobpcg_path}")
    print(f"[compare] MC/Group std: {group_path}")
    print(f"[compare] Output dir  : {out_dir}")

    # ── Load data ────────────────────────────────────────────────────────────
    lobpcg_std = np.abs(np.load(lobpcg_path))   # (Ny, Nx, N_seq)
    mc_std     = np.abs(np.load(group_path))    # (Ny, Nx, N_seq)

    if lobpcg_std.shape != mc_std.shape:
        raise ValueError(
            f"Shape mismatch: LOBPCG {lobpcg_std.shape} vs group {mc_std.shape}. "
            "Check --dim and that both scripts used the same subject/sequence length."
        )

    N_SEQ = lobpcg_std.shape[-1]
    print(f"[compare] Array shape: {lobpcg_std.shape}")

    # ── Spectrum axis ────────────────────────────────────────────────────────
    load_scan_params(args, data_dir, k_key="k_mrsi")
    TS         = (args.k_points / N_SEQ) * args.dwelltime
    sweepwidth = 1.0 / TS
    FREQ_AXIS  = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS   = FREQ_AXIS / args.center_freq + args.ppm_center

    # ── Brain mask ───────────────────────────────────────────────────────────
    wref_img = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_norm, brain_mask, _ = make_brain_mask(wref_img, args.brain_threshold, args.brain_erosion,
                                                cleanup=args.brain_mask_cleanup)

    # ── Optional ppm restriction ─────────────────────────────────────────────
    if args.ppm_range is not None:
        plo, phi = sorted(args.ppm_range)
        ppm_sel  = (PPM_AXIS >= plo) & (PPM_AXIS <= phi)
        print(f"[compare] ppm restriction: {plo}–{phi} ppm  ({ppm_sel.sum()} bins)")
        lobpcg_sel = lobpcg_std[:, :, ppm_sel]
        mc_sel     = mc_std[:, :, ppm_sel]
        PPM_sel    = PPM_AXIS[ppm_sel]
    else:
        lobpcg_sel = lobpcg_std
        mc_sel     = mc_std
        PPM_sel    = PPM_AXIS

    # ── Ratio map (mean over selected ppm range) ─────────────────────────────
    lobpcg_mean = lobpcg_sel.mean(axis=-1)   # (Ny, Nx)
    mc_mean     = mc_sel.mean(axis=-1)

    lobpcg_mean[~brain_mask] = np.nan
    mc_mean[~brain_mask]     = np.nan

    ratio_map = mc_mean / (lobpcg_mean + 1e-30)   # (Ny, Nx)
    ratio_map[~brain_mask] = np.nan

    plot_ratio_map(
        ratio_map, brain_mask, wref_norm,
        os.path.join(out_dir, "fig_16_ratio_map.png"),
        title=f"MC std / LOBPCG std  (mean over ppm)\nk_eig={args.k_eig}  run={args.run_tag}",
    )

    # ── Ratio histogram ──────────────────────────────────────────────────────
    ratios_flat = ratio_map[brain_mask].ravel()
    median_r, iqr_r, cv = plot_ratio_histogram(
        ratios_flat,
        os.path.join(out_dir, "fig_16_ratio_histogram.png"),
    )

    # ── Scatter plot ─────────────────────────────────────────────────────────
    lobpcg_3d = lobpcg_sel.copy()
    mc_3d     = mc_sel.copy()
    brain_3d  = np.broadcast_to(brain_mask[:, :, np.newaxis], lobpcg_3d.shape)
    lobpcg_3d[~brain_3d] = np.nan
    mc_3d[~brain_3d]     = np.nan

    pr, sr = plot_scatter(
        lobpcg_3d.ravel(),
        mc_3d.ravel(),
        os.path.join(out_dir, "fig_16_scatter.png"),
    )

    # ── Per-voxel spectral overlay ───────────────────────────────────────────
    vy, vx = args.voxel_y, args.voxel_x
    plot_voxel_spectra(
        lobpcg_std, mc_std, PPM_AXIS, vy, vx,
        os.path.join(out_dir, "fig_16_voxel_spectra.png"),
    )

    # ── Summary statistics ───────────────────────────────────────────────────
    lobpcg_brain = lobpcg_mean[brain_mask]
    mc_brain     = mc_mean[brain_mask]

    # robust stats: clip at 2.5/97.5 to avoid 800x outliers skewing mean
    ratio_clipped = np.clip(ratios_flat, *np.nanpercentile(ratios_flat, [2.5, 97.5]))

    stats_lines = [
        "=== Uncert_10: MC vs LOBPCG pre-fitting std comparison ===",
        f"LOBPCG std path  : {lobpcg_path}",
        f"MC/Group std path: {group_path}",
        f"k_eig            : {args.k_eig}",
        f"Brain voxels     : {brain_mask.sum()}",
        f"ppm range        : {PPM_sel[0]:.3f} – {PPM_sel[-1]:.3f}",
        "",
        "--- Ratio (MC / LOBPCG), mean over ppm, within brain ---",
        f"Median ratio     : {median_r:.4f}",
        f"Mean ratio (raw) : {float(np.nanmean(ratio_map[brain_mask])):.4f}  (may be inflated by outliers)",
        f"Mean ratio (p2.5–p97.5 clipped): {float(np.nanmean(ratio_clipped)):.4f}",
        f"Max ratio        : {float(np.nanmax(ratios_flat)):.2f}",
        f"Std of ratio     : {float(np.nanstd(ratio_map[brain_mask])):.4f}",
        f"IQR of ratio     : {iqr_r:.4f}",
        f"IQR / Median (CV): {cv:.4f}",
        f"5th–95th pctile  : {float(np.nanpercentile(ratios_flat, 5)):.4f} – "
                            f"{float(np.nanpercentile(ratios_flat, 95)):.4f}",
        "",
        "--- Brain-mean std values ---",
        f"LOBPCG brain mean: {float(np.nanmean(lobpcg_brain)):.4e}",
        f"MC     brain mean: {float(np.nanmean(mc_brain)):.4e}",
        "",
        "--- Spatial correlation (full spectral dimension) ---",
        f"Pearson  r       : {pr:.4f}",
        f"Spearman r       : {sr:.4f}",
        "",
        "--- Diagnosis ---",
    ]

    if cv < 0.15:
        diag = (
            f"CV={cv:.3f} < 0.15  → ratio is SPATIALLY CONCENTRATED.\n"
            f"Dominant component is SYSTEMATIC bias.\n"
            f"Median ratio={median_r:.3f}: LOBPCG std is {1/median_r:.2f}× MC std on average.\n"
            f"Possible causes: k_eig too small (insufficient eigenpairs), "
            f"sigma miscalibration, or strong regularization compressing LOBPCG posterior."
        )
    elif cv < 0.35:
        diag = (
            f"CV={cv:.3f} moderate → MIXED: systematic bias + spatial variation.\n"
            f"Median ratio={median_r:.3f} is a rough overall calibration factor.\n"
            f"Residual spatial variation may reflect motion, field drift, or k_eig truncation."
        )
    else:
        diag = (
            f"CV={cv:.3f} > 0.35  → ratio is SPATIALLY VARIABLE.\n"
            f"Error dominated by spatially non-uniform sources (motion, field drift, etc.).\n"
            f"Check median_r={median_r:.3f} for any global bias."
        )
    stats_lines.append(diag)

    stats_text = "\n".join(stats_lines)
    print("\n" + stats_text)

    stats_path = os.path.join(out_dir, "compare_stats.txt")
    Path(stats_path).write_text(stats_text)
    print(f"\n[compare] Saved {stats_path}")
    print("[compare] Done.")


if __name__ == "__main__":
    main()
