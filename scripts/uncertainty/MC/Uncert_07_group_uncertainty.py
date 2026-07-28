#!/usr/bin/env python3
"""
Step 14 — Group uncertainty across 260623 series subjects.

Pre-fitting uncertainty:
    Compute std of SPICE-reconstructed spectra across subjects, per voxel.
    → output/group_260623_w5000_l0.0001/prefitting_std.npy   (Ny, Nx, N_seq)

Concentration uncertainty (both raw and internal):
    Compute std of fsl_mrsi concentration maps across subjects, per voxel.
    → output/group_260623_w5000_l0.0001/conc_std_raw.npy      (Ny, Nx, n_metab)
    → output/group_260623_w5000_l0.0001/conc_std_internal.npy (Ny, Nx, n_metab)

Output directory is auto-derived as <out-root>/group_<prefix>_<run-tag> when --out-dir is not given.

Usage:
    python scripts/uncertainty/MC/Uncert_07_group_uncertainty.py \
        --subjects invivo_260623_01 invivo_260623_02 invivo_260623_03 invivo_260623_04 invivo_260623_05 \
        --run-tag  w5000_l0.0001 \
        --dim 64 64 \
        --plot-metabs NAA Cr Ins Glu PCh
"""

import argparse
import os
import sys
from pathlib import Path
from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nibabel as nib
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)
from utils.scan_params import load_scan_params
from utils.pipeline_utils import make_brain_mask


# ── helpers ───────────────────────────────────────────────────────────────────

def fid_to_spec(fid):
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm="ortho"), axes=-1)


def load_conc_maps(fit_dir, use_internal):
    """Load all metabolite maps from fsl_mrsi concs/raw or concs/internal.
    Returns (dict{name: (Ny,Nx)}, list[str]).  (Nx,Ny) → (Ny,Nx) transpose applied.
    """
    subdir  = "internal" if use_internal else "raw"
    src_dir = Path(fit_dir) / "concs" / subdir
    if not src_dir.exists():
        raise FileNotFoundError(f"Not found: {src_dir}")
    conc_maps   = {}
    metab_names = []
    for f in sorted(src_dir.glob("*.nii*")):
        name = f.name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        data = np.squeeze(nib.load(str(f)).get_fdata())
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[:, :, 0]
        conc_maps[name] = data.T   # (Nx,Ny) → (Ny,Nx)
        metab_names.append(name)
    return conc_maps, metab_names


def _plot_map(title, data_2d, wref_norm, brain_mask, out_path,
              vmax=None, cmap="Reds", cbar_label=""):
    import copy
    masked = np.where(brain_mask, data_2d, np.nan)
    cmap_obj = copy.copy(plt.cm.get_cmap(cmap))
    cmap_obj.set_bad(color="black", alpha=1.0)   # NaN → fully black
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")
    # wref only inside brain so outside stays black
    wref_brain = np.where(brain_mask, wref_norm, np.nan)
    wref_cmap  = copy.copy(plt.cm.get_cmap("gray"))
    wref_cmap.set_bad(color="black", alpha=1.0)
    ax.imshow(wref_brain, origin="lower", cmap=wref_cmap, alpha=0.6, zorder=0)
    im = ax.imshow(masked, origin="lower", vmin=0, vmax=vmax,
                   cmap=cmap_obj, alpha=0.9, zorder=1)
    ax.contour(brain_mask, levels=[0.5], colors="white", linewidths=0.7, zorder=2)
    ax.set_title(title, color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    )
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)


def _plot_prefitting_voxel_uncert(mean_map, std_map, mean_spec, std_spec,
                                   PPM_AXIS, brain_mask, wref_norm,
                                   vy, vx, out_path, threshold=None):
    """3-panel figure: mean magnitude map | std map | spectrum ± std at (vy, vx)."""
    import copy
    import matplotlib.ticker as mticker

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor="black")
    for ax in axs:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for sp in ax.spines.values():
            sp.set_color("white")

    # Panel 1: mean spectral magnitude map
    mean_masked = np.where(brain_mask, mean_map, np.nan)
    cmap_g = copy.copy(plt.cm.get_cmap("viridis"))
    cmap_g.set_bad(color="black")
    im0 = axs[0].imshow(mean_masked, origin="lower", cmap=cmap_g)
    axs[0].set_title("Mean spectral magnitude (avg over ppm)")
    cbar0 = plt.colorbar(im0, ax=axs[0], fraction=0.046)
    cbar0.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar0.ax.get_yticklabels(), color="white")
    axs[0].plot(vx, vy, "g+", markersize=10, markeredgewidth=2)

    # Panel 2: std map with numerical colorbar
    std_masked = np.where(brain_mask, std_map, np.nan)
    vmax_std = threshold if threshold is not None else (
        float(np.nanpercentile(std_masked[brain_mask], 95)) if brain_mask.any() else None
    )
    cmap_r = copy.copy(plt.cm.get_cmap("Reds"))
    cmap_r.set_bad(color="black")
    wref_brain = np.where(brain_mask, wref_norm, np.nan)
    wref_cmap = copy.copy(plt.cm.get_cmap("gray"))
    wref_cmap.set_bad(color="black")
    axs[1].imshow(wref_brain, origin="lower", cmap=wref_cmap, alpha=0.5, zorder=0)
    im1 = axs[1].imshow(std_masked, origin="lower", vmin=0, vmax=vmax_std,
                         cmap=cmap_r, alpha=0.9, zorder=1)
    axs[1].contour(brain_mask, levels=[0.5], colors="white", linewidths=0.7, zorder=2)
    axs[1].set_title("Pre-fitting std (mean over ppm)")
    cbar1 = plt.colorbar(im1, ax=axs[1], fraction=0.046)
    cbar1.ax.yaxis.set_tick_params(color="white")
    cbar1.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
    plt.setp(cbar1.ax.get_yticklabels(), color="white")
    axs[1].plot(vx, vy, "g+", markersize=10, markeredgewidth=2)

    # Panel 3: spectrum ± std at selected voxel
    mu  = mean_spec[vy, vx, :]
    sig = std_spec[vy, vx, :]
    axs[2].plot(PPM_AXIS, mu, color="white", label="Mean")
    axs[2].fill_between(PPM_AXIS, mu - sig, mu + sig, alpha=0.35, color="tomato",
                         label="±1 std")
    axs[2].set_xlabel("ppm")
    axs[2].set_ylabel("|Spectrum|")
    axs[2].set_title(f"Group spectrum uncertainty  voxel ({vy},{vx})")
    axs[2].invert_xaxis()
    axs[2].legend(labelcolor="white", facecolor="black")
    axs[2].grid(True, alpha=0.3, color="gray")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"[group-uncert] Saved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Group std uncertainty across subjects — step 14")
    p.add_argument("--subjects",        nargs="+", required=True,
                   help="Subject IDs (e.g. invivo_260623_01 ... invivo_260623_05)")
    p.add_argument("--out-root",        default="./output",
                   help="Root dir containing per-subject output folders (default: ./output)")
    p.add_argument("--data-root",       default="./data/processed",
                   help="Root dir containing per-subject data folders (default: ./data/processed)")
    p.add_argument("--out-dir",         default=None,
                   help="Output directory for group results (default: <out-root>/group_<prefix>)")
    p.add_argument("--run-tag",         default="",
                   help="Run tag used in recon/fitting dirs (e.g. w5000_l0.0001); "
                        "auto-derives fit-subdir as fitting_<tag>/spice_fit")
    p.add_argument("--fit-subdir",      default=None,
                   help="Subdir under each subject's output for fsl_mrsi results "
                        "(default: fitting_<run-tag>/spice_fit, or fitting/spice_fit if no tag)")
    p.add_argument("--dim",             type=int, nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    p.add_argument("--n-seq-points",    type=int, default=300)
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int,   default=None)
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--brain-threshold", type=float, default=0.16,
                   help="wref_o normalised threshold for brain mask (default 0.07, matching scripts 02/03)")
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component and fill enclosed holes. Default: off.")
    p.add_argument("--plot-metabs",     nargs="+",
                   default=["NAA", "NAA+NAAG", "Cr", "Cr+PCr", "Ins", "Glu", "PCh", "PCh+GPC"])
    p.add_argument("--ref-subject",     default=None,
                   help="Subject used for brain mask / wref (default: first subject)")
    p.add_argument("--threshold",       type=float, default=None,
                   help="vmax for prefitting std map colorbar (default: auto 95th percentile)")
    p.add_argument("--voxel-x",        type=int, default=38,
                   help="Column index for spectrum uncertainty plot (default: 38)")
    p.add_argument("--voxel-y",        type=int, default=20,
                   help="Row index for spectrum uncertainty plot (default: 20)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    Ny, Nx  = args.dim
    N_SEQ   = args.n_seq_points
    N_SUBJ  = len(args.subjects)

    if args.fit_subdir is None:
        _fit_base = f"fitting_{args.run_tag}" if args.run_tag else "fitting"
        args.fit_subdir = f"{_fit_base}/spice_fit"

    # Derive out-dir from subject list prefix + run-tag if not given
    if args.out_dir is None:
        prefix = os.path.commonprefix(args.subjects).rstrip("_0")
        base   = f"group_{prefix or 'group'}"
        args.out_dir = os.path.join(args.out_root, f"{base}_{args.run_tag}" if args.run_tag else base)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[group-uncert] Subjects ({N_SUBJ}): {args.subjects}")
    print(f"[group-uncert] Output: {args.out_dir}")

    # Use first subject's scan_params for axis construction
    ref_subj     = args.ref_subject or args.subjects[0]
    ref_data_dir = os.path.join(args.data_root, ref_subj) + "/"
    load_scan_params(args, ref_data_dir, k_key="k_mrsi")

    TS         = (args.k_points / N_SEQ) * args.dwelltime
    sweepwidth = 1.0 / TS
    FREQ_AXIS  = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS   = FREQ_AXIS / args.center_freq + args.ppm_center

    # Brain mask from ref subject (threshold=0.07, consistent with scripts 02/03)
    wref_ref = np.load(os.path.join(ref_data_dir, "wref_o.npy"), mmap_mode="r")
    wref_norm, brain_mask, _ = make_brain_mask(wref_ref, args.brain_threshold, args.brain_erosion,
                                                cleanup=args.brain_mask_cleanup)

    # ── Pre-fitting uncertainty ────────────────────────────────────────────────
    print("\n[group-uncert] === Pre-fitting uncertainty ===")
    spice_specs = []
    _tg = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    for subj in args.subjects:
        spice_path = os.path.join(args.out_root, subj, _tg("spice"), "SPICE_f.npy")
        if not os.path.exists(spice_path):
            print(f"[warn] Missing {spice_path}, skipping subject.")
            continue
        fid = np.load(spice_path)            # (N_vox, N_seq)
        spec = fid_to_spec(fid.reshape(Ny, Nx, N_SEQ))  # (Ny, Nx, N_seq)
        spice_specs.append(spec)
        print(f"  Loaded {subj}: SPICE_f {fid.shape}")

    if len(spice_specs) < 2:
        print("[warn] Need at least 2 subjects for pre-fitting std.")
    else:
        spice_stack = np.stack(spice_specs, axis=0)   # (N_subj, Ny, Nx, N_seq)
        prefitting_std  = np.std(spice_stack, axis=0)            # (Ny, Nx, N_seq) complex std, consistent with analytical posterior_std
        prefitting_mean = np.mean(np.abs(spice_stack), axis=0)
        # mask outside brain to NaN
        prefitting_std[~brain_mask, :]  = np.nan
        prefitting_mean[~brain_mask, :] = np.nan

        np.save(os.path.join(args.out_dir, "prefitting_std.npy"),  prefitting_std)
        np.save(os.path.join(args.out_dir, "prefitting_mean.npy"), prefitting_mean)
        print(f"[group-uncert] Saved prefitting_std.npy  shape={prefitting_std.shape}")

        # Summary plot: mean std over ppm axis
        mean_std_map = prefitting_std.mean(axis=-1)   # (Ny, Nx)
        _plot_map(
            "Pre-fitting std (mean over ppm)",
            mean_std_map, wref_norm, brain_mask,
            os.path.join(args.out_dir, "fig_14_prefitting_std_map.png"),
            cmap="Reds", cbar_label="Std (spectral magnitude)",
        )

        # Per-subject spectra overlay at selected voxel
        vy, vx = args.voxel_y, args.voxel_x
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="black")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
        for k, spec in enumerate(spice_specs):
            ax.plot(PPM_AXIS, np.abs(spec[vy, vx, :]), alpha=0.7, label=args.subjects[k])
        ax.set_xlabel("ppm", color="white")
        ax.set_ylabel("|Spectrum|", color="white")
        ax.set_title(f"Per-subject spectra at voxel ({vy},{vx})", color="white")
        ax.invert_xaxis()
        ax.legend(fontsize=7, labelcolor="white", facecolor="black")
        ax.grid(True, alpha=0.3, color="gray")
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_14_prefitting_spectra.png"),
                    dpi=150, facecolor="black")
        plt.close(fig)

        # 3-panel uncertainty figure (map | std map | spectrum ± std)
        _plot_prefitting_voxel_uncert(
            mean_map  = prefitting_mean.mean(axis=-1),
            std_map   = mean_std_map,
            mean_spec = prefitting_mean,
            std_spec  = prefitting_std,
            PPM_AXIS  = PPM_AXIS,
            brain_mask = brain_mask,
            wref_norm  = wref_norm,
            vy=vy, vx=vx,
            out_path  = os.path.join(args.out_dir, "fig_14_prefitting_voxel_uncert.png"),
            threshold = args.threshold,
        )
        print(f"[group-uncert] Saved fig_14_prefitting_std_map.png + fig_14_prefitting_spectra.png")

    # ── Concentration uncertainty ──────────────────────────────────────────────
    print("\n[group-uncert] === Concentration uncertainty ===")
    conc_raw_list      = []
    conc_internal_list = []
    metab_names_ref    = None

    for subj in args.subjects:
        fit_dir = os.path.join(args.out_root, subj, args.fit_subdir)
        if not os.path.exists(fit_dir):
            print(f"[warn] Missing {fit_dir}, skipping subject.")
            continue
        try:
            raw_maps, metab_names = load_conc_maps(fit_dir, use_internal=False)
            int_maps, _           = load_conc_maps(fit_dir, use_internal=True)
        except FileNotFoundError as e:
            print(f"[warn] {subj}: {e}")
            continue

        if metab_names_ref is None:
            metab_names_ref = metab_names
            print(f"  Metabolites: {metab_names_ref}")

        nan_map = np.full((Ny, Nx), np.nan)
        raw_stack = np.stack([raw_maps.get(m, nan_map) for m in metab_names_ref], axis=-1)
        int_stack = np.stack([int_maps.get(m, nan_map) for m in metab_names_ref], axis=-1)

        conc_raw_list.append(raw_stack)
        conc_internal_list.append(int_stack)
        print(f"  Loaded {subj}: {len(metab_names_ref)} metabolites")

    if len(conc_raw_list) < 2:
        print("[warn] Need at least 2 subjects for concentration std.")
        return

    conc_raw_arr = np.stack(conc_raw_list, axis=0)       # (N_subj, Ny, Nx, n_metab)
    conc_int_arr = np.stack(conc_internal_list, axis=0)

    conc_std_raw      = np.nanstd(conc_raw_arr,  axis=0)  # (Ny, Nx, n_metab)
    conc_std_internal = np.nanstd(conc_int_arr,  axis=0)
    conc_mean_raw     = np.nanmean(conc_raw_arr, axis=0)
    conc_mean_internal = np.nanmean(conc_int_arr, axis=0)
    # mask outside brain to NaN
    conc_std_raw[~brain_mask, :]       = np.nan
    conc_std_internal[~brain_mask, :]  = np.nan
    conc_mean_raw[~brain_mask, :]      = np.nan
    conc_mean_internal[~brain_mask, :] = np.nan

    np.save(os.path.join(args.out_dir, "conc_std_raw.npy"),       conc_std_raw)
    np.save(os.path.join(args.out_dir, "conc_std_internal.npy"),  conc_std_internal)
    np.save(os.path.join(args.out_dir, "conc_mean_raw.npy"),      conc_mean_raw)
    np.save(os.path.join(args.out_dir, "conc_mean_internal.npy"), conc_mean_internal)
    np.save(os.path.join(args.out_dir, "metab_names.npy"),        np.array(metab_names_ref))
    np.save(os.path.join(args.out_dir, "conc_raw_all.npy"),       conc_raw_arr)
    np.save(os.path.join(args.out_dir, "conc_internal_all.npy"),  conc_int_arr)

    print(f"[group-uncert] Saved conc_std_raw.npy       shape={conc_std_raw.shape}")
    print(f"[group-uncert] Saved conc_std_internal.npy  shape={conc_std_internal.shape}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    for meta in args.plot_metabs:
        if meta not in metab_names_ref:
            print(f"[warn] '{meta}' not in metabolites, skipping.")
            continue
        idx = metab_names_ref.index(meta)

        # raw std
        std_raw  = conc_std_raw[:, :, idx]
        mean_raw = conc_mean_raw[:, :, idx]
        vmax_s = float(np.nanpercentile(std_raw[brain_mask],  90)) if brain_mask.any() else None
        vmax_m = float(np.nanpercentile(mean_raw[brain_mask], 90)) if brain_mask.any() else None

        _plot_map(f"Raw conc. std: {meta}", std_raw, wref_norm, brain_mask,
                  os.path.join(args.out_dir, f"fig_14_std_raw_{meta}.png"),
                  vmax=vmax_s, cmap="Reds", cbar_label="Std (arb. units)")
        _plot_map(f"Raw conc. mean: {meta}", mean_raw, wref_norm, brain_mask,
                  os.path.join(args.out_dir, f"fig_14_mean_raw_{meta}.png"),
                  vmax=vmax_m, cmap="inferno", cbar_label="Mean (arb. units)")

        # internal std
        std_int  = conc_std_internal[:, :, idx]
        mean_int = conc_mean_internal[:, :, idx]
        vmax_si = float(np.nanpercentile(std_int[brain_mask],  90)) if brain_mask.any() else None
        vmax_mi = float(np.nanpercentile(mean_int[brain_mask], 90)) if brain_mask.any() else None

        _plot_map(f"Internal conc. std: {meta}", std_int, wref_norm, brain_mask,
                  os.path.join(args.out_dir, f"fig_14_std_internal_{meta}.png"),
                  vmax=vmax_si, cmap="Blues", cbar_label="Std (ratio units)")
        _plot_map(f"Internal conc. mean: {meta}", mean_int, wref_norm, brain_mask,
                  os.path.join(args.out_dir, f"fig_14_mean_internal_{meta}.png"),
                  vmax=vmax_mi, cmap="inferno", cbar_label="Mean (ratio units)")

        print(f"[group-uncert] Saved plots for {meta}")

    print("\n[group-uncert] Step 14 complete.")


if __name__ == "__main__":
    main()
