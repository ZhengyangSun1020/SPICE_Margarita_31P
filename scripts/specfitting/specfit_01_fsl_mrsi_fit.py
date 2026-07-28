#!/usr/bin/env python3
"""
Step 7 — Spectral fitting of SPICE reconstruction.

Reads  : <out_dir>/spice_<run_tag>/SPICE_phcorr.nii.gz  (xcorr-aligned output from recon_01)
         OR --spice-phcorr <path>                        (explicit override)
         <data_dir>/wref_o.npy                           (water reference for brain mask)
         <fit-basis-dir>/                                (FSL-MRS fitting basis)
Writes : <out_dir>/fitting_<run_tag>/brain_mask.nii.gz
         <out_dir>/fitting_<run_tag>/spice_fit/   (fsl_mrsi output directory)
         <out_dir>/fitting_<run_tag>/conc_maps.npy
         <out_dir>/fitting_<run_tag>/fig_05_*.png

Usage:
    python scripts/specfitting/specfit_01_fsl_mrsi_fit.py \
        --data-dir      data/processed/invivo_250305_01 \
        --basis-dir     ./basis/ \
        --run-tag       w5000_l0.0001 \
        --dim 64 64 \
        --combine NAA NAAG --combine PCh GPC --combine Cr PCr \
        --rescale --brain-erosion 2 --brain-threshold 0.16 \
        [--plot-metabs NAA Cr Ins Glu PCh]
        [--ppmlim 0.0 7.5]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from warnings import filterwarnings
filterwarnings("ignore")

from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.core.nifti_mrs import NIFTI_MRS as NIFTI_MRS_fsl
from fsl.data.image import Image

# project root → utils package
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from utils.scan_params import load_scan_params

from utils.utils import plot_voxel_spectrum_and_maps


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_concentration_maps(fit_dir):
    """
    Load all metabolite concentration maps from fsl_mrsi output.
    Returns dict {metab_name: 2-D np.ndarray (Ny, Nx)}.
    """
    raw_dir = Path(fit_dir) / "concs" / "internal"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Concentration folder not found: {raw_dir}")

    conc_maps = {}
    for f in sorted(raw_dir.glob("*.nii*")):
        name = f.name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        data = np.squeeze(nib.load(str(f)).get_fdata())
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[:, :, 0]
        if data.ndim != 2:
            raise ValueError(f"{f.name}: expected 2-D after squeeze, got {data.shape}")
        conc_maps[name] = data.T[:, ::-1]
    return conc_maps


def plot_metab_map(name, conc_maps, out_path, brain_mask=None,
                   vmin=None, vmax=None, cmap="inferno", wref_img_2d=None):
    if name not in conc_maps:
        print(f"[warn] '{name}' not in conc_maps, skipping.")
        return

    raw = conc_maps[name]
    arr = np.where(brain_mask, raw, np.nan) if brain_mask is not None else raw

    import copy
    _cmap = copy.copy(plt.cm.get_cmap(cmap))
    _cmap.set_bad(color="black")

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")

    if wref_img_2d is not None:
        _gray = copy.copy(plt.cm.get_cmap("gray"))
        _gray.set_bad(color="black")
        wref_masked = np.where(brain_mask, wref_img_2d, np.nan) if brain_mask is not None else wref_img_2d
        ax.imshow(wref_masked, origin="lower", cmap=_gray, alpha=0.6, zorder=0)

    im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=_cmap,
                   alpha=0.9 if wref_img_2d is not None else 1.0, zorder=1)

    ax.set_title(f"Concentration: {name}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Concentration (arb. units)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    )
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SPICE spectral fitting — step 5")
    p.add_argument("--data-dir",         required=True,
                   help="Scan data directory (contains wref_o.npy)")
    p.add_argument("--basis-dir",        required=True,
                   help="Training basis dir (same as recon_01; also used as fit basis if --fit-basis-dir not set)")
    p.add_argument("--fit-basis-dir",    default=None,
                   help="Fitting basis directory for fsl_mrsi (defaults to --basis-dir)")
    p.add_argument("--out-dir",          default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--spice-phcorr",     default=None,
                   help="Path to SPICE_phcorr.nii.gz (overrides default "
                        "<out_dir>/spice_<run_tag>/SPICE_phcorr.nii.gz)")
    p.add_argument("--ref-nii",          default=None,
                   help="Reference NIfTI for affine (optional)")
    # Spectral / acquisition (must match step 04)
    p.add_argument("--dwelltime",        type=float, default=None)
    p.add_argument("--k-points",         type=int, default=None)
    p.add_argument("--n-seq-points",     type=int,   default=300)
    p.add_argument("--center-freq",      type=float, default=None)
    p.add_argument("--ppm-center",       type=float, default=3.027)
    p.add_argument("--dim",              type=int,   nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    # Brain mask (must match step 04)
    p.add_argument("--brain-threshold",  type=float, default=0.08)
    p.add_argument("--brain-erosion",    type=int,   default=3)
    # fsl_mrsi options
    p.add_argument("--ppmlim",           type=float, nargs=2, default=[0.0, 7.5],
                   metavar=("LO", "HI"))
    p.add_argument("--baseline",         default="poly, 0")
    p.add_argument("--combine",          nargs="+", action="append", default=[],
                   metavar="METAB",
                   help="Metabolite combine groups (repeat for multiple): "
                        "--combine NAA NAAG --combine PCh GPC")
    p.add_argument("--no-conj-basis",    action="store_true")
    p.add_argument("--no-conj-fid-flag", action="store_true")
    p.add_argument("--rescale",          action="store_true",
                   help="Pass rescale to fsl_mrsi (default: --no_rescale)")
    # Visualisation
    p.add_argument("--plot-metabs",      nargs="+",
                   default=["NAA","NAA+NAAG", "Cr","Cr+PCr", "Ins", "Glu", "PCh","PCh+GPC"])
    p.add_argument("--voxel-x",          type=int, default=32)
    p.add_argument("--voxel-y",          type=int, default=32)
    p.add_argument("--run-tag",          default="",
                   help="Run identifier from recon_01 (e.g. w5000_l0.0001); "
                        "appended to spice/fitting subdir names")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    data_dir      = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    _tg           = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    spice_dir     = os.path.join(args.out_dir, _tg("spice"))
    fit_dir       = os.path.join(args.out_dir, _tg("fitting"))
    fit_basis_dir = args.fit_basis_dir or args.basis_dir
    os.makedirs(fit_dir, exist_ok=True)

    Ny, Nx    = args.dim[0], args.dim[1]
    N_SEQ     = args.n_seq_points
    TS        = (args.k_points / N_SEQ) * args.dwelltime
    sweepwidth = 1.0 / TS
    center_freq = args.center_freq
    PPM_CENTER  = args.ppm_center
    FREQ_AXIS   = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS    = FREQ_AXIS / center_freq + PPM_CENTER

    # ── Reference NIfTI affine ────────────────────────────────────────────────
    # Priority: (1) --ref-nii arg, (2) affine.npy saved by data_proc_01_twix2npy,
    #           (3) fallback to identity
    _affine_npy = data_dir + "affine.npy"
    if args.ref_nii:
        affine = Image(args.ref_nii).voxToWorldMat
    elif os.path.exists(_affine_npy):
        affine = np.load(_affine_npy)
        print(f"[fitting] Loaded affine from {_affine_npy}")
    else:
        ref_nii_path = data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz"
        try:
            affine = Image(ref_nii_path).voxToWorldMat
        except Exception:
            affine = np.eye(4)
            print("[fitting] WARNING: affine.npy not found; using identity — run data_proc_01_twix2npy")

    # ── Load SPICE_phcorr (xcorr-aligned output from recon_01) ───────────────
    phcorr_path = args.spice_phcorr or os.path.join(spice_dir, "SPICE_phcorr.nii.gz")
    if not os.path.exists(phcorr_path):
        raise FileNotFoundError(
            f"SPICE_phcorr not found at {phcorr_path}\n"
            "Run recon_01 first or pass --spice-phcorr to an alternate path."
        )
    print(f"[fitting] Loading {phcorr_path} ...")
    aligned_nmrs = NIFTI_MRS_fsl(phcorr_path)
    aligned_nii  = phcorr_path
    aligned_data = np.array(aligned_nmrs.image[:, :, 0, :]).transpose(1, 0, 2)[:, ::-1, :].conj()  # (Ny, Nx, N_SEQ)

    # ── Brain mask ────────────────────────────────────────────────────────────
    wref_img  = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # save magnitude-weighted mask for fsl_mrsi
    mask_nii = os.path.join(fit_dir, "brain_mask.nii.gz")
    mask_data = np.ascontiguousarray((wref_2d * brain_mask).astype(np.float32).T[::-1, :])
    mask_nii_obj = nib.Nifti1Image(mask_data[:, :, np.newaxis], affine)
    mask_nii_obj.header.set_xyzt_units("mm")
    nib.save(mask_nii_obj, mask_nii)
    print(f"[fitting] Brain mask saved → {mask_nii}")

    # ── Visualise phcorr spectrum ─────────────────────────────────────────────
    _, fig_aln, _ = plot_voxel_spectrum_and_maps(
        FIDToSpec(aligned_data, axis=-1), (Ny, Nx, N_SEQ),
        voxel_x=args.voxel_x, voxel_y=args.voxel_y,
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS, show=False,
    )
    fig_aln.savefig(os.path.join(fit_dir, "fig_05a_spice_phcorr.png"), dpi=120)
    plt.close(fig_aln)

    # ── fsl_mrsi ──────────────────────────────────────────────────────────────
    fsl_out = os.path.join(fit_dir, "spice_fit")
    if os.path.exists(fsl_out):
        shutil.rmtree(fsl_out)
    cmd = [
        "fsl_mrsi",
        "--data",     aligned_nii,
        "--basis",    fit_basis_dir,
        "--mask",     mask_nii,
        "--baseline", args.baseline,
        "--ppmlim",   str(args.ppmlim[0]), str(args.ppmlim[1]),
        "--output",   fsl_out,
        "--overwrite",
        "--no_rescale",
        "--report",
    ]
    if not args.no_conj_basis:
        cmd.append("--conj_basis")
    if not args.no_conj_fid_flag:
        cmd.append("--no_conj_fid")
    if args.rescale:
        cmd.remove("--no_rescale")
    combine_groups = args.combine if args.combine else [["NAA", "NAAG"]]
    for group in combine_groups:
        cmd += ["--combine"] + list(group)

    print("[fitting/fsl_mrsi] Running:", " ".join(cmd))
    subprocess.run(cmd, env=os.environ.copy(), check=True)
    print(f"[fitting/fsl_mrsi] Done → {fsl_out}")

    # ── Symlink pipeline outputs into spice_fit/ subfolders for FSLeyes ─────
    fit_subdir = os.path.join(fsl_out, "fit")

    def _symlink(src, dst_name, subdir):
        """Absolute-path symlink subdir/dst_name → src. No-op if src missing."""
        if not os.path.exists(src):
            return False
        dst = os.path.join(subdir, dst_name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.unlink(dst)
        os.symlink(os.path.abspath(src), dst)
        print(f"[fitting] Symlinked {dst_name}")
        return True

    b0map_dir      = os.path.join(args.out_dir, "b0map")
    lprm_dir_link  = os.path.join(args.out_dir, "lipid_removal")
    adj_dir        = os.path.join(args.out_dir, "adjoint_test")
    coilmap_dir    = os.path.join(args.out_dir, "coilmap")

    # Files that go into spice_fit/fit/ (standard 4D NIfTI-MRS only)
    fit_symlinks = [
        # (dst_name, src_path, tree_type_label)  — missing src → silently skipped
        ("spice_aligned.nii.gz",      aligned_nii,                                                "fit-aligned"),
    ]

    # Files that go into spice_fit/data/ (non-standard dims or pipeline data)
    data_symlinks = [
        ("SPICE_result.nii.gz",       os.path.join(spice_dir, "SPICE_result.nii.gz"),             "data-spice-result"),
        ("adj_recon.nii.gz",          os.path.join(adj_dir,   "adj_recon.nii.gz"),                "data-adj-recon"),
        ("wref_adj_nufft.nii.gz",     os.path.join(b0map_dir, "wref_adj_nufft.nii.gz"),          "data-wref"),
        ("my_mrsi_lprm_f.nii.gz",     os.path.join(lprm_dir_link, "my_mrsi_lprm_f.nii.gz"),     "data-lprm-mrsi"),
        ("U_subspace.nii.gz",         os.path.join(spice_dir, "U_subspace.nii.gz"),               "data-subspace-U"),
        ("V_subspace.nii.gz",         os.path.join(spice_dir, "V_subspace.nii.gz"),               "data-subspace-V"),
        ("wref_masked.nii.gz",        os.path.join(spice_dir, "wref_masked.nii.gz"),              "data-wref-masked"),
        ("adj_recon_aligned.nii.gz",  os.path.join(adj_dir,   "adj_recon_aligned.nii.gz"),        "data-adj-aligned"),
        ("wref_phcorr.nii.gz",        os.path.join(b0map_dir, "wref_phcorr_nifti.nii.gz"),       "data-wref-phcorr"),
        ("b0_map.nii.gz",             os.path.join(b0map_dir, "B0_map.nii.gz"),                  "data-b0-map"),
        ("coilmap.nii.gz",            os.path.join(coilmap_dir, "ecalib_pp.nii.gz"),             "data-coilmap"),
        ("adj_bf_lprm.nii.gz",        os.path.join(lprm_dir_link, "adj_bf_lprm.nii.gz"),         "data-lprm-bf"),
        ("adj_bf_crs.nii.gz",         os.path.join(lprm_dir_link, "adj_bf_spice_crs_cr.nii.gz"), "data-lprm-crs"),
        ("lipid_basis.nii.gz",        os.path.join(lprm_dir_link, "lipid_basis.nii.gz"),          "data-lipid-basis"),
        ("mrsi_lprm_pre.nii.gz",      os.path.join(lprm_dir_link, "mrsi_lprm_pre_phcorr.nii.gz"), "data-lprm-pre"),
        ("brain_mask.nii.gz",         os.path.join(fit_dir, "brain_mask.nii.gz"),                "data-brain-mask"),
    ]

    data_subdir = os.path.join(fsl_out, "data")
    os.makedirs(data_subdir, exist_ok=True)

    fit_created  = [(n, t) for n, s, t in fit_symlinks  if _symlink(s, n, fit_subdir)]
    data_created = [(n, t) for n, s, t in data_symlinks if _symlink(s, n, data_subdir)]

    # ── Patch mrsi.tree ───────────────────────────────────────────────────────
    tree_path = os.path.join(fsl_out, "mrsi.tree")
    if os.path.exists(tree_path):
        with open(tree_path) as f:
            tree_txt = f.read()

        # fit/ entries: append into fit section (before data or uncertainties)
        new_fit = "".join(
            f"    {name:<38} ({label})\n"
            for name, label in fit_created
            if name not in tree_txt
        )
        if new_fit:
            anchor = "data\n" if "data\n" in tree_txt else "uncertainties\n"
            tree_txt = tree_txt.replace(anchor, new_fit + anchor)

        # data/ entries: append into data section (create section if missing)
        new_data = "".join(
            f"    {name:<38} ({label})\n"
            for name, label in data_created
            if name not in tree_txt
        )
        if new_data:
            if "data\n" in tree_txt:
                tree_txt = tree_txt.replace("data\n", "data\n" + new_data)
            else:
                tree_txt = tree_txt.replace("uncertainties\n", "data\n" + new_data + "uncertainties\n")

        with open(tree_path, "w") as f:
            f.write(tree_txt)
        print(f"[fitting] Patched mrsi.tree (+{len(fit_created)} fit, +{len(data_created)} data entries)")

    # ── Load & plot concentration maps ────────────────────────────────────────
    try:
        conc_maps = load_concentration_maps(fsl_out)
        np.save(os.path.join(fit_dir, "conc_maps.npy"), conc_maps)
        print(f"[fitting] Metabolites fitted: {sorted(conc_maps.keys())}")

        for meta in args.plot_metabs:
            out_png = os.path.join(fit_dir, f"fig_05c_{meta}.png")
            if meta in conc_maps:
                arr = conc_maps[meta]
                vmax = float(np.nanpercentile(arr[brain_mask_inner], 90))
            else:
                vmax = None
            plot_metab_map(
                meta, conc_maps, out_png,
                brain_mask=brain_mask_inner,
                wref_img_2d=wref_norm,
                vmin=0,
                vmax=vmax,
            )
            if meta in conc_maps:
                print(f"[fitting] Saved {out_png}")
    except FileNotFoundError as e:
        print(f"[warn] Could not load concentration maps: {e}")

    print("[fitting] Step 5 complete.")


if __name__ == "__main__":
    main()
