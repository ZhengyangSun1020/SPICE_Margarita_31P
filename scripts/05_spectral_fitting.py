#!/usr/bin/env python3
"""
Step 5 — Spectral fitting of SPICE reconstruction.

Reads  : <out_dir>/spice/SPICE_f.npy             (raw SPICE FID from step 04)
         <data_dir>/wref_o.npy                   (water reference for brain mask)
         <fit-basis-dir>/                         (FSL-MRS fitting basis)
Writes : <out_dir>/fitting/spice_aligned.nii.gz  (xcorr freq-aligned NIfTI-MRS)
         <out_dir>/fitting/brain_mask.nii.gz
         <out_dir>/fitting/spice_fit.nii.gz/      (fsl_mrsi output directory)
         <out_dir>/fitting/conc_maps.npy
         <out_dir>/fitting/fig_05_*.png

Usage:
    python scripts/05_spectral_fitting.py \\
        --data-dir ./invivo_260305/cr/ \\
        --basis-dir ./2pi_csap_SMF_MRSI/ \\
        --fit-basis-dir ./ISMRM2026_BASIS_fit/ \\
        [--out-dir ./output] [--ppmlim -5 10]
"""

import argparse
import os
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

from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.utils.synthetic import syntheticFromBasisFile
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl.data.image import Image

# project root → utils package + xcorr
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from utils.xcorr import my_mrsi_freq_align
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
        conc_maps[name] = data.T
    return conc_maps


def plot_metab_map(name, conc_maps, out_path, brain_mask=None,
                   vmin=None, vmax=None, cmap="inferno", wref_img_2d=None):
    if name not in conc_maps:
        print(f"[warn] '{name}' not in conc_maps, skipping.")
        return

    arr = conc_maps[name]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")

    if wref_img_2d is not None:
        ax.imshow(wref_img_2d, origin="lower", cmap="gray", alpha=0.6, zorder=0)

    im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap,
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

    if brain_mask is not None:
        ax.contour(brain_mask, levels=[0.5], colors="white", linewidths=0.7, zorder=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SPICE spectral fitting — step 5")
    p.add_argument("--data-dir",         required=True,
                   help="Scan data directory (contains wref_o.npy)")
    p.add_argument("--basis-dir",        required=True,
                   help="Training basis dir (same as step 04; used to build basis_nmrs for xcorr)")
    p.add_argument("--fit-basis-dir",    default=None,
                   help="Fitting basis directory for fsl_mrsi (defaults to --basis-dir)")
    p.add_argument("--out-dir",          default="./output")
    p.add_argument("--ref-nii",          default=None,
                   help="Reference NIfTI for affine (optional)")
    # Spectral / acquisition (must match step 04)
    p.add_argument("--dwelltime",        type=float, default=5e-6)
    p.add_argument("--k-points",         type=int,   default=39842)
    p.add_argument("--n-seq-points",     type=int,   default=300)
    p.add_argument("--center-freq",      type=float, default=297.219338)
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
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    data_dir      = args.data_dir.rstrip("/") + "/"
    spice_dir     = os.path.join(args.out_dir, "spice")
    fit_dir       = os.path.join(args.out_dir, "fitting")
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
    ref_nii_path = args.ref_nii or (data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz")
    try:
        ref_img = Image(ref_nii_path)
        affine  = ref_img.voxToWorldMat
    except Exception:
        ref_img = None
        affine  = np.eye(4)

    # ── Load raw SPICE FID ────────────────────────────────────────────────────
    spice_f_path = os.path.join(spice_dir, "SPICE_f.npy")
    if not os.path.exists(spice_f_path):
        raise FileNotFoundError(
            f"SPICE_f.npy not found at {spice_f_path}\n"
            "Run step 04 (04_run_spice.py) first."
        )
    print(f"[fitting] Loading {spice_f_path} ...")
    spice_est = np.load(spice_f_path)               # (N_Vox, N_SEQ) or (Ny*Nx, N_SEQ)
    spice_3d  = spice_est.reshape(Ny, Nx, N_SEQ)    # (Ny, Nx, N_SEQ), FID

    # ── Brain mask ────────────────────────────────────────────────────────────
    wref_img  = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # save magnitude-weighted mask for fsl_mrsi
    mask_nii = os.path.join(fit_dir, "brain_mask.nii.gz")
    Image((wref_2d * brain_mask).astype(np.float32).T).save(mask_nii)
    print(f"[fitting] Brain mask saved → {mask_nii}")

    # ── Visualise raw SPICE spectrum ──────────────────────────────────────────
    _, fig_pre, _ = plot_voxel_spectrum_and_maps(
        FIDToSpec(spice_3d, axis=-1), (Ny, Nx, N_SEQ),
        voxel_x=args.voxel_x, voxel_y=args.voxel_y,
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS, show=False,
    )
    fig_pre.savefig(os.path.join(fit_dir, "fig_05a_spice_raw.png"), dpi=120)
    plt.close(fig_pre)

    # ── Build basis_nmrs (reference for xcorr) ────────────────────────────────
    print("[fitting] Building basis_nmrs for xcorr ...")
    fullbasis = mrs_io.read_basis(args.basis_dir)
    fid_ref, emptymrs, _ = syntheticFromBasisFile(
        fullbasis,
        noisecovariance=[[0]],
        bandwidth=sweepwidth,
        points=N_SEQ,
    )
    basis_nmrs = gen_nifti_mrs(
        fid_ref.conj().reshape(1, 1, 1, N_SEQ),
        dwelltime=emptymrs.dwellTime,
        spec_freq=emptymrs.centralFrequency,
        affine=affine,
    )
    fig_basis = basis_nmrs.plot(ppmlim=(0, 10))
    fig_basis.savefig(os.path.join(fit_dir, "fig_05_basis_nmrs.png"), dpi=150)
    plt.close(fig_basis)
    print(f"[fitting] basis_nmrs plot saved → {os.path.join(fit_dir, 'fig_05_basis_nmrs.png')}")

    # ── Wrap SPICE FID in NIfTI-MRS ──────────────────────────────────────────
    # NIfTI-MRS layout: (Nx, Ny, 1, npts)  (x-axis first)
    mrsi_f_4 = spice_3d.transpose(1, 0, 2)[:, :, np.newaxis, :]
    nifti_spice = gen_nifti_mrs(
        mrsi_f_4,
        dwelltime=TS,
        spec_freq=297.219,
        affine=affine,
    )

    # ── xcorr frequency alignment ─────────────────────────────────────────────
    print("[fitting] Running xcorr frequency alignment ...")
    aligned_nmrs, _ = my_mrsi_freq_align(nifti_spice, basis_nmrs)

    aligned_nii = os.path.join(fit_dir, "spice_aligned.nii.gz")
    aligned_nmrs.save(aligned_nii)
    print(f"[fitting] Freq-aligned NIfTI saved → {aligned_nii}")

    # Visualise after alignment
    aligned_data = np.array(aligned_nmrs.image[:, :, 0, :]).transpose(1, 0, 2).conj()  # (Ny, Nx, N_SEQ)
    _, fig_aln, _ = plot_voxel_spectrum_and_maps(
        FIDToSpec(aligned_data, axis=-1), (Ny, Nx, N_SEQ),
        voxel_x=args.voxel_x, voxel_y=args.voxel_y,
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS, show=False,
    )
    fig_aln.savefig(os.path.join(fit_dir, "fig_05b_spice_aligned.png"), dpi=120)
    plt.close(fig_aln)

    # ── fsl_mrsi ──────────────────────────────────────────────────────────────
    fsl_out = os.path.join(fit_dir, "spice_fit.nii.gz")
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
