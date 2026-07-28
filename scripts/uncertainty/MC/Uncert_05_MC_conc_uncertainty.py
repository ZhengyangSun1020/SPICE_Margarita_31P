#!/usr/bin/env python3
"""
Step 12 — Monte Carlo concentration concentration uncertainty.

For each posterior SPICE sample (from Laplace approximation), runs the full
spectral fitting pipeline (xcorr alignment + fsl_mrsi), then computes
concentration uncertainty (std) across all samples.

Supports two modes matching step 09:
  voxelwise   Samples from per-voxel mHm Hessian (step 08/09).  Each sample
              is drawn from CN(mu_vox, cov_vox) — already centred on the mean.
  lobpcg      Samples from LOBPCG eigenpairs (step 10).  Perturbations from
              CN(0, H^{-1}) are added to the mean SPICE estimate.

Reads  : <out_dir>/spice_<run_tag>/SPICE_f.npy        (tag e.g. w5000_l0.0001)
         <out_dir>/spice_<run_tag>/V_subspace.npy
         <out_dir>/spice_<run_tag>/U_est.npy
         <data_dir>/wref_o.npy
         [voxelwise] <out_dir>/hessian_<run_tag>/mHm_*.npy  +  <data_dir>/sigma_noise.npy
         [lobpcg]    <out_dir>/lobpcg_<run_tag>/lobpcg_Q.npy  lobpcg_vals.npy
         <basis_dir>/            (training basis for xcorr reference)
         <fit_basis_dir>/        (FSL-MRS fitting basis)

Writes : <out_dir>/conc_uncertainty_<run_tag>/
             mc_{i:04d}_phcorr.nii.gz      (xcorr-aligned sample NIfTI-MRS)
             mc_{i:04d}_fit.nii.gz/         (fsl_mrsi output directory)
             output_concs.npy               (N_success, Ny, Nx, n_metab)
             conc_std.npy                   (Ny, Nx, n_metab)
             conc_mean.npy                  (Ny, Nx, n_metab)
             metab_names.npy
             fig_11_conc_std_<metab>.png

Usage:
    # voxelwise mode (default)
    python scripts/uncertainty/MC/Uncert_05_MC_conc_uncertainty.py \
        --data-dir  data/processed/invivo_250305_01 \
        --basis-dir ./basis/ \
        --run-tag   w5000_l0.0001 \
        --rank 20 --n-samples 20 \
        --combine NAA NAAG --combine PCh GPC --combine Cr PCr \
        --plot-metabs NAA Cr Ins Glu PCh

    # lobpcg mode (faster)
    python scripts/uncertainty/MC/Uncert_05_MC_conc_uncertainty.py \
        --data-dir  data/processed/invivo_250305_01 \
        --basis-dir ./basis/ \
        --run-tag   w5000_l0.0001 \
        --mode lobpcg --rank 20 --n-samples 20
"""

import argparse
import os
import shutil
import subprocess
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
from scipy.ndimage import binary_erosion
from tqdm import tqdm

from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.synthetic import syntheticFromBasisFile
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs, NIFTI_MRS
from fsl.data.image import Image

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)
from utils.scan_params import load_scan_params

from utils.xcorr import my_mrsi_freq_align

D_TYPE = np.complex64


# ── posterior sampling (ported from 09/10) ────────────────────────────────────

def _sample_complex_mvn(mean, cov, rng):
    """Draw one sample from CN(mean, cov)."""
    cov  = 0.5 * (cov + cov.conj().T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals.real, 0.0, None)
    L = evecs * np.sqrt(evals)[None, :]
    d = len(mean)
    z = (rng.standard_normal(d) + 1j * rng.standard_normal(d)) / np.sqrt(2.0)
    return (mean + z @ L.conj().T).astype(D_TYPE)


def build_voxelwise_samples(mHm_dir, N_voxel, V, mu_map, cov_scale,
                              n_samples, rng, dtype=D_TYPE):
    """
    Pre-generate all n_samples from the per-voxel Hessian mHm.
    Loads each mHm file once.  Returns (n_samples, N_voxel, N_seq) FID array.
    """
    mHm_dir = Path(mHm_dir)
    N_seq   = mu_map.shape[1]
    samples = np.broadcast_to(mu_map[np.newaxis], (n_samples, N_voxel, N_seq)).copy().astype(dtype)

    files = sorted(mHm_dir.glob("mHm_*.npy"))
    print(f"[mc-fit] Found {len(files)} mHm files in {mHm_dir}")
    if len(files) == 0:
        raise FileNotFoundError(f"No mHm_*.npy files in {mHm_dir}")

    for f in tqdm(files, desc="Precomputing voxel posteriors"):
        try:
            vox_idx = int(f.stem.split("_")[1])
        except Exception:
            continue
        if vox_idx < 0 or vox_idx >= N_voxel:
            continue
        mHm   = np.load(f).astype(np.complex128)
        Sigma = cov_scale * (V @ mHm @ V.conj().T)
        mu    = mu_map[vox_idx].astype(np.complex128)
        # Precompute L once, then draw n_samples
        Sigma = 0.5 * (Sigma + Sigma.conj().T)
        evals, evecs = np.linalg.eigh(Sigma)
        evals = np.clip(evals.real, 0.0, None)
        L = (evecs * np.sqrt(evals)[None, :]).astype(np.complex128)
        Z = (rng.standard_normal((n_samples, N_seq))
             + 1j * rng.standard_normal((n_samples, N_seq))) / np.sqrt(2.0)
        draws = (mu[np.newaxis] + Z @ L.conj().T).astype(dtype)
        samples[:, vox_idx, :] = draws

    return samples   # (n_samples, N_voxel, N_seq)


def build_lobpcg_samples(Q, vals, mean_U, V, n_samples, sigma2, rng, rank, N_voxel, dtype=D_TYPE):
    """
    Draw n_samples from CN(mean, Q Λ^{-1} Q^H) in U space, map to FID space.
    Returns (n_samples, N_voxel, N_seq) FID array.
    """
    k = Q.shape[1]
    lam_inv_sqrt = np.sqrt(1.0 / (vals + 1e-20))
    mean_flat = mean_U.ravel()   # (N_voxel * rank,)
    N_seq = V.shape[0]

    samples = np.zeros((n_samples, N_voxel, N_seq), dtype=dtype)
    batch = 20
    for b0 in tqdm(range(0, n_samples, batch), desc="Sampling (lobpcg)"):
        b1  = min(n_samples, b0 + batch)
        bs  = b1 - b0
        Zk  = (rng.standard_normal((bs, k)) + 1j * rng.standard_normal((bs, k))) / np.sqrt(2)
        dU  = ((Zk * lam_inv_sqrt[np.newaxis, :]) @ Q.T).astype(dtype)
        if sigma2 != 1.0:
            dU *= np.sqrt(sigma2)
        u_full = (mean_flat[np.newaxis] + dU).reshape(bs, N_voxel, rank)
        samples[b0:b1] = (u_full @ V.conj().T).astype(dtype)

    return samples   # (n_samples, N_voxel, N_seq)


# ── fsl_mrsi helpers ──────────────────────────────────────────────────────────

def _run_fsl_mrsi(data_file, basis_path, mask_file, ppmlim, out_file,
                   baseline, combine_groups,
                   conj_basis=True, no_conj_fid=True, no_rescale=True):
    cmd = [
        "fsl_mrsi",
        "--data",     str(data_file),
        "--basis",    str(basis_path),
        "--mask",     str(mask_file),
        "--baseline", baseline,
        "--ppmlim",   str(ppmlim[0]), str(ppmlim[1]),
        "--overwrite",
        "--output",   str(out_file),
    ]
    if conj_basis:
        cmd.append("--conj_basis")
    if no_conj_fid:
        cmd.append("--no_conj_fid")
    if no_rescale:
        cmd.append("--no_rescale")
    for group in combine_groups:
        cmd += ["--combine"] + list(group)
    subprocess.run(cmd, env=os.environ.copy(), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _load_conc_maps(fit_dir, use_internal=False):
    """Load concentrations from fsl_mrsi output.

    use_internal=False → concs/raw      (raw amplitudes; correct when --no_rescale)
    use_internal=True  → concs/internal (ratio-normalised; use when fsl_mrsi rescales
                         so the per-sample scale factor cancels in the ratio)

    Returns (dict{name: (Ny,Nx) array}, list[str]).
    Transposes from fsl_mrsi (Nx,Ny) → (Ny,Nx) for display consistency.
    """
    subdir  = "internal" if use_internal else "raw"
    raw_dir = Path(fit_dir) / "concs" / subdir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Concentration folder not found: {raw_dir}")
    conc_maps = {}
    metab_names = []
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
            raise ValueError(f"{f.name}: expected 2D after squeeze, got {data.shape}")
        conc_maps[name] = data.T   # (Nx,Ny) → (Ny,Nx)
        metab_names.append(name)
    return conc_maps, metab_names


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_std_map(name, conc_std, metab_names, wref_norm, brain_mask, out_path, vmax=None):
    if name not in metab_names:
        return
    idx = metab_names.index(name)
    std_map = conc_std[:, :, idx]
    masked  = np.where(brain_mask, std_map, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(wref_norm, origin="lower", cmap="gray", alpha=0.6, zorder=0)
    im = ax.imshow(masked, origin="lower", vmin=0, vmax=vmax, cmap="Reds",
                   alpha=0.9, zorder=1)
    ax.contour(brain_mask, levels=[0.5], colors="white", linewidths=0.7, zorder=2)
    ax.set_title(f"Conc. std: {name}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Std (arb. units)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    )
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)


def _plot_mean_map(name, conc_mean, metab_names, wref_norm, brain_mask, out_path, vmax=None):
    if name not in metab_names:
        return
    idx = metab_names.index(name)
    mean_map = conc_mean[:, :, idx]
    masked   = np.where(brain_mask, mean_map, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(wref_norm, origin="lower", cmap="gray", alpha=0.6, zorder=0)
    im = ax.imshow(masked, origin="lower", vmin=0, vmax=vmax, cmap="inferno",
                   alpha=0.9, zorder=1)
    ax.contour(brain_mask, levels=[0.5], colors="white", linewidths=0.7, zorder=2)
    ax.set_title(f"Conc. mean: {name}", color="white")
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
    p = argparse.ArgumentParser(description="Laplacian MC concentration uncertainty — step 11")
    p.add_argument("--mode",            choices=["voxelwise", "lobpcg"], default="voxelwise",
                   help="Source of posterior samples (matches step 09 --mode)")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--basis-dir",       required=True,
                   help="Training basis dir (for xcorr reference, same as step 04/05)")
    p.add_argument("--fit-basis-dir",   default=None,
                   help="Fitting basis for fsl_mrsi (defaults to --basis-dir)")
    p.add_argument("--out-dir",         default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--hess-dir",        default=None,
                   help="[voxelwise] mHm_*.npy directory (default: <out-dir>/hessian)")
    p.add_argument("--lobpcg-dir",      default=None,
                   help="[lobpcg] lobpcg_Q/vals directory (default: <out-dir>/lobpcg)")
    p.add_argument("--sigma2",          type=float, default=1.0,
                   help="[lobpcg] Posterior variance scale (default 1.0 — H already lacks 1/σ²)")
    p.add_argument("--ref-nii",         default=None,
                   help="Reference NIfTI for affine (optional)")
    # Acquisition params (must match step 04/05)
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int, default=None)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    p.add_argument("--rank",            type=int,   default=20)
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    # Sampling
    p.add_argument("--n-samples",       type=int,   default=20,
                   help="Number of MC samples (each runs fsl_mrsi, so keep small ~10-50)")
    p.add_argument("--seed",            type=int,   default=0)
    # fsl_mrsi options (match step 05)
    p.add_argument("--ppmlim",          type=float, nargs=2, default=[0.0, 7.5],
                   metavar=("LO", "HI"))
    p.add_argument("--baseline",        default="poly, 0")
    p.add_argument("--combine",         nargs="+", action="append", default=[],
                   metavar="METAB",
                   help="Metabolite combine groups: --combine NAA NAAG --combine PCh GPC")
    p.add_argument("--no-conj-basis",   action="store_true")
    p.add_argument("--no-conj-fid",     action="store_true")
    p.add_argument("--rescale",         action="store_true")
    # Output
    p.add_argument("--plot-metabs",     nargs="+",
                   default=["NAA", "NAA+NAAG", "Cr", "Cr+PCr", "Ins", "Glu", "PCh"])
    p.add_argument("--cleanup",         action="store_true",
                   help="Delete intermediate per-sample NIfTI and fit dirs after collecting")
    p.add_argument("--run-tag",         default="",
                   help="Run identifier from recon_01 (e.g. w5000_l0.0001); "
                        "appended to spice/fitting/conc_uncertainty subdir names")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_dir      = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    _tg           = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    spice_dir     = os.path.join(args.out_dir, _tg("spice"))
    out_dir       = os.path.join(args.out_dir, _tg("conc_uncertainty"))
    fit_basis_dir = args.fit_basis_dir or args.basis_dir
    os.makedirs(out_dir, exist_ok=True)

    Ny, Nx   = args.dim
    N_SEQ    = args.n_seq_points
    N_VOXEL  = Ny * Nx
    TS       = (args.k_points / N_SEQ) * args.dwelltime
    sweepwidth  = 1.0 / TS

    # ── Common loads ─────────────────────────────────────────────────────────
    print("[mc-fit] Loading common data …")
    wref_img  = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    V_full = np.load(os.path.join(spice_dir, "V_subspace.npy"))
    V      = V_full[:, :args.rank].astype(D_TYPE)               # (N_seq, rank)
    est_U  = np.load(os.path.join(spice_dir, "U_est.npy"))
    mean_U = est_U[:N_VOXEL, :args.rank].astype(D_TYPE)         # (N_vox, rank)

    # ── Reference affine ─────────────────────────────────────────────────────
    _affine_npy = data_dir + "affine.npy"
    if args.ref_nii:
        affine = Image(args.ref_nii).voxToWorldMat
    elif os.path.exists(_affine_npy):
        affine = np.load(_affine_npy)
        print(f"[mc-fit] Loaded affine from {_affine_npy}")
    else:
        ref_nii_path = data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz"
        try:
            affine = Image(ref_nii_path).voxToWorldMat
        except Exception:
            affine = np.eye(4)
            print("[mc-fit] WARNING: affine.npy not found; using identity")

    # ── Build xcorr reference (basis_nmrs) ───────────────────────────────────
    print("[mc-fit] Building xcorr reference basis …")
    fullbasis = mrs_io.read_basis(args.basis_dir)
    fid_ref, emptymrs, _ = syntheticFromBasisFile(
        fullbasis, noisecovariance=[[0]], bandwidth=sweepwidth, points=N_SEQ,
    )
    basis_nmrs = gen_nifti_mrs(
        fid_ref.conj().reshape(1, 1, 1, N_SEQ),
        dwelltime=emptymrs.dwellTime,
        spec_freq=emptymrs.centralFrequency,
        affine=affine,
    )

    # ── Save brain mask NIfTI for fsl_mrsi ───────────────────────────────────
    mask_nii = os.path.join(out_dir, "brain_mask.nii.gz")
    mask_data = np.ascontiguousarray((wref_2d * brain_mask).astype(np.float32).T[::-1, :])
    mask_nii_obj = nib.Nifti1Image(mask_data[:, :, np.newaxis], affine)
    mask_nii_obj.header.set_xyzt_units("mm")
    nib.save(mask_nii_obj, mask_nii)

    # ── Generate all posterior samples ───────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    if args.mode == "voxelwise":
        hess_dir    = args.hess_dir or os.path.join(args.out_dir, _tg("hessian"))
        sigma_noise = float(np.load(data_dir + "sigma_noise.npy"))
        cov_scale   = 2.0 * sigma_noise ** 2  # σ_complex² = 2σ_real²; sigma_noise stores σ_real
        mu_map      = (mean_U @ V.conj().T).astype(D_TYPE)      # (N_vox, N_seq)
        print(f"[mc-fit] voxelwise  sigma_noise={sigma_noise:.3e}  cov_scale={cov_scale:.3e}")
        all_samples = build_voxelwise_samples(
            hess_dir, N_VOXEL, V, mu_map, cov_scale, args.n_samples, rng
        )   # (n_samples, N_voxel, N_seq) FID

    else:  # lobpcg
        lobpcg_dir = args.lobpcg_dir or os.path.join(args.out_dir, _tg("lobpcg"))
        Q    = np.load(os.path.join(lobpcg_dir, "lobpcg_Q.npy"))
        vals = np.load(os.path.join(lobpcg_dir, "lobpcg_vals.npy"))
        print(f"[mc-fit] lobpcg  Q={Q.shape}  vals={vals.shape}  sigma2={args.sigma2}")
        all_samples = build_lobpcg_samples(
            Q, vals, mean_U, V, args.n_samples, args.sigma2, rng,
            rank=args.rank, N_voxel=N_VOXEL
        )   # (n_samples, N_voxel, N_seq) FID

    print(f"[mc-fit] Generated samples shape={all_samples.shape}")

    # Per-voxel scale factor from the xcorr-aligned ORIGINAL FID that fsl_mrsi saw.
    # fsl_mrsi rescales each voxel by s = max(|FID|).  For the original fit this
    # gives s_j (from spice_aligned.nii.gz); for sample i it gives s_{ij} from the
    # xcorr-aligned perturbed FID.  We correct each MC sample so its c_raw is in
    # the same units as the original fit's c_raw (i.e. c_fit / s_j).
    if args.rescale:
        _orig_nii      = NIFTI_MRS(os.path.join(spice_dir, "SPICE_phcorr.nii.gz"))
        _orig_fid_data = np.array(_orig_nii.image[:, :, 0, :])      # (Nx, Ny, N_seq)
        orig_scale_map = np.max(np.abs(_orig_fid_data), axis=-1)    # (Nx, Ny)
        orig_scale_map = np.where(orig_scale_map > 0, orig_scale_map, 1.0)
        print(f"[mc-fit] Loaded orig_scale from SPICE_phcorr.nii.gz, "
              f"mean={orig_scale_map.mean():.3e}  max={orig_scale_map.max():.3e}")

    combine_groups = args.combine if args.combine else [["NAA", "NAAG"]]
    # Always include combined metabolite names in plots (user may omit from --plot-metabs)
    combined_names = ["+".join(grp) for grp in combine_groups]
    plot_metabs    = list(args.plot_metabs) + [n for n in combined_names if n not in args.plot_metabs]

    # ── Monte Carlo fitting loop ──────────────────────────────────────────────
    # Use fixed temp filenames — overwritten each iteration so only one copy
    # exists on disk at a time.  Always cleaned up in `finally`.
    tmp_phcorr_nii = os.path.join(out_dir, "_mc_tmp_phcorr.nii.gz")
    tmp_fit_dir    = os.path.join(out_dir, "_mc_tmp_fit.nii.gz")

    output_concs    = []
    metab_names_ref = None
    success_idx     = []
    failed_idx      = []

    for i in tqdm(range(args.n_samples), desc="MC fitting"):
        phcorr_nii = tmp_phcorr_nii
        fit_dir    = tmp_fit_dir

        try:
            # 1) Reshape sample FID → NIfTI-MRS layout (Nx, Ny, 1, N_seq)
            spice_3d = all_samples[i].reshape(Ny, Nx, N_SEQ)
            mrsi_f_4 = spice_3d.transpose(1, 0, 2)[:, :, np.newaxis, :]

            # 2) Wrap in NIfTI-MRS
            nifti_sample = gen_nifti_mrs(mrsi_f_4, dwelltime=TS,
                                          spec_freq=297.219, affine=affine)

            # 3) xcorr frequency alignment
            aligned_nmrs, _ = my_mrsi_freq_align(nifti_sample, basis_nmrs)

            # 3b) If the original fit used rescaling, pre-divide by the FIXED per-voxel
            #     scale factor s_j = max|original_FID_j| so that all MC samples are in
            #     the same normalised domain as the original fit.  Then run with
            #     --no_rescale so fsl_mrsi does not re-scale by a sample-dependent s_ij.
            #     This avoids the positive bias of E[max|FID+noise|] > max|FID|.
            if args.rescale:
                _aln_fid = np.array(aligned_nmrs.image[:, :, 0, :])        # (Nx, Ny, N_seq)
                _aln_fid = _aln_fid / orig_scale_map[:, :, np.newaxis]     # (Nx, Ny, N_seq)
                _pre_nii = gen_nifti_mrs(_aln_fid[:, :, np.newaxis, :],
                                         dwelltime=TS, spec_freq=297.219, affine=affine)
                _pre_nii.save(phcorr_nii)
            else:
                aligned_nmrs.save(phcorr_nii)

            # 4) fsl_mrsi — always no_rescale: scaling is handled above (or not needed)
            _run_fsl_mrsi(
                data_file      = phcorr_nii,
                basis_path     = fit_basis_dir,
                mask_file      = mask_nii,
                ppmlim         = args.ppmlim,
                out_file       = fit_dir,
                baseline       = args.baseline,
                combine_groups = combine_groups,
                conj_basis     = not args.no_conj_basis,
                no_conj_fid    = not args.no_conj_fid,
                no_rescale     = True,
            )

            # 5) Load concentration maps and restore to original-FID units.
            #    Pre-scaling divided each voxel FID by s_j before fitting, so c values
            #    are c_raw = c_true/s_j.  Multiply back by s_j to match the analytical
            #    result (script 12 computes Cov in the unscaled c_true domain).
            conc_maps, metab_names = _load_conc_maps(fit_dir, use_internal=False)
            if args.rescale:
                # orig_scale_map: (Nx, Ny); conc_maps values: (Ny, Nx) after transpose
                for _mn in list(conc_maps.keys()):
                    conc_maps[_mn] = conc_maps[_mn] * orig_scale_map.T   # (Ny, Nx)
            if metab_names_ref is None:
                metab_names_ref = metab_names
                print(f"[mc-fit] Metabolites: {metab_names_ref}")

            # Stack maps into (Ny, Nx, n_metab)
            nan_map = np.full((Ny, Nx), np.nan, dtype=float)
            one_iter = np.stack(
                [conc_maps.get(m, nan_map) for m in metab_names_ref],
                axis=-1
            )   # (Ny, Nx, n_metab)

            output_concs.append(one_iter)
            success_idx.append(i)

        except Exception as e:
            failed_idx.append(i)
            tqdm.write(f"[MC {i}] failed: {repr(e)}")
            continue

        finally:
            # Always remove temp files — they're overwritten each iteration anyway
            if os.path.exists(phcorr_nii):
                os.remove(phcorr_nii)
            if os.path.isdir(fit_dir):
                shutil.rmtree(fit_dir)

    if not output_concs:
        print("[mc-fit] ERROR: all MC iterations failed.")
        return

    print(f"[mc-fit] MC done: success={len(success_idx)}/{args.n_samples}  "
          f"failed={len(failed_idx)}")

    # ── Compute and save stats ────────────────────────────────────────────────
    output_concs_arr = np.stack(output_concs, axis=0)       # (N_success, Ny, Nx, n_metab)
    conc_std         = np.nanstd(output_concs_arr, axis=0)  # (Ny, Nx, n_metab)
    conc_mean        = np.nanmean(output_concs_arr, axis=0)

    np.save(os.path.join(out_dir, "output_concs.npy"),  output_concs_arr)
    np.save(os.path.join(out_dir, "conc_std.npy"),      conc_std)
    np.save(os.path.join(out_dir, "conc_mean.npy"),     conc_mean)
    np.save(os.path.join(out_dir, "metab_names.npy"),   np.array(metab_names_ref))
    print(f"[mc-fit] Saved output_concs shape={output_concs_arr.shape}")
    print(f"[mc-fit] Saved conc_std shape={conc_std.shape}")

    # ── Save NIfTI + symlink into spice_fit/uncertainties/ for FSLeyes ───────
    spice_fit_dir  = Path(args.out_dir) / _tg("fitting") / "spice_fit"
    fit_uncert_dir = spice_fit_dir / "uncertainties"
    if fit_uncert_dir.exists():
        _ref_conc = spice_fit_dir / "concs" / "raw" / f"{metab_names_ref[0]}.nii.gz"
        _nii_aff  = nib.load(str(_ref_conc)).affine if _ref_conc.exists() else np.eye(4)

        def _save_and_link_mc(arr_ny_nx, name):
            nii_data = np.ascontiguousarray(arr_ny_nx.T[::-1, :]).astype(np.float32)
            nii_path = os.path.join(out_dir, f"mc_{name}_sd.nii.gz")
            nib.save(nib.Nifti1Image(nii_data[:, :, np.newaxis], _nii_aff), nii_path)
            dst = fit_uncert_dir / f"mc_{name}_sd.nii.gz"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(os.path.abspath(nii_path))

        for k_idx, name in enumerate(metab_names_ref):
            _save_and_link_mc(conc_std[:, :, k_idx], name)
        print(f"[mc-fit] Saved + symlinked mc_*_sd.nii.gz → {fit_uncert_dir}")

        tree_path = spice_fit_dir / "mrsi.tree"
        if tree_path.exists():
            tree_txt = tree_path.read_text()
            entry = "    mc_{metab}_sd.nii.gz                    (mc-sd)\n"
            if "mc_{metab}_sd.nii.gz" not in tree_txt:
                tree_txt = tree_txt.replace(
                    "    {metab}_sd.nii.gz                       (sd)\n",
                    "    {metab}_sd.nii.gz                       (sd)\n" + entry,
                )
                tree_path.write_text(tree_txt)
                print(f"[mc-fit] Patched mrsi.tree with mc-sd entry")

    # ── Plot ──────────────────────────────────────────────────────────────────
    for meta in plot_metabs:
        if meta not in metab_names_ref:
            print(f"[warn] '{meta}' not in results, skipping plot.")
            continue
        idx = metab_names_ref.index(meta)

        std_slice  = conc_std[:, :, idx]
        mean_slice = conc_mean[:, :, idx]
        vmax_std   = float(np.nanpercentile(std_slice[brain_mask],  90)) if brain_mask.any() else None
        vmax_mean  = float(np.nanpercentile(mean_slice[brain_mask], 90)) if brain_mask.any() else None

        _plot_std_map(
            meta, conc_std, metab_names_ref, wref_norm, brain_mask,
            os.path.join(out_dir, f"fig_11_conc_std_{meta}.png"),
            vmax=vmax_std,
        )
        _plot_mean_map(
            meta, conc_mean, metab_names_ref, wref_norm, brain_mask,
            os.path.join(out_dir, f"fig_11_conc_mean_{meta}.png"),
            vmax=vmax_mean,
        )
        print(f"[mc-fit] Saved fig_11_conc_std_{meta}.png / fig_11_conc_mean_{meta}.png")

    print("[mc-fit] Step 11 complete.")


if __name__ == "__main__":
    main()
