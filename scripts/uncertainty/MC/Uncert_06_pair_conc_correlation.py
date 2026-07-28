#!/usr/bin/env python3
"""
Compare fitted concentration covariance/correlation for metabolite pairs.

This script uses the same linearised covariance propagation as
11_analytical_conc_uncertainty.py, then extracts pairwise concentration
covariance and correlation maps, e.g. Cr-PCr and NAA-NAAG.

Reads  : <out_dir>/spice_<run_tag>/V_subspace.npy         (tag e.g. w5000_l0.0001)
         <out_dir>/fitting_<run_tag>/spice_fit/fit/spice_aligned.nii.gz  (symlink → SPICE_phcorr.nii.gz)
         <out_dir>/fitting_<run_tag>/spice_fit/concs/raw/*.nii.gz
         <out_dir>/fitting_<run_tag>/spice_fit/nuisance/*.nii.gz
         <data_dir>/wref_o.npy
         <data_dir>/sigma_noise.npy
         <out_dir>/hessian_<run_tag>/mHm_*.npy
         <basis_dir>/
Writes : <out_dir>/pair_conc_correlation/
             corr_<A>_<B>.npy
             cov_<A>_<B>.npy
             fig_corr_map_<A>_<B>.png
             fig_corr_hist_<A>_<B>.png
             pair_correlation_summary.csv

Usage:
    python scripts/uncertainty/MC/Uncert_06_pair_conc_correlation.py \
        --data-dir data/processed/invivo_250305_01 \
        --basis-dir ./basis \
        --run-tag   w5000_l0.0001
"""

import argparse
import csv
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
from scipy.ndimage import binary_erosion
from tqdm import tqdm

from fsl_mrs.utils import mrs_io
from fsl_mrs.core import MRS
from fsl_mrs.core.nifti_mrs import NIFTI_MRS
from fsl_mrs.utils.baseline import Baseline
from fsl_mrs import models as fsl_models

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)


def load_scan_params(args, data_dir, k_key="k_mrsi"):
    """Local lightweight copy to avoid importing the whole utils package."""
    import json

    path = os.path.join(data_dir.rstrip("/"), "scan_params.json")
    params = {}
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
    else:
        print(f"[scan_params] WARNING: {path} not found -- using CLI defaults only.")

    if hasattr(args, "dwelltime") and args.dwelltime is None:
        args.dwelltime = params.get("dwelltime", 5e-6)
    if hasattr(args, "k_points") and args.k_points is None:
        args.k_points = params.get(k_key, 39762)
    if hasattr(args, "center_freq") and args.center_freq is None:
        args.center_freq = params.get("center_freq", 297.219338)
    return args


def spec_to_fid(x, axis=0):
    return np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho")


def concentration_covariance(Sigma, J_fid, n_metabs):
    """Return concentration covariance from FID covariance and model Jacobian."""
    J_plus = np.linalg.pinv(J_fid)
    Sigma_theta = np.real(J_plus @ Sigma @ J_plus.conj().T)
    return Sigma_theta[:n_metabs, :n_metabs]


def parse_pair(text):
    if ":" in text:
        a, b = text.split(":", 1)
    elif "," in text:
        a, b = text.split(",", 1)
    else:
        raise argparse.ArgumentTypeError(
            "Pairs must be formatted as A:B or A,B, e.g. Cr:PCr")
    return a.strip(), b.strip()


def first_existing(paths, what):
    for p in paths:
        if p and os.path.exists(p):
            return p
    tried = "\n  ".join(str(p) for p in paths if p)
    raise FileNotFoundError(f"Could not find {what}. Tried:\n  {tried}")


def load_nii_2d(path):
    return np.squeeze(nib.load(str(path)).get_fdata())


def plot_corr_map(name, corr_map, wref_norm, brain_mask, out_path):
    masked = np.where(brain_mask, corr_map, np.nan)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(wref_norm, origin="lower", cmap="gray", alpha=0.55, zorder=0)
    im = ax.imshow(masked, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                   alpha=0.92, zorder=1)
    ax.contour(brain_mask, levels=[0.5], colors="white", linewidths=0.7, zorder=2)
    ax.set_title(f"Correlation: {name}", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("rho", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)


def plot_corr_hist(name, values, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=60, range=(-1, 1), color="#4C78A8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.axvline(np.nanmedian(values), color="#E45756", linewidth=1.5,
               label=f"median={np.nanmedian(values):.3f}")
    ax.set_title(f"Pairwise concentration correlation: {name}")
    ax.set_xlabel("rho")
    ax.set_ylabel("Voxel count")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare concentration covariance/correlation for metabolite pairs.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--basis-dir", required=True)
    p.add_argument("--out-dir", default=None,
                   help="Output directory root (default: ./output/<subject_id>)")
    p.add_argument("--hess-dir", default=None,
                   help="Directory containing mHm_*.npy. Default: <out-dir>/hessian.")
    p.add_argument("--fit-dir", default=None,
                   help="fsl_mrsi output directory. Auto-tries fitting/spice_fit variants.")
    p.add_argument("--pairs", nargs="+", type=parse_pair,
                   default=[("Cr", "PCr"), ("NAA", "NAAG")],
                   help="Pairs as A:B or A,B. Default: Cr:PCr NAA:NAAG")
    p.add_argument("--dwelltime", type=float, default=None)
    p.add_argument("--k-points", type=int, default=None)
    p.add_argument("--n-seq-points", type=int, default=300)
    p.add_argument("--center-freq", type=float, default=None)
    p.add_argument("--ppm-center", type=float, default=3.027)
    p.add_argument("--dim", type=int, nargs=2, default=[64, 64], metavar=("NY", "NX"))
    p.add_argument("--rank", type=int, default=20)
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion", type=int, default=3)
    p.add_argument("--mask", choices=["brain", "inner"], default="brain",
                   help="Voxel mask for reporting. Default matches step 11 loop.")
    p.add_argument("--ppmlim", type=float, nargs=2, default=[3.5, 5.0])
    p.add_argument("--ppmlim-jac", action="store_true",
                   help="Restrict Jacobian to ppmlim range, matching step 11 option.")
    p.add_argument("--run-tag",   default="",
                   help="Run identifier from recon_01 (e.g. w5000_l0.0001); "
                        "appended to spice/hessian/fitting subdir names")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")

    Ny, Nx = args.dim
    N_SEQ = args.n_seq_points

    _tg = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    hess_dir = args.hess_dir or first_existing(
        [
            os.path.join(args.out_dir, _tg("hessian")),
            os.path.join(args.out_dir, "Hess_1e4"),
        ],
        "Hessian directory",
    )
    fit_dir = args.fit_dir or first_existing(
        [
            os.path.join(args.out_dir, _tg("fitting"), "spice_fit"),
            os.path.join(args.out_dir, _tg("fitting"), "spice_fit"),
        ],
        "fsl_mrsi fit directory",
    )
    spice_dir = os.path.join(args.out_dir, _tg("spice"))
    aligned_nii_path = first_existing(
        [
            os.path.join(fit_dir, "fit", "spice_aligned.nii.gz"),
            os.path.join(spice_dir, "SPICE_phcorr.nii.gz"),
        ],
        "aligned SPICE NIfTI-MRS",
    )
    out_dir = os.path.join(args.out_dir, "pair_conc_correlation")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[pair-corr] data_dir={data_dir}")
    print(f"[pair-corr] out_dir={args.out_dir}")
    print(f"[pair-corr] hess_dir={hess_dir}")
    print(f"[pair-corr] fit_dir={fit_dir}")
    print(f"[pair-corr] aligned={aligned_nii_path}")

    wref_img = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_2d = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)
    report_mask = brain_mask_inner if args.mask == "inner" else brain_mask

    V_full = np.load(os.path.join(spice_dir, "V_subspace.npy"))
    V = V_full[:, :args.rank].astype(np.complex128)
    sigma_noise = float(np.load(data_dir + "sigma_noise.npy"))
    sigma2 = sigma_noise ** 2
    print(f"[pair-corr] sigma_noise={sigma_noise:.4e}")

    basis = mrs_io.read_basis(args.basis_dir)
    K = basis.n_metabs
    metab_names = list(basis.names)
    print(f"[pair-corr] Basis metabolites: {metab_names}")

    for a, b in args.pairs:
        missing = [m for m in (a, b) if m not in metab_names]
        if missing:
            raise ValueError(f"Pair {a}:{b} includes metabolites not in basis: {missing}")

    # Some older outputs in this project are valid complex NIfTI files but do
    # not satisfy the current nifti_mrs version check.  We only need a
    # representative FID plus acquisition metadata, so nibabel + scan_params is
    # enough and matches the SPICE scripts' TS convention.
    aligned_img = nib.load(aligned_nii_path)
    aligned_data = np.asanyarray(aligned_img.dataobj)
    bw_data = 1.0 / ((args.k_points / N_SEQ) * args.dwelltime)
    cf_data = float(args.center_freq)
    fid_ref = np.array(aligned_data[Nx // 2, Ny // 2, 0, :])
    mrs_ref = MRS(FID=fid_ref, cf=cf_data, bw=bw_data, basis=basis)
    mrs_ref.conj_Basis = True
    mrs_ref.check_Basis(ppmlim=tuple(args.ppmlim))

    t = mrs_ref.timeAxis
    nu = mrs_ref.frequencyAxis
    m = mrs_ref.basis.copy()
    G = np.zeros(K, dtype=int)
    g = 1

    baseline = Baseline(mrs_ref, ppmlim=tuple(args.ppmlim),
                        baseline_selection="poly, 0", baseline_order=None)
    B_baseline = baseline.regressor
    if args.ppmlim_jac:
        jac_first, jac_last = mrs_ref.ppmlim_to_range(tuple(args.ppmlim))
    else:
        jac_first, jac_last = 0, N_SEQ
    voigt_jac = fsl_models.getModelJac("voigt")

    n_nuisance = g + g + g + 1 + 1 + 2
    n_params = K + n_nuisance
    print(f"[pair-corr] Model params: {K} conc + {n_nuisance} nuisance = {n_params}")
    print(f"[pair-corr] Jacobian range: [{jac_first}:{jac_last}] / {N_SEQ}")

    fit_path = Path(fit_dir)
    conc_raw = np.zeros((Nx, Ny, K), dtype=np.float64)
    for k_idx, name in enumerate(metab_names):
        conc_raw[:, :, k_idx] = load_nii_2d(fit_path / "concs" / "raw" / f"{name}.nii.gz")

    gamma_map = load_nii_2d(fit_path / "nuisance" / "gamma_group0.nii.gz")
    sigma_map = load_nii_2d(fit_path / "nuisance" / "sigma_group0.nii.gz")
    eps_map = load_nii_2d(fit_path / "nuisance" / "shift_group0.nii.gz")
    phi0_map = load_nii_2d(fit_path / "nuisance" / "p0.nii.gz")
    phi1_map = load_nii_2d(fit_path / "nuisance" / "p1.nii.gz")

    pair_maps = {}
    for a, b in args.pairs:
        key = f"{a}_{b}"
        pair_maps[key] = {
            "pair": (a, b),
            "corr": np.full((Ny, Nx), np.nan),
            "cov": np.full((Ny, Nx), np.nan),
            "var_a": np.full((Ny, Nx), np.nan),
            "var_b": np.full((Ny, Nx), np.nan),
        }

    voxels = np.argwhere(report_mask)
    n_success, n_missing, n_fail = 0, 0, 0

    for iy, ix in tqdm(voxels, desc="Pair correlations"):
        flat_idx = iy * Nx + ix
        mhm_path = os.path.join(hess_dir, f"mHm_{flat_idx}.npy")
        if not os.path.exists(mhm_path):
            n_missing += 1
            continue

        try:
            mHm_v = np.load(mhm_path).astype(np.complex128)
            Sigma = sigma2 * (V @ mHm_v @ V.conj().T)

            c = conc_raw[ix, iy, :]
            gamma = np.array([gamma_map[ix, iy]])
            sigma = np.array([sigma_map[ix, iy]])
            eps = np.array([eps_map[ix, iy]])
            phi0 = phi0_map[ix, iy]
            phi1 = phi1_map[ix, iy]
            bline = np.zeros(2)
            x_params = np.concatenate([c, gamma, sigma, eps, [phi0, phi1], bline])

            J = voigt_jac(x_params, nu, t, m, B_baseline, G, g, jac_first, jac_last)
            if args.ppmlim_jac:
                J_spec = np.zeros((N_SEQ, n_params), dtype=complex)
                J_spec[jac_first:jac_last, :] = J
            else:
                J_spec = J
            J_fid = spec_to_fid(J_spec, axis=0)
            cov_c = concentration_covariance(Sigma, J_fid, K)

            for key, entry in pair_maps.items():
                a, b = entry["pair"]
                ia, ib = metab_names.index(a), metab_names.index(b)
                cov_ab = float(np.real(cov_c[ia, ib]))
                var_a = float(np.real(cov_c[ia, ia]))
                var_b = float(np.real(cov_c[ib, ib]))
                if var_a > 0 and var_b > 0:
                    rho = cov_ab / np.sqrt(var_a * var_b)
                    rho = float(np.clip(rho, -1.0, 1.0))
                else:
                    rho = np.nan
                entry["corr"][iy, ix] = rho
                entry["cov"][iy, ix] = cov_ab
                entry["var_a"][iy, ix] = var_a
                entry["var_b"][iy, ix] = var_b

            n_success += 1
        except Exception as e:
            n_fail += 1
            tqdm.write(f"[pair-corr] voxel ({iy},{ix}) flat={flat_idx} failed: {repr(e)}")

    print(f"[pair-corr] Done: success={n_success} missing_mHm={n_missing} failed={n_fail}")

    summary_rows = []
    for key, entry in pair_maps.items():
        a, b = entry["pair"]
        name = f"{a}-{b}"
        corr = entry["corr"]
        cov = entry["cov"]
        valid = corr[report_mask & np.isfinite(corr)]
        cov_valid = cov[report_mask & np.isfinite(cov)]

        np.save(os.path.join(out_dir, f"corr_{key}.npy"), corr)
        np.save(os.path.join(out_dir, f"cov_{key}.npy"), cov)
        np.save(os.path.join(out_dir, f"var_{a}.npy"), entry["var_a"])
        np.save(os.path.join(out_dir, f"var_{b}.npy"), entry["var_b"])

        if valid.size:
            neg_frac = float(np.mean(valid < 0))
            row = {
                "pair": name,
                "n_voxels": int(valid.size),
                "negative_fraction": neg_frac,
                "mean_rho": float(np.mean(valid)),
                "median_rho": float(np.median(valid)),
                "p05_rho": float(np.percentile(valid, 5)),
                "p95_rho": float(np.percentile(valid, 95)),
                "min_rho": float(np.min(valid)),
                "max_rho": float(np.max(valid)),
                "mean_cov": float(np.mean(cov_valid)) if cov_valid.size else np.nan,
                "median_cov": float(np.median(cov_valid)) if cov_valid.size else np.nan,
            }
        else:
            row = {
                "pair": name,
                "n_voxels": 0,
                "negative_fraction": np.nan,
                "mean_rho": np.nan,
                "median_rho": np.nan,
                "p05_rho": np.nan,
                "p95_rho": np.nan,
                "min_rho": np.nan,
                "max_rho": np.nan,
                "mean_cov": np.nan,
                "median_cov": np.nan,
            }
        summary_rows.append(row)

        if valid.size:
            plot_corr_map(name, corr, wref_norm, report_mask,
                          os.path.join(out_dir, f"fig_corr_map_{key}.png"))
            plot_corr_hist(name, valid,
                           os.path.join(out_dir, f"fig_corr_hist_{key}.png"))
            print(
                f"[pair-corr] {name}: n={row['n_voxels']} "
                f"negative={row['negative_fraction']:.1%} "
                f"median rho={row['median_rho']:.3f} "
                f"p05/p95=({row['p05_rho']:.3f}, {row['p95_rho']:.3f})"
            )
        else:
            print(f"[pair-corr] {name}: no valid voxels")

    summary_path = os.path.join(out_dir, "pair_correlation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[pair-corr] Saved {summary_path}")


if __name__ == "__main__":
    main()
