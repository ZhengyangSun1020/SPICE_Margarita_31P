#!/usr/bin/env python3
"""
Step 11 — Analytical concentration uncertainty (Laplace approximation).

Propagates SPICE reconstruction uncertainty through the spectral fitting model
analytically, without Monte Carlo sampling.

Method (Taylor / linearised error propagation):
    SPICE posterior: x_v ~ CN(x_hat, Sigma_x)  where Sigma_x = sigma^2 * V @ mHm_v @ V^H
    Fitting function (fsl_mrsi): theta = F(x)   linearise around x_hat:
        delta_theta = J_{theta|x} @ delta_x
    where J_{theta|x} = pinv(J_fid)  (pseudo-inverse of the model Jacobian, by IFT)

    Covariance propagation:
        Sigma_theta = J_{theta|x} @ Sigma_x @ J_{theta|x}^H
                    = sigma^2 * M @ mHm_v @ M^H      (M = pinv(J_fid) @ V)
    Concentration covariance:
        Cov(c) = Sigma_theta[:K, :K]

Reads  : <out_dir>/spice_<run_tag>/V_subspace.npy        (tag e.g. w5000_l0.0001)
         <out_dir>/spice_<run_tag>/SPICE_phcorr.nii.gz
         <out_dir>/fitting_<run_tag>/spice_fit/
         <data_dir>/wref_o.npy
         <data_dir>/sigma_noise.npy
         <out_dir>/hessian_<run_tag>/mHm_*.npy
         <basis_dir>/

Writes : <out_dir>/conc_uncertainty_analytical_<run_tag>/
             conc_std_analytical.npy           (Ny, Nx, K)  raw fit uncertainty
             conc_std_analytical_internal.npy  (Ny, Nx, K)  internal (ratio) uncertainty
             metab_names.npy
             fig_12_conc_std_<metab>.png
             fig_12_internal_std_<metab>.png

Usage:
    python scripts/uncertainty/analytical/Uncert_04_analytical_conc_uncertainty.py \
        --data-dir  data/processed/invivo_250305_01 \
        --basis-dir ./basis/ \
        --run-tag   w5000_l0.0001 \
        --rank 20 \
        --combine NAA NAAG --combine PCh GPC --combine Cr PCr \
        --brain-threshold 0.16 --brain-erosion 1 \
        --plot-metabs NAA Cr Ins Glu PCh \
        [--vmax 4e-5]
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
from scipy.ndimage import binary_erosion
from tqdm import tqdm

from fsl_mrs.utils import mrs_io
from fsl_mrs.core import MRS
from fsl_mrs.core.nifti_mrs import NIFTI_MRS
from fsl_mrs.utils.baseline import Baseline
from fsl_mrs import models as fsl_models

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)
from utils.scan_params import load_scan_params


# ── SpecToFID (inverse of fsl_mrs FIDToSpec convention) ───────────────────────

def spec_to_fid(x, axis=0):
    return np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho")


# ── Analytical uncertainty per voxel ──────────────────────────────────────────

def analytical_conc_uncertainty(Sigma, J_fid, K):
    """
    Linearised error propagation: Sigma_theta = J_{F|x} @ Sigma_x @ J_{F|x}^H
    where J_{F|x} = pinv(J_fid)  (implicit function theorem, IFT on OLS fit).

    Sigma : (T, T) complex PSD  — SPICE FID covariance (sigma^2 * V @ mHm @ V^H)
    J_fid : (T, n_params) complex — forward model Jacobian J_{x|theta} in FID domain
    K     : int — number of concentration parameters (first K columns)

    Returns
    -------
    cov_c : (K, K) real  — raw concentration covariance
    std_c : (K,)   real  — raw concentration std
    """
    J_plus      = np.linalg.pinv(J_fid)                        # (n_params, T)
    Sigma_theta = np.real(J_plus @ Sigma @ J_plus.conj().T)    # (n_params, n_params)
    cov_c       = Sigma_theta[:K, :K]                           # (K, K)
    std_c       = np.sqrt(np.abs(np.diag(cov_c)))
    return cov_c, std_c


def internal_conc_uncertainty(cov_c, c_raw, ref_idxs):
    """
    Ratio error propagation for internal concentrations: f_i = c_i / sum(c_ref).

    Uses supervisor formula:
        sigma_f^2 = f^2 * [(sigma_A/A)^2 + (sigma_B/B)^2 - 2*sigma_AB/(A*B)]
    where A = c_i, B = sum(c_ref), sigma_AB = Cov(c_i, sum c_ref).

    cov_c    : (K, K) real — raw concentration covariance from analytical_conc_uncertainty
    c_raw    : (K,)   real — raw concentration values at this voxel
    ref_idxs : list[int]  — indices of reference metabolites (e.g. Cr, PCr)

    Returns
    -------
    std_internal : (K,) real — std of internal (ratio) concentrations
    """
    K         = len(c_raw)
    ref_conc  = float(np.sum(c_raw[ref_idxs]))           # B = sum(c_ref)
    sigma_B2  = float(np.sum(cov_c[np.ix_(ref_idxs, ref_idxs)]))  # Var(B)

    std_internal = np.full(K, np.nan)
    for i in range(K):
        A = float(c_raw[i])
        if abs(A) < 1e-30 or abs(ref_conc) < 1e-30:
            continue
        sigma_A2  = float(cov_c[i, i])
        sigma_AB  = float(np.sum(cov_c[i, ref_idxs]))    # Cov(c_i, sum c_ref)
        f         = A / ref_conc
        var_f     = f**2 * (sigma_A2/A**2 + sigma_B2/ref_conc**2
                            - 2.0 * sigma_AB / (A * ref_conc))
        std_internal[i] = np.sqrt(max(0.0, var_f))
    return std_internal


# ── Plotting ────────────────────────────────────────────────────────────────────

def _plot_std_map(name, std_2d, wref_norm, brain_mask_inner, out_path,
                  vmax=None, title_prefix="Analytical σ(c)", cbar_label="Std (arb. units)"):
    masked     = np.where(brain_mask_inner, std_2d, np.nan)
    wref_masked = np.where(brain_mask_inner, wref_norm, np.nan)
    import copy
    _gray = copy.copy(plt.cm.get_cmap("gray"));  _gray.set_bad(color="black")
    _reds = copy.copy(plt.cm.get_cmap("Reds"));  _reds.set_bad(color="black")
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(wref_masked, origin="lower", cmap=_gray, alpha=0.6, zorder=0)
    im = ax.imshow(masked, origin="lower", vmin=0, vmax=vmax,
                   cmap=_reds, alpha=0.9, zorder=1)
    ax.set_title(f"{title_prefix}: {name}", color="white")
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


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analytical concentration uncertainty — step 12")
    p.add_argument("--data-dir",         required=True,
                   help="Scan data dir (contains wref_o.npy, sigma_noise.npy)")
    p.add_argument("--basis-dir",        required=True,
                   help="Fitting basis dir (same as used in step 05)")
    p.add_argument("--out-dir",          default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--hess-dir",         default=None,
                   help="Dir with mHm_*.npy (default: <out-dir>/hessian)")
    p.add_argument("--fit-dir",          default=None,
                   help="fsl_mrsi output dir (default: <out-dir>/fitting_<run-tag>/spice_fit)")
    # Acquisition (must match 04/05)
    p.add_argument("--dwelltime",        type=float, default=None)
    p.add_argument("--k-points",         type=int, default=None)
    p.add_argument("--n-seq-points",     type=int,   default=300)
    p.add_argument("--center-freq",      type=float, default=None)
    p.add_argument("--ppm-center",       type=float, default=3.027)
    p.add_argument("--dim",              type=int,   nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    p.add_argument("--rank",             type=int,   default=20)
    p.add_argument("--brain-threshold",  type=float, default=0.08)
    p.add_argument("--brain-erosion",    type=int,   default=3)
    # Fitting options (must match step 05)
    p.add_argument("--ppmlim",           type=float, nargs=2, default=[3.5, 5.0])
    p.add_argument("--ppmlim-jac",       action="store_true",
                   help="Restrict Jacobian to ppmlim range instead of full spectrum")
    # Output
    p.add_argument("--plot-metabs",      nargs="+",
                   default=["NAA", "Cr", "Ins", "Glu", "PCh"])
    p.add_argument("--vmax",             type=float, default=None,
                   help="Fixed colorbar upper limit for all std plots "
                        "(default: 90th-percentile per map)")
    p.add_argument("--combine",          nargs="+", action="append", default=[],
                   metavar="METAB",
                   help="Metabolite combine groups for std output: "
                        "--combine NAA NAAG --combine PCh GPC")
    p.add_argument("--internal-ref",     nargs="+", default=["Cr", "PCr"],
                   metavar="METAB",
                   help="Reference metabolites for internal concentration ratio "
                        "(default: Cr PCr)")
    p.add_argument("--rescale",          action="store_true",
                   help="fsl_mrsi was run WITH rescaling (default: no_rescale). "
                        "When set, the per-voxel rescale factor s=max|FID| is used "
                        "to restore original-scale concentrations for Jacobian "
                        "evaluation, fixing ill-conditioning of J_fid.")
    p.add_argument("--run-tag",          default="",
                   help="Run identifier from recon_01 (e.g. w5000_l0.0001); "
                        "appended to spice/hessian/fitting subdir names")
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    _tg       = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    spice_dir = os.path.join(args.out_dir, _tg("spice"))
    hess_dir  = args.hess_dir or os.path.join(args.out_dir, _tg("hessian"))
    fit_dir   = args.fit_dir  or os.path.join(args.out_dir, _tg("fitting"), "spice_fit")
    out_dir   = os.path.join(args.out_dir, _tg("conc_uncertainty_analytical"))
    os.makedirs(out_dir, exist_ok=True)

    Ny, Nx  = args.dim
    N_SEQ   = args.n_seq_points
    TS      = (args.k_points / N_SEQ) * args.dwelltime

    # ── Load common data ───────────────────────────────────────────────────────
    print("[step12] Loading data …")
    wref_img  = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_2d   = np.abs(wref_img.squeeze(-1))                    # (Ny, Nx)
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # Subspace V (FID domain) — matches script 09 convention: Sigma = sigma^2 * V @ mHm @ V^H
    V_full = np.load(os.path.join(spice_dir, "V_subspace.npy"))      # (N_seq, rank_full)
    V      = V_full[:, :args.rank].astype(np.complex128)              # (N_seq, rank)

    # Noise variance (sigma_noise^2 scaling for mHm posterior covariance)
    sigma_noise = float(np.load(data_dir + "sigma_noise.npy"))
    sigma2      = 2.0 * sigma_noise ** 2  # σ_complex² = 2σ_real²; sigma_noise stores σ_real
    print(f"[step12] sigma_noise={sigma_noise:.4e}")

    # ── Set up fsl_mrs voigt model ─────────────────────────────────────────────
    print("[step12] Setting up fsl_mrs model …")
    basis   = mrs_io.read_basis(args.basis_dir)
    K       = basis.n_metabs
    metab_names = list(basis.names)
    print(f"[step12] Basis: {K} metabolites → {metab_names}")

    # Reference metabolites for internal concentration ratio
    ref_names = args.internal_ref
    ref_idxs  = [metab_names.index(r) for r in ref_names if r in metab_names]
    if len(ref_idxs) != len(ref_names):
        missing = [r for r in ref_names if r not in metab_names]
        print(f"[step12] WARNING: internal-ref metabolites not in basis: {missing}")
    print(f"[step12] Internal ref: {[metab_names[i] for i in ref_idxs]} (idxs {ref_idxs})")

    # Load aligned SPICE NIfTI to get dwelltime / cf for MRS object
    aligned_nii_path = os.path.join(spice_dir, "SPICE_phcorr.nii.gz")
    nii_mrs  = NIFTI_MRS(aligned_nii_path)
    bw_data  = 1.0 / nii_mrs.dwelltime
    cf_data  = float(nii_mrs.spectrometer_frequency[0])

    # Build reference MRS object → extract t, nu, m (basis in FID domain), B
    fid_ref = np.array(nii_mrs.image[Nx // 2, Ny // 2, 0, :])
    mrs_ref = MRS(FID=fid_ref, cf=cf_data, bw=bw_data, basis=basis)
    mrs_ref.conj_Basis = True
    mrs_ref.check_Basis(ppmlim=tuple(args.ppmlim))

    t  = mrs_ref.timeAxis                  # (T, 1) in s  — keep 2D for jac broadcasting
    nu = mrs_ref.frequencyAxis             # (T, 1) in Hz
    m  = mrs_ref.basis.copy()              # (T, K) complex FID basis (conj applied)

    # All metabolites in one group (matches --metab_groups 0 in step 05)
    G = np.zeros(K, dtype=int)
    g = 1

    # Baseline regressor (poly,0 → constant, nonzero only within ppmlim)
    bl         = Baseline(mrs_ref, ppmlim=tuple(args.ppmlim),
                          baseline_selection="poly, 0", baseline_order=None)
    B_baseline = bl.regressor     # (T, 2) complex

    # Jacobian frequency range
    if args.ppmlim_jac:
        jac_first, jac_last = mrs_ref.ppmlim_to_range(tuple(args.ppmlim))
    else:
        jac_first, jac_last = 0, N_SEQ

    # n_params = K(conc) + g(gamma) + g(sigma) + g(eps) + 1(phi0) + 1(phi1) + 2(baseline)
    n_nuisance = g + g + g + 1 + 1 + 2    # 7 with g=1
    n_params   = K + n_nuisance
    print(f"[step12] Model params: {K} conc + {n_nuisance} nuisance = {n_params} total")
    print(f"[step12] Jacobian range: [{jac_first}:{jac_last}] / {N_SEQ}")

    voigt_jac = fsl_models.getModelJac("voigt")

    # ── Load fitted parameters from fsl_mrsi output ───────────────────────────
    # fsl_mrsi outputs (Nx_flipped, Ny); apply [::-1,:] on load → (Nx, Ny), access as [ix, iy]
    print("[step12] Loading fitted parameters …")
    fit_path = Path(fit_dir)

    conc_raw = np.zeros((Nx, Ny, K), dtype=np.float64)
    for k_idx, name in enumerate(metab_names):
        f = fit_path / "concs" / "raw" / f"{name}.nii.gz"
        conc_raw[:, :, k_idx] = np.squeeze(nib.load(str(f)).get_fdata())[::-1, :]  # (Nx_flipped,Ny)→(Nx,Ny)

    def _load_nii_map(path):
        return np.squeeze(nib.load(str(path)).get_fdata())[::-1, :]  # (Nx_flipped,Ny)→(Nx,Ny)

    gamma_map = _load_nii_map(fit_path / "nuisance" / "gamma_group0.nii.gz")
    sigma_map = _load_nii_map(fit_path / "nuisance" / "sigma_group0.nii.gz")
    eps_map   = _load_nii_map(fit_path / "nuisance" / "shift_group0.nii.gz")
    phi0_map  = _load_nii_map(fit_path / "nuisance" / "p0.nii.gz")
    phi1_map  = _load_nii_map(fit_path / "nuisance" / "p1.nii.gz")

    # ── Per-voxel analytical uncertainty ─────────────────────────────────────
    # conc_std stored in (Ny, Nx, K) layout matching brain_mask
    conc_std          = np.full((Ny, Nx, K), np.nan)
    conc_std_internal = np.full((Ny, Nx, K), np.nan)

    # Handle combine groups: compute combined std from covariance
    combine_groups        = args.combine if args.combine else [["NAA", "NAAG"]]
    combined_stds         = {}   # {combined_name: (Ny, Nx) array}  raw
    combined_stds_internal = {}  # {combined_name: (Ny, Nx) array}  internal
    for grp in combine_groups:
        name = "+".join(grp)
        combined_stds[name]          = np.full((Ny, Nx), np.nan)
        combined_stds_internal[name] = np.full((Ny, Nx), np.nan)

    brain_voxels = np.argwhere(brain_mask)    # (n_brain, 2) with (iy, ix)
    n_success, n_missing, n_fail = 0, 0, 0

    for iy, ix in tqdm(brain_voxels, desc="Analytical uncertainty"):
        flat_idx = iy * Nx + ix
        mhm_path = os.path.join(hess_dir, f"mHm_{flat_idx}.npy")

        if not os.path.exists(mhm_path):
            n_missing += 1
            continue

        try:
            mHm_v = np.load(mhm_path).astype(np.complex128)   # (rank, rank)

            # Step 1: SPICE FID covariance — identical to script 09
            Sigma = sigma2 * (V @ mHm_v @ V.conj().T)          # (T, T)

            # Step 2: Jacobian in FID domain
            # NIfTI is (Nx, Ny) → index as [ix, iy]
            c     = conc_raw[ix, iy, :]
            gamma = np.array([gamma_map[ix, iy]])
            sigma = np.array([sigma_map[ix, iy]])
            eps   = np.array([eps_map[ix, iy]])
            phi0  = phi0_map[ix, iy]
            phi1  = phi1_map[ix, iy]
            b     = np.zeros(2)
            x_params = np.concatenate([c, gamma, sigma, eps, [phi0, phi1], b])

            J = voigt_jac(x_params, nu, t, m, B_baseline, G, g, jac_first, jac_last)
            if args.ppmlim_jac:
                J_spec = np.zeros((N_SEQ, n_params), dtype=complex)
                J_spec[jac_first:jac_last, :] = J
            else:
                J_spec = J
            J_fid = spec_to_fid(J_spec, axis=0)                # (T, n_params)

            # Step 3: linearised propagation Sigma_theta = pinv(J) @ Sigma @ pinv(J)^H
            cov_c, std_c = analytical_conc_uncertainty(Sigma, J_fid, K)

            conc_std[iy, ix, :] = std_c

            # Step 4: internal concentration uncertainty via ratio propagation
            # Reference metabolites themselves: internal ≡ c_ref_i/ref (valid ratio,
            # not degenerate unlike the full group), so ratio propagation still applies.
            # But for plotting, conc_std_internal for ref metabolites is of limited
            # scientific interest; it's included for completeness.
            if ref_idxs:
                std_int = internal_conc_uncertainty(cov_c, c, ref_idxs)
                # For any metabolite whose index set == entire ref group, fall back to raw
                # (only relevant for single-metabolite ref; multi-ref is handled in combined)
                if len(ref_idxs) == 1:
                    std_int[ref_idxs[0]] = std_c[ref_idxs[0]]
                conc_std_internal[iy, ix, :] = std_int

            # Combined metabolite uncertainties (raw and internal)
            for grp in combine_groups:
                name = "+".join(grp)
                idxs = [metab_names.index(m_) for m_ in grp if m_ in metab_names]
                if len(idxs) == len(grp):
                    # Var(sum c_k) = sum_i sum_j Cov(c_i, c_j)
                    var_sum = np.sum(cov_c[np.ix_(idxs, idxs)])
                    combined_stds[name][iy, ix] = np.sqrt(max(0.0, np.real(var_sum)))

                    # Internal: ratio propagation for sum / ref
                    # If this group IS the reference group, internal ≡ 1 (degenerate);
                    # fall back to raw std instead.
                    if ref_idxs:
                        if set(idxs) == set(ref_idxs):
                            combined_stds_internal[name][iy, ix] = combined_stds[name][iy, ix]
                        else:
                            A_sum    = float(np.sum(c[idxs]))
                            ref_conc = float(np.sum(c[ref_idxs]))
                            if abs(A_sum) > 1e-30 and abs(ref_conc) > 1e-30:
                                sigma_A2 = float(np.sum(cov_c[np.ix_(idxs, idxs)]))
                                sigma_B2 = float(np.sum(cov_c[np.ix_(ref_idxs, ref_idxs)]))
                                sigma_AB = float(np.sum(cov_c[np.ix_(idxs, ref_idxs)]))
                                f        = A_sum / ref_conc
                                var_f    = f**2 * (sigma_A2/A_sum**2 + sigma_B2/ref_conc**2
                                                   - 2.0*sigma_AB/(A_sum*ref_conc))
                                combined_stds_internal[name][iy, ix] = np.sqrt(max(0.0, var_f))

            n_success += 1

        except (np.linalg.LinAlgError, Exception) as e:
            n_fail += 1
            tqdm.write(f"[step12] vox ({iy},{ix}) flat={flat_idx} failed: {repr(e)}")

    print(f"[step12] Done: success={n_success}  "
          f"missing_mHm={n_missing}  failed={n_fail}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "conc_std_analytical.npy"),          conc_std)
    np.save(os.path.join(out_dir, "conc_std_analytical_internal.npy"), conc_std_internal)
    np.save(os.path.join(out_dir, "metab_names.npy"), np.array(metab_names))
    for name, arr in combined_stds.items():
        safe = name.replace("+", "_plus_")
        np.save(os.path.join(out_dir, f"combined_std_{safe}.npy"), arr)
        np.save(os.path.join(out_dir, f"combined_std_internal_{safe}.npy"),
                combined_stds_internal[name])
    print(f"[step12] Saved conc_std_analytical.npy          shape={conc_std.shape}")
    print(f"[step12] Saved conc_std_analytical_internal.npy shape={conc_std_internal.shape}")

    # ── Save NIfTI + symlink into spice_fit/analytical/ for FSLeyes ─────────
    fit_anal_dir = Path(fit_dir) / "analytical"
    if Path(fit_dir).exists():
        fit_anal_dir.mkdir(exist_ok=True)
        _ref_conc = Path(fit_dir) / "concs" / "raw" / f"{metab_names[0]}.nii.gz"
        _nii_aff  = nib.load(str(_ref_conc)).affine if _ref_conc.exists() else np.eye(4)

        def _save_and_link(arr_ny_nx, name):
            nii_data = np.ascontiguousarray(arr_ny_nx.T[::-1, :]).astype(np.float32)
            nii_path = os.path.join(out_dir, f"anal_{name}_sd.nii.gz")
            nib.save(nib.Nifti1Image(nii_data[:, :, np.newaxis], _nii_aff), nii_path)
            dst = fit_anal_dir / f"anal_{name}_sd.nii.gz"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(os.path.abspath(nii_path))

        for k_idx, name in enumerate(metab_names):
            _save_and_link(conc_std[:, :, k_idx], name)
        for name, arr in combined_stds.items():
            _save_and_link(arr, name)
        print(f"[step12] Saved + symlinked anal_*_sd.nii.gz → {fit_anal_dir}")

        tree_path = Path(fit_dir) / "mrsi.tree"
        if tree_path.exists():
            tree_txt = tree_path.read_text()
            anal_section = "analytical\n    anal_{metab}_sd.nii.gz              (anal-sd)\n"
            if "analytical" not in tree_txt:
                tree_txt = tree_txt.replace("qc\n", anal_section + "qc\n")
                tree_path.write_text(tree_txt)
                print("[step12] Patched mrsi.tree with analytical section")

    # ── Plot ─────────────────────────────────────────────────────────────────
    for meta in args.plot_metabs:
        # Individual metabolite — raw
        if meta in metab_names:
            idx    = metab_names.index(meta)
            std_2d = conc_std[:, :, idx]
            if args.vmax is not None:
                vmax = args.vmax
            else:
                valid = std_2d[brain_mask_inner & np.isfinite(std_2d)]
                vmax  = float(np.percentile(valid, 90)) if len(valid) else None
            _plot_std_map(
                meta, std_2d, wref_norm, brain_mask_inner,
                os.path.join(out_dir, f"fig_12_conc_std_{meta}.png"),
                vmax=vmax,
            )
            print(f"[step12] Saved fig_12_conc_std_{meta}.png")

            # Individual metabolite — internal
            std_2d_int = conc_std_internal[:, :, idx]
            if args.vmax is not None:
                vmax_int = args.vmax
            else:
                valid_int = std_2d_int[brain_mask_inner & np.isfinite(std_2d_int)]
                vmax_int  = float(np.percentile(valid_int, 90)) if len(valid_int) else None
            _plot_std_map(
                meta, std_2d_int, wref_norm, brain_mask_inner,
                os.path.join(out_dir, f"fig_12_internal_std_{meta}.png"),
                vmax=vmax_int,
                title_prefix="Internal σ(c/ref)",
                cbar_label="Std (ratio units)",
            )
            print(f"[step12] Saved fig_12_internal_std_{meta}.png")

        # Combined metabolites — raw and internal
        for name, arr in combined_stds.items():
            if meta in name:
                if args.vmax is not None:
                    vmax = args.vmax
                else:
                    valid = arr[brain_mask_inner & np.isfinite(arr)]
                    vmax  = float(np.percentile(valid, 90)) if len(valid) else None
                safe  = name.replace("+", "_plus_")
                _plot_std_map(
                    name, arr, wref_norm, brain_mask_inner,
                    os.path.join(out_dir, f"fig_12_conc_std_{safe}.png"),
                    vmax=vmax,
                )
                print(f"[step12] Saved fig_12_conc_std_{safe}.png")

                arr_int = combined_stds_internal[name]
                if args.vmax is not None:
                    vmax_int = args.vmax
                else:
                    valid_int = arr_int[brain_mask_inner & np.isfinite(arr_int)]
                    vmax_int  = float(np.percentile(valid_int, 90)) if len(valid_int) else None
                _plot_std_map(
                    name, arr_int, wref_norm, brain_mask_inner,
                    os.path.join(out_dir, f"fig_12_internal_std_{safe}.png"),
                    vmax=vmax_int,
                    title_prefix="Internal σ(c/ref)",
                    cbar_label="Std (ratio units)",
                )
                print(f"[step12] Saved fig_12_internal_std_{safe}.png")

    print("[step12] Step 12 complete.")


if __name__ == "__main__":
    main()
