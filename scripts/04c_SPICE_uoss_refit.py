#!/usr/bin/env python3
"""
Step 4c — UoSS-style lipid removal + metabolite (or joint) SPICE refit.

Standalone alternative to the 03(L2) -> 04 -> 04b chain: 04c does NOT use
step 03's L2 lipid-removed data (kt_mrsi_lprm.npy) or step 04's U_est.npy
(which is fit against that L2-removed data) anywhere. The only step-04
artefact it reuses is V_subspace.npy, the metabolite spectral subspace —
that one is learned purely from simulated training spectra
(Sig_func_Multi_Peak_2D in 04_run_spice.py), never touches measured data,
so reusing it carries none of the L2 lineage. U_metab is cold-started and
solved fresh in stage 3 below, against the UoSS-cleaned data this script
produces itself.

Three-stage pipeline, closer to the Lam et al. 2020 SPICE lipid-handling
architecture than 04b:

  1. Lipid-only SPICE fit on the with-lipid (no-ring) k-t data from step 03
     (kt_mrsi_withlip_noring.npy): a single-block CG solve for U_lipid using
     only V_lipid, spatially regularized by the in-brain LSS GMM mask
     (W_lip, same construction as 04b) — the mask is kept here (rather than
     dropping it entirely, as Lam 2020 does) as a safeguard: an unregularized
     lipid-only fit can converge poorly or overfit metabolite signal in
     brain-mask voxels as if it were lipid, since nothing in a single-block
     fit forces U_lipid to leave clean voxels alone.
  2. Subtract the lipid-only model's predicted k-t contribution from the
     original with-lipid k-t data (true UoSS-style removal in the data
     domain — F{U_lipid_fit @ V_lipid.H} subtracted, not a uniform Tikhonov
     shrinkage filter applied to every voxel regardless of actual lipid
     content). Result saved as kt_mrsi_uoss_lprm.npy.
  3. Final metabolite estimation on this UoSS-cleaned data: by default a
     metab-only refit (no lipid block at all — closest to Lam 2020, since
     lipid has already been fully removed and is never revisited); pass
     --final-mode joint to instead do one more joint [U_lipid, U_metab]
     mop-up refit (warm-started from step 1's U_lipid_fit) in case step 1's
     single-pass removal leaves visible residual.

Reads  : <out_dir>/spice/V_subspace.npy                       (step 04, simulation-derived only — NOT U_est.npy)
         <out_dir>/lipid_removal/V_lipid.npy                  (step 03, SVD-truncated)
         <out_dir>/lipid_removal/lss_map.npy                  (step 03, raw LSS map)
         <out_dir>/lipid_removal/kt_mrsi_withlip_noring.npy   (step 03)
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <data_dir>/wref_o.npy
Writes : <out_dir>/spice_refit_uoss/U_lipid_step1.npy          (stage-1 lipid-only fit)
         <out_dir>/spice_refit_uoss/kt_mrsi_uoss_lprm.npy      (stage-2 UoSS-cleaned k-t data)
         <out_dir>/spice_refit_uoss/SPICE_refit_f.npy          (stage-3 result, metab-only or joint)
         <out_dir>/spice_refit_uoss/U_metab_final.npy          (+ U_lipid_final.npy if --final-mode joint)
         <out_dir>/spice_refit_uoss/w_lip_vec.npy
         <out_dir>/spice_refit_uoss/SPICE_refit_result.nii.gz
         <out_dir>/spice_refit_uoss/metab_clean_result.nii.gz
         <out_dir>/spice_refit_uoss/fig_04c_*.png

Usage:
    python scripts/04c_SPICE_uoss_refit.py \
        --data-dir ./data/ --out-dir ./output \
        --final-mode metab_only \
        --save-plots
    (defaults below match 04b's tuned command-line: --lamda-lip 1e-6
    --lambda1 1e-4 --maxiter-lipid 120 --maxiter 120 --nsigma-gmm-inbrain 2.0
    --n-lipid-voxels-inbrain 2000 --topn-fallback 100 --min-lip-penalty 0.001
    --max-lip-penalty 1e2 --patience 10)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.sparse import diags as sp_diags
from warnings import filterwarnings
filterwarnings("ignore")

from fsl_mrs.utils.misc import FIDToSpec
from fsl.data.image import Image
from nifti_mrs.create_nmrs import gen_nifti_mrs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    calc_Bmatrix,
    SPICEWithSpatialConstrain_cg_nufft_joint,
    NUFFTOp,
    plot_voxel_spectrum_and_maps,
    Calc_B0_matrix,
    phase_corr,
    build_nufft_ops,
)
from utils.lipid import select_lipid_mask_gmm_simple


def parse_args():
    p = argparse.ArgumentParser(description="UoSS lipid removal + metab/joint SPICE refit — step 4c")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--out-dir",         default="./output")
    p.add_argument("--backend",         default="torchnufft",
                   choices=["torchnufft", "finufft"],
                   help="NUFFT backend: torchnufft (default) or finufft")
    p.add_argument("--dwelltime",       type=float, default=5e-6)
    p.add_argument("--k-points",        type=int,   default=39762)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--n-coils",         type=int,   default=32)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64], metavar=("NX", "NY"))
    p.add_argument("--center-freq",     type=float, default=297.219338)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--n-shots",         type=int,   default=360)
    p.add_argument("--phase-ppmlim",    type=float, nargs=2, default=[0.0, 5.0], metavar=("LO", "HI"))
    # SPICE / regularization
    p.add_argument("--lambda1",         type=float, default=1e-4,
                   help="Spatial reg weight for U_metab block (match step 04)")
    p.add_argument("--lamda-lip",       type=float, default=1e-6,
                   help="Spatial reg weight for U_lipid block in stage 1 (and stage 3 if "
                        "--final-mode joint), scales WW_lip = W_lip.H @ W_lip")
    # in-brain lipid-contamination GMM (drives W_lip; separate from step 03's
    # ring-extraction GMM, which uses --nsigma-gmm/--n-lipid-voxels there)
    p.add_argument("--nsigma-gmm-inbrain", type=float, default=2.0,
                   help="nsigma for the in-brain LSS GMM classification")
    p.add_argument("--n-lipid-voxels-inbrain", type=int, default=2000,
                   help="Max voxel cap for the in-brain GMM classification before --topn-fallback kicks in")
    p.add_argument("--topn-fallback",   type=int,   default=100,
                   help="Top-N voxel fallback when the in-brain GMM threshold selects "
                        "more than --n-lipid-voxels-inbrain")
    p.add_argument("--min-lip-penalty", type=float, default=0.001,
                   help="W_lip diagonal weight for lipid-contaminated voxels: a small floor "
                        "penalty, not zero — U_lipid is still mildly regularized there")
    p.add_argument("--max-lip-penalty", type=float, default=1e2,
                   help="W_lip diagonal weight for clean in-brain voxels: shrinks U_lipid toward zero")
    p.add_argument("--final-mode",      choices=["metab_only", "joint"], default="metab_only",
                   help="Stage-3 refit: 'metab_only' drops the lipid block entirely (closest "
                        "to Lam 2020 — lipid removed once in stages 1-2, never revisited); "
                        "'joint' does one more [U_lipid, U_metab] mop-up refit warm-started "
                        "from stage 1's U_lipid_fit, for residual cleanup")
    p.add_argument("--wmax",            type=float, default=5e3)
    p.add_argument("--adj",             type=int,   default=8)
    p.add_argument("--pool-size",       type=int,   default=1)
    p.add_argument("--minpool",         action="store_true")
    p.add_argument("--maxiter-lipid",   type=int,   default=120,
                   help="CG maxiter for stage 1 (lipid-only fit)")
    p.add_argument("--maxiter",         type=int,   default=120,
                   help="CG maxiter for stage 3 (final metab-only or joint refit)")
    p.add_argument("--patience",        type=int,   default=10)
    p.add_argument("--patience-dx",     type=int,   default=3)
    p.add_argument("--rtol-lipid",      type=float, default=1e-5,
                   help="CG rtol for stage 1 lipid fit (default 1e-5; solver default was 1e-3)")
    p.add_argument("--rtol-metab",      type=float, default=1e-5,
                   help="CG rtol for stage 3 metab/joint refit (default 1e-5)")
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--ref-nii",         default=None)
    p.add_argument("--save-plots",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    out_dir  = os.path.join(args.out_dir, "spice_refit_uoss")
    os.makedirs(out_dir, exist_ok=True)

    spice_dir   = os.path.join(args.out_dir, "spice")
    coilmap_dir = os.path.join(args.out_dir, "coilmap")
    b0map_dir   = os.path.join(args.out_dir, "b0map")
    lprm_dir    = os.path.join(args.out_dir, "lipid_removal")

    D_TYPE      = np.complex64
    Trej_D_TYPE = np.float32

    K_POINTS  = args.k_points
    N_SEQ     = args.n_seq_points
    N_COILS   = args.n_coils
    Dim_Voxel = args.dim
    N_VOXEL   = Dim_Voxel[0] * Dim_Voxel[1]
    Ny, Nx, T = Dim_Voxel[0], Dim_Voxel[1], N_SEQ
    im_size   = (Ny, Nx, T)

    TS          = (K_POINTS / N_SEQ) * args.dwelltime
    sweepwidth  = 1.0 / TS
    center_freq = args.center_freq
    PPM_CENTER  = args.ppm_center
    FREQ_AXIS   = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS    = FREQ_AXIS / center_freq + PPM_CENTER
    TIME_AXIS   = np.linspace(TS, TS * N_SEQ, N_SEQ)
    print(f"[spice_uoss] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s  final-mode={args.final_mode}")

    # ── V_metab subspace from step 04 (simulation-derived basis only — U_est.npy
    # is NOT loaded here, since it was fit against step 03's L2-removed data and
    # would quietly reintroduce the L2 lineage this script is meant to avoid) ──
    V_metab = np.load(os.path.join(spice_dir, "V_subspace.npy")).astype(D_TYPE)   # (T, R_met)
    R_met   = V_metab.shape[1]
    print(f"[spice_uoss] Loaded V_metab {V_metab.shape} (U_metab will be cold-started in stage 3)")

    # ── V_lipid (SVD-truncated in step 03) ────────────────────────────────────
    V_lipid = np.load(os.path.join(lprm_dir, "V_lipid.npy")).astype(D_TYPE)       # (T, R_lip)
    R_lip   = V_lipid.shape[1]
    print(f"[spice_uoss] Loaded V_lipid {V_lipid.shape}")

    # ── Load data for forward model ───────────────────────────────────────────
    print("[spice_uoss] Loading data …")
    mrsi_withlip    = np.load(os.path.join(lprm_dir, "kt_mrsi_withlip_noring.npy"), mmap_mode="r").astype(D_TYPE)
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir, "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"), mmap_mode="r")
    B0_map          = np.load(os.path.join(b0map_dir, "B0_map.npy"))
    wref_img        = np.load(data_dir + "wref_o.npy", mmap_mode="r")

    trej     = mrsi_ksp_scaled.T.astype(Trej_D_TYPE)
    NUM_CMAP = coil_smap_raw.shape[0]

    coil_smap = np.repeat(
        coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1
    ).astype(D_TYPE)
    smap_time = coil_smap.squeeze(0)

    # ── Brain mask ───────────────────────────────────────────────────────────
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # ── Build NUFFT operators ─────────────────────────────────────────────────
    F_OP, Gram_OP, F1D, device_str = build_nufft_ops(
        args.backend, trej, im_size, coil_smap_raw, NUM_CMAP, D_TYPE,
        osamp=2.0, ost=2.0,
    )

    # ── B0 modulation matrix ─────────────────────────────────────────────────
    print("[spice_uoss] Building B0 modulation matrix …")
    B0_map_clean = np.nan_to_num(B0_map, nan=0.0)
    B0_mat = Calc_B0_matrix(B0_map_clean, TIME_AXIS).reshape(N_VOXEL, N_SEQ)

    # ── Spatial regularization: anatomical WW for metab, w_lip diag for lipid ──
    print("[spice_uoss] Building spatial regularization (B matrix) …")
    W_edge, _, _W, Nb = calc_Bmatrix(
        wref_norm, wmax=args.wmax, adj=args.adj,
        pool_size=args.pool_size,
        minpooling_Handler=args.minpool,
        brain_mask=brain_mask,
        mask_dilate_layers=3,
    )
    WW = W_edge.conj().T @ W_edge

    # ── In-brain LSS GMM → W_lip (same construction as 04b) ───────────────────
    print("[spice_uoss] In-brain LSS GMM classification …")
    lss_map      = np.load(os.path.join(lprm_dir, "lss_map.npy"))
    lss_2d_brain = np.squeeze(lss_map) * brain_mask
    res_inbrain  = select_lipid_mask_gmm_simple(
        lss_2d_brain, out_dir=out_dir, nsigma=args.nsigma_gmm_inbrain,
        max_voxels=args.n_lipid_voxels_inbrain, topN_fallback=args.topn_fallback,
        save_plots=args.save_plots, out_fname="fig_04c_lss_in_brain.png",
        title_prefix="In-brain LSS",
    )
    print(f"[spice_uoss] In-brain GMM  method={res_inbrain['method']}  "
          f"n_contaminated={res_inbrain['n_selected']}")

    lipid_contam_mask = res_inbrain["lipid_mask"]
    w_lip_vec = np.full(brain_mask.shape, args.max_lip_penalty, dtype=np.float32)
    w_lip_vec[brain_mask & lipid_contam_mask] = args.min_lip_penalty
    w_lip_vec = w_lip_vec.ravel()
    np.save(os.path.join(out_dir, "w_lip_vec.npy"), w_lip_vec)
    print(f"[spice_uoss] Saved w_lip_vec.npy  n_outside_brain={(~brain_mask).sum()}  "
          f"n_inbrain_contaminated={(brain_mask & lipid_contam_mask).sum()}  "
          f"n_inbrain_clean={(brain_mask & ~lipid_contam_mask).sum()}")

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(w_lip_vec.reshape(brain_mask.shape), origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, label="W_lip weight")
        ax.set_title("W_lip (lipid spatial-prior weight map)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_04c_wlip_map.png"), dpi=120)
        plt.close(fig)

    W_lip   = sp_diags(w_lip_vec)
    WW_lip  = W_lip.conj().T @ W_lip

    ref_img_path = args.ref_nii or (data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz")
    try:
        ref_img_obj = Image(ref_img_path)
        affine      = ref_img_obj.voxToWorldMat
    except Exception:
        ref_img_obj = None
        affine      = np.eye(4)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 1 — lipid-only SPICE fit on with-lipid k-t data
    # ═══════════════════════════════════════════════════════════════════════
    print(f"[spice_uoss] Stage 1: lipid-only fit  R_lip={R_lip}  lamda_lip={args.lamda_lip}  "
          f"maxiter={args.maxiter_lipid} …")
    B0_mat_ones  = np.ones((N_VOXEL, N_SEQ), dtype=D_TYPE)   # no B0 for lipid stage
    U_lipid_init = np.zeros((N_VOXEL, R_lip), dtype=D_TYPE)
    _, U_lipid_fit, info1 = SPICEWithSpatialConstrain_cg_nufft_joint(
        noisy_kt_spaces = mrsi_withlip,
        img_shape       = im_size,
        F=F_OP, Gram_OP=Gram_OP, F1D_OP=F1D,
        B0_mat=B0_mat_ones, V=V_lipid,
        N_Vox=N_VOXEL, rank_blocks=[R_lip],
        WW_blocks=[WW_lip], lamda_blocks=[args.lamda_lip],
        x0=U_lipid_init,
        maxiter=args.maxiter_lipid,
        rtol=args.rtol_lipid,
        save_folder=os.path.join(out_dir, "cg_iters_stage1_lipid"),
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS,
        patience=args.patience,
        patience_dx=args.patience_dx,
    )
    np.save(os.path.join(out_dir, "U_lipid_step1.npy"), U_lipid_fit)
    print(f"[spice_uoss] Stage 1 done. U_lipid_fit {U_lipid_fit.shape}")

    lipid_fit_fid = (U_lipid_fit @ V_lipid.conj().T).reshape(Ny, Nx, N_SEQ)
    _lf_raw_save  = lipid_fit_fid.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(_lf_raw_save, dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "lipid_fit_stage1_raw.nii.gz"))
    print("[spice_uoss] Saved lipid_fit_stage1_raw.nii.gz")

    if args.save_plots:
        lipid_step1_est = (U_lipid_fit @ V_lipid.conj().T).reshape(Ny, Nx, N_SEQ)
        plot_voxel_spectrum_and_maps(
            FIDToSpec(lipid_step1_est, axis=-1), im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04c_stage1_lipid_fit.png"), dpi=120)
        plt.close("all")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 2 — UoSS subtraction: data - F{U_lipid_fit @ V_lipid.H}
    # ═══════════════════════════════════════════════════════════════════════
    print("[spice_uoss] Stage 2: subtracting lipid model from with-lipid k-t data …")
    AA_lipid      = (U_lipid_fit @ V_lipid.conj().T).astype(D_TYPE)
    lipid_kt_pred = F_OP.matvec(AA_lipid.ravel()).reshape(mrsi_withlip.shape).astype(D_TYPE)
    kt_mrsi_uoss  = (mrsi_withlip - lipid_kt_pred).astype(D_TYPE)
    np.save(os.path.join(out_dir, "kt_mrsi_uoss_lprm.npy"), kt_mrsi_uoss)
    print(f"[spice_uoss] Saved kt_mrsi_uoss_lprm.npy  shape={kt_mrsi_uoss.shape}")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 3 — final refit on UoSS-cleaned data: metab-only (default) or joint
    # ═══════════════════════════════════════════════════════════════════════
    U_metab_init = np.zeros((N_VOXEL, R_met), dtype=D_TYPE)   # cold-started — see module docstring

    if args.final_mode == "metab_only":
        print(f"[spice_uoss] Stage 3: metab-only refit (cold-start)  R_met={R_met}  "
              f"lambda1={args.lambda1}  maxiter={args.maxiter} …")
        V_final          = V_metab
        rank_blocks      = [R_met]
        WW_blocks         = [WW]
        lamda_blocks      = [args.lambda1]
        x0_final          = U_metab_init
    else:
        print(f"[spice_uoss] Stage 3: joint mop-up refit (lipid warm-start from stage 1, "
              f"metab cold-start)  R_lip={R_lip}  R_met={R_met}  "
              f"lamda_lip={args.lamda_lip}  lambda1={args.lambda1}  maxiter={args.maxiter} …")
        V_final          = np.hstack([V_lipid, V_metab]).astype(D_TYPE)
        rank_blocks      = [R_lip, R_met]
        WW_blocks         = [WW_lip, WW]
        lamda_blocks      = [args.lamda_lip, args.lambda1]
        x0_final          = np.hstack([U_lipid_fit, U_metab_init]).astype(D_TYPE)

    spice_refit_est, U_final, info3 = SPICEWithSpatialConstrain_cg_nufft_joint(
        noisy_kt_spaces = kt_mrsi_uoss,
        img_shape       = im_size,
        F=F_OP, Gram_OP=Gram_OP, F1D_OP=F1D,
        B0_mat=B0_mat, V=V_final,
        N_Vox=N_VOXEL, rank_blocks=rank_blocks,
        WW_blocks=WW_blocks, lamda_blocks=lamda_blocks,
        x0=x0_final,
        maxiter=args.maxiter,
        rtol=args.rtol_metab,
        save_folder=os.path.join(out_dir, "cg_iters_stage3"),
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS,
        patience=args.patience,
        patience_dx=args.patience_dx,
    )
    print(f"[spice_uoss] Stage 3 done. est shape: {spice_refit_est.shape}")

    if args.final_mode == "joint":
        U_lipid_final = U_final[:, :R_lip]
        U_metab_final = U_final[:, R_lip:]
        np.save(os.path.join(out_dir, "U_lipid_final.npy"), U_lipid_final)
    else:
        U_metab_final = U_final

    np.save(os.path.join(out_dir, "SPICE_refit_f.npy"), spice_refit_est)
    np.save(os.path.join(out_dir, "U_metab_final.npy"), U_metab_final)

    spice_3d = spice_refit_est.reshape(Ny, Nx, N_SEQ)

    # ── Phase correction ─────────────────────────────────────────────────────
    print(f"[spice_uoss] Phase correction  ppmlim={args.phase_ppmlim} …")
    spice_phcorr_f = phase_corr(
        spice_3d,
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = Dim_Voxel,
        out_dir    = out_dir,
        ppmlim     = args.phase_ppmlim,
        ref_img    = ref_img_obj,
        out_fname  = "SPICE_refit_uoss_phcorr",
    )
    spice_phcorr = FIDToSpec(spice_phcorr_f, axis=-1)

    spice_save = spice_phcorr_f.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(spice_save.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "SPICE_refit_result.nii.gz"))
    print("[spice_uoss] Saved SPICE_refit_result.nii.gz")

    # ── xcorr from SPICE result → align adjoint recon and lipid fit ───────────
    from utils.xcorr import xcorr_align_complex
    from fsl_mrs.utils.preproc import applyPhase, freqshift as _fsl_freqshift

    def _spec2fid(x):
        xr = np.asarray(x).reshape(N_VOXEL, -1)
        return np.fft.ifft(np.fft.ifftshift(xr, axes=-1), axis=-1, norm='ortho').astype(D_TYPE)

    _br, _bc = np.where(brain_mask_inner)
    _, _shifts_hz, _xcorr_ph = xcorr_align_complex(spice_3d[_br, _bc, :], TS)

    def _apply_xcorr_3d(fid_3d):
        out = fid_3d.copy()
        out[_br, _bc, :] = np.stack([
            applyPhase(_fsl_freqshift(fid, TS, shi), phs)
            for fid, shi, phs in zip(fid_3d[_br, _bc, :], _shifts_hz, _xcorr_ph)
        ])
        return out

    # adjoint recon of with-lipid data: B0.conj * F1D.H @ F.H(y)
    adj_withlip_fid = (B0_mat.conj() *
                       _spec2fid(F_OP.rmatvec(mrsi_withlip.ravel())).reshape(N_VOXEL, N_SEQ)
                       ).reshape(Ny, Nx, N_SEQ)
    _adj_save = _apply_xcorr_3d(adj_withlip_fid).transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(_adj_save, dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "adj_withlip_xcorr.nii.gz"))
    print("[spice_uoss] Saved adj_withlip_xcorr.nii.gz")

    # lipid fit (Stage-1 U_lipid @ V_lipid.H) with same xcorr correction
    _lip_save = _apply_xcorr_3d(lipid_fit_fid).transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(_lip_save, dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "lipid_fit_stage1_xcorr.nii.gz"))
    print("[spice_uoss] Saved lipid_fit_stage1_xcorr.nii.gz")

    # ── Clean metab-only MRSI: U_metab_final @ V_metab.H ──────────────────────
    metab_est_3d = (U_metab_final @ V_metab.conj().T).reshape(Ny, Nx, N_SEQ)
    print(f"[spice_uoss] Phase correction (metab-only)  ppmlim={args.phase_ppmlim} …")
    metab_phcorr_f = phase_corr(
        metab_est_3d,
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = Dim_Voxel,
        out_dir    = out_dir,
        ppmlim     = args.phase_ppmlim,
        ref_img    = ref_img_obj,
        out_fname  = "metab_clean_uoss_phcorr",
    )
    metab_phcorr = FIDToSpec(metab_phcorr_f, axis=-1)

    metab_save = metab_phcorr_f.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(metab_save.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "metab_clean_result.nii.gz"))
    print("[spice_uoss] Saved metab_clean_result.nii.gz")

    if args.save_plots:
        plot_voxel_spectrum_and_maps(
            spice_phcorr, im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04c_spice_refit_result.png"), dpi=120)
        plt.close("all")

        plot_voxel_spectrum_and_maps(
            metab_phcorr, im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04c_metab_clean_result.png"), dpi=120)
        plt.close("all")

    print("[spice_uoss] Done.")


if __name__ == "__main__":
    main()
