#!/usr/bin/env python3
"""
Step 4d — Lam 2020 UoSS reconstruction (Eq. 4 + optional Eq. 5 free refit).

Implements the method from:
  Lam et al., "Ultrafast MRSI Using SPICE with Learned Subspaces",
  Magn Reson Med 83(2):377-390, 2020.  DOI: 10.1002/mrm.27980

Pipeline (equations refer to the paper):

  Stage 1 — Lipid-only subspace CG solve:
    Fit U_lipid on the raw with-lipid k-t data (kt_mrsi_withlip_noring.npy)
    using the lipid basis V_lipid from step 03.  W_lip diagonal prior keeps
    U_lipid near zero in clean brain voxels.

  Stage 2 — Lipid k-space subtraction:
    y_metab = y_withlip - F{B0 * (U_lipid @ V_lipid.H)}

  Stage 3 — Standard SPICE subspace solve on lipid-removed data (Lam 2020 Eq. 4):
    Exactly like step 04 (04_run_spice.py) but applied to y_metab.
    Solves for U_metab_new with metabolite basis V_metab.
    Warm-started from step 04's U_est.npy.
    Output: U_metab_lam20.npy, SPICE_lam20_subspace.npy.

  Stage 4 — Eq. 5 free image refit (optional, disable with --no-refit):
    Tikhonov-regularized free-image solve on y_metab:
        min  ||F{B0 * rho_m} - y_metab||^2  +  lambda_refit ||rho_m - rho_spice||^2
    where rho_spice = U_metab_new @ V_metab.H from Stage 3.
    Output: SPICE_lam20_refit.npy.

Reads  : <out_dir>/spice/V_subspace.npy
         <out_dir>/spice/U_est.npy
         <out_dir>/lipid_removal/V_lipid.npy
         <out_dir>/lipid_removal/lss_map.npy
         <out_dir>/lipid_removal/kt_mrsi_withlip_noring.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <data_dir>/wref_o.npy
Writes : <out_dir>/spice_lam20/U_lipid.npy
         <out_dir>/spice_lam20/kt_mrsi_lam20_lprm.npy
         <out_dir>/spice_lam20/U_metab_lam20.npy
         <out_dir>/spice_lam20/SPICE_lam20_subspace.npy  (Stage 3 subspace result)
         <out_dir>/spice_lam20/SPICE_lam20_refit.npy     (Stage 4 free image, if --refit)
         <out_dir>/spice_lam20/w_lip_vec.npy
         <out_dir>/spice_lam20/subspace_result.nii.gz    (Stage 3, phase-corrected)
         <out_dir>/spice_lam20/refit_result.nii.gz       (Stage 4, if --refit)
         <out_dir>/spice_lam20/fig_04d_*.png

Usage:
    python scripts/04d_SPICE_lam20_refit.py \\
        --data-dir ./data/ --out-dir ./output \\
        --lambda1 1e-4 --lamda-lip 1e-2 \\
        --lambda-refit 1e-3 \\
        --save-plots
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
from scipy.sparse.linalg import cg, LinearOperator
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

from fsl_mrs.utils.misc import FIDToSpec
from fsl.data.image import Image
from nifti_mrs.create_nmrs import gen_nifti_mrs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    calc_Bmatrix,
    SPICEWithSpatialConstrain_cg_nufft,
    plot_voxel_spectrum_and_maps,
    Calc_B0_matrix,
    phase_corr,
    build_nufft_ops,
)
from utils.lipid import select_lipid_mask_gmm_simple


def parse_args():
    p = argparse.ArgumentParser(description="Lam 2020 UoSS SPICE -- step 4d")
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
    # Stage 1: lipid-only SPICE
    p.add_argument("--lamda-lip",       type=float, default=1e-2,
                   help="Spatial reg for Stage 1 lipid-only subspace solve (default 1e-2)")
    p.add_argument("--maxiter-lip",     type=int,   default=120,
                   help="CG maxiter for Stage 1 lipid solve (default 120)")
    p.add_argument("--rtol-lip",        type=float, default=1e-5,
                   help="CG rtol for Stage 1 lipid solve (default 1e-5)")
    # Stage 3: standard SPICE on kt_metab (Lam 2020 Eq. 4)
    p.add_argument("--lambda1",         type=float, default=1e-4,
                   help="Spatial reg for Stage 3 metabolite SPICE solve (default 1e-4)")
    p.add_argument("--maxiter",         type=int,   default=120,
                   help="CG maxiter for Stage 3 metabolite SPICE (default 120)")
    p.add_argument("--rtol",            type=float, default=1e-5,
                   help="CG rtol for Stage 3 metabolite SPICE (default 1e-5)")
    # Stage 4: free refit (Lam 2020 Eq. 5)
    p.add_argument("--lambda-refit",    type=float, default=1e-3,
                   help="Tikhonov weight for Stage 4 free refit (default 1e-3)")
    p.add_argument("--maxiter-refit",   type=int,   default=60,
                   help="CG maxiter for Stage 4 free refit (default 60)")
    p.add_argument("--rtol-refit",      type=float, default=1e-5,
                   help="CG rtol for Stage 4 free refit (default 1e-5)")
    p.add_argument("--no-refit",        action="store_true",
                   help="Skip Stage 4 Eq. 5 free refit")
    # In-brain lipid GMM (W_lip)
    p.add_argument("--nsigma-gmm-inbrain", type=float, default=2.0)
    p.add_argument("--n-lipid-voxels-inbrain", type=int, default=2000)
    p.add_argument("--topn-fallback",   type=int,   default=100)
    p.add_argument("--min-lip-penalty", type=float, default=0.001)
    p.add_argument("--max-lip-penalty", type=float, default=1e2)
    # Misc
    p.add_argument("--wmax",            type=float, default=5e3)
    p.add_argument("--adj",             type=int,   default=8)
    p.add_argument("--pool-size",       type=int,   default=1)
    p.add_argument("--minpool",         action="store_true")
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--ref-nii",         default=None)
    p.add_argument("--save-plots",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    out_dir  = os.path.join(args.out_dir, "spice_lam20")
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
    print(f"[lam20] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s"
          f"  refit={'disabled' if args.no_refit else 'enabled'}")

    # Subspace bases
    V_metab = np.load(os.path.join(spice_dir, "V_subspace.npy")).astype(D_TYPE)
    U_metab = np.load(os.path.join(spice_dir, "U_est.npy")).astype(D_TYPE)   # warm start for Stage 3
    R_met   = V_metab.shape[1]
    print(f"[lam20] Loaded V_metab {V_metab.shape}  U_metab {U_metab.shape}")

    V_lipid = np.load(os.path.join(lprm_dir, "V_lipid.npy")).astype(D_TYPE)
    R_lip   = V_lipid.shape[1]
    print(f"[lam20] Loaded V_lipid {V_lipid.shape}")

    # Data
    print("[lam20] Loading data ...")
    mrsi_withlip    = np.load(os.path.join(lprm_dir, "kt_mrsi_withlip_noring.npy"),
                              mmap_mode="r").astype(D_TYPE)
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir, "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"), mmap_mode="r")
    B0_map          = np.load(os.path.join(b0map_dir, "B0_map.npy"))
    wref_img        = np.load(data_dir + "wref_o.npy", mmap_mode="r")

    trej     = mrsi_ksp_scaled.T.astype(Trej_D_TYPE)
    NUM_CMAP = coil_smap_raw.shape[0]

    # Brain mask
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # Build NUFFT operators
    F_OP, Gram_OP, F1D, device_str = build_nufft_ops(
        args.backend, trej, im_size, coil_smap_raw, NUM_CMAP, D_TYPE,
        osamp=2.0, ost=2.0,
    )

    # B0 matrix
    print("[lam20] Building B0 modulation matrix ...")
    B0_mat = Calc_B0_matrix(np.nan_to_num(B0_map, nan=0.0), TIME_AXIS).reshape(N_VOXEL, N_SEQ)

    # Spatial regularization (metabolite)
    print("[lam20] Building spatial regularization ...")
    W_edge, _, _W, Nb = calc_Bmatrix(
        wref_norm, wmax=args.wmax, adj=args.adj,
        pool_size=args.pool_size,
        minpooling_Handler=args.minpool,
        brain_mask=brain_mask,
        mask_dilate_layers=3,
    )
    WW = W_edge.conj().T @ W_edge

    # In-brain LSS GMM -> W_lip (for Stage 1)
    print("[lam20] In-brain LSS GMM classification ...")
    lss_map      = np.load(os.path.join(lprm_dir, "lss_map.npy"))
    lss_2d_brain = np.squeeze(lss_map) * brain_mask
    res_inbrain  = select_lipid_mask_gmm_simple(
        lss_2d_brain, out_dir=out_dir, nsigma=args.nsigma_gmm_inbrain,
        max_voxels=args.n_lipid_voxels_inbrain, topN_fallback=args.topn_fallback,
        save_plots=args.save_plots, out_fname="fig_04d_lss_in_brain.png",
        title_prefix="In-brain LSS",
    )
    print(f"[lam20] In-brain GMM  method={res_inbrain['method']}  "
          f"n_contaminated={res_inbrain['n_selected']}")

    lipid_contam_mask = res_inbrain["lipid_mask"]
    w_lip_vec = np.full(brain_mask.shape, args.max_lip_penalty, dtype=np.float32)
    w_lip_vec[brain_mask & lipid_contam_mask] = args.min_lip_penalty
    w_lip_vec = w_lip_vec.ravel()
    np.save(os.path.join(out_dir, "w_lip_vec.npy"), w_lip_vec)

    W_lip  = sp_diags(w_lip_vec)
    WW_lip = W_lip.conj().T @ W_lip

    ref_img_path = args.ref_nii or (data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz")
    try:
        ref_img_obj = Image(ref_img_path)
        affine      = ref_img_obj.voxToWorldMat
    except Exception:
        ref_img_obj = None
        affine      = np.eye(4)

    # =========================================================================
    # Stage 1 -- lipid-only subspace CG solve on with-lipid k-t data
    # =========================================================================
    print(f"[lam20] Stage 1: lipid-only SPICE  R_lip={R_lip}  "
          f"lamda_lip={args.lamda_lip}  maxiter={args.maxiter_lip} ...")

    _, U_lipid, _ = SPICEWithSpatialConstrain_cg_nufft(
        noisy_kt_spaces = mrsi_withlip,
        img_shape       = im_size,
        F=F_OP, Gram_OP=Gram_OP, F1D_OP=F1D,
        B0_mat=B0_mat, V=V_lipid,
        N_Vox=N_VOXEL, NUM_SPICE_RANK=R_lip,
        WW=WW_lip, Solver="cg",
        lamda_1=args.lamda_lip, maxiter=args.maxiter_lip,
        rtol=args.rtol_lip,
        x0=np.zeros((N_VOXEL * R_lip,), dtype=D_TYPE),
        save_folder=os.path.join(out_dir, "cg_iters_stage1"),
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS,
    )
    np.save(os.path.join(out_dir, "U_lipid.npy"), U_lipid)
    print(f"[lam20] Stage 1 done.  U_lipid {U_lipid.shape}")

    # =========================================================================
    # Stage 2 -- subtract lipid k-space prediction -> kt_metab
    # =========================================================================
    print("[lam20] Stage 2: subtracting lipid k-space model ...")
    lipid_kt_pred = F_OP.matvec(
        (B0_mat * (U_lipid @ V_lipid.conj().T)).ravel()
    ).reshape(mrsi_withlip.shape).astype(D_TYPE)
    kt_metab = (mrsi_withlip - lipid_kt_pred).astype(D_TYPE)
    np.save(os.path.join(out_dir, "kt_mrsi_lam20_lprm.npy"), kt_metab)
    print(f"[lam20] Stage 2 done.  kt_metab shape={kt_metab.shape}")

    # =========================================================================
    # Stage 3 -- standard SPICE on lipid-removed data (Lam 2020 Eq. 4)
    #
    # Identical structure to step 04 (04_run_spice.py) but applied to
    # kt_metab instead of kt_mrsi_lprm.  Warm-started from step 04's U_est.
    # =========================================================================
    print(f"[lam20] Stage 3: SPICE on kt_metab (Eq. 4)  R_met={R_met}  "
          f"lambda1={args.lambda1}  maxiter={args.maxiter} ...")

    rho_spice, U_metab_new, _ = SPICEWithSpatialConstrain_cg_nufft(
        noisy_kt_spaces = kt_metab,
        img_shape       = im_size,
        F=F_OP, Gram_OP=Gram_OP, F1D_OP=F1D,
        B0_mat=B0_mat, V=V_metab,
        N_Vox=N_VOXEL, NUM_SPICE_RANK=R_met,
        WW=WW, Solver="cg",
        lamda_1=args.lambda1, maxiter=args.maxiter,
        rtol=args.rtol,
        x0=U_metab.ravel().astype(D_TYPE),  # warm start from step 04
        save_folder=os.path.join(out_dir, "cg_iters_stage3"),
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS,
    )
    np.save(os.path.join(out_dir, "U_metab_lam20.npy"), U_metab_new)
    np.save(os.path.join(out_dir, "SPICE_lam20_subspace.npy"), rho_spice)
    print(f"[lam20] Stage 3 done.  U_metab_new {U_metab_new.shape}")

    if args.save_plots:
        plot_voxel_spectrum_and_maps(
            FIDToSpec(rho_spice.reshape(Ny, Nx, N_SEQ), axis=-1), im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04d_stage3_subspace.png"), dpi=120)
        plt.close("all")

    # Phase-correct and save Stage 3 result
    print(f"[lam20] Phase correction (Stage 3)  ppmlim={args.phase_ppmlim} ...")
    subspace_phcorr_f = phase_corr(
        rho_spice.reshape(Ny, Nx, N_SEQ),
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = Dim_Voxel,
        out_dir    = out_dir,
        ppmlim     = args.phase_ppmlim,
        ref_img    = ref_img_obj,
        out_fname  = "subspace_phcorr",
    )
    sub_save = subspace_phcorr_f.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(sub_save.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "subspace_result.nii.gz"))
    print("[lam20] Saved subspace_result.nii.gz")

    if args.no_refit:
        print("[lam20] --no-refit: skipping Stage 4.  Done.")
        return

    # =========================================================================
    # Stage 4 -- Eq. 5 free image refit (optional)
    #
    # Solves  (Gram_B0 + lambda_refit * I) rho_m = b  where
    #   Gram_B0(rho) = B0.conj * Gram_OP(B0 * rho)
    #   b            = B0.conj * F1D.H(F.H(y_metab)) + lambda_refit * rho_spice
    #
    # rho_spice (Stage 3 output) is the Tikhonov prior.
    # rho_m is a free spatiotemporal image -- NOT constrained to V_metab.
    # =========================================================================
    print(f"[lam20] Stage 4: free refit (Eq. 5)  lambda_refit={args.lambda_refit}  "
          f"maxiter={args.maxiter_refit}  rtol={args.rtol_refit} ...")

    lam_r = np.float32(args.lambda_refit)

    adj_y_metab = (B0_mat.conj() *
                   F1D.rmatvec(F_OP.rmatvec(kt_metab.ravel())).reshape(N_VOXEL, N_SEQ))
    b_refit = (adj_y_metab + lam_r * rho_spice).ravel().astype(D_TYPE)

    def mv_refit(rho_vec):
        rho  = rho_vec.reshape(N_VOXEL, N_SEQ)
        gram = B0_mat.conj() * Gram_OP.matvec((B0_mat * rho).ravel()).reshape(N_VOXEL, N_SEQ)
        return (gram + lam_r * rho).ravel().astype(D_TYPE)

    A_refit = LinearOperator(
        (N_VOXEL * N_SEQ, N_VOXEL * N_SEQ),
        matvec=mv_refit, dtype=D_TYPE,
    )

    x0_refit    = rho_spice.ravel().astype(D_TYPE)
    iters_done  = [0]
    pbar_r      = tqdm(total=args.maxiter_refit, desc="CG refit", unit="iter")

    def _cb(xk):
        iters_done[0] += 1
        pbar_r.update(1)

    rho_flat, cg_info = cg(A_refit, b_refit, x0=x0_refit,
                            maxiter=args.maxiter_refit, rtol=args.rtol_refit,
                            callback=_cb)
    pbar_r.close()
    print(f"[lam20] Stage 4 done  cg_info={cg_info}  iters={iters_done[0]}")

    rho_m = rho_flat.reshape(N_VOXEL, N_SEQ).astype(D_TYPE)
    np.save(os.path.join(out_dir, "SPICE_lam20_refit.npy"), rho_m)
    print("[lam20] Saved SPICE_lam20_refit.npy")

    if args.save_plots:
        plot_voxel_spectrum_and_maps(
            FIDToSpec(rho_m.reshape(Ny, Nx, N_SEQ), axis=-1), im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04d_stage4_refit.png"), dpi=120)
        plt.close("all")

    # Phase-correct and save Stage 4 result
    print(f"[lam20] Phase correction (Stage 4)  ppmlim={args.phase_ppmlim} ...")
    refit_phcorr_f = phase_corr(
        rho_m.reshape(Ny, Nx, N_SEQ),
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = Dim_Voxel,
        out_dir    = out_dir,
        ppmlim     = args.phase_ppmlim,
        ref_img    = ref_img_obj,
        out_fname  = "refit_phcorr",
    )
    refit_save = refit_phcorr_f.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(refit_save.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "refit_result.nii.gz"))
    print("[lam20] Saved refit_result.nii.gz")

    print("[lam20] Done.")


if __name__ == "__main__":
    main()
