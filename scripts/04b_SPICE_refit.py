#!/usr/bin/env python3
"""
Step 4b — Joint lipid+metabolite SPICE refit (on raw with-lipid data).

Warm-starts U_metab/V_metab from step 04, builds V_lipid via SVD of the
lipid basis voxels saved in step 03, and jointly refits U_joint =
[U_lipid, U_metab] on the raw with-lipid (no-ring) k-t data from step 03
(kt_mrsi_withlip_noring.npy). U_lipid absorbs all in-skull lipid signal
directly; U_metab is refit jointly from the same data. The in-brain GMM
W_lip prior constrains U_lipid to known-contaminated voxels so that
clean-brain voxels are not fitted by the lipid block.

Spatial regularization is block-wise: lamda1 * WW (anatomical-edge Laplacian
Gram, W_edge.H @ W_edge) on U_metab, lamda-lip * WW_lip (W_lip.H @ W_lip) on
U_lipid. W_lip is built here from an in-brain LSS GMM classification (own
nsigma/cap, separate from step 03's ring-extraction GMM) with two tiers:
in-brain voxels the GMM flags as contaminated (leakage near skull/edges) ->
min-lip-penalty (mild floor, not zero); everything else, i.e. clean in-brain
voxels AND outside the brain mask -> max-lip-penalty (shrink-to-zero) — the
lipid ring is already zeroed out of the with-lipid k-t data, so there's no
real signal outside brain for U_lipid to explain there either.

Reads  : <out_dir>/spice/U_est.npy, V_subspace.npy            (step 04)
         <out_dir>/lipid_removal/V_lipid.npy                  (step 03, SVD-truncated)
         <out_dir>/lipid_removal/lss_map.npy                  (step 03, raw LSS map)
         <out_dir>/lipid_removal/kt_mrsi_withlip_noring.npy     (step 03, with-lipid no-ring)
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <data_dir>/wref_o.npy
Writes : <out_dir>/spice_refit/SPICE_refit_f.npy
         <out_dir>/spice_refit/U_joint.npy, V_joint.npy
         <out_dir>/spice_refit/U_lipid_final.npy, U_metab_final.npy
         <out_dir>/spice_refit/w_lip_vec.npy
         <out_dir>/spice_refit/SPICE_refit_result.nii.gz        (full U_joint @ V_joint.H)
         <out_dir>/spice_refit/metab_clean_result.nii.gz        (metab-only, U_metab_final @ V_metab.H)
         <out_dir>/spice_refit/lipid_clean_result.nii.gz         (lipid-only, U_lipid_final @ V_lipid.H; reference only)
         <out_dir>/spice_refit/fig_04b_*.png

Usage:
    python scripts/04b_SPICE_refit.py \
        --data-dir ./data/ --out-dir ./output \
        --save-plots
    (defaults below already match the tuned command-line: --lamda-lip 1e-6
    --lambda1 1e-4 --maxiter 120 --nsigma-gmm-inbrain 2.0
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
from fsl_mrs.utils.plotting import FID2Spec
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
    p = argparse.ArgumentParser(description="Joint lipid+metabolite SPICE refit — step 4b")
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
                   help="Spatial reg weight for U_lipid block (scales WW_lip = "
                        "W_lip.H @ W_lip, built here from the in-brain GMM weight map)")
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
                   help="W_lip diagonal weight for lipid-contaminated voxels (skull/ring + "
                        "in-brain leakage): a small floor penalty, not zero — U_lipid is "
                        "still mildly regularized there, just much less than clean voxels")
    p.add_argument("--max-lip-penalty", type=float, default=1e2,
                   help="W_lip diagonal weight for clean in-brain voxels: shrinks U_lipid toward zero")
    p.add_argument("--wmax",            type=float, default=5e3)
    p.add_argument("--adj",             type=int,   default=8)
    p.add_argument("--pool-size",       type=int,   default=1)
    p.add_argument("--minpool",         action="store_true")
    p.add_argument("--maxiter",         type=int,   default=120)
    p.add_argument("--patience",        type=int,   default=10,
                   help="CG early-stop: consecutive non-improving iters tolerated before stopping early")
    p.add_argument("--patience-dx",     type=int,   default=3,
                   help="CG early-stop: consecutive tiny-step (rel_dx<1e-6) iters tolerated before stopping early")
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--ref-nii",         default=None)
    p.add_argument("--save-plots",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    out_dir  = os.path.join(args.out_dir, "spice_refit")
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
    print(f"[spice_refit] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s")

    # ── Warm-start U_metab / V_metab from step 04 ─────────────────────────────
    U_metab = np.load(os.path.join(spice_dir, "U_est.npy")).astype(D_TYPE)        # (N_Vox, R_met)
    V_metab = np.load(os.path.join(spice_dir, "V_subspace.npy")).astype(D_TYPE)   # (T, R_met)
    R_met   = V_metab.shape[1]
    print(f"[spice_refit] Loaded U_metab {U_metab.shape}, V_metab {V_metab.shape}")

    # ── V_lipid (SVD-truncated in step 03) ────────────────────────────────────
    V_lipid = np.load(os.path.join(lprm_dir, "V_lipid.npy")).astype(D_TYPE)       # (T, R_lip)
    R_lip   = V_lipid.shape[1]
    print(f"[spice_refit] Loaded V_lipid {V_lipid.shape}")

    # ── Joint V ──────────────────────────────────────────────────────────────
    V_joint      = np.hstack([V_lipid, V_metab]).astype(D_TYPE)                   # (T, R_lip+R_met)
    rank_blocks  = [R_lip, R_met]

    # ── Load data for forward model ───────────────────────────────────────────
    print("[spice_refit] Loading data …")
    mrsi_lprm       = np.load(os.path.join(lprm_dir, "kt_mrsi_withlip_noring.npy"), mmap_mode="r").astype(D_TYPE)
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
    print("[spice_refit] Building B0 modulation matrix …")
    B0_map_clean = np.nan_to_num(B0_map, nan=0.0)
    B0_mat = Calc_B0_matrix(B0_map_clean, TIME_AXIS).reshape(N_VOXEL, N_SEQ)

    # ── Spatial regularization: anatomical WW for metab, w_lip diag for lipid ──
    print("[spice_refit] Building spatial regularization (B matrix) …")
    W_edge, _, _W, Nb = calc_Bmatrix(
        wref_norm, wmax=args.wmax, adj=args.adj,
        pool_size=args.pool_size,
        minpooling_Handler=args.minpool,
        brain_mask=brain_mask,
        mask_dilate_layers=3,
    )
    WW = W_edge.conj().T @ W_edge

    # ── In-brain LSS GMM → W_lip (own nsigma/cap, independent of step 03's
    # ring-extraction GMM) ──────────────────────────────────────────────────
    print("[spice_refit] In-brain LSS GMM classification …")
    lss_map      = np.load(os.path.join(lprm_dir, "lss_map.npy"))
    lss_2d_brain = np.squeeze(lss_map) * brain_mask
    res_inbrain  = select_lipid_mask_gmm_simple(
        lss_2d_brain, out_dir=out_dir, nsigma=args.nsigma_gmm_inbrain,
        max_voxels=args.n_lipid_voxels_inbrain, topN_fallback=args.topn_fallback,
        save_plots=args.save_plots, out_fname="fig_04b_lss_in_brain.png",
        title_prefix="In-brain LSS",
    )
    print(f"[spice_refit] In-brain GMM  method={res_inbrain['method']}  "
          f"n_contaminated={res_inbrain['n_selected']}")

    # Two-tier weight: in-brain GMM-flagged contaminated -> min_lip_penalty
    # (mild floor, not zero); everything else (in-brain clean AND outside the
    # brain mask) -> max_lip_penalty (shrink to zero). With the lipid ring
    # already zeroed out (kt_mrsi_lprm.npy is masked to brain_nolip_mask in step
    # 03), there's no real signal
    # outside brain for U_lipid to explain, so it gets the same strong
    # suppression as clean in-brain voxels.
    lipid_contam_mask = res_inbrain["lipid_mask"]
    w_lip_vec = np.full(brain_mask.shape, args.max_lip_penalty, dtype=np.float32)
    w_lip_vec[brain_mask & lipid_contam_mask] = args.min_lip_penalty
    w_lip_vec = w_lip_vec.ravel()
    np.save(os.path.join(out_dir, "w_lip_vec.npy"), w_lip_vec)
    print(f"[spice_refit] Saved w_lip_vec.npy  n_outside_brain={(~brain_mask).sum()}  "
          f"n_inbrain_contaminated={(brain_mask & lipid_contam_mask).sum()}  "
          f"n_inbrain_clean={(brain_mask & ~lipid_contam_mask).sum()}")

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(w_lip_vec.reshape(brain_mask.shape), origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, label="W_lip weight")
        ax.set_title("W_lip (lipid spatial-prior weight map)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_04b_wlip_map.png"), dpi=120)
        plt.close(fig)

    # W_lip is the bare diagonal operator (penalty = ||W_lip @ x||^2); WW_lip is
    # its Gram, passed to the CG solver just like WW = W_edge.H @ W_edge above.
    W_lip   = sp_diags(w_lip_vec)
    WW_lip  = W_lip.conj().T @ W_lip

    WW_blocks    = [WW_lip, WW]
    lamda_blocks = [args.lamda_lip, args.lambda1]

    # ── U_lipid initial guess ───────────────────────────────────────────────
    # Zero cold-start. (An adjoint/matched-filter warm-start was tried but
    # produced irregular under-fit bands in clean, non-lipid brain regions —
    # the joint Hessian's cross-coupling term let a poorly-scaled U_lipid
    # initial guess pull U_metab away from its already-converged step-04
    # value before CG could correct it within maxiter iterations.)
    U_lipid_init = np.zeros((N_VOXEL, R_lip), dtype=D_TYPE)
    U_joint_init = np.hstack([U_lipid_init, U_metab]).astype(D_TYPE)

    # ── Run joint SPICE refit ──────────────────────────────────────────────────
    print(f"[spice_refit] Running joint refit  R_lip={R_lip}  R_met={R_met}  "
          f"lamda_lip={args.lamda_lip}  lambda1={args.lambda1}  maxiter={args.maxiter} …")
    spice_refit_est, U_joint, info = SPICEWithSpatialConstrain_cg_nufft_joint(
        noisy_kt_spaces = mrsi_lprm,
        img_shape       = im_size,
        F=F_OP, Gram_OP=Gram_OP, F1D_OP=F1D,
        B0_mat=B0_mat, V=V_joint,
        N_Vox=N_VOXEL, rank_blocks=rank_blocks,
        WW_blocks=WW_blocks, lamda_blocks=lamda_blocks,
        x0=U_joint_init,
        maxiter=args.maxiter,
        save_folder=os.path.join(out_dir, "cg_iters"),
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS,
        patience=args.patience,
        patience_dx=args.patience_dx,
    )
    print(f"[spice_refit] Done. est shape: {spice_refit_est.shape}")

    # ── Split U_joint back into lipid / metab blocks ──────────────────────────
    U_lipid_final = U_joint[:, :R_lip]
    U_metab_final = U_joint[:, R_lip:]

    # ── Save raw outputs ──────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "SPICE_refit_f.npy"), spice_refit_est)
    np.save(os.path.join(out_dir, "U_joint.npy"), U_joint)
    np.save(os.path.join(out_dir, "V_joint.npy"), V_joint)
    np.save(os.path.join(out_dir, "U_lipid_final.npy"), U_lipid_final)
    np.save(os.path.join(out_dir, "U_metab_final.npy"), U_metab_final)

    ref_img_path = args.ref_nii or (data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz")
    try:
        ref_img_obj = Image(ref_img_path)
        affine      = ref_img_obj.voxToWorldMat
    except Exception:
        ref_img_obj = None
        affine      = np.eye(4)

    spice_3d = spice_refit_est.reshape(Ny, Nx, N_SEQ)

    # ── Phase correction ─────────────────────────────────────────────────────
    print(f"[spice_refit] Phase correction  ppmlim={args.phase_ppmlim} …")
    spice_phcorr_f = phase_corr(
        spice_3d,
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = Dim_Voxel,
        out_dir    = out_dir,
        ppmlim     = args.phase_ppmlim,
        ref_img    = ref_img_obj,
        out_fname  = "SPICE_refit_phcorr",
    )
    spice_phcorr = FIDToSpec(spice_phcorr_f, axis=-1)

    spice_save = spice_phcorr_f.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(spice_save.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "SPICE_refit_result.nii.gz"))
    print("[spice_refit] Saved SPICE_refit_result.nii.gz")

    # ── Save U_lipid / U_metab as NIfTI ────────────────────────────────────────
    U_lip_nii = U_lipid_final.reshape(Ny, Nx, R_lip).transpose(1, 0, 2)[:, :, np.newaxis, :].conj().astype(np.complex64)
    Image(U_lip_nii, xform=affine).save(os.path.join(out_dir, "U_lipid_subspace.nii.gz"))

    U_met_nii = U_metab_final.reshape(Ny, Nx, R_met).transpose(1, 0, 2)[:, :, np.newaxis, :].conj().astype(np.complex64)
    Image(U_met_nii, xform=affine).save(os.path.join(out_dir, "U_metab_subspace.nii.gz"))
    print("[spice_refit] Saved U_lipid_subspace.nii.gz and U_metab_subspace.nii.gz")

    # ── Clean metab-only MRSI: U_metab_final @ V_metab.H, no lipid component ──
    metab_est_3d = (U_metab_final @ V_metab.conj().T).reshape(Ny, Nx, N_SEQ)
    print(f"[spice_refit] Phase correction (metab-only)  ppmlim={args.phase_ppmlim} …")
    metab_phcorr_f = phase_corr(
        metab_est_3d,
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = Dim_Voxel,
        out_dir    = out_dir,
        ppmlim     = args.phase_ppmlim,
        ref_img    = ref_img_obj,
        out_fname  = "metab_clean_phcorr",
    )
    metab_phcorr = FIDToSpec(metab_phcorr_f, axis=-1)

    metab_save = metab_phcorr_f.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(metab_save.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "metab_clean_result.nii.gz"))
    print("[spice_refit] Saved metab_clean_result.nii.gz")

    # ── Pure lipid-only MRSI (reference): U_lipid_final @ V_lipid.H ───────────
    # Saved purely as a sanity-check reference for verifying the U_joint split —
    # not consumed by any downstream step.
    lipid_est_3d = (U_lipid_final @ V_lipid.conj().T).reshape(Ny, Nx, N_SEQ)
    print(f"[spice_refit] Phase correction (lipid-only)  ppmlim={args.phase_ppmlim} …")
    lipid_phcorr_f = phase_corr(
        lipid_est_3d,
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = Dim_Voxel,
        out_dir    = out_dir,
        ppmlim     = args.phase_ppmlim,
        ref_img    = ref_img_obj,
        out_fname  = "lipid_clean_phcorr",
    )
    lipid_phcorr = FIDToSpec(lipid_phcorr_f, axis=-1)

    lipid_save = lipid_phcorr_f.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(lipid_save.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "lipid_clean_result.nii.gz"))
    print("[spice_refit] Saved lipid_clean_result.nii.gz")

    if args.save_plots:
        plot_voxel_spectrum_and_maps(
            spice_phcorr, im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04b_spice_refit_result.png"), dpi=120)
        plt.close("all")

        plot_voxel_spectrum_and_maps(
            metab_phcorr, im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04b_metab_clean_result.png"), dpi=120)
        plt.close("all")

        plot_voxel_spectrum_and_maps(
            lipid_phcorr, im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04b_lipid_clean_result.png"), dpi=120)
        plt.close("all")

    print("[spice_refit] Done.")


if __name__ == "__main__":
    main()
