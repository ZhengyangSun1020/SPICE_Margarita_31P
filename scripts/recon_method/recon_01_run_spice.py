#!/usr/bin/env python3
"""
Step 4 — SPICE reconstruction with spatial regularization.

Two backends available via --backend:
  torchnufft  (default) : torchkbnufft + Toeplitz Gram — faster per CG iter, needs torch
  finufft               : mrinufft finufft, Gram = F.H@F — no torch dep

Reads  : <data_dir>/wref_o.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <out_dir>/lipid_removal/kt_mrsi_lprm.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <basis_dir>/
Writes : <out_dir>/spice_w<wmax>_l<lambda1>/SPICE_result.nii.gz
         <out_dir>/spice_w<wmax>_l<lambda1>/SPICE_f.npy
         <out_dir>/spice_w<wmax>_l<lambda1>/U_est.npy
         <out_dir>/spice_w<wmax>_l<lambda1>/V_subspace.npy
         (run tag e.g. w5000_l0.0001; printed on startup for use with --run-tag downstream)

Usage:
    # torchnufft (default)
    python scripts/recon_method/recon_01_run_spice.py \
        --data-dir data/processed/invivo_250305_01 --basis-dir ./basis/ \
        --save-plots [--brain-threshold 0.16] [--brain-erosion 1] \
        [--backend torchnufft] [--rank 20] [--lambda1 1e-4] [--maxiter 120]

    # finufft
    python scripts/recon_method/recon_01_run_spice.py \
        --data-dir data/processed/invivo_250305_01 --basis-dir ./basis/ \
        --backend finufft --save-plots [--brain-threshold 0.16] [--brain-erosion 3]
        [--rank 15] [--lambda1 1e-4] [--maxiter 120] [--brain-mask-cleanup]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
])
import numpy as np
import mrinufft
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage import binary_erosion
from warnings import filterwarnings
filterwarnings("ignore")
import nibabel as nib

from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.utils.plotting import FID2Spec
from fsl_mrs.utils.synthetic import syntheticFromBasisFile
from fsl.data.image import Image
from nifti_mrs.create_nmrs import gen_nifti_mrs
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs as gen_nifti_mrs_fsl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.scan_params import load_scan_params
from utils.pipeline_utils import make_brain_mask
from utils.utils import (
    calc_Bmatrix,
    save_training_data_as_csv,
    read_training_data_from_csv,
    Sig_func_Multi_Peak_2D,
    SPICEWithSpatialConstrain_cg_nufft,
    NUFFTLinearOperator,
    plot_voxel_spectrum_and_maps,
    plot_anatomical_mask_points_size_directional,
    plot_voxel_sum_map,
    Calc_B0_matrix,
    NUFFTOp,
    build_nufft_ops,
)
from utils.xcorr import my_mrsi_freq_align


def parse_args():
    p = argparse.ArgumentParser(description="SPICE reconstruction — step 4")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--basis-dir",       required=True)
    p.add_argument("--out-dir",         default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--backend",         default="torchnufft",
                   choices=["torchnufft", "finufft"],
                   help="NUFFT backend: torchnufft (default) or finufft")
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int, default=None)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--n-coils",         type=int, default=None)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64], metavar=("NX","NY"))
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--n-shots",         type=int,   default=360,
                   help="Number of shots (torchnufft only, default: 360)")
    p.add_argument("--phase-ppmlim",    type=float, nargs=2, default=[0.0, 5.0], metavar=("LO","HI"))
    # SPICE
    p.add_argument("--rank",            type=int,   default=20)
    p.add_argument("--lambda1",         type=float, default=1e-4)
    p.add_argument("--wmax",            type=float, default=5e3)
    p.add_argument("--adj",             type=int,   default=8)
    p.add_argument("--pool-size",       type=int,   default=1)
    p.add_argument("--minpool",         action="store_true")
    p.add_argument("--maxiter",         type=int,   default=120)
    p.add_argument("--dx-tol",          type=float, default=1e-6,
                   help="Early-stop step-size tolerance for CG (default: 1e-6)")
    p.add_argument("--patience",        type=int,   default=4,
                   help="No-improvement steps before early stop (default: 4)")
    p.add_argument("--patience-dx",     type=int,   default=3,
                   help="Small-dx steps before early stop (default: 3)")
    # Training
    p.add_argument("--training-size",   type=int,   default=10000)
    p.add_argument("--csv-name",        default="SS_training")
    # Metabolites
    p.add_argument("--metabs",          nargs="+",
                   default=["Cr","GABA","Glu","Gln","GPC","GSH",
                             "Lac","NAA","NAAG","Ins","PCh","PCr","Tau","Asp","PE"])
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component (drops disconnected noise blobs outside "
                        "the brain) and fill enclosed holes (e.g. a central signal void). "
                        "Default: off, use only if a single global threshold isn't enough.")
    p.add_argument("--ref-nii",         default=None)
    p.add_argument("--save-plots",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    run_tag  = f"w{args.wmax:g}_l{args.lambda1:g}"
    out_dir  = os.path.join(args.out_dir, f"spice_{run_tag}")
    print(f"[spice] Run tag: {run_tag}  (pass --run-tag {run_tag} to downstream scripts)")
    os.makedirs(out_dir, exist_ok=True)

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
    print(f"[spice/{args.backend}] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s")

    META_LIST = args.metabs
    NUM_METAB = len(META_LIST)

    # ── Basis ────────────────────────────────────────────────────────────────────
    print("[spice] Loading basis …")
    fullbasis = mrs_io.read_basis(args.basis_dir)
    basis_mat = fullbasis.get_formatted_basis(bandwidth=sweepwidth, points=N_SEQ)
    bm_FIDs   = []
    for meta in META_LIST:
        try:
            j = fullbasis.names.index(meta)
        except ValueError:
            raise ValueError(f"'{meta}' not found in basis. Available: {fullbasis.names}")
        bm_FIDs.append(basis_mat[:, j].conj())

    # ── Load inputs ──────────────────────────────────────────────────────────────
    print("[spice] Loading data …")
    mrsi_lprm       = np.load(os.path.join(lprm_dir,    "kt_mrsi_lprm.npy"),    mmap_mode="r").astype(D_TYPE)
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir,    "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"),        mmap_mode="r")
    B0_map          = np.load(os.path.join(b0map_dir,   "B0_map.npy"))
    wref_img        = np.load(data_dir + "wref_o.npy", mmap_mode="r")

    trej     = mrsi_ksp_scaled.T.astype(Trej_D_TYPE)
    NUM_CMAP = coil_smap_raw.shape[0]

    coil_smap = np.repeat(
        coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1
    ).astype(D_TYPE)
    smap_time = coil_smap.squeeze(0)   # (C, Ny, Nx, T)

    # ── Brain mask ───────────────────────────────────────────────────────────────
    wref_norm, brain_mask, brain_mask_inner = make_brain_mask(
        wref_img, args.brain_threshold, args.brain_erosion,
        cleanup=args.brain_mask_cleanup)

    # ── Build NUFFT operators ─────────────────────────────────────────────────────
    F_OP, Gram_OP, F1D, device_str = build_nufft_ops(
        args.backend, trej, im_size, coil_smap_raw, NUM_CMAP, D_TYPE,
        osamp=2.0, ost=2.0,
    )

    # ── B0 modulation matrix ─────────────────────────────────────────────────────
    print("[spice] Building B0 modulation matrix …")
    B0_map_clean = np.nan_to_num(B0_map, nan=0.0)
    B0_mat = Calc_B0_matrix(B0_map_clean, TIME_AXIS).reshape(N_VOXEL, N_SEQ)

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.abs(np.mean(B0_mat.reshape(Ny, Nx, N_SEQ), axis=-1)),
                       origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, label="B0 avg magnitude")
        ax.set_title("B0 modulation matrix (avg over time)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_04_B0_mat.png"), dpi=120)
        plt.close(fig)

    # ── Spatial regularization ───────────────────────────────────────────────────
    print("[spice] Building spatial regularization (B matrix) …")
    W_edge, _, _W, Nb = calc_Bmatrix(
        wref_norm, wmax=args.wmax, adj=args.adj,
        pool_size=args.pool_size,
        minpooling_Handler=args.minpool,
        brain_mask=brain_mask,
        mask_dilate_layers=3,
    )
    WW = W_edge.conj().T @ W_edge

    wref_masked = wref_norm * brain_mask

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(wref_masked, origin="lower", cmap="gray")
        ax.set_title("wref_o (masked)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_04_wref_masked.png"), dpi=120)
        plt.close(fig)
        print("[spice] Saved fig_04_wref_masked.png")

    if args.save_plots:
        edge_index = [tuple(pair) for pair in Nb]

        # wref anatomical prior
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(wref_norm, origin="lower", cmap="gray")
        ax.imshow(brain_mask_inner, origin="lower", cmap="Reds", alpha=0.35)
        ax.set_title(f"wref prior + brain mask (thr={args.brain_threshold})")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_04_prior_mask.png"), dpi=120)
        plt.close(fig)

        # edge prior directional plot
        plot_anatomical_mask_points_size_directional(
            mask=_W, anatomical_prior=wref_norm, edge_index=edge_index)
        plt.savefig(os.path.join(out_dir, "fig_04_edge_prior.png"), dpi=120)
        plt.close("all")

        # voxel sum map
        voxel_sum_map = plot_voxel_sum_map(
            mask=_W, anatomical_prior=wref_norm, edge_index=edge_index,
            threshold=1e-6, use_abs=True)
        plt.savefig(os.path.join(out_dir, "fig_04_voxel_sum_map.png"), dpi=120)
        plt.close("all")

    # ── Subspace training ────────────────────────────────────────────────────────
    csv_path = os.path.join(args.basis_dir, args.csv_name + ".csv")
    if os.path.exists(csv_path):
        print(f"[spice] Loading existing training data: {csv_path}")
        training_dataset = read_training_data_from_csv(args.basis_dir, args.csv_name).astype(D_TYPE)
    else:
        print(f"[spice] Generating {args.training_size} synthetic training samples …")
        rng      = np.random.default_rng()
        train_cs = rng.random((NUM_METAB, args.training_size)) * 1.5
        train_fs = (2 * rng.random((NUM_METAB, args.training_size)) - 1) * 0.005 * center_freq
        train_ws = (2 * rng.random((args.training_size,)) - 1) * 0.04 * center_freq
        train_lw = 10.0 + rng.standard_normal((NUM_METAB, args.training_size)) * 2.0
        training_dataset = Sig_func_Multi_Peak_2D(
            bm_FIDs, train_lw, train_cs, TIME_AXIS,
            args.training_size, freq_shift=train_fs, whole_shift=train_ws,
            N_SEQ_POINTS=N_SEQ,
        ).astype(D_TYPE)
        save_training_data_as_csv(training_dataset, args.basis_dir, args.csv_name, savecondition=True)

    print(f"[spice] SVD of training data {training_dataset.shape} …")
    _, s, Vh = np.linalg.svd(training_dataset)
    V = Vh[:args.rank, :].conj().T
    np.save(os.path.join(out_dir, "V_subspace.npy"), V)

    if args.save_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(s[:30], "x-"); ax1.set_title("Singular values")
        ax2.plot(PPM_AXIS, np.abs(FID2Spec(Vh[:6, :].T))); ax2.invert_xaxis()
        ax2.set_title("Top-6 subspace vectors")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_04a_subspace.png"), dpi=120)
        plt.close(fig)

    # ── Run SPICE ─────────────────────────────────────────────────────────────────
    print(f"[spice/{args.backend}] Running SPICE  rank={args.rank}  λ={args.lambda1}  maxiter={args.maxiter} …")

    spice_est, est_U, _ = SPICEWithSpatialConstrain_cg_nufft(
        noisy_kt_spaces  = mrsi_lprm,
        img_shape        = im_size,
        F=F_OP, Gram_OP=Gram_OP, F1D_OP=F1D,
        B0_mat=B0_mat, V=V,
        N_Vox=N_VOXEL, NUM_SPICE_RANK=args.rank,
        WW=WW, Solver="cg",
        lamda_1=args.lambda1, maxiter=args.maxiter,
        dx_tol           = args.dx_tol,
        patience         = args.patience,
        patience_dx      = args.patience_dx,
        brain_mask_inner = brain_mask_inner,
        PPM_AXIS         = PPM_AXIS,
    )

    print(f"[spice] Done. est shape: {spice_est.shape}")

    # ── Save raw outputs ──────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "SPICE_f.npy"), spice_est)
    np.save(os.path.join(out_dir, "U_est.npy"),   est_U)

    # Priority: (1) --ref-nii arg, (2) affine.npy saved by data_proc_01_twix2npy,
    # (3) subject-specific reference NIfTI in data_dir, (4) identity
    _affine_npy = data_dir + "affine.npy"
    if args.ref_nii:
        ref_img_obj = Image(args.ref_nii)
        affine      = ref_img_obj.voxToWorldMat
    elif os.path.exists(_affine_npy):
        affine      = np.load(_affine_npy)
        ref_img_obj = None
        print(f"[spice] Loaded affine from {_affine_npy}")
    else:
        ref_img_path = data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz"
        try:
            ref_img_obj = Image(ref_img_path)
            affine      = ref_img_obj.voxToWorldMat
        except Exception:
            ref_img_obj = None
            affine      = np.eye(4)

    wref_nii = nib.Nifti1Image(
        np.ascontiguousarray(wref_masked.T[::-1, :])[:, :, np.newaxis].astype(np.float32),
        affine)
    wref_nii.header.set_xyzt_units("mm")
    nib.save(wref_nii, os.path.join(out_dir, "wref_masked.nii.gz"))
    print("[spice] Saved wref_masked.nii.gz")

    spice_3d = spice_est.reshape(Ny, Nx, N_SEQ)

    # ── Save raw SPICE result (pre-alignment) ─────────────────────────────────────
    spice_raw_save = np.ascontiguousarray(spice_3d.transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :]
    gen_nifti_mrs(spice_raw_save, dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "SPICE_result.nii.gz"))
    print("[spice] Saved SPICE_result.nii.gz")

    # ── xcorr frequency alignment (replaces FSL-MRS phase_corr) ─────────────────
    print("[spice] xcorr frequency alignment …")
    fid_ref, emptymrs, _ = syntheticFromBasisFile(
        fullbasis, noisecovariance=[[0]], bandwidth=sweepwidth, points=N_SEQ)
    basis_nmrs = gen_nifti_mrs_fsl(
        fid_ref.conj().reshape(1, 1, 1, N_SEQ),
        dwelltime=emptymrs.dwellTime,
        spec_freq=emptymrs.centralFrequency,
        affine=affine,
    )
    spice_nii = gen_nifti_mrs(
        np.ascontiguousarray(spice_3d.transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :],
        dwelltime=TS, spec_freq=297.219, affine=affine,
    )
    spice_aligned_nmrs, _ = my_mrsi_freq_align(spice_nii, basis_nmrs)
    spice_aligned_nmrs.save(os.path.join(out_dir, "SPICE_phcorr.nii.gz"))
    print("[spice] Saved SPICE_phcorr.nii.gz")
    spice_phcorr_f = np.array(spice_aligned_nmrs.image[:, :, 0, :]).transpose(1, 0, 2)[:, ::-1, :].conj()
    spice_phcorr = FIDToSpec(spice_phcorr_f, axis=-1)

    # ── Save U and V as NIfTI ─────────────────────────────────────────────────
    U_nii = np.ascontiguousarray(est_U.reshape(Ny, Nx, args.rank).transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :].conj().astype(np.complex64)
    Image(U_nii, xform=affine).save(os.path.join(out_dir, "U_subspace.nii.gz"))

    V_nmrs = np.tile(V[np.newaxis, np.newaxis, np.newaxis, :, :], (Nx, Ny, 1, 1, 1)).conj()
    gen_nifti_mrs(V_nmrs, dwelltime=TS, spec_freq=297.219, affine=affine,
                  dim_tags=['DIM_USER_0', None, None]).save(os.path.join(out_dir, "V_subspace.nii.gz"))
    print("[spice] Saved U_subspace.nii.gz and V_subspace.nii.gz")

    if args.save_plots:
        plot_voxel_spectrum_and_maps(
            spice_phcorr, im_size,
            voxel_x=Nx // 2, voxel_y=Ny // 2,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        plt.savefig(os.path.join(out_dir, "fig_04b_spice_result.png"), dpi=120)
        plt.close("all")

    print("[spice] Done.")


if __name__ == "__main__":
    main()
