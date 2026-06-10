#!/usr/bin/env python3
"""
Step 4 — SPICE reconstruction with spatial regularization.

Two backends available via --backend:
  toeplitz  (default) : torchkbnufft + Toeplitz Gram — faster per CG iter, needs torch
  finufft             : mrinufft finufft, Gram = F.H@F — no torch dep, has early-stop CG

Reads  : <data_dir>/wref_o.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <out_dir>/lipid_removal/kt_mrsi_lprm.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <basis_dir>/
Writes : <out_dir>/spice/SPICE_result.nii.gz
         <out_dir>/spice/SPICE_f.npy
         <out_dir>/spice/U_est.npy
         <out_dir>/spice/V_subspace.npy

Usage:
    # Toeplitz (default)
    python scripts/04_run_spice.py \\
        --data-dir ./invivo_260305/cr/ --basis-dir ./2pi_csap_SMF_MRSI/ \\
        [--backend toeplitz] [--rank 20] [--lambda1 1e-4] [--maxiter 120]

    # finufft
    python scripts/04_run_spice.py \\
        --data-dir ./invivo_260305/cr/ --basis-dir ./2pi_csap_SMF_MRSI/ \\
        --backend finufft [--rank 15] [--lambda1 1e-6] [--maxiter 120]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mrinufft
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage import binary_erosion
from warnings import filterwarnings
filterwarnings("ignore")

from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.utils.plotting import FID2Spec
from fsl.data.image import Image
from nifti_mrs.create_nmrs import gen_nifti_mrs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spice_mrsi.utils import (
    calc_Bmatrix,
    save_training_data_as_csv,
    read_training_data_from_csv,
    Sig_func_Multi_Peak_2D,
    SPICEWithSpatialConstrain_cg_nufft,
    SPICEWithSpatialConstrain_cg_finufft,
    NUFFTLinearOperator,
    plot_voxel_spectrum_and_maps,
    plot_anatomical_mask_points_size_directional,
    plot_voxel_sum_map,
    Calc_B0_matrix,
    NUFFTOp,
    phase_corr,
)


def parse_args():
    p = argparse.ArgumentParser(description="SPICE reconstruction — step 4")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--basis-dir",       required=True)
    p.add_argument("--out-dir",         default="./output")
    p.add_argument("--backend",         default="toeplitz",
                   choices=["toeplitz", "finufft"],
                   help="NUFFT backend: toeplitz (default) or finufft")
    p.add_argument("--dwelltime",       type=float, default=5e-6)
    p.add_argument("--k-points",        type=int,   default=39762)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--n-coils",         type=int,   default=32)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64], metavar=("NX","NY"))
    p.add_argument("--center-freq",     type=float, default=297.219338)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--n-shots",         type=int,   default=360,
                   help="Number of shots (toeplitz only, default: 360)")
    p.add_argument("--phase-ppmlim",    type=float, nargs=2, default=[0.0, 5.0], metavar=("LO","HI"))
    # SPICE
    p.add_argument("--rank",            type=int,   default=20)
    p.add_argument("--lambda1",         type=float, default=1e-4)
    p.add_argument("--wmax",            type=float, default=5e3)
    p.add_argument("--adj",             type=int,   default=8)
    p.add_argument("--pool-size",       type=int,   default=1)
    p.add_argument("--minpool",         action="store_true")
    p.add_argument("--maxiter",         type=int,   default=120)
    # Training
    p.add_argument("--training-size",   type=int,   default=10000)
    p.add_argument("--csv-name",        default="SS_training")
    p.add_argument("--reuse-training",  action="store_true")
    # Metabolites
    p.add_argument("--metabs",          nargs="+",
                   default=["Cr","GABA","Glu","Gln","GPC","GSH",
                             "Lac","NAA","NAAG","Ins","PCh","PCr","Tau","Asp","PE"])
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--ref-nii",         default=None)
    p.add_argument("--save-plots",      action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    out_dir  = os.path.join(args.out_dir, "spice")
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
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # ── Build NUFFT operators (backend-specific) ──────────────────────────────────
    if args.backend == "toeplitz":
        import torch
        import torchkbnufft as tkbn

        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T_D_TYPE   = torch.complex64
        print(f"[spice/toeplitz] Building Toeplitz NUFFT  device={device} …")

        osamp, ost = 2.0, 2.0
        grid_size  = (int(np.ceil(osamp * Ny)),
                      int(np.ceil(osamp * Nx)),
                      int(np.ceil(ost * T)))

        ktraj      = torch.from_numpy(trej).permute(2, 0, 1).reshape(3, -1).to(device)
        smap_torch = torch.from_numpy(coil_smap).to(device, dtype=T_D_TYPE)
        tnufft_ob  = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(device)
        tadjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(device)

        F_tkbn = NUFFTOp(
            im_size=im_size, grid_size=grid_size,
            omega=ktraj, smaps=coil_smap,
            norm="ortho", device=device,
            nufft_ob=tnufft_ob, adjnufft_ob=tadjnufft_ob,
        )

        toep_ob = tkbn.ToepNufft()
        kernel  = tkbn.calc_toeplitz_kernel(ktraj, im_size, grid_size=grid_size, norm="ortho")

        def toep_matvec(x_np):
            x_t = torch.from_numpy(x_np.astype(D_TYPE)).reshape(1, 1, *im_size).to(device, dtype=T_D_TYPE)
            return toep_ob(x_t, kernel, smaps=smap_torch, norm="ortho").squeeze().cpu().numpy().astype(D_TYPE).ravel()

        Gram_OP = LinearOperator((N_VOXEL * N_SEQ, N_VOXEL * N_SEQ), matvec=toep_matvec, dtype=D_TYPE)

        def _mv(x):  return F_tkbn.A_np(x.astype(D_TYPE).reshape(Ny, Nx, N_SEQ)).ravel()
        def _rmv(y): return F_tkbn.AH_np(y.astype(D_TYPE).reshape(NUM_CMAP, -1)).ravel()
        F_OP = LinearOperator(
            (K_POINTS * N_COILS * args.n_shots, N_VOXEL * N_SEQ),
            matvec=_mv, rmatvec=_rmv, dtype=D_TYPE,
        )
        def _fid2spec(x):
            xr = np.asarray(x).reshape(Ny, Nx, N_SEQ)
            return np.fft.fftshift(np.fft.fft(xr, axis=-1, norm='ortho'), axes=-1).ravel().astype(D_TYPE, copy=False)

        def _spec2fid(x):
            xr = np.asarray(x).reshape(Ny, Nx, N_SEQ)
            return np.fft.ifft(np.fft.ifftshift(xr, axes=-1), axis=-1, norm='ortho').ravel().astype(D_TYPE, copy=False)

        F1D = LinearOperator(
            (N_VOXEL * N_SEQ, N_VOXEL * N_SEQ),
            matvec=_fid2spec, rmatvec=_spec2fid, dtype=D_TYPE,
        )

    else:  # finufft
        print("[spice/finufft] Building finufft NUFFT operator …")
        NufftOp    = mrinufft.get_operator("finufft")
        nufft_mrsi = NufftOp(trej, shape=im_size, n_coils=NUM_CMAP,
                              n_batchs=1, squeeze_dims=True, smaps=smap_time)
        fop  = NUFFTLinearOperator(nufft_mrsi, img_shape=im_size,
                                   n_samples=mrsi_lprm.shape[1],
                                   n_coils=mrsi_lprm.shape[0])
        F_OP = fop.to_scipy()

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
    csv_path = os.path.join(out_dir, args.csv_name + ".csv")
    if args.reuse_training and os.path.exists(csv_path):
        print(f"[spice] Loading existing training data: {csv_path}")
        training_dataset = read_training_data_from_csv(out_dir, args.csv_name).astype(D_TYPE)
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
        save_training_data_as_csv(training_dataset, out_dir, args.csv_name, savecondition=True)

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

    # ── Run SPICE (backend-specific) ──────────────────────────────────────────────
    print(f"[spice/{args.backend}] Running SPICE  rank={args.rank}  λ={args.lambda1}  maxiter={args.maxiter} …")

    if args.backend == "toeplitz":
        spice_est, est_U, _ = SPICEWithSpatialConstrain_cg_nufft(
            noisy_kt_spaces  = mrsi_lprm,
            img_shape        = im_size,
            F=F_OP, Gram_OP=Gram_OP, F1D_OP=F1D,
            B0_mat=B0_mat, V=V,
            N_Vox=N_VOXEL, NUM_SPICE_RANK=args.rank,
            WW=WW, Solver="cg",
            lamda_1=args.lambda1, maxiter=args.maxiter,
            brain_mask_inner = brain_mask_inner,
            PPM_AXIS         = PPM_AXIS,
        )
    else:  # finufft
        spice_est, est_U, _ = SPICEWithSpatialConstrain_cg_finufft(
            noisy_kt_spaces=mrsi_lprm,
            img_shape=im_size,
            F=F_OP,
            B0_mat=B0_mat, V=V,
            N_Vox=N_VOXEL, NUM_SPICE_RANK=args.rank,
            WW=WW,
            lamda_1=args.lambda1, maxiter=args.maxiter,
            save_folder=os.path.join(out_dir, "cg_iters"),
        )

    print(f"[spice] Done. est shape: {spice_est.shape}")

    # ── Save raw outputs ──────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "SPICE_f.npy"), spice_est)
    np.save(os.path.join(out_dir, "U_est.npy"),   est_U)

    ref_img_path = args.ref_nii or (data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz")
    try:
        ref_img_obj = Image(ref_img_path)
        affine      = ref_img_obj.voxToWorldMat
    except Exception:
        ref_img_obj = None
        affine      = np.eye(4)

    spice_3d = spice_est.reshape(Ny, Nx, N_SEQ)

    # ── Phase correction ─────────────────────────────────────────────────────────
    print(f"[spice] Phase correction  ppmlim={args.phase_ppmlim} …")
    spice_phcorr_f = phase_corr(
        spice_3d,
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = Dim_Voxel,
        out_dir    = out_dir,
        ppmlim     = args.phase_ppmlim,
        ref_img    = ref_img_obj,
        out_fname  = "SPICE_phcorr",
    )
    spice_phcorr = FIDToSpec(spice_phcorr_f, axis=-1)

    # ── Save NIfTI-MRS ────────────────────────────────────────────────────────────
    spice_save = spice_phcorr_f.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(spice_save.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "SPICE_result.nii.gz"))
    print("[spice] Saved SPICE_result.nii.gz")

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
