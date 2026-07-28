#!/usr/bin/env python3
"""
Step 5 — Adjoint NUFFT reconstruction from kt_mrsi_lprm.npy.

No Gram operator — just F^H @ y. Saves the result as NIfTI-MRS and plots
spatial magnitude + mean spectrum over brain voxels. Use this to verify that
kt_mrsi_lprm.npy and the trajectory are consistent.

Outputs (in <out-dir>/adjoint_test/):
  adj_recon.nii.gz          — adjoint recon in spectrum domain
  fig_adj_magnitude.png     — spatial mean-|spectrum| map
  fig_adj_spectrum.png      — mean spectrum over brain voxels
  fig_adj_center_voxel.png  — single center-voxel spectrum

Usage:
    python scripts/recon_method/recon_02_adjoint_recon.py \
        --data-dir data/processed/invivo_250305_01 --basis-dir ./basis/ \
        --rank 20 --dim 64 64 --n-seq-points 300
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from warnings import filterwarnings
filterwarnings("ignore")

import mrinufft
import torch
import torchkbnufft as tkbn
from scipy.sparse.linalg import LinearOperator
from fsl.data.image import Image
from nifti_mrs.create_nmrs import gen_nifti_mrs
from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.misc import FIDToSpec, SpecToFID
from fsl_mrs.utils.synthetic import syntheticFromBasisFile
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs as gen_nifti_mrs_fsl

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from utils.scan_params import load_scan_params
from utils.xcorr import my_mrsi_freq_align
from utils.pipeline_utils import patch_spicefit_tree
from utils.utils import NUFFTOp, plot_voxel_spectrum_and_maps, read_training_data_from_csv, phase_corr, Calc_B0_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Adjoint NUFFT diagnostic")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--out-dir",         default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int, default=None)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--n-coils",         type=int, default=None)
    p.add_argument("--n-shots",         type=int,   default=360)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64], metavar=("NX", "NY"))
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--backend",         default="finufft", choices=["torchnufft", "finufft"])
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--rank",            type=int,   default=20)
    p.add_argument("--csv-name",        default="SS_training")
    p.add_argument("--align-method",    default="xcorr", choices=["xcorr", "phase_corr"],
                   help="Alignment method: xcorr (freq align) or phase_corr (0th-order phase)")
    p.add_argument("--basis-dir",       default=None,
                   help="FSL-MRS basis directory (required for --align-method xcorr)")
    p.add_argument("--phase-ppmlim",    type=float, nargs=2, default=[3.5, 5.0],
                   metavar=("LO", "HI"), help="ppm window for phase_corr (default: 0 5)")
    p.add_argument("--ref-nii",         default=None)
    p.add_argument("--no-b0",           action="store_true",
                   help="Skip B0 correction (use fake all-ones matrix) for testing")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir    = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    lprm_dir    = os.path.join(args.out_dir, "lipid_removal")
    coilmap_dir = os.path.join(args.out_dir, "coilmap")
    b0map_dir   = os.path.join(args.out_dir, "b0map")
    out_dir     = os.path.join(args.out_dir, "adjoint_test")
    os.makedirs(out_dir, exist_ok=True)

    D_TYPE   = np.complex64
    T_D_TYPE = torch.complex64

    K_POINTS  = args.k_points
    N_SEQ     = args.n_seq_points
    N_COILS   = args.n_coils
    Dim_Voxel = args.dim
    Ny, Nx    = Dim_Voxel[0], Dim_Voxel[1]
    im_size   = (Ny, Nx, N_SEQ)

    TS          = (K_POINTS / N_SEQ) * args.dwelltime
    sweepwidth  = 1.0 / TS
    FREQ_AXIS   = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS    = FREQ_AXIS / args.center_freq + args.ppm_center
    TIME_AXIS   = np.linspace(TS, TS * N_SEQ, N_SEQ)
    print(f"[adj-test] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s")

    # ── Load data ────────────────────────────────────────────────────────────────
    print("[adj-test] Loading data …")
    mrsi_lprm       = np.load(os.path.join(lprm_dir,    "kt_mrsi_lprm.npy"),    mmap_mode="r").astype(D_TYPE)
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir,    "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"),        mmap_mode="r")
    B0_map          = np.load(os.path.join(b0map_dir,   "B0_map.npy"))
    wref_img        = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    print(f"[adj-test] kt_mrsi_lprm  shape={mrsi_lprm.shape}  dtype={mrsi_lprm.dtype}")
    print(f"[adj-test] mrsi_ksp_scaled shape={mrsi_ksp_scaled.shape}")
    print(f"[adj-test] coil_smap_raw  shape={coil_smap_raw.shape}")

    # ── Brain mask ───────────────────────────────────────────────────────────────
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    NUM_CMAP = coil_smap_raw.shape[0]
    N_VOXEL  = Ny * Nx
    trej     = mrsi_ksp_scaled.T.astype(np.float32)   # (N_shots, K, 3)

    # ── B0 modulation matrix ─────────────────────────────────────────────────────
    print("[adj-test] Building B0 matrix …")
    B0_map_clean = np.nan_to_num(B0_map, nan=0.0)
    B0_mat  = Calc_B0_matrix(B0_map_clean, TIME_AXIS).reshape(Ny, Nx, N_SEQ)  # (Ny, Nx, N_SEQ)
    B0_FAKE = np.ones((Ny, Nx, N_SEQ), dtype=B0_mat.dtype)                    # no B0 correction

    if args.backend == "torchnufft":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[adj-test] Building toeplitz NUFFT  device={device} …")

        ktraj = torch.from_numpy(trej).permute(2, 0, 1).reshape(3, -1).to(device)
        print(f"[adj-test] ktraj shape={tuple(ktraj.shape)}")

        osamp, ost = 2.0, 2.0
        grid_size  = (int(np.ceil(osamp * Ny)),
                      int(np.ceil(osamp * Nx)),
                      int(np.ceil(ost * N_SEQ)))

        coil_smap  = np.repeat(
            coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1
        ).astype(D_TYPE)   # (1, C, Ny, Nx, N_SEQ)

        tnufft_ob    = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(device)
        tadjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(device)

        F_tkbn = NUFFTOp(
            im_size=im_size, grid_size=grid_size,
            omega=ktraj, smaps=coil_smap,
            norm="ortho", device=device,
            nufft_ob=tnufft_ob, adjnufft_ob=tadjnufft_ob,
        )

        n_ksamples = ktraj.shape[-1]

        def _mv(x):  return F_tkbn.A_np(x.astype(D_TYPE).reshape(Ny, Nx, N_SEQ)).ravel()
        def _rmv(y): return F_tkbn.AH_np(y.astype(D_TYPE).reshape(NUM_CMAP, -1)).ravel()

        F_OP = LinearOperator(
            (n_ksamples * NUM_CMAP, N_VOXEL * N_SEQ),
            matvec=_mv, rmatvec=_rmv, dtype=D_TYPE,
        )

    else:  # finufft — same as 03_lipid_removal.py (nufft_mrsi.adj_op)
        print("[adj-test] Building finufft NUFFT operator …")

        # Cell 11: coilmap → (1, C, Ny, Nx, T) → squeeze → (C, Ny, Nx, T)
        NUM_CMAP_CHANNEL = coil_smap_raw.shape[0]
        coil_smap = np.repeat(
            coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1
        ).astype(D_TYPE)                          # (1, C, Ny, Nx, T)
        smap_time = coil_smap.squeeze(0)          # (C, Ny, Nx, T)

        # Cell 15: finufft NUFFT operator
        NufftOperator = mrinufft.get_operator("finufft")
        nufft_mrsi = NufftOperator(
            trej, shape=im_size,
            n_coils=NUM_CMAP_CHANNEL, n_batchs=1,
            squeeze_dims=True,
            smaps=smap_time,
        )

    # ── Adjoint reconstruction ────────────────────────────────────────────────────
    b0_corr_mat = B0_FAKE if args.no_b0 else B0_mat   # (Ny, Nx, N_SEQ)
    print(f"[adj-test/{args.backend}] Running adjoint NUFFT  b0={'fake' if args.no_b0 else 'real'} …")
    if args.backend == "torchnufft":
        adj_raw   = (F_OP.H @ mrsi_lprm.ravel().astype(D_TYPE)).reshape(Ny, Nx, N_SEQ)
    else:  # finufft — adj_op expects (1, N_COILS, n_total_samples)
        adj_raw   = nufft_mrsi.adj_op(mrsi_lprm.reshape(1, NUM_CMAP, -1))   # (Ny, Nx, N_SEQ) spectrum
    # B0 correction: FID domain element-wise multiply by conj(B0_mat) to undo phase
    fid_raw   = SpecToFID(adj_raw, axis=-1)                                  # (Ny, Nx, N_SEQ)
    fid_b0    = fid_raw * b0_corr_mat.conj()                                 # (Ny, Nx, N_SEQ)
    image_adj = FIDToSpec(fid_b0, axis=-1)                                   # (Ny, Nx, N_SEQ)
    print(f"[adj-test] image_adj  shape={image_adj.shape}  |max|={np.abs(image_adj).max():.4e}")

    # ── Save as NIfTI-MRS ─────────────────────────────────────────────────────────
    # Priority: (1) --ref-nii arg, (2) affine.npy saved by data_proc_01_twix2npy,
    # (3) subject-specific reference NIfTI in data_dir, (4) identity
    _affine_npy = data_dir + "affine.npy"
    if args.ref_nii:
        ref_img_obj = Image(args.ref_nii)
        affine      = ref_img_obj.voxToWorldMat
    elif os.path.exists(_affine_npy):
        affine      = np.load(_affine_npy)
        ref_img_obj = None
        print(f"[adj-test] Loaded affine from {_affine_npy}")
    else:
        ref_img_path = data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz"
        try:
            ref_img_obj = Image(ref_img_path)
            affine      = ref_img_obj.voxToWorldMat
        except Exception:
            ref_img_obj = None
            affine      = np.eye(4)

    # ── Save raw adjoint as NIfTI-MRS (FID) ──────────────────────────────────
    _fid_lr   = np.ascontiguousarray(SpecToFID(image_adj, axis=-1).transpose(1, 0, 2)[::-1, :, :])
    nii_data  = _fid_lr[:, :, np.newaxis, :]  # (Nx, Ny, 1, N_SEQ); ascontiguousarray avoids negative-stride issues in gen_nifti_mrs
    nifti_adj = gen_nifti_mrs(nii_data, dwelltime=TS, spec_freq=297.219, affine=affine)
    nifti_adj.save(os.path.join(out_dir, "adj_recon.nii.gz"))
    print("[adj-test] Saved adj_recon.nii.gz")

    # ── Alignment ─────────────────────────────────────────────────────────────
    adj_fid = SpecToFID(image_adj, axis=-1)   # (Ny, Nx, N_SEQ) FID

    if args.align_method == "xcorr" and args.basis_dir is None:
        print("[adj-test] WARNING: --basis-dir not provided; falling back to phase_corr")
        args.align_method = "phase_corr"

    if args.align_method == "xcorr":
        print("[adj-test] Building basis_nmrs for xcorr …")
        fullbasis = mrs_io.read_basis(args.basis_dir)
        fid_ref, emptymrs, _ = syntheticFromBasisFile(
            fullbasis, noisecovariance=[[0]], bandwidth=sweepwidth, points=N_SEQ)
        basis_nmrs = gen_nifti_mrs_fsl(
            fid_ref.conj().reshape(1, 1, 1, N_SEQ),
            dwelltime=emptymrs.dwellTime,
            spec_freq=emptymrs.centralFrequency,
            affine=affine,
        )
        print("[adj-test] Running xcorr frequency alignment …")
        aligned_nmrs, _ = my_mrsi_freq_align(nifti_adj, basis_nmrs)
        aligned_nmrs.save(os.path.join(out_dir, "adj_recon_aligned.nii.gz"))
        print("[adj-test] Saved adj_recon_aligned.nii.gz")
        aligned_fid = np.array(aligned_nmrs.image[:, :, 0, :]).transpose(1, 0, 2)

    else:  # phase_corr
        print(f"[adj-test] Running phase_corr  ppmlim={args.phase_ppmlim} …")
        aligned_fid = phase_corr(
            adj_fid,
            mag_map_2d = wref_2d,
            brain_mask = brain_mask_inner,
            TS         = TS,
            img_shape  = (Ny, Nx),
            out_dir    = out_dir,
            ppmlim     = tuple(args.phase_ppmlim),
            ref_img    = ref_img_obj,
            out_fname  = "adj_recon_phcorr",
        )
        print("[adj-test] Saved adj_recon_phcorr.nii.gz")

    # ── Plot: spatial magnitude ───────────────────────────────────────────────────
    mag_map = np.mean(np.abs(image_adj), axis=-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mag_map, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title("Adjoint recon — spatial magnitude (mean over ppm)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_adj_magnitude.png"), dpi=120)
    plt.close(fig)
    print("[adj-test] Saved fig_adj_magnitude.png")

    # ── Plot: voxel (16, 31) spectrum ─────────────────────────────────────────────
    vy, vx = 32, 32
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(PPM_AXIS, np.abs(image_adj[vy, vx, :]))
    ax.set_xlabel("ppm")
    ax.set_ylabel("|spectrum|")
    ax.set_title(f"Adjoint recon — voxel [{vy}, {vx}]")
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_adj_spectrum.png"), dpi=120)
    plt.close(fig)
    print("[adj-test] Saved fig_adj_spectrum.png")

    # ── Plot: aligned spectrum ────────────────────────────────────────────────
    _, fig_aln, _ = plot_voxel_spectrum_and_maps(
        FIDToSpec(aligned_fid, axis=-1), im_size,
        voxel_x=32, voxel_y=32,
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS, show=False,
    )
    fig_aln.savefig(os.path.join(out_dir, "fig_adj_aligned.png"), dpi=120)
    plt.close(fig_aln)
    print("[adj-test] Saved fig_adj_aligned.png")

    # ── SVD from SS training dataset → V → project adjoint ───────────────────────
    spice_dir = os.path.join(args.out_dir, "spice")
    csv_path  = os.path.join(args.basis_dir, args.csv_name + ".csv")
    if os.path.exists(csv_path):
        print(f"[adj-test] Loading SS training data: {csv_path} …")
        training_dataset = read_training_data_from_csv(args.basis_dir, args.csv_name).astype(np.complex64)
        print(f"[adj-test] training_dataset shape={training_dataset.shape}")

        print(f"[adj-test] SVD (rank={args.rank}) …")
        _, s, Vh = np.linalg.svd(training_dataset)
        V = Vh[:args.rank, :].conj().T          # (N_SEQ, rank)
        print(f"[adj-test] V shape={V.shape}  top singular values: {s[:10].round(2)}")

        # plot singular value curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(s[:30], "x-")
        ax.set_title("Singular values (top 30)")
        ax.set_xlabel("rank")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_adj_singular_values.png"), dpi=120)
        plt.close(fig)

        # project adjoint image onto subspace
        adjoint_flat = SpecToFID(image_adj.reshape(N_VOXEL, N_SEQ),axis=-1)           # (N_Vox, N_SEQ)
        U_init       = adjoint_flat @ np.linalg.pinv(V.conj().T)   # (N_Vox, rank)
        spice_recon  = FIDToSpec(
            (U_init @ V.conj().T).reshape(Ny, Nx, N_SEQ), axis=-1  # (Ny, Nx, N_SEQ) spectrum
        )
        print(f"[adj-test] U_init shape={U_init.shape}")

        _, fig_init, _ = plot_voxel_spectrum_and_maps(
            spice_recon, (Ny, Nx, N_SEQ),
            voxel_x=32, voxel_y=32,
            brain_mask_inner=brain_mask_inner,
            PPM_AXIS=PPM_AXIS, show=False,
        )
        fig_init.savefig(os.path.join(out_dir, "fig_adj_V_projection.png"), dpi=120)
        plt.close(fig_init)
        print("[adj-test] Saved fig_adj_V_projection.png")
    else:
        print(f"[adj-test] {csv_path} not found — skipping SVD projection")

    patch_spicefit_tree(args.out_dir, [
        ("adj_recon.nii.gz", os.path.join(out_dir, "adj_recon.nii.gz"), "data-adj-recon"),
        ("adj_recon_aligned.nii.gz", os.path.join(out_dir, "adj_recon_aligned.nii.gz"), "data-adj-aligned"),
    ])
    print("[adj-test] Done.")


if __name__ == "__main__":
    main()
