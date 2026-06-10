#!/usr/bin/env python3
"""
Diagnostic: adjoint NUFFT reconstruction from kt_mrsi_lprm.npy.

No Gram operator — just F^H @ y. Saves the result as NIfTI-MRS and plots
spatial magnitude + mean spectrum over brain voxels. Use this to verify that
kt_mrsi_lprm.npy and the trajectory are consistent.

Outputs (in <out-dir>/adjoint_test/):
  adj_recon.nii.gz          — adjoint recon in spectrum domain
  fig_adj_magnitude.png     — spatial mean-|spectrum| map
  fig_adj_spectrum.png      — mean spectrum over brain voxels
  fig_adj_center_voxel.png  — single center-voxel spectrum

Usage:
    python scripts/test_adjoint_recon.py \\
        --data-dir ./invivo_260305/cr/ \\
        [--out-dir ./output] \\
        [--k-points 39762] [--n-seq-points 300] [--n-coils 32] [--n-shots 360] \\
        [--dim 64 64] [--brain-threshold 0.08]
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
from fsl_mrs.utils.misc import FIDToSpec, SpecToFID

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spice_mrsi.utils import NUFFTOp, plot_voxel_spectrum_and_maps, read_training_data_from_csv


def parse_args():
    p = argparse.ArgumentParser(description="Adjoint NUFFT diagnostic")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--out-dir",         default="./output")
    p.add_argument("--dwelltime",       type=float, default=5e-6)
    p.add_argument("--k-points",        type=int,   default=39842)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--n-coils",         type=int,   default=32)
    p.add_argument("--n-shots",         type=int,   default=360)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64], metavar=("NX", "NY"))
    p.add_argument("--center-freq",     type=float, default=297.219338)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--backend",         default="toeplitz", choices=["toeplitz", "finufft"])
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--rank",            type=int,   default=20)
    p.add_argument("--csv-name",        default="SS_training")
    p.add_argument("--ref-nii",         default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir    = args.data_dir.rstrip("/") + "/"
    lprm_dir    = os.path.join(args.out_dir, "lipid_removal")
    coilmap_dir = os.path.join(args.out_dir, "coilmap")
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
    print(f"[adj-test] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s")

    # ── Load data ────────────────────────────────────────────────────────────────
    print("[adj-test] Loading data …")
    mrsi_lprm       = np.load(os.path.join(lprm_dir,    "kt_mrsi_lprm.npy"),    mmap_mode="r").astype(D_TYPE)
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir,    "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"),        mmap_mode="r")
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

    if args.backend == "toeplitz":
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
    print(f"[adj-test/{args.backend}] Running adjoint NUFFT …")
    if args.backend == "toeplitz":
        image_adj = (F_OP.H @ mrsi_lprm.ravel().astype(D_TYPE)).reshape(Ny, Nx, N_SEQ)
    else:  # finufft: direct adj_op, same as 03 image_blurry_numpy = nufft_mrsi.adj_op(mrsi_reordered)
        image_adj = nufft_mrsi.adj_op(mrsi_lprm)   # (Ny, Nx, N_SEQ) spectrum
    print(f"[adj-test] image_adj  shape={image_adj.shape}  |max|={np.abs(image_adj).max():.4e}")

    # ── Save as NIfTI-MRS ─────────────────────────────────────────────────────────
    ref_img_path = args.ref_nii or (data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz")
    try:
        affine = Image(ref_img_path).voxToWorldMat
    except Exception:
        affine = np.eye(4)

    nii_data = SpecToFID(image_adj,axis=-1).transpose(1, 0, 2)[:, :, np.newaxis, :]  # (Nx, Ny, 1, N_SEQ)
    gen_nifti_mrs(nii_data, dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "adj_recon.nii.gz"))
    print("[adj-test] Saved adj_recon.nii.gz")

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


    # ── SVD from SS training dataset → V → project adjoint ───────────────────────
    spice_dir = os.path.join(args.out_dir, "spice")
    csv_path  = os.path.join(spice_dir, args.csv_name + ".csv")
    if os.path.exists(csv_path):
        print(f"[adj-test] Loading SS training data: {csv_path} …")
        training_dataset = read_training_data_from_csv(spice_dir, args.csv_name).astype(np.complex64)
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

    print("[adj-test] Done.")


if __name__ == "__main__":
    main()
