#!/usr/bin/env python3
"""
Step 6 — Iterative NUFFT reconstruction (CG, with B0 correction).

Solves (A^H A) x = A^H y  where A = F_NUFFT @ F1D @ B0_diag.
Result x is in FID domain, saved as NIfTI-MRS for downstream use
(xcorr alignment + spectral fitting in step 05).

Backends:
  torchnufft (default) : torchkbnufft + Toeplitz Gram  — fast per CG iter
  finufft            : mrinufft finufft, Gram = F.H@F — no torch dep, slower

Reads  : <out_dir>/lipid_removal/kt_mrsi_lprm.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <data_dir>/wref_o.npy
Writes : <out_dir>/iter_recon/iter_recon.nii.gz         (FID, NIfTI-MRS)
         <out_dir>/iter_recon/iter_recon_phcorr.nii.gz  (phase-corrected FID)
         <out_dir>/iter_recon/adjoint.nii.gz             (b_init FID, NIfTI-MRS)
         <out_dir>/iter_recon/iter_recon.npy              (raw FID array)
         <out_dir>/iter_recon/fig_06_*.png

Usage:
    python scripts/recon_method/recon_03_iterative_nufft_recon.py \
        --data-dir data/processed/invivo_250305_01 \
        [--maxiter 150] [--solver cg]
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

from fsl.data.image import Image
from nifti_mrs.create_nmrs import gen_nifti_mrs
from fsl_mrs.utils.misc import FIDToSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.scan_params import load_scan_params
from utils.utils import (
    Calc_B0_matrix,
    NUFFTOp,
    NUFFTLinearOperator,
    iterative_nufft_recon,
    phase_corr,
    plot_voxel_spectrum_and_maps,
)


def parse_args():
    p = argparse.ArgumentParser(description="Iterative NUFFT reconstruction — step 6")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--out-dir",         default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--backend",         default="torchnufft",
                   choices=["torchnufft", "finufft"])
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int, default=None)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--n-coils",         type=int, default=None)
    p.add_argument("--n-shots",         type=int,   default=360)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--maxiter",         type=int,   default=150)
    p.add_argument("--rtol",            type=float, default=1e-3)
    p.add_argument("--solver",          default="cg", choices=["cg", "bicgstab"])
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--phase-ppmlim",    type=float, nargs=2, default=[0.0, 5.0],
                   metavar=("LO", "HI"), help="ppm window for phase correction")
    p.add_argument("--voxel-x",         type=int,   default=32)
    p.add_argument("--voxel-y",         type=int,   default=32)
    p.add_argument("--ref-nii",         default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    data_dir    = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    lprm_dir    = os.path.join(args.out_dir, "lipid_removal")
    coilmap_dir = os.path.join(args.out_dir, "coilmap")
    b0map_dir   = os.path.join(args.out_dir, "b0map")
    out_dir     = os.path.join(args.out_dir, "iter_recon")
    os.makedirs(out_dir, exist_ok=True)

    D_TYPE      = np.complex64
    Trej_D_TYPE = np.float32

    K_POINTS  = args.k_points
    N_SEQ     = args.n_seq_points
    N_COILS   = args.n_coils
    Ny, Nx    = args.dim[0], args.dim[1]
    N_VOXEL   = Ny * Nx
    im_size   = (Ny, Nx, N_SEQ)

    TS          = (K_POINTS / N_SEQ) * args.dwelltime
    sweepwidth  = 1.0 / TS
    center_freq = args.center_freq
    PPM_CENTER  = args.ppm_center
    FREQ_AXIS   = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS    = FREQ_AXIS / center_freq + PPM_CENTER
    TIME_AXIS   = np.linspace(TS, TS * N_SEQ, N_SEQ)
    print(f"[iter-recon/{args.backend}] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[iter-recon] Loading data …")
    mrsi_lprm       = np.load(os.path.join(lprm_dir,    "kt_mrsi_lprm.npy"),    mmap_mode="r").astype(D_TYPE)
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir,    "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"),        mmap_mode="r")
    B0_map          = np.load(os.path.join(b0map_dir,   "B0_map.npy"))
    wref_img        = np.load(data_dir + "wref_o.npy", mmap_mode="r")

    trej     = mrsi_ksp_scaled.T.astype(Trej_D_TYPE)
    NUM_CMAP = coil_smap_raw.shape[0]

    # ── Brain mask ────────────────────────────────────────────────────────────
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # ── Build NUFFT operators ─────────────────────────────────────────────────
    if args.backend == "torchnufft":
        import torch
        import torchkbnufft as tkbn

        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T_D_TYPE = torch.complex64
        print(f"[iter-recon/toeplitz] Building Toeplitz NUFFT  device={device} …")

        osamp, ost = 2.0, 2.0
        grid_size  = (int(np.ceil(osamp * Ny)),
                      int(np.ceil(osamp * Nx)),
                      int(np.ceil(ost * N_SEQ)))

        ktraj = torch.from_numpy(trej).permute(2, 0, 1).reshape(3, -1).to(device)

        coil_smap  = np.repeat(
            coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1
        ).astype(D_TYPE)
        smap_torch = torch.from_numpy(coil_smap).to(device, dtype=T_D_TYPE)

        tnufft_ob    = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(device)
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
        n_ksamples = ktraj.shape[-1]   # = K_POINTS * n_shots (all traj points)
        F_OP = LinearOperator(
            (n_ksamples * N_COILS, N_VOXEL * N_SEQ),
            matvec=_mv, rmatvec=_rmv, dtype=D_TYPE,
        )

        def _fid2spec(x):
            xr = np.asarray(x).reshape(Ny, Nx, N_SEQ)
            return np.fft.fftshift(np.fft.fft(xr, axis=-1, norm="ortho"), axes=-1).ravel().astype(D_TYPE, copy=False)

        def _spec2fid(x):
            xr = np.asarray(x).reshape(Ny, Nx, N_SEQ)
            return np.fft.ifft(np.fft.ifftshift(xr, axes=-1), axis=-1, norm="ortho").ravel().astype(D_TYPE, copy=False)

        F1D = LinearOperator(
            (N_VOXEL * N_SEQ, N_VOXEL * N_SEQ),
            matvec=_fid2spec, rmatvec=_spec2fid, dtype=D_TYPE,
        )

    else:  # finufft
        print("[iter-recon/finufft] Building finufft NUFFT operator …")
        coil_smap = np.repeat(
            coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1
        ).astype(D_TYPE)
        smap_time = coil_smap.squeeze(0)

        NufftOp    = mrinufft.get_operator("finufft")
        nufft_mrsi = NufftOp(trej, shape=im_size, n_coils=NUM_CMAP,
                              n_batchs=1, squeeze_dims=True, smaps=smap_time)
        fop  = NUFFTLinearOperator(nufft_mrsi, img_shape=im_size,
                                   n_samples=mrsi_lprm.shape[1],
                                   n_coils=mrsi_lprm.shape[0])
        F_OP = fop.to_scipy()

        def gram_mv(x):
            return F_OP.rmatvec(F_OP.matvec(x.astype(D_TYPE)))
        Gram_OP = LinearOperator((N_VOXEL * N_SEQ, N_VOXEL * N_SEQ), matvec=gram_mv, dtype=D_TYPE)

        F1D = LinearOperator(
            (N_VOXEL * N_SEQ, N_VOXEL * N_SEQ),
            matvec=lambda x: x, rmatvec=lambda x: x, dtype=D_TYPE,
        )

    # ── B0 modulation matrix ──────────────────────────────────────────────────
    print("[iter-recon] Building B0 matrix …")
    B0_map_clean = np.nan_to_num(-B0_map, nan=0.0)
    B0_mat = Calc_B0_matrix(B0_map_clean, TIME_AXIS).reshape(N_VOXEL, N_SEQ)
    B0_FAKE = np.ones_like(B0_mat) # for no b0 correction needed

    # ── Reference NIfTI affine ────────────────────────────────────────────────
    # Priority: (1) --ref-nii arg, (2) affine.npy saved by data_proc_01_twix2npy,
    # (3) subject-specific reference NIfTI in data_dir, (4) identity
    _affine_npy = data_dir + "affine.npy"
    if args.ref_nii:
        ref_img_obj = Image(args.ref_nii)
        affine      = ref_img_obj.voxToWorldMat
    elif os.path.exists(_affine_npy):
        affine      = np.load(_affine_npy)
        ref_img_obj = None
        print(f"[iter-recon] Loaded affine from {_affine_npy}")
    else:
        ref_nii = data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz"
        try:
            ref_img_obj = Image(ref_nii)
            affine      = ref_img_obj.voxToWorldMat
        except Exception:
            ref_img_obj = None
            affine      = np.eye(4)

    # ── Run iterative NUFFT recon ─────────────────────────────────────────────
    print(f"[iter-recon] Running CG  maxiter={args.maxiter}  solver={args.solver} …")
    recon, diagnostics, b_init = iterative_nufft_recon(
        kspace       = mrsi_lprm,
        image_shape  = im_size,
        B0_mat       = B0_mat,
        F_OP         = F_OP,
        Gram_OP      = Gram_OP,
        F1D_OP       = F1D,
        n_coils      = N_COILS,
        maxiter      = args.maxiter,
        rtol         = args.rtol,
        solver       = args.solver,
    )
    print(f"[iter-recon] Done  info={diagnostics['info']}")

    recon_3d  = recon.reshape(Ny, Nx, N_SEQ)    # FID
    b_init_3d = b_init.reshape(Ny, Nx, N_SEQ)   # FID (adjoint image)

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "iter_recon.npy"), recon_3d)
    np.save(os.path.join(out_dir, "b_init.npy"),     b_init_3d)
    print("[iter-recon] Saved iter_recon.npy + b_init.npy")

    # ── Save as NIfTI-MRS (FID) ───────────────────────────────────────────────
    for fname, data in [("iter_recon.nii.gz", recon_3d), ("adjoint.nii.gz", b_init_3d)]:
        nii_data = np.ascontiguousarray(data.transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :]  # (Nx, Ny, 1, N_SEQ)
        gen_nifti_mrs(nii_data.conj(), dwelltime=TS, spec_freq=297.219, affine=affine).save(
            os.path.join(out_dir, fname))
        print(f"[iter-recon] Saved {fname}")

    # ── Phase correction ──────────────────────────────────────────────────────
    print(f"[iter-recon] Phase correction  ppmlim={args.phase_ppmlim} …")
    recon_phcorr = phase_corr(
        recon_3d,
        mag_map_2d = wref_2d,
        brain_mask = brain_mask_inner,
        TS         = TS,
        img_shape  = (Ny, Nx),
        out_dir    = out_dir,
        ppmlim     = tuple(args.phase_ppmlim),
        ref_img    = ref_img_obj,
        out_fname  = "iter_recon_phcorr",
    )
    np.save(os.path.join(out_dir, "iter_recon_phcorr.npy"), recon_phcorr)
    print("[iter-recon] Saved iter_recon_phcorr.npy")

    # ── Visualise recon before phase correction ───────────────────────────────
    _, fig_recon, _ = plot_voxel_spectrum_and_maps(
        FIDToSpec(recon_3d, axis=-1), im_size,
        voxel_x=args.voxel_x, voxel_y=args.voxel_y,
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS, show=False,
    )
    fig_recon.savefig(os.path.join(out_dir, "fig_06a_iter_recon.png"), dpi=120)
    plt.close(fig_recon)

    # ── Visualise after phase correction ─────────────────────────────────────
    _, fig_phcorr, _ = plot_voxel_spectrum_and_maps(
        FIDToSpec(recon_phcorr, axis=-1), im_size,
        voxel_x=args.voxel_x, voxel_y=args.voxel_y,
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS, show=False,
    )
    fig_phcorr.savefig(os.path.join(out_dir, "fig_06b_phcorr.png"), dpi=120)
    plt.close(fig_phcorr)

    # ── Visualise adjoint (b_init) spectrum ───────────────────────────────────
    _, fig_adj, _ = plot_voxel_spectrum_and_maps(
        FIDToSpec(b_init_3d, axis=-1), im_size,
        voxel_x=args.voxel_x, voxel_y=args.voxel_y,
        brain_mask_inner=brain_mask_inner,
        PPM_AXIS=PPM_AXIS, show=False,
    )
    fig_adj.savefig(os.path.join(out_dir, "fig_06c_adjoint.png"), dpi=120)
    plt.close(fig_adj)

    print("[iter-recon] Done.")


if __name__ == "__main__":
    main()
