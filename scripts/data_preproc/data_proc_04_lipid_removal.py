#!/usr/bin/env python3
"""
Step 3 — Lipid removal (GMM + L2 only, no B0/xcorr correction).

Reads  : <data_dir>/mrsi_data.npy, mrsi_ksp.npy, wref_o.npy
         <out_dir>/coilmap/ecalib_pp.npy
Writes : <out_dir>/lipid_removal/kt_mrsi_lprm.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/lipid_removal/adj_bf_lprm.nii.gz
         <out_dir>/lipid_removal/my_mrsi_lprm_f.nii.gz
         <out_dir>/lipid_removal/V_lipid.npy           (SVD-truncated lipid spectral subspace)
         <out_dir>/lipid_removal/lss_map.npy           (raw LSS map, for 04b's in-brain GMM/W_lip)
         <out_dir>/lipid_removal/kt_mrsi_withlip_noring.npy
         <out_dir>/lipid_removal/fig_03b_*.png  (when --save-plots)

Usage:
    python scripts/data_preproc/data_proc_04_lipid_removal.py \
        --data-dir        data/processed/invivo_250305_01 \
        --k-points        39762 \
        --n-seq-points    300 \
        --n-coils         32 \
        --dim             64 64 \
        --ppm-center      3.027 \
        --brain-threshold 0.16 \
        --lipid-beta      200.0 \
        --n-lipid-voxels  500 \
        --nsigma-gmm      0.2 \
        --lipid-rank      10 \
        --topn-fallback   100 \
        --phase-ppmlim    3.5 3.9 \
        --plot-voxel      41 24 \
        --save-plots \
        --phase-method none

"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import binary_erosion
from warnings import filterwarnings
filterwarnings("ignore")

import mrinufft
from fsl_mrs.utils.misc import FIDToSpec, SpecToFID
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl.data.image import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.scan_params import load_scan_params
from utils.utils import phase_corr
from utils.lipid import compute_lss, select_lipid_mask_gmm_simple
from utils.pipeline_utils import make_brain_mask, patch_spicefit_tree


def parse_args():
    p = argparse.ArgumentParser(description="Lipid removal (GMM+L2 only) — step 3b")
    p.add_argument("--data-dir",         required=True)
    p.add_argument("--out-dir",          default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--dwelltime",        type=float, default=None)
    p.add_argument("--k-points",         type=int, default=None)
    p.add_argument("--n-seq-points",     type=int,   default=300)
    p.add_argument("--n-coils",          type=int, default=None)
    p.add_argument("--dim",              type=int,   nargs=2, default=[64, 64], metavar=("NX", "NY"))
    p.add_argument("--center-freq",      type=float, default=None)
    p.add_argument("--ppm-center",       type=float, default=3.027)
    p.add_argument("--mrsi-ksp-scale",   type=float, default=None)
    # brain masks
    p.add_argument("--brain-threshold",  type=float, default=0.07,
                   help="Normalized wref_o threshold for brain mask (default 0.07)")
    p.add_argument("--brain-erosion",    type=int,   default=3)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component (drops disconnected noise blobs outside "
                        "the brain) and fill enclosed holes (e.g. a central signal void). "
                        "Default: off, use only if a single global threshold isn't enough.")
    # lipid removal
    p.add_argument("--lss-ppm-low",      type=float, default=0.7)
    p.add_argument("--lss-ppm-high",     type=float, default=1.8)
    p.add_argument("--lipid-beta",       type=float, default=200.0)
    p.add_argument("--n-lipid-voxels",   type=int,   default=500)
    p.add_argument("--nsigma-gmm",       type=float, default=0.2,
                   help="nsigma for the ring-extraction LSS GMM (selects lipid-basis voxels for SVD)")
    p.add_argument("--topn-fallback",    type=int,   default=100,
                   help="Top-N voxel fallback when the ring-extraction GMM threshold "
                        "selects more than --n-lipid-voxels")
    p.add_argument("--lipid-rank",       type=int,   default=5,
                   help="SVD truncation rank for V_lipid (spectral subspace), default 5")
    p.add_argument("--phantom",          action="store_true",
                   help="Phantom scan: no lipid signal present, so skip LSS/GMM lipid "
                        "identification and L2 removal entirely. The adjoint-NUFFT data is "
                        "passed straight through and saved under the usual lipid_removal/ "
                        "filenames (with zeroed V_lipid/lss_map placeholders) so downstream "
                        "steps run unchanged. Default: off.")
    # phase correction
    p.add_argument("--phase-ppmlim",     type=float, nargs=2, default=[3.5, 3.9],
                   metavar=("LO", "HI"))
    p.add_argument("--phase-method",     type=str,   default="phasta",
                   help="phase correction method: 'phasta', 'max-real', 'xcorr-phase', or 'none' "
                        "(none: skip phase correction entirely; xcorr-phase: xcorr against "
                        "brain-mean, apply only 0th-order phase, leave frequency shift for B0 "
                        "in step 04) (default: phasta)")
    # misc
    p.add_argument("--plot-voxel",       type=int,   nargs=2, default=[41, 24],
                   metavar=("ROW", "COL"))
    p.add_argument("--ref-nii",          default=None)
    p.add_argument("--save-plots",       action="store_true")
    return p.parse_args()


def lipid_removal_l2(data, lipid_basis, beta):
    nx, ny, nz, npts = data.shape
    Linv = np.linalg.inv(np.eye(npts) + beta * (lipid_basis @ lipid_basis.conj().T))
    out  = np.zeros_like(data)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                out[ix, iy, iz, :] = Linv @ data[ix, iy, iz, :]
    return out


def plot_mag_and_voxel(spec_4d, PPM_AXIS, voxel_row, voxel_col, title, fname):
    mag_map = np.mean(np.abs(spec_4d), axis=-1)[:, :, 0]
    spec_v  = spec_4d[voxel_row, voxel_col, 0, :].astype(np.complex64)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    im = axs[0].imshow(np.abs(mag_map), origin="lower", cmap="viridis")
    axs[0].set_title("Avg spectral magnitude")
    plt.colorbar(im, ax=axs[0], fraction=0.046)
    axs[0].add_patch(Rectangle((voxel_col - 0.5, voxel_row - 0.5), 1, 1,
                                linewidth=2, edgecolor="red", facecolor="none"))
    axs[1].plot(PPM_AXIS, np.real(spec_v), label="Real")
    axs[1].plot(PPM_AXIS, np.abs(spec_v),  label="|S|", alpha=0.7)
    axs[1].set_title(f"{title}  ({voxel_row},{voxel_col})")
    axs[1].set_xlabel("ppm")
    axs[1].invert_xaxis()
    axs[1].legend()
    plt.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    out_dir  = os.path.join(args.out_dir, "lipid_removal")
    os.makedirs(out_dir, exist_ok=True)

    coilmap_dir = os.path.join(args.out_dir, "coilmap")

    D_TYPE      = np.complex64
    Trej_D_TYPE = np.float32

    K_POINTS = args.k_points
    N_SEQ    = args.n_seq_points
    N_COILS  = args.n_coils
    Ny, Nx   = args.dim[0], args.dim[1]

    TS         = (K_POINTS / N_SEQ) * args.dwelltime
    sweepwidth = 1.0 / TS
    FREQ_AXIS  = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS   = FREQ_AXIS / args.center_freq + args.ppm_center
    print(f"[lipidrm] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s")

    # ── Trajectory ───────────────────────────────────────────────────────────────
    ksp_cache = os.path.join(out_dir, "mrsi_ksp_scaled.npy")
    if not os.path.exists(ksp_cache):
        print("[lipidrm] Scaling MRSI trajectory …")
        mrsi_ksp = np.load(data_dir + "mrsi_ksp.npy")
        scale    = args.mrsi_ksp_scale or (30.37478212844472 / 32.0)
        mrsi_ksp[:2, ...] *= scale
        mrsi_ksp[2, ...]   = np.flip(mrsi_ksp[2, ...])
        np.save(ksp_cache, mrsi_ksp)
    trej = np.load(ksp_cache, mmap_mode="r").T.astype(Trej_D_TYPE)

    # ── Load data ─────────────────────────────────────────────────────────────────
    print("[lipidrm] Loading MRSI data …")
    mrsi_raw       = np.load(data_dir + "mrsi_data.npy", mmap_mode="r").astype(D_TYPE)
    wref_img       = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    mrsi_reordered = np.transpose(mrsi_raw, (0, 3, 2, 1)).reshape(1, N_COILS, -1)

    # ── Coil maps & NUFFT ────────────────────────────────────────────────────────
    coil_smap_raw = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"), mmap_mode="r")
    coil_smap     = np.repeat(coil_smap_raw[np.newaxis, :, :, :, np.newaxis],
                               N_SEQ, axis=-1).astype(D_TYPE)
    smap_time     = coil_smap.squeeze(0)                             # (C, Ny, Nx, T)

    NufftOp    = mrinufft.get_operator("finufft")
    print("[lipidrm] Building NUFFT operator …")
    nufft_mrsi = NufftOp(trej, shape=(Ny, Nx, N_SEQ), n_coils=N_COILS,
                          n_batchs=1, squeeze_dims=True, smaps=smap_time)

    # ── Adjoint NUFFT ─────────────────────────────────────────────────────────────
    print("[lipidrm] Adjoint NUFFT …")
    image_blurry = nufft_mrsi.adj_op(mrsi_reordered)                # (Ny, Nx, N_SEQ) spectrum
    mag_map_2d   = np.mean(np.abs(image_blurry), axis=-1)

    # ── ref image / affine ───────────────────────────────────────────────────────
    _affine_npy = data_dir + "affine.npy"
    if args.ref_nii:
        ref_img_obj = Image(args.ref_nii)
        affine      = ref_img_obj.voxToWorldMat
    elif os.path.exists(_affine_npy):
        affine      = np.load(_affine_npy)
        ref_img_obj = None
        print(f"[lipidrm] Loaded affine from {_affine_npy}")
    else:
        ref_img_path = data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz"
        try:
            ref_img_obj = Image(ref_img_path)
            affine      = ref_img_obj.voxToWorldMat
        except Exception:
            ref_img_obj = None
            affine      = np.eye(4)
            print("[lipidrm] WARNING: affine.npy not found; using identity")

    # ── Brain mask (single, normalised wref_o) ───────────────────────────────────
    wref_norm, brain_nolip_mask, _ = make_brain_mask(
        wref_img, args.brain_threshold, args.brain_erosion,
        cleanup=args.brain_mask_cleanup)

    # ── Save adj NUFFT before lipid removal (unmasked, lipid ring visible) ──────
    fid_adj = np.ascontiguousarray(SpecToFID(image_blurry, axis=-1).transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :]
    gen_nifti_mrs(fid_adj, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "adj_bf_lprm.nii.gz"))
    print("[lipidrm] Saved adj_bf_lprm.nii.gz")
    img_masked = image_blurry * brain_nolip_mask[:, :, np.newaxis]

    img_4d = image_blurry[:, :, np.newaxis, :]                      # (Ny, Nx, 1, T)
    vr, vc = args.plot_voxel

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(brain_nolip_mask, origin="lower", cmap="gray")
        ax.set_title(f"brain_nolip_mask  (thr={args.brain_threshold:.2e})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03a_brain_nolip_mask.png"), dpi=120)
        plt.close(fig)
        print("[lipidrm] Saved fig_03a_brain_nolip_mask.png")

    if args.phantom:
        # Phantoms have no lipid signal — skip LSS/GMM/SVD/L2 removal entirely and
        # pass the adjoint-NUFFT image straight through as the "lipid-removed" data,
        # writing zero placeholders for V_lipid/lss_map so 04b/04c/04d's lipid
        # regularization terms are inert (still needed since those scripts load them).
        print("[lipidrm] --phantom: skipping lipid identification & L2 removal "
              "(no lipid signal expected); passing adjoint-NUFFT data straight through.")
        np.save(os.path.join(out_dir, "lss_map.npy"),
                np.zeros(brain_nolip_mask.shape, dtype=np.float32))
        print("[lipidrm] Saved lss_map.npy (zeros — phantom)")

        V_lipid = np.zeros((N_SEQ, args.lipid_rank), dtype=D_TYPE)
        np.save(os.path.join(out_dir, "V_lipid.npy"), V_lipid)
        print(f"[lipidrm] Saved V_lipid.npy  shape={V_lipid.shape}  (zeros — phantom)")

        mrsi_lprm_4d = img_4d.copy()                                # (Ny,Nx,1,T) spectrum
    else:
        # ── LSS map ───────────────────────────────────────────────────────────────
        print("[lipidrm] Computing LSS map …")
        lss_map, _ = compute_lss(img_4d, PPM_AXIS,
                                  low_ppm=args.lss_ppm_low, high_ppm=args.lss_ppm_high)

        if args.save_plots:
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(np.squeeze(lss_map), origin="lower", cmap="viridis")
            plt.colorbar(im, ax=ax, label="LSS")
            ax.set_title("LSS map")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_03b_lss_map.png"), dpi=120)
            plt.close(fig)

        # Raw LSS map for 04b (in-brain GMM classification / W_lip now built there,
        # next to where the joint refit's spatial regularization is assembled).
        np.save(os.path.join(out_dir, "lss_map.npy"), np.squeeze(lss_map))
        print("[lipidrm] Saved lss_map.npy")

        # ── GMM lipid mask ────────────────────────────────────────────────────────
        res        = select_lipid_mask_gmm_simple(
            lss_map, out_dir=out_dir, nsigma=args.nsigma_gmm,
            max_voxels=args.n_lipid_voxels, topN_fallback=args.topn_fallback,
            save_plots=args.save_plots,
        )
        lipid_mask = res["lipid_mask"]
        print(f"[lipidrm] Lipid mask: n={res['n_selected']}  method={res['method']}  thr={res['threshold']:.4e}")

        if args.save_plots:
            plot_mag_and_voxel(img_4d, PPM_AXIS, vr, vc,
                               "Before L2 lipid removal",
                               os.path.join(out_dir, "fig_03b_pre_l2.png"))

        # ── L2 lipid removal ──────────────────────────────────────────────────────
        print("[lipidrm] L2 lipid removal …")
        mrsi_fid_4d        = SpecToFID(image_blurry, axis=-1)[:, :, np.newaxis, :]  # (Ny,Nx,1,T) FID
        lipid_fids         = mrsi_fid_4d[lipid_mask[:, :, np.newaxis]]              # (N_vox, T)
        lipid_basis        = lipid_fids.T                                            # (T, N_vox)

        # ── Save lipid basis matrix (for SVD reuse in 04b joint refit) ──────────
        np.save(os.path.join(out_dir, "lipid_basis_matrix.npy"), lipid_basis)
        print(f"[lipidrm] Saved lipid_basis_matrix.npy  shape={lipid_basis.shape}")

        # ── SVD truncation → V_lipid (spectral subspace for 04b joint refit) ────
        U_lip_full, s_lip_full, _ = np.linalg.svd(lipid_basis, full_matrices=False)
        V_lipid = U_lip_full[:, :args.lipid_rank]                                    # (T, lipid_rank)
        np.save(os.path.join(out_dir, "V_lipid.npy"), V_lipid)
        print(f"[lipidrm] Saved V_lipid.npy  shape={V_lipid.shape}  "
              f"top singvals={s_lip_full[:args.lipid_rank]}")

        if args.save_plots:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(s_lip_full[:30], "x-")
            ax.axvline(args.lipid_rank - 0.5, color="r", ls="--", label=f"lipid_rank={args.lipid_rank}")
            ax.set_title("Lipid basis singular values")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_03e_lipid_singvals.png"), dpi=120)
            plt.close(fig)

        # ── Save lipid basis as NIfTI-MRS (tiled to image size) ──────────────────
        lipid_nmrs = np.tile(lipid_basis.T[np.newaxis, np.newaxis, :, :], (Nx, Ny, 1, 1))
        gen_nifti_mrs(lipid_nmrs, dwelltime=TS, spec_freq=297.219,
                      affine=affine).save(os.path.join(out_dir, "lipid_basis.nii.gz"))
        print(f"[lipidrm] Saved lipid_basis.nii.gz  shape={lipid_nmrs.shape}")

        mrsi_fid_lprm_4d   = lipid_removal_l2(mrsi_fid_4d, lipid_basis, beta=args.lipid_beta)
        mrsi_lprm_4d       = FIDToSpec(mrsi_fid_lprm_4d, axis=-1)                   # (Ny,Nx,1,T) spectrum

        if args.save_plots:
            plot_mag_and_voxel(mrsi_lprm_4d, PPM_AXIS, vr, vc,
                               "After L2 lipid removal",
                               os.path.join(out_dir, "fig_03b_post_l2.png"))

    # ── Mask outside brain ────────────────────────────────────────────────────────
    mrsi_lprm_masked                  = mrsi_lprm_4d[:, :, 0, :].copy()
    mrsi_lprm_masked[~brain_nolip_mask] = 0
    mrsi_lprm_4d                      = mrsi_lprm_masked[:, :, np.newaxis, :]

    # ── Save lipid-free NIfTI before phase correction ─────────────────────────────
    _pre_ph_fid = np.ascontiguousarray(SpecToFID(mrsi_lprm_masked, axis=-1).transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :]
    gen_nifti_mrs(_pre_ph_fid, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "mrsi_lprm_pre_phcorr.nii.gz"))
    print("[lipidrm] Saved mrsi_lprm_pre_phcorr.nii.gz")

    # ── Phase correction ──────────────────────────────────────────────────────────
    if args.phase_method.lower() == 'none':
        print("[lipidrm] Phase correction skipped (--phase-method none).")
    else:
        print(f"[lipidrm] Phase correction  ppmlim={args.phase_ppmlim}  method={args.phase_method} …")
        lpfree_phcorr_f = phase_corr(
            SpecToFID(mrsi_lprm_masked, axis=-1),   # (Ny, Nx, T) FID — from masked spectrum
            mag_map_2d = mag_map_2d,
            brain_mask = brain_nolip_mask,
            TS         = TS,
            img_shape  = (Ny, Nx),
            out_dir    = out_dir,
            ppmlim     = tuple(args.phase_ppmlim),
            ref_img    = ref_img_obj,
            out_fname  = "lpfree_phcorr_nifti",
            method     = args.phase_method,
        )
        lpfree_phcorr_spec = FIDToSpec(lpfree_phcorr_f, axis=-1)    # (Ny, Nx, T)
        mrsi_lprm_4d       = lpfree_phcorr_spec[:, :, np.newaxis, :] # (Ny, Nx, 1, T)
        print("[lipidrm] Phase correction done.")

    if args.save_plots:
        plot_mag_and_voxel(mrsi_lprm_4d, PPM_AXIS, vr, vc,
                           "After phase correction",
                           os.path.join(out_dir, "fig_03b_final.png"))

    # ── Save NIfTI ────────────────────────────────────────────────────────────────
    lprm_3d  = mrsi_lprm_4d[:, :, 0, :]                             # (Ny, Nx, T) spectrum
    save_fid = np.ascontiguousarray(SpecToFID(lprm_3d, axis=-1).transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :]
    gen_nifti_mrs(save_fid, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "adj_bf_spice_crs_cr.nii.gz"))
    print("[lipidrm] Saved adj_bf_spice_crs_cr.nii.gz")

    # ── Forward NUFFT → kt space (for SPICE) ─────────────────────────────────────
    print("[lipidrm] Forward NUFFT → kt_mrsi_lprm.npy …")
    kt_mrsi_lprm = nufft_mrsi.op(lprm_3d)
    np.save(os.path.join(out_dir, "kt_mrsi_lprm.npy"), kt_mrsi_lprm)
    print(f"[lipidrm] Saved kt_mrsi_lprm.npy  shape={kt_mrsi_lprm.shape}")

    # ── Save NIfTI for SPICE ──────────────────────────────────────────────────────
    mrsi_lprm_f   = np.ascontiguousarray(SpecToFID(lprm_3d, axis=-1).transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :]
    gen_nifti_mrs(mrsi_lprm_f, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "my_mrsi_lprm_f.nii.gz"))
    print("[lipidrm] Saved my_mrsi_lprm_f.nii.gz")

    # ── With-lipid (no ring) data for 04b joint refit ────────────────────────────
    # Same brain_nolip_mask crop as the L2 path, but skip L2 — lipid signal still
    # present inside the mask (only the extreme outer ring used to build the lipid
    # basis is excluded by construction of brain_nolip_mask).
    print("[lipidrm] Building with-lipid (no-ring) data for 04b joint refit …")
    if args.phase_method.lower() == 'none':
        print("[lipidrm] With-lipid phase correction skipped (--phase-method none).")
        withlip_spec_3d = img_masked                                     # (Ny, Nx, T) spectrum
    else:
        withlip_phcorr_f = phase_corr(
            SpecToFID(img_masked, axis=-1),
            mag_map_2d = mag_map_2d,
            brain_mask = brain_nolip_mask,
            TS         = TS,
            img_shape  = (Ny, Nx),
            out_dir    = out_dir,
            ppmlim     = tuple(args.phase_ppmlim),
            ref_img    = ref_img_obj,
            out_fname  = "withlip_phcorr_nifti",
            method     = args.phase_method,
        )
        withlip_spec_3d = FIDToSpec(withlip_phcorr_f, axis=-1)          # (Ny, Nx, T)

    if args.save_plots:
        plot_mag_and_voxel(withlip_spec_3d[:, :, np.newaxis, :], PPM_AXIS, vr, vc,
                           "With-lipid (no ring), phase-corrected",
                           os.path.join(out_dir, "fig_03d_withlip_noring.png"))

    kt_mrsi_withlip = nufft_mrsi.op(withlip_spec_3d)
    np.save(os.path.join(out_dir, "kt_mrsi_withlip_noring.npy"), kt_mrsi_withlip)
    print(f"[lipidrm] Saved kt_mrsi_withlip_noring.npy  shape={kt_mrsi_withlip.shape}")

    patch_spicefit_tree(args.out_dir, [
        ("my_mrsi_lprm_f.nii.gz", os.path.join(out_dir, "my_mrsi_lprm_f.nii.gz"), "data-lprm-mrsi"),
        ("adj_bf_lprm.nii.gz",   os.path.join(out_dir, "adj_bf_lprm.nii.gz"),          "data-lprm-bf"),
        ("adj_bf_crs.nii.gz",    os.path.join(out_dir, "adj_bf_spice_crs_cr.nii.gz"),  "data-lprm-crs"),
        ("lipid_basis.nii.gz",   os.path.join(out_dir, "lipid_basis.nii.gz"),           "data-lipid-basis"),
        ("mrsi_lprm_pre.nii.gz", os.path.join(out_dir, "mrsi_lprm_pre_phcorr.nii.gz"), "data-lprm-pre"),
    ])
    print("[lipidrm] Done.")


if __name__ == "__main__":
    main()
