#!/usr/bin/env python3
"""
Step 3 — Lipid removal (GMM + L2 only, no B0/xcorr correction).

Reads  : <data_dir>/mrsi_data.npy, mrsi_ksp.npy, wref_o.npy
         <out_dir>/coilmap/ecalib_pp.npy
Writes : <out_dir>/lipid_removal/kt_mrsi_lprm.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/lipid_removal/adj_bf_lprm.nii.gz
         <out_dir>/lipid_removal/my_mrsi_lprm_f.nii.gz
         <out_dir>/lipid_removal/fig_03b_*.png  (when --save-plots)

Usage:
    python scripts/03_lipid_removal.py \
        --data-dir ./data/ \
        --out-dir  ./output \
        --dim 64 64 --n-seq-points 300 --k-points 39842 \
        [--lipid-beta 200] [--save-plots]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import phase_corr


def parse_args():
    p = argparse.ArgumentParser(description="Lipid removal (GMM+L2 only) — step 3b")
    p.add_argument("--data-dir",         required=True)
    p.add_argument("--out-dir",          default="./output")
    p.add_argument("--dwelltime",        type=float, default=5e-6)
    p.add_argument("--k-points",         type=int,   default=39842)
    p.add_argument("--n-seq-points",     type=int,   default=300)
    p.add_argument("--n-coils",          type=int,   default=32)
    p.add_argument("--dim",              type=int,   nargs=2, default=[64, 64], metavar=("NX", "NY"))
    p.add_argument("--center-freq",      type=float, default=297.219338)
    p.add_argument("--ppm-center",       type=float, default=3.027)
    p.add_argument("--mrsi-ksp-scale",   type=float, default=None)
    # brain masks
    p.add_argument("--brain-threshold",  type=float, default=0.00034,
                   help="Raw wref threshold for lipid mask (default 0.00034)")
    p.add_argument("--brain-threshold2", type=float, default=0.08,
                   help="Normalized wref threshold for phase-corr / SPICE mask (default 0.08)")
    p.add_argument("--brain-erosion",    type=int,   default=3)
    # lipid removal
    p.add_argument("--lss-ppm-low",      type=float, default=0.7)
    p.add_argument("--lss-ppm-high",     type=float, default=1.8)
    p.add_argument("--lipid-beta",       type=float, default=200.0)
    p.add_argument("--n-lipid-voxels",   type=int,   default=500)
    p.add_argument("--nsigma-gmm",       type=float, default=0.2)
    # phase correction
    p.add_argument("--phase-ppmlim",     type=float, nargs=2, default=[3.5, 5.0],
                   metavar=("LO", "HI"))
    p.add_argument("--phase-method",     type=str,   default="max_real",
                   help="fsl_mrs phase correction method (default: max_real)")
    # misc
    p.add_argument("--plot-voxel",       type=int,   nargs=2, default=[41, 24],
                   metavar=("ROW", "COL"))
    p.add_argument("--ref-nii",          default=None)
    p.add_argument("--save-plots",       action="store_true")
    return p.parse_args()


def compute_lss(data, ppm_axis, low_ppm=0.7, high_ppm=1.8):
    lipid_idx = np.where((ppm_axis >= low_ppm) & (ppm_axis <= high_ppm))[0]
    return np.sum(np.abs(data[..., lipid_idx]), axis=-1), lipid_idx


def select_lipid_mask_gmm_simple(lss_map, out_dir, nsigma=0.2,
                                  max_voxels=500, topN_fallback=100,
                                  save_plots=False):
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm

    lss2d = np.squeeze(np.asarray(lss_map))
    flat  = lss2d.ravel()
    vals  = flat[np.isfinite(flat) & (flat > 0)]

    thr, method, gmm_model = None, None, None
    try:
        logvals = np.log(vals).reshape(-1, 1)
        gmm     = GaussianMixture(n_components=2, random_state=0).fit(logvals)
        means   = gmm.means_.ravel()
        covs    = gmm.covariances_.ravel()
        li      = int(np.argmax(means))
        thr     = float(np.exp(means[li] - nsigma * np.sqrt(covs[li])))
        method, gmm_model = "gmm", gmm
    except Exception:
        thr    = float(np.percentile(vals, 90))
        method = "percentile"

    mask = lss2d >= thr
    nsel = int(np.sum(mask))

    if nsel > max_voxels:
        sorted_flat = np.sort(vals)[::-1]
        thr    = float(sorted_flat[topN_fallback - 1]) if len(sorted_flat) >= topN_fallback else thr
        mask   = lss2d >= thr
        nsel   = int(np.sum(mask))
        method = "topN"

    if save_plots:
        from scipy.stats import norm as _norm
        logvals_all = np.log(np.maximum(vals, 1e-12))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(logvals_all, bins=80, density=True, alpha=0.6, color="C0")
        xs = np.linspace(logvals_all.min(), logvals_all.max(), 400)
        if gmm_model is not None:
            for mu, cov, w in zip(gmm_model.means_.ravel(),
                                   gmm_model.covariances_.ravel(),
                                   gmm_model.weights_.ravel()):
                ax1.plot(xs, w * _norm.pdf(xs, loc=mu, scale=np.sqrt(cov)), lw=2)
        ax1.axvline(np.log(thr), color="k", ls="--", label=f"thr log={np.log(thr):.3f}")
        ax1.set_xlabel("log(LSS)")
        ax1.legend()
        ax1.set_title(f"LSS GMM  method={method}  n={nsel}")
        im = ax2.imshow(lss2d, origin="lower", cmap="viridis")
        ax2.set_title("LSS map + lipid mask")
        plt.colorbar(im, ax=ax2, fraction=0.046)
        for yy, xx in zip(*np.where(mask)):
            ax2.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                    edgecolor="red", facecolor="none", linewidth=1.2))
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03b_lss_gmm.png"), dpi=120)
        plt.close(fig)

    return {"lipid_mask": mask, "threshold": thr, "n_selected": nsel, "method": method}


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
    ref_img_path = args.ref_nii or (data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz")
    try:
        ref_img_obj = Image(ref_img_path)
        affine      = ref_img_obj.voxToWorldMat
    except Exception:
        ref_img_obj = None
        affine      = np.eye(4)

    # ── Brain masks ───────────────────────────────────────────────────────────────
    img_ref          = np.abs(wref_img[:, :, 0])
    brain_nolip_mask = img_ref > args.brain_threshold             # raw (for lipid / save)

    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask2      = wref_norm > args.brain_threshold2          # normalised (for phase-corr)

    # ── Save adj NUFFT before lipid removal ──────────────────────────────────────
    img_masked = image_blurry * brain_nolip_mask[:, :, np.newaxis]
    fid_adj    = SpecToFID(img_masked, axis=-1).transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(fid_adj, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "adj_bf_lprm.nii.gz"))
    print("[lipidrm] Saved adj_bf_lprm.nii.gz")

    # ── LSS map ───────────────────────────────────────────────────────────────────
    print("[lipidrm] Computing LSS map …")
    img_4d          = image_blurry[:, :, np.newaxis, :]            # (Ny, Nx, 1, T)
    lss_map, _      = compute_lss(img_4d, PPM_AXIS,
                                   low_ppm=args.lss_ppm_low, high_ppm=args.lss_ppm_high)

    vr, vc = args.plot_voxel
    if args.save_plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(np.squeeze(lss_map), origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, label="LSS")
        ax.set_title("LSS map")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03b_lss_map.png"), dpi=120)
        plt.close(fig)

    # ── GMM lipid mask ────────────────────────────────────────────────────────────
    res        = select_lipid_mask_gmm_simple(
        lss_map, out_dir=out_dir, nsigma=args.nsigma_gmm,
        max_voxels=args.n_lipid_voxels, topN_fallback=100,
        save_plots=args.save_plots,
    )
    lipid_mask = res["lipid_mask"]
    print(f"[lipidrm] Lipid mask: n={res['n_selected']}  method={res['method']}  thr={res['threshold']:.4e}")

    if args.save_plots:
        plot_mag_and_voxel(img_4d, PPM_AXIS, vr, vc,
                           "Before L2 lipid removal",
                           os.path.join(out_dir, "fig_03b_pre_l2.png"))

    # ── L2 lipid removal ──────────────────────────────────────────────────────────
    print("[lipidrm] L2 lipid removal …")
    mrsi_fid_4d        = SpecToFID(image_blurry, axis=-1)[:, :, np.newaxis, :]  # (Ny,Nx,1,T) FID
    lipid_fids         = mrsi_fid_4d[lipid_mask[:, :, np.newaxis]]              # (N_vox, T)
    lipid_basis        = lipid_fids.T                                            # (T, N_vox)
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

    # ── Phase correction ──────────────────────────────────────────────────────────
    print(f"[lipidrm] Phase correction  ppmlim={args.phase_ppmlim}  method={args.phase_method} …")
    lpfree_phcorr_f = phase_corr(
        SpecToFID(mrsi_lprm_masked, axis=-1),   # (Ny, Nx, T) FID — from masked spectrum
        mag_map_2d = mag_map_2d,
        brain_mask = brain_mask2,
        TS         = TS,
        img_shape  = (Ny, Nx),
        out_dir    = out_dir,
        ppmlim     = tuple(args.phase_ppmlim),
        ref_img    = ref_img_obj,
        out_fname  = "lpfree_phcorr_nifti",
        method     = args.phase_method,
    )
    lpfree_phcorr_spec = FIDToSpec(lpfree_phcorr_f, axis=-1)        # (Ny, Nx, T)
    mrsi_lprm_4d       = lpfree_phcorr_spec[:, :, np.newaxis, :]    # (Ny, Nx, 1, T)
    print("[lipidrm] Phase correction done.")

    if args.save_plots:
        plot_mag_and_voxel(mrsi_lprm_4d, PPM_AXIS, vr, vc,
                           "After phase correction",
                           os.path.join(out_dir, "fig_03b_final.png"))

    # ── Save NIfTI ────────────────────────────────────────────────────────────────
    lprm_3d  = mrsi_lprm_4d[:, :, 0, :]                             # (Ny, Nx, T) spectrum
    save_fid = SpecToFID(lprm_3d, axis=-1).transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(save_fid, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "adj_bf_spice_crs_cr.nii.gz"))
    print("[lipidrm] Saved adj_bf_spice_crs_cr.nii.gz")

    # ── Forward NUFFT → kt space (for SPICE) ─────────────────────────────────────
    print("[lipidrm] Forward NUFFT → kt_mrsi_lprm.npy …")
    kt_mrsi_lprm = nufft_mrsi.op(lprm_3d)
    np.save(os.path.join(out_dir, "kt_mrsi_lprm.npy"), kt_mrsi_lprm)
    print(f"[lipidrm] Saved kt_mrsi_lprm.npy  shape={kt_mrsi_lprm.shape}")

    # ── Save NIfTI for SPICE ──────────────────────────────────────────────────────
    mrsi_lprm_f   = SpecToFID(lprm_3d, axis=-1).transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(mrsi_lprm_f, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "my_mrsi_lprm_f.nii.gz"))
    print("[lipidrm] Saved my_mrsi_lprm_f.nii.gz")

    print("[lipidrm] Done.")


if __name__ == "__main__":
    main()
