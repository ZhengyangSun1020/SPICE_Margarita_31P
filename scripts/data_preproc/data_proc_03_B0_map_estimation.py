#!/usr/bin/env python3
"""
Step 2 — B0 map estimation, following B0_corr_finufft.ipynb cell-by-cell.

Reads  : <data_dir>/wref_data.npy, wref_ksp.npy
         <out_dir>/coilmap/ecalib_pp.npy
Writes : <out_dir>/b0map/B0_map.npy   (= B0_map_pk.npy)
         <out_dir>/b0map/wref_resampled.npy
         <out_dir>/b0map/wref_ksp_scaled.npy
         <out_dir>/b0map/wref_phcorr_nifti.nii.gz
         <out_dir>/b0map/fig_02*.png   (when --save-plots)

Usage:
    python scripts/data_preproc/data_proc_03_B0_map_estimation.py \
        --data-dir  data/processed/invivo_250305_01 \
        --basis-dir ./basis/ \
        --dim 64 64 --n-seq-points 300 --k-points 39842 \
        --save-plots \
        --phase-method max-real
        [--brain-threshold 0.16]
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
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, distance_transform_edt
from warnings import filterwarnings
filterwarnings("ignore")
import nibabel as nib

import mrinufft
from fsl_mrs.utils.misc import FIDToSpec, SpecToFID
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl_mrs.core.mrs import MRS, Basis
from fsl.data.image import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.scan_params import load_scan_params
from utils.utils import phase_corr
from utils.pipeline_utils import try_symlink_shared_output, make_brain_mask, patch_spicefit_tree


def parse_args():
    p = argparse.ArgumentParser(description="B0 map estimation — step 2")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--basis-dir",       required=True)
    p.add_argument("--out-dir",         default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int, default=None)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--n-coils",         type=int, default=None)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64], metavar=("NX", "NY"))
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--ksp-scale",       type=float, default=None)
    p.add_argument("--brain-threshold", type=float, default=0.07,
                   help="Normalized wref_o threshold for brain mask (default 0.07)")
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component (drops disconnected noise blobs outside "
                        "the brain) and fill enclosed holes (e.g. a central signal void). "
                        "Default: off, use only if a single global threshold isn't enough.")
    p.add_argument("--phase-ppmlim",    type=float, nargs=2, default=[0.0, 7.0], metavar=("LO", "HI"))
    p.add_argument("--phase-method",    type=str,   default=None,
                   help="Phase correction method passed to fsl_mrs: 'max-real' or 'phasta' (default: None = fsl_mrs default)")
    p.add_argument("--singlet-lw",      type=float, default=5.0)
    p.add_argument("--no-smooth",       action="store_true",
                   help="Skip Gaussian fill+smooth (notebook Cell 28 no-smooth option).")
    p.add_argument("--smooth-sigma",    type=float, default=1.0)
    p.add_argument("--plot-voxel",      type=int,   nargs=2, default=[30, 15],
                   metavar=("ROW", "COL"))
    p.add_argument("--ref-nii",         default=None)
    p.add_argument("--save-plots",      action="store_true")
    return p.parse_args()


# ── Cell 26: fill + smooth helpers ─────────────────────────────────────────────

def fill_outside_with_nearest(B0_map, brain_mask):
    outside = ~brain_mask
    indices = distance_transform_edt(brain_mask == 0, return_distances=False, return_indices=True)
    filled  = B0_map.copy()
    filled[outside] = B0_map[indices[0][outside], indices[1][outside]]
    return filled


def smooth_fill_then_gaussian(B0_map, brain_mask, sigma=1.0):
    filled = fill_outside_with_nearest(B0_map, brain_mask)
    sm     = gaussian_filter(filled, sigma=sigma)
    result = np.zeros_like(B0_map)
    result[brain_mask] = sm[brain_mask]
    return result


# ── Cells 8 / 14: plot helpers ─────────────────────────────────────────────────

def plot_mag_and_spectrum(spec_3d, PPM_AXIS, voxel_row, voxel_col, title, fname):
    mag_map = np.mean(np.abs(spec_3d), axis=-1)
    spec_v  = spec_3d[voxel_row, voxel_col, :].astype(np.complex64)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im = axs[0].imshow(mag_map, cmap="viridis", origin="lower")
    axs[0].set_title("Average spectral magnitude")
    axs[0].set_xlabel("x (cols)")
    axs[0].set_ylabel("y (rows)")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].add_patch(Rectangle((voxel_col - 0.5, voxel_row - 0.5), 1, 1,
                                linewidth=2, edgecolor="red", facecolor="none"))
    axs[1].plot(PPM_AXIS, np.real(spec_v), label="Real")
    axs[1].plot(PPM_AXIS, np.abs(spec_v),  label="|S|", alpha=0.7)
    axs[1].set_title(f"{title}  voxel (row={voxel_row}, col={voxel_col})")
    axs[1].set_xlabel("ppm")
    axs[1].invert_xaxis()
    axs[1].set_ylabel("Signal")
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    plt.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# ── Cells 16–18: water singlet basis ───────────────────────────────────────────

def make_singlet_fid(PPM_CENTER, center_freq, TIME_AXIS, linewidth_hz=5.0):
    t   = np.asarray(TIME_AXIS).astype(float)
    fid = np.exp(-linewidth_hz * np.pi * t)   # on-resonance, 0 phase
    m   = np.max(np.abs(FIDToSpec(fid)))
    return fid / m if m != 0 else fid


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_wref")

    if try_symlink_shared_output(data_dir, args.out_dir, "b0map"):
        return

    out_dir  = os.path.join(args.out_dir, "b0map")
    os.makedirs(out_dir, exist_ok=True)

    coilmap_dir = os.path.join(args.out_dir, "coilmap")

    D_TYPE      = np.complex64
    Trej_D_TYPE = np.float32

    K_POINTS  = args.k_points
    N_SEQ     = args.n_seq_points
    N_COILS   = args.n_coils
    Dim_Voxel = args.dim
    Ny, Nx    = Dim_Voxel[0], Dim_Voxel[1]

    TS          = (K_POINTS / N_SEQ) * args.dwelltime
    sweepwidth  = 1.0 / TS
    center_freq = args.center_freq
    PPM_CENTER  = args.ppm_center
    FREQ_AXIS   = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS    = FREQ_AXIS / center_freq + PPM_CENTER
    TIME_AXIS   = np.linspace(TS, TS * N_SEQ, N_SEQ)
    print(f"[b0_corr] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s")
    phase_method = args.phase_method

    # ── Cell 2: scale & cache wref trajectory ─────────────────────────────────────
    ksp_cache = os.path.join(out_dir, "wref_ksp_scaled.npy")
    if not os.path.exists(ksp_cache):
        print("[b0_corr] Scaling wref trajectory …")
        wref_ksp = np.load(data_dir + "wref_ksp.npy")
        scale    = args.ksp_scale or (15.713940692571413 / 16.0)
        wref_ksp[:2, ...] *= scale
        wref_ksp[2, ...]   = np.flip(wref_ksp[2, ...])
        np.save(ksp_cache, wref_ksp)
    wref_ksp_scaled = np.load(ksp_cache, mmap_mode="r")
    trej = wref_ksp_scaled.T.astype(Trej_D_TYPE)

    # ── Cell 2: load wref data & reorder ──────────────────────────────────────────
    print("[b0_corr] Loading wref data …")
    wref_raw       = np.load(data_dir + "wref_data.npy", mmap_mode="r")
    wref_reordered = np.transpose(wref_raw, (0, 3, 2, 1)).reshape(1, N_COILS, -1)

    # ── Cell 4: load coilmap ──────────────────────────────────────────────────────
    coil_smap_raw = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"), mmap_mode="r")
    coil_smap     = np.repeat(coil_smap_raw[np.newaxis, :, :, :, np.newaxis],
                               N_SEQ, axis=-1).astype(D_TYPE)
    smap_time     = coil_smap.squeeze(0)                        # (C, Ny, Nx, T)
    print(f"[b0_corr] coil_smap shape={coil_smap.shape}")

    # ── Cell 5: build finufft NUFFT operator ──────────────────────────────────────
    result_shape = (Ny, Nx, N_SEQ)
    NufftOp      = mrinufft.get_operator("finufft")
    print("[b0_corr] Building finufft NUFFT operator …")
    nufft_mrsi = NufftOp(trej, shape=result_shape, n_coils=N_COILS,
                          n_batchs=1, squeeze_dims=True, smaps=smap_time)

    # ── Cell 6: adjoint NUFFT → image_blurry_numpy ────────────────────────────────
    print("[b0_corr] Adjoint NUFFT …")
    image_blurry_numpy = nufft_mrsi.adj_op(wref_reordered)    # (Ny, Nx, N_SEQ) spectrum
    print(f"[b0_corr] image_blurry_numpy shape={image_blurry_numpy.shape}")

    mag_map_2d = np.mean(np.abs(image_blurry_numpy), axis=-1)

    # ── Cell 7: save wref NIfTI ───────────────────────────────────────────────────
    _wref_affine_npy = data_dir + "wref_affine.npy"
    if args.ref_nii:
        ref_img_obj = Image(args.ref_nii)
        affine      = ref_img_obj.voxToWorldMat
    elif os.path.exists(_wref_affine_npy):
        affine      = np.load(_wref_affine_npy)
        ref_img_obj = None
        print(f"[b0_corr] Loaded wref affine from {_wref_affine_npy}")
    else:
        ref_img_path = data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz"
        try:
            ref_img_obj = Image(ref_img_path)
            affine      = ref_img_obj.voxToWorldMat
        except Exception:
            ref_img_obj = None
            affine      = np.eye(4)
            print("[b0_corr] WARNING: wref_affine.npy not found; using identity")

    fid_save = np.ascontiguousarray(SpecToFID(image_blurry_numpy.reshape(Ny, Nx, -1), axis=-1).transpose(1, 0, 2)[::-1, :, :])
    gen_nifti_mrs(fid_save[:, :, np.newaxis, :], dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "wref_adj_nufft.nii.gz"))
    print("[b0_corr] Saved wref_adj_nufft.nii.gz")

    # ── Cell 8: plot raw adj NUFFT ────────────────────────────────────────────────
    vr, vc = args.plot_voxel
    if args.save_plots:
        plot_mag_and_spectrum(image_blurry_numpy, PPM_AXIS, vr, vc,
                              "Raw adj NUFFT wref",
                              os.path.join(out_dir, "fig_02a_adjnufft_raw.png"))
        print("[b0_corr] Saved fig_02a_adjnufft_raw.png")

    # ── Cell 9: brain mask (normalised wref_o) ───────────────────────────────────
    wref_o_img  = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_norm, brain_mask, brain_mask_inner = make_brain_mask(
        wref_o_img, args.brain_threshold, args.brain_erosion,
        cleanup=args.brain_mask_cleanup)

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(wref_norm, origin="lower", cmap="gray")
        ax.imshow(brain_mask, origin="lower", cmap="Reds", alpha=0.35)
        ax.set_title(f"wref_o (norm) + brain mask (thr={args.brain_threshold:.3g})")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_02b_brain_mask.png"), dpi=120)
        plt.close(fig)
        print("[b0_corr] Saved fig_02b_brain_mask.png")

    # ── Cell 13: phase correction via utils.recon.phase_corr ─────────────────
    # Notebook call: phase_corr(SpecToFID(image_blurry_numpy, axis=-1), mag_map_2d,
    #                           brain_mask, TS, (Ny, Nx), out_dir, ppmlim=(2,4), ref_img=ref_img)
    wref_fid = SpecToFID(image_blurry_numpy, axis=-1)   # (Ny, Nx, T) FID
    if phase_method is not None and phase_method.lower() == 'none':
        print("[b0_corr] Phase correction skipped (--phase-method none).")
        wref_phcorr_f = wref_fid
    else:
        print(f"[b0_corr] Phase correction  ppmlim={args.phase_ppmlim} …")
        wref_phcorr_f = phase_corr(
            wref_fid,
            mag_map_2d = mag_map_2d,
            brain_mask = brain_mask,
            TS         = TS,
            img_shape  = (Ny, Nx),
            out_dir    = out_dir,
            ppmlim     = tuple(args.phase_ppmlim),
            ref_img    = ref_img_obj,
            out_fname  = "wref_phcorr_nifti",
            method     = phase_method,
            affine     = affine,
        )
    # wref_phcorr_f: (Ny, Nx, T) — conjugated phase-corrected FID (matches notebook)
    wref_phcorr = FIDToSpec(np.array(wref_phcorr_f), axis=-1)   # spectrum for plotting

    # ── Cell 14: plot phase-corrected wref ────────────────────────────────────────
    if args.save_plots:
        plot_mag_and_spectrum(wref_phcorr, PPM_AXIS, vr, vc,
                              "Phase-corrected wref",
                              os.path.join(out_dir, "fig_02c_adjnufft_phcorr.png"))
        print("[b0_corr] Saved fig_02c_adjnufft_phcorr.png")

    # ── Cells 16–18: water singlet basis ─────────────────────────────────────────
    print("[b0_corr] Building water singlet basis …")
    hdr = {"dwelltime": TS, "bandwidth": 1.0 / TS,
           "centralFrequency": center_freq, "fwhm": 0, "nucleus": "1H"}
    fid0          = make_singlet_fid(PPM_CENTER, center_freq, TIME_AXIS, args.singlet_lw)
    water_basis_0 = Basis(fid0, ["water_0"], headers=[hdr])

    # ── Cell 25: fit B0 voxel-by-voxel ───────────────────────────────────────────
    print("[b0_corr] Fitting B0 shift voxel-by-voxel …")
    nx, ny, _ = wref_phcorr.shape                    # (Ny, Nx, T) → nx=Ny, ny=Nx
    nvox       = nx * ny
    b0_is      = wref_phcorr_f.reshape(-1, N_SEQ)   # (nvox, T) FID
    brain_flat = brain_mask.reshape(-1)
    b0_shift   = np.full(nvox, np.nan, dtype=float)

    for i in range(nvox):
        if not brain_flat[i]:
            continue
        water_mrs = MRS(b0_is[i].copy(), cf=center_freq, bw=1.0 / TS, nucleus="1H")
        water_mrs._nuc_info.ppm_shift = PPM_CENTER
        water_mrs._calculate_axes()
        water_mrs.basis = water_basis_0
        water_mrs.rescaleForFitting()
        try:
            res        = water_mrs.fit(model="voigt",
                                       metab_groups=water_mrs.parse_metab_groups("separate_all"),
                                       baseline="poly, 0")
            b0_shift[i] = res.getShiftParams(units="hz")[0]
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            print(f"[b0_corr]   {i+1}/{nvox} voxels")

    b0_shift_hz_img = b0_shift.reshape(nx, ny)

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(b0_shift_hz_img, origin="lower", cmap="gray")
        plt.colorbar(im, ax=ax, label="B0 offset (Hz)")
        ax.set_title("B0 shift (Hz) — raw fit")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_02d_b0map_raw.png"), dpi=120)
        plt.close(fig)
        print("[b0_corr] Saved fig_02d_b0map_raw.png")

    # ── Cell 26: fill + smooth (default ON; --no-smooth skips it) ───────────────
    if not args.no_smooth:
        B0_fill_smoothed = smooth_fill_then_gaussian(b0_shift_hz_img, brain_mask,
                                                      sigma=args.smooth_sigma)
        B0_fill_smoothed[~brain_mask] = np.nan
        if args.save_plots:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(B0_fill_smoothed, origin="lower", cmap="gray")
            plt.colorbar(im, ax=ax, label="B0 offset smoothed (Hz)")
            ax.set_title(f"B0 map smoothed  σ={args.smooth_sigma}")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_02e_b0map_smoothed.png"), dpi=120)
            plt.close(fig)
    else:
        B0_fill_smoothed = b0_shift_hz_img   # --no-smooth: skip, use raw

    # ── Cell 27: large-shift voxels ───────────────────────────────────────────────
    large_shift_mask = np.abs(b0_shift_hz_img) > 40
    coords = np.argwhere(large_shift_mask)
    if coords.size > 0:
        print(f"[b0_corr] {len(coords)} voxels with |shift| > 40 Hz:")
        for (x, y) in coords:
            print(f"  voxel (row={x}, col={y}) -> {b0_shift_hz_img[x, y]:.1f} Hz")

    # ── Cell 29: save B0 map ──────────────────────────────────────────────────────
    B0_final = B0_fill_smoothed
    np.save(os.path.join(out_dir, "B0_map_pk.npy"), B0_final)
    np.save(os.path.join(out_dir, "B0_map.npy"),    B0_final)
    b0_nii = nib.Nifti1Image(
        np.ascontiguousarray(B0_final.T[::-1, :])[:, :, np.newaxis].astype(np.float32),
        affine)
    b0_nii.header.set_xyzt_units("mm")
    nib.save(b0_nii, os.path.join(out_dir, "B0_map.nii.gz"))
    print(f"[b0_corr] Saved B0_map.npy / B0_map.nii.gz  "
          f"range=[{np.nanmin(B0_final):.2f}, {np.nanmax(B0_final):.2f}] Hz")

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(B0_final, origin="lower", cmap="gray")
        plt.colorbar(im, ax=ax, label="B0 offset (Hz)")
        ax.set_title("B0 map — final")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_02f_b0map_final.png"), dpi=120)
        plt.close(fig)
        print("[b0_corr] Saved fig_02f_b0map_final.png")

    # ── Cell 30: save wref_resampled ──────────────────────────────────────────────
    np.save(os.path.join(out_dir, "wref_resampled.npy"), image_blurry_numpy)
    print(f"[b0_corr] Saved wref_resampled.npy  shape={image_blurry_numpy.shape}")

    patch_spicefit_tree(args.out_dir, [
        ("wref_adj_nufft.nii.gz", os.path.join(out_dir, "wref_adj_nufft.nii.gz"), "data-wref"),
        ("b0_map.nii.gz",      os.path.join(out_dir, "B0_map.nii.gz"),            "data-b0-map"),
        ("wref_phcorr.nii.gz", os.path.join(out_dir, "wref_phcorr_nifti.nii.gz"), "data-wref-phcorr"),
    ])
    print("[b0_corr] Done.")


if __name__ == "__main__":
    main()
