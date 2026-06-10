#!/usr/bin/env python3
"""
Step 3 — Lipid removal, following L2_lipidrm_finufft.ipynb cell-by-cell.

Reads  : <data_dir>/mrsi_data.npy, mrsi_ksp.npy, wref_o.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         [<data_dir>/ref_vox_spec.npy]  (optional, for cross-correlation step)
Writes : <out_dir>/lipid_removal/kt_mrsi_lprm.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/lipid_removal/adj_bf_lprm.nii.gz
         <out_dir>/lipid_removal/adj_bf_spice_crs_cr.nii.gz
         <out_dir>/lipid_removal/my_mrsi_lprm_f.nii.gz
         <out_dir>/lipid_removal/lpfree_phcorr_nifti.nii.gz
         <out_dir>/lipid_removal/fig_03*.png  (when --save-plots)

Usage:
    python scripts/03_lipid_removal.py \\
        --data-dir  ./invivo_260305/cr/ \\
        [--out-dir  ./output] \\
        [--ref-spec ./invivo_260305/cr/ref_vox_spec.npy] \\
        [--save-plots]
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
from scipy.signal import correlate
from warnings import filterwarnings
filterwarnings("ignore")

import mrinufft
from fsl_mrs.utils.misc import FIDToSpec, SpecToFID
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl.data.image import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import phase_corr


def parse_args():
    p = argparse.ArgumentParser(description="Lipid removal — step 3")
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
    # brain masks (two thresholds as in notebook)
    p.add_argument("--brain-threshold",  type=float, default=0.00034,
                   help="Raw wref threshold for lipid mask (notebook default 0.00034)")
    p.add_argument("--brain-threshold2", type=float, default=0.08,
                   help="Normalized wref threshold for phase-corr / SPICE mask (notebook default 0.08)")
    p.add_argument("--brain-erosion",    type=int,   default=3)
    # B0
    p.add_argument("--global-b0-shift",  type=float, default=None,
                   help="Manual global B0 shift in Hz (default: auto from spectral peak offset)")
    p.add_argument("--center-box",       type=int,   nargs=4, default=[22, 38, 16, 44],
                   metavar=("R0", "R1", "C0", "C1"),
                   help="Row/col bounds of high-SNR center box for B0 peak-offset (default: 22 38 16 44)")
    p.add_argument("--b0-peak-window",   type=float, default=0.1,
                   help="PPM window around PPM_CENTER for peak search (default 0.1)")
    # lipid removal
    p.add_argument("--lss-ppm-low",      type=float, default=0.7)
    p.add_argument("--lss-ppm-high",     type=float, default=1.8)
    p.add_argument("--lipid-beta",       type=float, default=200.0)
    p.add_argument("--n-lipid-voxels",   type=int,   default=500)
    p.add_argument("--nsigma-gmm",       type=float, default=0.2,
                   help="GMM sigma offset for lipid threshold (notebook default 0.2)")
    # phase correction
    p.add_argument("--phase-ppmlim",     type=float, nargs=2, default=[3.5, 5.0],
                   metavar=("LO", "HI"),
                   help="PPM limits for FSL phase correction (1.8+1.65 to 3.8+1.65 = 3.45 5.45)")
    # cross-correlation
    p.add_argument("--ref-spec",         default=None,
                   help="Path to ref_vox_spec.npy for cross-correlation frequency alignment")
    p.add_argument("--crosscorr-window", type=float, nargs=2, default=[2.6, 3.3],
                   metavar=("LO", "HI"))
    p.add_argument("--highsnr-box",      type=int,   nargs=4, default=[25, 35, 30, 40],
                   metavar=("R0", "R1", "C0", "C1"),
                   help="Row/col bounds for high-SNR mean spectrum in cross-correlation")
    # plot voxels
    p.add_argument("--plot-voxel",       type=int,   nargs=2, default=[41, 24],
                   metavar=("ROW", "COL"),
                   help="Voxel (row, col) to highlight in intermediate plots")
    p.add_argument("--plot-voxel-raw",   type=int,   nargs=2, default=[41, 24],
                   metavar=("ROW", "COL"),
                   help="Voxel for raw adj-NUFFT plot (Cell 9 default: row=41,col=24)")
    # misc
    p.add_argument("--ref-nii",          default=None)
    p.add_argument("--save-plots",       action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: LSS (Cells 20–21)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lss(data, ppm_axis, low_ppm=0.7, high_ppm=1.8):
    """data: (nx, ny, nz, npts)"""
    lipid_idx = np.where((ppm_axis >= low_ppm) & (ppm_axis <= high_ppm))[0]
    lss_map   = np.sum(np.abs(data[..., lipid_idx]), axis=-1)
    return lss_map, lipid_idx


# ─────────────────────────────────────────────────────────────────────────────
# Helper: GMM lipid mask with plot (Cells 22–25)
# ─────────────────────────────────────────────────────────────────────────────

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
        gmm = GaussianMixture(n_components=2, random_state=0).fit(logvals)
        means = gmm.means_.ravel()
        covs  = gmm.covariances_.ravel()
        lipid_idx = int(np.argmax(means))
        thr = float(np.exp(means[lipid_idx] - nsigma * np.sqrt(covs[lipid_idx])))
        method, gmm_model = "gmm", gmm
    except Exception:
        thr    = float(np.percentile(vals, 90))
        method = "percentile"

    mask = (lss2d >= thr)
    nsel = int(np.sum(mask))

    if nsel > max_voxels:
        sorted_flat = np.sort(vals)[::-1]
        thr    = float(sorted_flat[topN_fallback - 1]) if len(sorted_flat) >= topN_fallback else thr
        mask   = (lss2d >= thr)
        nsel   = int(np.sum(mask))
        method = "topN"

    if save_plots:
        logvals_all = np.log(np.maximum(vals, 1e-12))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # histogram
        ax1.hist(logvals_all, bins=80, density=True, alpha=0.6, color="C0",
                 label="log(LSS) histogram")
        xs = np.linspace(logvals_all.min(), logvals_all.max(), 400)
        if gmm_model is not None:
            for mu, cov, w in zip(gmm_model.means_.ravel(),
                                   gmm_model.covariances_.ravel(),
                                   gmm_model.weights_.ravel()):
                ax1.plot(xs, w * norm.pdf(xs, loc=mu, scale=np.sqrt(cov)), lw=2)
        ax1.axvline(np.log(thr), color="k", ls="--", label=f"threshold log={np.log(thr):.3f}")
        ax1.set_xlabel("log(LSS)")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.set_title(f"LSS histogram  method={method}  n={nsel}")

        # map + mask overlay
        im = ax2.imshow(lss2d, origin="lower", cmap="viridis")
        ax2.set_title("LSS map + lipid mask")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ys_m, xs_m = np.where(mask)
        for yy, xx in zip(ys_m, xs_m):
            ax2.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                    edgecolor="red", facecolor="none", linewidth=1.2))
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03c_lss_gmm.png"), dpi=120)
        plt.close(fig)

    return {"lipid_mask": mask, "threshold": thr, "n_selected": nsel, "method": method}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: L2 lipid removal (Cell 27)
# ─────────────────────────────────────────────────────────────────────────────

def lipid_removal_l2(data, lipid_basis, beta):
    """data: (nx, ny, nz, npts)  lipid_basis: (npts, rank)"""
    nx, ny, nz, npts = data.shape
    Linv = np.linalg.inv(np.eye(npts) + beta * (lipid_basis @ lipid_basis.conj().T))
    out  = np.zeros_like(data)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                out[ix, iy, iz, :] = Linv @ data[ix, iy, iz, :]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helper: voxel plot (magnitude map + spectrum) — Cells 28 / 30 / 36 / 39
# ─────────────────────────────────────────────────────────────────────────────

def plot_mag_and_voxel(spec_4d, PPM_AXIS, voxel_row, voxel_col, title, fname):
    """spec_4d: (nx, ny, nz, npts)  — uses spec_4d[voxel_row, voxel_col, 0, :]"""
    mag_map = np.mean(np.abs(spec_4d), axis=-1)[:, :, 0]
    spec_v  = spec_4d[voxel_row, voxel_col, 0, :].astype(np.complex64)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    im = axs[0].imshow(np.abs(mag_map), origin="lower", cmap="viridis")
    axs[0].set_title(f"Avg spectral magnitude")
    axs[0].set_xlabel("x (cols)")
    axs[0].set_ylabel("y (rows)")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].add_patch(Rectangle((voxel_col - 0.5, voxel_row - 0.5), 1, 1,
                                linewidth=2, edgecolor="red", facecolor="none"))

    axs[1].plot(PPM_AXIS, np.real(spec_v), label=f"Real")
    axs[1].plot(PPM_AXIS, np.abs(spec_v),  label="|S| (magnitude)", alpha=0.7)
    axs[1].set_title(f"{title}  voxel (row={voxel_row}, col={voxel_col})")
    axs[1].set_xlabel("ppm")
    axs[1].invert_xaxis()
    axs[1].set_ylabel("Signal")
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    plt.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: magnitude + B0 + spectrum plot — Cells 15 / 17
# ─────────────────────────────────────────────────────────────────────────────

def plot_mag_b0_spectrum(spec_3d, B0_map, PPM_AXIS, FREQ_AXIS, PPM_CENTER,
                          voxel_row, voxel_col, title, fname, phi0=0, window_ppm=0.1):
    mag_map = np.mean(np.abs(spec_3d), axis=-1)
    spec_v  = spec_3d[voxel_row, voxel_col, :].astype(np.complex64)
    spec_v *= np.exp(1j * np.deg2rad(phi0))
    spec_abs = np.abs(spec_v)

    idx_target = np.argmin(np.abs(PPM_AXIS - PPM_CENTER))
    freq_at_target = FREQ_AXIS[idx_target]
    window_mask = np.abs(PPM_AXIS - PPM_CENTER) <= window_ppm
    idxs = np.nonzero(window_mask)[0]
    peak_idx = idxs[np.argmax(spec_abs[idxs])]
    delta_hz = float(FREQ_AXIS[peak_idx] - freq_at_target)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axs[0].imshow(mag_map, cmap="viridis", origin="lower")
    axs[0].set_title("Avg spectral magnitude")
    axs[0].set_xlabel("x (cols)")
    axs[0].set_ylabel("y (rows)")
    axs[0].add_patch(Rectangle((voxel_col - 0.5, voxel_row - 0.5), 1, 1,
                                linewidth=2, edgecolor="red", facecolor="none"))
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(B0_map, cmap="coolwarm", origin="lower")
    axs[1].set_title("B0 map")
    axs[1].set_xlabel("x (cols)")
    axs[1].set_ylabel("y (rows)")
    axs[1].add_patch(Rectangle((voxel_col - 0.5, voxel_row - 0.5), 1, 1,
                                linewidth=2, edgecolor="red", facecolor="none"))
    b0_val = float(B0_map[voxel_row, voxel_col]) if B0_map.shape == mag_map.shape else 0.0
    axs[1].text(0.02, 0.98, f"b0={b0_val:.3f}\ndelta={delta_hz:.2f} Hz",
                transform=axs[1].transAxes, fontsize=10, va="top", ha="left", color="white",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.4"))
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].plot(PPM_AXIS, np.real(spec_v), label=f"Real (rot {phi0:.0f}°)")
    axs[2].plot(PPM_AXIS, spec_abs, label="|S|", alpha=0.7)
    axs[2].axvline(x=PPM_CENTER, color="k", ls="--", lw=1, label=f"{PPM_CENTER} ppm")
    axs[2].axvline(x=PPM_AXIS[peak_idx], color="r", ls="-", lw=1, label="Detected peak")
    axs[2].set_title(f"{title}  voxel (row={voxel_row}, col={voxel_col})")
    axs[2].set_xlabel("ppm")
    axs[2].invert_xaxis()
    axs[2].set_ylabel("Signal")
    axs[2].grid(alpha=0.3)
    axs[2].legend()
    plt.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: global B0 shift from spectral peaks (Cell 13)
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_b0_from_peaks(spec_3d, B0_map, PPM_AXIS, FREQ_AXIS,
                                  PPM_CENTER, brain_mask, center_box,
                                  window_ppm=0.1, b0_zero_thr_hz=2.0):
    r0, r1, c0, c1 = center_box
    box_mask = np.zeros_like(brain_mask, dtype=bool)
    box_mask[r0:r1, c0:c1] = True
    combined = box_mask & brain_mask & (np.abs(B0_map) <= b0_zero_thr_hz)

    idx_target = np.argmin(np.abs(PPM_AXIS - PPM_CENTER))
    freq_at_target = FREQ_AXIS[idx_target]
    window_mask = np.abs(PPM_AXIS - PPM_CENTER) <= window_ppm
    idxs = np.nonzero(window_mask)[0]

    deltas = []
    for (row, col) in np.argwhere(combined):
        mag = np.abs(spec_3d[row, col, :])
        if np.all(mag[idxs] == 0):
            continue
        peak_idx = idxs[np.argmax(mag[idxs])]
        deltas.append(float(FREQ_AXIS[peak_idx] - freq_at_target))

    if not deltas:
        # fallback to B0 map median inside brain
        valid = B0_map[brain_mask & np.isfinite(B0_map)]
        return float(np.median(valid)) if valid.size > 0 else 0.0
    return float(np.mean(deltas))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: cross-correlation frequency shift (Cell 37)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_for_corr(spec):
    s = np.abs(np.asarray(spec)).astype(float)
    nz = s > 0
    if nz.sum() > 2:
        s = s - np.mean(s[nz])
        x = np.arange(len(s))
        p = np.polyfit(x[nz], s[nz], 1)
        s[nz] -= np.polyval(p, x[nz])
    return s


def compute_freq_shift_hz(spec_ref, spec_target, PPM_AXIS, center_freq_mhz, window=None):
    n   = len(spec_ref)
    ppm = np.asarray(PPM_AXIS)

    if window is not None:
        lo, hi = window
        mask   = (ppm >= lo) & (ppm <= hi)
        s_ref  = _preprocess_for_corr(spec_ref  * mask)
        s_tar  = _preprocess_for_corr(spec_target * mask)
    else:
        s_ref = _preprocess_for_corr(spec_ref)
        s_tar = _preprocess_for_corr(spec_target)

    corr  = correlate(s_ref, s_tar, mode="full")
    lags  = np.arange(-n + 1, n)
    peak  = int(np.argmax(corr))
    # sub-sample refinement
    if 0 < peak < len(corr) - 1:
        y0, y1, y2 = corr[peak - 1], corr[peak], corr[peak + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    frac_lag = lags[peak] + delta
    ppm_step = ppm[1] - ppm[0]
    return float(frac_lag * ppm_step * center_freq_mhz)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    out_dir  = os.path.join(args.out_dir, "lipid_removal")
    os.makedirs(out_dir, exist_ok=True)

    coilmap_dir = os.path.join(args.out_dir, "coilmap")
    b0map_dir   = os.path.join(args.out_dir, "b0map")

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
    print(f"[lipidrm] sweep={sweepwidth:.1f} Hz  TS={TS:.3e} s  PPM_CENTER={PPM_CENTER}")

    # ── Cell 2: scale & cache MRSI trajectory ────────────────────────────────────
    ksp_cache = os.path.join(out_dir, "mrsi_ksp_scaled.npy")
    if not os.path.exists(ksp_cache):
        print("[lipidrm] Scaling MRSI trajectory …")
        mrsi_ksp = np.load(data_dir + "mrsi_ksp.npy")
        scale    = args.mrsi_ksp_scale or (30.37478212844472 / 32.0)
        mrsi_ksp[:2, ...] *= scale
        mrsi_ksp[2, ...]   = np.flip(mrsi_ksp[2, ...])
        np.save(ksp_cache, mrsi_ksp)
    mrsi_ksp_scaled = np.load(ksp_cache, mmap_mode="r")
    trej = mrsi_ksp_scaled.T.astype(Trej_D_TYPE)
    print(f"[lipidrm] traj range x:[{trej[:,:,0].min():.3f},{trej[:,:,0].max():.3f}]  "
          f"y:[{trej[:,:,1].min():.3f},{trej[:,:,1].max():.3f}]  "
          f"z:[{trej[:,:,2].min():.3f},{trej[:,:,2].max():.3f}]")

    # load MRSI raw data & reorder
    print("[lipidrm] Loading MRSI data …")
    mrsi_raw = np.load(data_dir + "mrsi_data.npy", mmap_mode="r").astype(D_TYPE)
    wref_img = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    mrsi_reordered = np.transpose(mrsi_raw, (0, 3, 2, 1)).reshape(1, N_COILS, -1)

    # ── Cell 6: load coilmap ──────────────────────────────────────────────────────
    coil_smap_raw = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"), mmap_mode="r")
    coil_smap     = np.repeat(coil_smap_raw[np.newaxis, :, :, :, np.newaxis],
                               N_SEQ, axis=-1).astype(D_TYPE)   # (1, C, Ny, Nx, T)
    smap_time     = coil_smap.squeeze(0)                        # (C, Ny, Nx, T)
    print(f"[lipidrm] coil_smap shape={coil_smap.shape}")

    # ── Cell 7: build finufft operator ───────────────────────────────────────────
    result_shape = (Ny, Nx, N_SEQ)
    NufftOp    = mrinufft.get_operator("finufft")
    print("[lipidrm] Building finufft NUFFT operator …")
    nufft_mrsi = NufftOp(trej, shape=result_shape, n_coils=N_COILS,
                          n_batchs=1, squeeze_dims=True, smaps=smap_time)

    # ── Cell 8: adjoint NUFFT ─────────────────────────────────────────────────────
    print("[lipidrm] Adjoint NUFFT …")
    image_blurry_numpy = nufft_mrsi.adj_op(mrsi_reordered)   # (Ny, Nx, N_SEQ)
    print(f"[lipidrm] image_blurry_numpy shape={image_blurry_numpy.shape}")

    mag_map_2d = np.mean(np.abs(image_blurry_numpy), axis=-1)   # kept for phase-corr mask

    # ── Cell 9: plot raw adj NUFFT ────────────────────────────────────────────────
    plot_row, plot_col = args.plot_voxel_raw
    if args.save_plots:
        spec_v   = image_blurry_numpy[plot_row, plot_col, :].astype(D_TYPE)
        brain_er = binary_erosion(np.abs(wref_img[:, :, 0]) > args.brain_threshold, iterations=3)
        mag_masked = np.where(brain_er, mag_map_2d, np.nan)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        im = axs[0].imshow(mag_masked, cmap="viridis", origin="lower")
        axs[0].set_title("Avg spectral magnitude (adj NUFFT)")
        axs[0].set_xlabel("x (cols)")
        axs[0].set_ylabel("y (rows)")
        plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].add_patch(Rectangle((plot_col - 0.5, plot_row - 0.5), 1, 1,
                                    linewidth=2, edgecolor="red", facecolor="none"))
        axs[1].plot(PPM_AXIS, np.real(spec_v), label="Real")
        axs[1].plot(PPM_AXIS, np.abs(spec_v),  label="|S|", alpha=0.7)
        axs[1].set_title(f"Spectrum at voxel (row={plot_row}, col={plot_col})")
        axs[1].set_xlabel("ppm")
        axs[1].invert_xaxis()
        axs[1].set_ylabel("Signal")
        axs[1].grid(alpha=0.3)
        axs[1].legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03a_adjnufft_spectrum.png"), dpi=120)
        plt.close(fig)
        print("[lipidrm] Saved fig_03a_adjnufft_spectrum.png")

    # ── Cell 10: save NIfTI before lipid removal ──────────────────────────────────
    ref_img_path = args.ref_nii or (data_dir + "meas_MID00125_FID81014_mrsi_64_cr_adj300.nii.gz")
    try:
        ref_img_obj = Image(ref_img_path)
        affine      = ref_img_obj.voxToWorldMat
    except Exception:
        ref_img_obj = None
        affine      = np.eye(4)

    img_ref          = np.abs(wref_img[:, :, 0])
    brain_nolip_mask = img_ref > args.brain_threshold         # raw threshold (Cell 12)

    img_masked = image_blurry_numpy * brain_nolip_mask[:, :, np.newaxis]
    fid_adj    = SpecToFID(img_masked, axis=-1).transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(fid_adj, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "adj_bf_lprm.nii.gz"))
    print("[lipidrm] Saved adj_bf_lprm.nii.gz")

    # ── Cell 12: brain mask 1 (raw wref) — plot ───────────────────────────────────
    if args.save_plots:
        center_box    = args.center_box
        r0, r1, c0, c1 = center_box
        center_box_mask = np.zeros_like(brain_nolip_mask, dtype=bool)
        center_box_mask[r0:r1, c0:c1] = True
        brain_center_mask = center_box_mask & brain_nolip_mask

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_ref, origin="lower", cmap="gray")
        axes[0].imshow(brain_nolip_mask, origin="lower", cmap="Reds", alpha=0.35)
        axes[0].set_title(f"wref + brain mask (thr={args.brain_threshold:.3g})")
        plt.colorbar(axes[0].images[0], ax=axes[0])

        axes[1].imshow(img_ref, origin="lower", cmap="gray")
        axes[1].imshow(brain_nolip_mask, origin="lower", cmap="Reds", alpha=0.3)
        axes[1].imshow(brain_center_mask, origin="lower", cmap="Blues", alpha=0.5)
        axes[1].set_title("Brain mask + center box")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03b1_brain_mask.png"), dpi=120)
        plt.close(fig)
        print("[lipidrm] Saved fig_03b1_brain_mask.png")

    # ── Cell 13–14: load B0 map, compute global shift ─────────────────────────────
    B0_map = np.load(os.path.join(b0map_dir, "B0_map.npy"))

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(B0_map, origin="lower")
        plt.colorbar(im, ax=ax, label="B0 offset (Hz)")
        ax.set_title("Voxel-wise B0 shift (Hz)")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03b2_b0map.png"), dpi=120)
        plt.close(fig)
        print("[lipidrm] Saved fig_03b2_b0map.png")

    if args.global_b0_shift is not None:
        global_b0_shift = args.global_b0_shift
    else:
        global_b0_shift = compute_global_b0_from_peaks(
            image_blurry_numpy, B0_map, PPM_AXIS, FREQ_AXIS,
            PPM_CENTER, brain_nolip_mask, args.center_box,
            window_ppm=args.b0_peak_window,
        )
    print(f"[lipidrm] global_b0_shift = {global_b0_shift:.3f} Hz")

    # ── Cell 15: plot pre-B0-correction ──────────────────────────────────────────
    vr, vc = args.plot_voxel
    if args.save_plots:
        plot_mag_b0_spectrum(
            image_blurry_numpy, B0_map, PPM_AXIS, FREQ_AXIS, PPM_CENTER,
            vr, vc, "Pre-B0-correction",
            os.path.join(out_dir, "fig_03b3_pre_b0corr.png"),
            window_ppm=args.b0_peak_window,
        )
        print("[lipidrm] Saved fig_03b3_pre_b0corr.png")

    # ── Cell 16: apply global B0 correction ──────────────────────────────────────
    mrsi_raw_fid      = SpecToFID(image_blurry_numpy, axis=-1)
    global_b0_arr     = np.exp(-1j * 2 * np.pi * global_b0_shift * TIME_AXIS)
    mrsi_fid_corr     = global_b0_arr * mrsi_raw_fid
    mrsi_corr         = FIDToSpec(mrsi_fid_corr, axis=-1)

    # ── Cell 17: plot post-B0-correction ─────────────────────────────────────────
    if args.save_plots:
        plot_mag_b0_spectrum(
            mrsi_corr, B0_map, PPM_AXIS, FREQ_AXIS, PPM_CENTER,
            vr, vc, "Post-B0-correction",
            os.path.join(out_dir, "fig_03b4_post_b0corr.png"),
            window_ppm=args.b0_peak_window,
        )
        print("[lipidrm] Saved fig_03b4_post_b0corr.png")

    # ── Cell 21: compute LSS map + plot ──────────────────────────────────────────
    print("[lipidrm] Computing LSS map …")
    mrsi_b0_corr_4dim = mrsi_corr[:, :, np.newaxis, :]          # (Ny, Nx, 1, T)
    lss_map, _        = compute_lss(mrsi_b0_corr_4dim, PPM_AXIS,
                                    low_ppm=args.lss_ppm_low, high_ppm=args.lss_ppm_high)
    lss_2d = np.squeeze(lss_map)

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(lss_2d, origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, label="LSS")
        ax.set_title("LSS map")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03c_lss_map.png"), dpi=120)
        plt.close(fig)
        print("[lipidrm] Saved fig_03c_lss_map.png")

    # ── Cell 25: GMM lipid mask + plots ──────────────────────────────────────────
    res        = select_lipid_mask_gmm_simple(
        lss_map, out_dir=out_dir, nsigma=args.nsigma_gmm,
        max_voxels=args.n_lipid_voxels, topN_fallback=100,
        save_plots=args.save_plots,
    )
    lipid_mask = res["lipid_mask"]
    print(f"[lipidrm] Lipid mask: n={res['n_selected']}  method={res['method']}  thr={res['threshold']:.4e}")

    # ── Cell 28: plot spectrum before L2 ─────────────────────────────────────────
    if args.save_plots:
        plot_mag_and_voxel(
            mrsi_b0_corr_4dim, PPM_AXIS, vr, vc,
            "Before L2 lipid removal",
            os.path.join(out_dir, "fig_03d1_pre_l2.png"),
        )
        print("[lipidrm] Saved fig_03d1_pre_l2.png")

    # ── Cell 29: L2 lipid removal ─────────────────────────────────────────────────
    print("[lipidrm] L2 lipid removal …")
    mask3d              = lipid_mask[:, :, np.newaxis]
    mrsi_fid_b0corr_4d  = mrsi_fid_corr[:, :, np.newaxis, :]   # (Ny, Nx, 1, T) FID
    lipid_fids          = mrsi_fid_b0corr_4d[mask3d]            # (N_vox, T)
    lipid_basis         = lipid_fids.T                           # (T, N_vox)
    mrsi_fid_lprm_4dim  = lipid_removal_l2(mrsi_fid_b0corr_4d, lipid_basis, beta=args.lipid_beta)
    mrsi_lprm_4dim      = FIDToSpec(mrsi_fid_lprm_4dim, axis=-1)  # (Ny, Nx, 1, T) spectrum

    # ── Cell 30: plot spectrum after L2 ──────────────────────────────────────────
    if args.save_plots:
        plot_mag_and_voxel(
            mrsi_lprm_4dim, PPM_AXIS, vr, vc,
            "After L2 lipid removal",
            os.path.join(out_dir, "fig_03d2_post_l2.png"),
        )
        print("[lipidrm] Saved fig_03d2_post_l2.png")

    # ── Cell 31: brain mask 2 (normalized wref) ───────────────────────────────────
    wref_2d      = np.abs(wref_img.squeeze(-1))
    wref_min, wref_max = wref_2d.min(), wref_2d.max()
    wref_norm    = (wref_2d - wref_min) / (wref_max - wref_min + 1e-12)
    brain_mask2  = wref_norm > args.brain_threshold2
    brain_mask_inner = binary_erosion(brain_mask2, iterations=args.brain_erosion)

    if args.save_plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(wref_norm, origin="lower", cmap="gray")
        ax.imshow(brain_mask2, origin="lower", cmap="Reds", alpha=0.35)
        ax.set_title(f"Normalized wref + brain mask (thr={args.brain_threshold2})")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_03e_brain_mask2.png"), dpi=120)
        plt.close(fig)
        print("[lipidrm] Saved fig_03e_brain_mask2.png")

    # ── Cell 32: phase correction via utils.recon.phase_corr ────────────────
    print(f"[lipidrm] Phase correction  ppmlim={args.phase_ppmlim}  method=max-real …")
    lpfree_phcorr_f = phase_corr(
        mrsi_fid_lprm_4dim[:, :, 0, :],    # (Ny, Nx, T) FID
        mag_map_2d = mag_map_2d,
        brain_mask = brain_mask2,
        TS         = TS,
        img_shape  = (Ny, Nx),
        out_dir    = out_dir,
        ppmlim     = tuple(args.phase_ppmlim),
        ref_img    = ref_img_obj,
        out_fname  = "lpfree_phcorr_nifti",
        method     = "max-real",
    )
    # lpfree_phcorr_f : (Ny, Nx, T) conjugated phase-corrected FID
    lpfree_phcorr_spec = FIDToSpec(lpfree_phcorr_f, axis=-1)      # (Ny, Nx, T) spectrum
    mrsi_lprm_4dim     = lpfree_phcorr_spec[:, :, np.newaxis, :]  # (Ny, Nx, 1, T)
    print("[lipidrm] Phase correction done.")

    # ── Cell 35: mask outside brain ───────────────────────────────────────────────
    mrsi_lprm_masked = mrsi_lprm_4dim[:, :, 0, :].copy()
    mrsi_lprm_masked[~brain_nolip_mask] = 0
    mrsi_lprm_4dim   = mrsi_lprm_masked[:, :, np.newaxis, :]

    # ── Cell 36: plot after masking ───────────────────────────────────────────────
    if args.save_plots:
        plot_mag_and_voxel(
            mrsi_lprm_4dim, PPM_AXIS, vr, vc,
            "After masking (before cross-corr)",
            os.path.join(out_dir, "fig_03f1_after_mask.png")
        )
        print("[lipidrm] Saved fig_03f1_after_mask.png")

    # ── Cell 37: cross-correlation frequency alignment ────────────────────────────
    ref_spec_path = args.ref_spec or (data_dir + "ref_vox_spec.npy")
    if os.path.exists(ref_spec_path):
        print("[lipidrm] Cross-correlation frequency alignment …")
        r0h, r1h, c0h, c1h = args.highsnr_box
        brain_highsnr = np.zeros_like(brain_nolip_mask, dtype=bool)
        brain_highsnr[r0h:r1h, c0h:c1h] = True
        brain_highsnr &= brain_nolip_mask
        brain_highsnr_3d = brain_highsnr[:, :, np.newaxis]

        mrsi_lprm_selected = lpfree_phcorr_spec * brain_highsnr_3d  # note: 3D spectrum

        # mean FID from high-SNR box then convert to spectrum
        fid_box    = SpecToFID(mrsi_lprm_selected, axis=-1)[r0h:r1h, c0h:c1h]
        mean_spec  = FIDToSpec(np.mean(fid_box, axis=(0, 1))).astype(D_TYPE)

        ref_spec = np.load(ref_spec_path, mmap_mode="r").astype(D_TYPE).squeeze()

        df_hz = compute_freq_shift_hz(
            ref_spec, mean_spec, PPM_AXIS, center_freq,
            window=tuple(args.crosscorr_window),
        )
        print(f"[lipidrm] Cross-correlation shift = {df_hz:.3f} Hz")

        if args.save_plots:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(PPM_AXIS, np.abs(mean_spec), label="Mean Spec (ROI)")
            ax.plot(PPM_AXIS, np.abs(ref_spec),  label="Reference Spec")
            ax.invert_xaxis()
            ax.legend()
            ax.set_xlabel("PPM")
            ax.set_ylabel("Magnitude")
            ax.set_title("Mean Spectrum vs Reference (before correction)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_03f2_crosscorr_before.png"), dpi=120)
            plt.close(fig)

        # apply shift
        mrsi_lprm_masked_fid  = SpecToFID(mrsi_lprm_4dim, axis=-1)
        shift_arr             = np.exp(-1j * 2 * np.pi * (-df_hz) * TIME_AXIS)
        mrsi_fid_lprm_4dim_cr = shift_arr * mrsi_lprm_masked_fid
        mrsi_lprm_4dim        = FIDToSpec(mrsi_fid_lprm_4dim_cr, axis=-1)

        if args.save_plots:
            fid_box_corr = mrsi_fid_lprm_4dim_cr[r0h:r1h, c0h:c1h, 0]
            mean_spec_corr = FIDToSpec(np.mean(fid_box_corr, axis=(0, 1))).astype(D_TYPE)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(PPM_AXIS, np.abs(mean_spec_corr), label="Corrected Mean Spec (ROI)")
            ax.plot(PPM_AXIS, np.abs(ref_spec),       label="Reference Spec")
            ax.invert_xaxis()
            ax.legend()
            ax.set_xlabel("PPM")
            ax.set_ylabel("Magnitude")
            ax.set_title(f"Corrected Mean Spectrum vs Reference (shift={df_hz:.2f} Hz)")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_03f3_crosscorr_after.png"), dpi=120)
            plt.close(fig)
            print("[lipidrm] Saved fig_03f2/f3_crosscorr.png")
    else:
        print(f"[lipidrm] ref_vox_spec not found at {ref_spec_path} — skipping cross-correlation")

    # ── Cell 39: plot final corrected spectrum ────────────────────────────────────
    if args.save_plots:
        plot_mag_and_voxel(
            mrsi_lprm_4dim, PPM_AXIS, vr, vc,
            "Final (after all corrections)",
            os.path.join(out_dir, "fig_03g_final.png"),
        )
        print("[lipidrm] Saved fig_03g_final.png")

    # ── Cell 38: save NIfTI of final lipid-free result ───────────────────────────
    lprm_3d     = mrsi_lprm_4dim[:, :, 0, :]                           # (Ny, Nx, T) spectrum
    save_fid    = SpecToFID(lprm_3d.reshape(Ny, Nx, N_SEQ), axis=-1)   # FID
    save_4d     = save_fid.transpose(1, 0, 2)[:, :, np.newaxis, :]
    gen_nifti_mrs(save_4d, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "adj_bf_spice_crs_cr.nii.gz"))
    print("[lipidrm] Saved adj_bf_spice_crs_cr.nii.gz")

    # ── Cell 42: forward NUFFT → kt space ────────────────────────────────────────
    print("[lipidrm] Forward NUFFT → kt_mrsi_lprm.npy …")
    mrsi_lprm_globalbeta = mrsi_lprm_4dim[:, :, 0, :]          # (Ny, Nx, T) spectrum
    kt_mrsi_lprm         = nufft_mrsi.op(mrsi_lprm_globalbeta)
    np.save(os.path.join(out_dir, "kt_mrsi_lprm.npy"), kt_mrsi_lprm)
    print(f"[lipidrm] Saved kt_mrsi_lprm.npy  shape={kt_mrsi_lprm.shape}")

    # ── Cell 43: save NIfTI for SPICE ────────────────────────────────────────────
    mrsi_lprm_f   = SpecToFID(mrsi_lprm_globalbeta, axis=-1)   # FID (Ny, Nx, T)
    mrsi_lprm_f   = mrsi_lprm_f.transpose(1, 0, 2)             # (Nx, Ny, T)
    mrsi_lprm_f_4 = mrsi_lprm_f[:, :, np.newaxis, :]           # (Nx, Ny, 1, T)
    gen_nifti_mrs(mrsi_lprm_f_4, dwelltime=TS, spec_freq=297.219,
                  affine=affine).save(os.path.join(out_dir, "my_mrsi_lprm_f.nii.gz"))
    print("[lipidrm] Saved my_mrsi_lprm_f.nii.gz")

    print("[lipidrm] Done.")


if __name__ == "__main__":
    main()
