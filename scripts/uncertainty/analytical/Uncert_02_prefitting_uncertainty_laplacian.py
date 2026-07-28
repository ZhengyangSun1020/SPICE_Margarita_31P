#!/usr/bin/env python3
"""
Step 9 — Pre-fitting uncertainty (Laplacian).

Supports two modes selected by --mode:

  voxelwise (default)
    Loads mHm_*.npy files from the Hessian directory, draws posterior samples
    via Laplace approximation, computes spectral std map, and saves plots.

    Reads  : <data_dir>/wref_o.npy
             <data_dir>/sigma_noise.npy
             <out_dir>/spice_<run_tag>/V_subspace.npy
             <out_dir>/spice_<run_tag>/U_est.npy
             <out_dir>/hessian_<run_tag>/mHm_*.npy

  lobpcg
    Loads pre-computed LOBPCG eigenpairs (from step 10), draws low-rank
    posterior samples of U, maps through V to spectrum domain, and plots.

    Reads  : <data_dir>/wref_o.npy
             <data_dir>/sigma_noise.npy  (optional; fallback to --sigma2)
             <out_dir>/spice_<run_tag>/V_subspace.npy
             <out_dir>/lobpcg_<run_tag>/lobpcg_Q.npy
             <out_dir>/lobpcg_<run_tag>/lobpcg_vals.npy

Writes : <out_dir>/uncertainty_<run_tag>/fig_09_uncert_map.png
         <out_dir>/uncertainty_<run_tag>/fig_09_spice_mean.png
         <out_dir>/uncertainty_<run_tag>/posterior_std.npy
         (tag e.g. w5000_l0.0001; omit --run-tag for legacy names without suffix)

Usage:
    # voxelwise (default) — requires step 08 hessian outputs
    python scripts/uncertainty/analytical/Uncert_02_prefitting_uncertainty_laplacian.py \
        --data-dir  data/processed/invivo_260623_01 \
        --run-tag   w5000_l0.0001 \
        --rank 20 --n-samples 100 --brain-threshold 0.16 --brain-erosion 1 [--threshold 2.5e-6]

    # lobpcg — requires step 10 LOBPCG outputs
    python scripts/uncertainty/analytical/Uncert_02_prefitting_uncertainty_laplacian.py \
        --mode lobpcg \
        --data-dir  data/processed/invivo_260623_01 \
        --run-tag   w5000_l0.0001 \
        --rank 20 --n-samples 100 --brain-threshold 0.16 --brain-erosion 1 [--threshold 2.5e-6]
"""

import argparse
import os
import sys
from pathlib import Path
from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
from nifti_mrs.create_nmrs import gen_nifti_mrs
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.ticker as mticker
import numpy as np
from scipy.ndimage import binary_erosion
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.scan_params import load_scan_params

D_TYPE = np.complex64


# ── helpers ───────────────────────────────────────────────────────────────────

def fid_to_spec(fid):
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm="ortho"), axes=-1)


def spec_to_fid(spec):
    return np.fft.ifft(np.fft.ifftshift(spec, axes=-1), axis=-1, norm="ortho")


def _save_niftimrs(std_img, mean_spec, out_dir, TS, affine):
    """Save posterior mean and std as NIfTI-MRS. Input shape (Ny, Nx, N_SEQ) spectrum domain."""
    def _to_nii(arr):
        fid = spec_to_fid(np.asarray(arr, dtype=np.complex64))
        return np.ascontiguousarray(fid.transpose(1, 0, 2)[::-1, :, :])[:, :, np.newaxis, :]
    gen_nifti_mrs(_to_nii(mean_spec), dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "posterior_mean.nii.gz"))
    gen_nifti_mrs(_to_nii(std_img),   dwelltime=TS, spec_freq=297.219, affine=affine).save(
        os.path.join(out_dir, "posterior_std.nii.gz"))
    print("[uncert-post] Saved posterior_mean.nii.gz  posterior_std.nii.gz")


def sample_complex_mvnormal(mean, cov, n_samples=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    cov = np.asarray(cov)
    cov = 0.5 * (cov + cov.conj().T)
    d   = cov.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals.real, 0.0, None)
    L = evecs * np.sqrt(evals)[None, :]
    z = (rng.standard_normal((n_samples, d))
         + 1j * rng.standard_normal((n_samples, d))) / np.sqrt(2.0)
    return mean[None, :] + z @ L.conj().T


def sample_lowrank(Q, vals_safe, n_samples, sigma2=1.0, seed=None, batch_size=100):
    """Draw samples from CN(0, sigma2 * Q Λ^{-1} Q^H) — low-rank posterior over U."""
    rng = np.random.default_rng(seed)
    d, k = Q.shape
    lam_inv_sqrt = np.sqrt(1.0 / (vals_safe + 1e-20))
    samples = np.zeros((n_samples, d), dtype=D_TYPE)
    for b0 in tqdm(range(0, n_samples, batch_size), desc="Sampling (lobpcg)"):
        b1  = min(n_samples, b0 + batch_size)
        bs  = b1 - b0
        Zk  = rng.standard_normal((bs, k)) + 1j * rng.standard_normal((bs, k))
        Zk /= np.sqrt(2)
        low = ((Zk * lam_inv_sqrt[None, :]) @ Q.T).astype(D_TYPE)
        if sigma2 != 1.0:
            low *= np.sqrt(sigma2)
        samples[b0:b1] = low
    return samples


def build_dataset_auto(mHm_dir, num_voxels, V, mu_map,
                        n_samples=100, dtype=D_TYPE, seed=0, cov_scale=1e-4):
    """
    mHm_dir   : folder with mHm_<idx>.npy files
    num_voxels: total image voxels (Ny*Nx)
    V         : (N_seq, rank)
    mu_map    : (num_voxels, N_seq) — mean spectrum per voxel
    returns   : data (n_samples, num_voxels, N_seq), mask (num_voxels,)
    """
    rng      = np.random.default_rng(seed)
    mHm_dir  = Path(mHm_dir)
    N_seq    = V.shape[0]
    data     = np.zeros((n_samples, num_voxels, N_seq), dtype=dtype)
    mask     = np.zeros(num_voxels, dtype=bool)

    files = sorted(mHm_dir.glob("mHm_*.npy"))
    print(f"[uncert-post] Found {len(files)} mHm files in {mHm_dir}")

    for f in tqdm(files, desc="Sampling voxels"):
        try:
            vox_idx = int(f.stem.split("_")[1])
        except Exception:
            print(f"[WARN] skipping bad filename: {f.name}")
            continue
        if vox_idx < 0 or vox_idx >= num_voxels:
            continue

        mHm   = np.load(f).astype(np.complex128)      # (rank, rank)
        Sigma = cov_scale * (V @ mHm @ V.conj().T)    # (N_seq, N_seq)
        mu    = mu_map[vox_idx].astype(np.complex128)  # (N_seq,)

        samples = sample_complex_mvnormal(mu, Sigma, n_samples=n_samples, rng=rng)
        data[:, vox_idx, :] = samples.astype(dtype, copy=False)
        mask[vox_idx] = True

    n_covered = int(mask.sum())
    print(f"[uncert-post] Covered {n_covered}/{num_voxels} voxels")
    return data, mask


# ── plot functions ────────────────────────────────────────────────────────────

def plot_average_variation(
    mean_spec:  np.ndarray,
    std_img:    np.ndarray,
    img_shape:  tuple,
    voxel_x:    int,
    voxel_y:    int,
    wref_norm:  np.ndarray = None,
    brain_mask: np.ndarray = None,
    threshold:  float = None,
    PPM_AXIS:   np.ndarray = None,
    dark_mode:  bool = True,
):
    import copy
    nx, ny, nt = img_shape
    mean_spec = np.asarray(mean_spec).reshape(nx, ny, nt)
    std_img   = np.asarray(std_img).reshape(nx, ny, nt)

    mask_2d = brain_mask if brain_mask is not None else np.ones((nx, ny), dtype=bool)

    mean_map = np.mean(np.abs(mean_spec), axis=-1)       # (nx, ny)
    std_map  = np.mean(np.abs(std_img),   axis=-1)       # (nx, ny)
    mean_map = np.where(mask_2d, mean_map, np.nan)
    std_map  = np.where(mask_2d, std_map,  np.nan)

    c = "white" if dark_mode else "black"
    bg = "black" if dark_mode else "white"

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    if dark_mode:
        fig.patch.set_facecolor("black")
    for ax in axs:
        if dark_mode:
            ax.set_facecolor("black")
        ax.title.set_color(c)
        ax.xaxis.label.set_color(c)
        ax.yaxis.label.set_color(c)
        ax.tick_params(colors=c)
        for sp in ax.spines.values():
            sp.set_color(c)

    def _wref_overlay(ax):
        if wref_norm is not None:
            wref_brain = np.where(mask_2d, wref_norm, np.nan)
            wref_cmap  = copy.copy(plt.cm.get_cmap("gray"))
            wref_cmap.set_bad(color=bg)
            ax.imshow(wref_brain, origin="lower", cmap=wref_cmap, alpha=0.5, zorder=0)

    # Panel 1 — mean spectral magnitude
    cmap_g = copy.copy(plt.cm.get_cmap("viridis"))
    cmap_g.set_bad(color=bg)
    _wref_overlay(axs[0])
    im0 = axs[0].imshow(mean_map, origin="lower", cmap=cmap_g, zorder=1)
    axs[0].set_title("Mean spectral magnitude")
    cbar0 = plt.colorbar(im0, ax=axs[0], fraction=0.046)
    cbar0.ax.yaxis.set_tick_params(color=c)
    plt.setp(cbar0.ax.get_yticklabels(), color=c)
    axs[0].plot(voxel_x, voxel_y, "g+", markersize=10, markeredgewidth=2, zorder=3)

    # Panel 2 — posterior std map
    vmax = threshold if threshold is not None else (
        float(np.nanpercentile(std_map[mask_2d], 95)) if mask_2d.any() else None
    )
    cmap_r = copy.copy(plt.cm.get_cmap("Reds"))
    cmap_r.set_bad(color=bg)
    _wref_overlay(axs[1])
    im1 = axs[1].imshow(std_map, origin="lower", vmin=0, vmax=vmax,
                          cmap=cmap_r, alpha=0.9, zorder=1)
    axs[1].set_title("Posterior std (mean over spectrum)")
    cbar1 = plt.colorbar(im1, ax=axs[1], fraction=0.046)
    cbar1.ax.yaxis.set_tick_params(color=c)
    cbar1.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
    plt.setp(cbar1.ax.get_yticklabels(), color=c)
    axs[1].plot(voxel_x, voxel_y, "g+", markersize=10, markeredgewidth=2, zorder=3)

    # Panel 3 — mean ± std spectrum at selected voxel
    mu  = np.abs(mean_spec[voxel_y, voxel_x, :])
    sig = np.abs(std_img[voxel_y,   voxel_x, :])
    x_axis = PPM_AXIS if PPM_AXIS is not None else np.arange(nt)
    axs[2].plot(x_axis, mu, color=c, label="Mean")
    axs[2].fill_between(x_axis, mu - sig, mu + sig, alpha=0.35,
                         color="tomato", label="±1 std")
    axs[2].set_title(f"Posterior uncertainty  voxel ({voxel_y},{voxel_x})")
    axs[2].set_xlabel("ppm" if PPM_AXIS is not None else "index")
    axs[2].set_ylabel("|Spectrum|")
    if PPM_AXIS is not None:
        axs[2].invert_xaxis()
    axs[2].grid(True, alpha=0.3, color="gray")
    axs[2].legend(labelcolor=c, facecolor=bg)
    if dark_mode:
        axs[2].set_facecolor("black")
        axs[2].tick_params(colors="white")
        for sp in axs[2].spines.values():
            sp.set_color("white")

    plt.tight_layout()
    return fig


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Uncertainty post-processing — step 9")
    p.add_argument("--mode",           choices=["voxelwise", "lobpcg"],
                   default="voxelwise",
                   help="voxelwise: Laplace/mHm approach; lobpcg: load Q/vals from step 10")
    p.add_argument("--data-dir",       required=True)
    p.add_argument("--out-dir",        default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--hess-dir",       default=None,
                   help="[voxelwise] Directory with mHm_*.npy (default: <out-dir>/hessian)")
    p.add_argument("--lobpcg-dir",     default=None,
                   help="[lobpcg] Directory with lobpcg_Q.npy / lobpcg_vals.npy "
                        "(default: <out-dir>/lobpcg)")
    p.add_argument("--sigma2",         type=float, default=1.0,
                   help="[lobpcg] Posterior variance scale; overridden by sigma_noise.npy if found")
    p.add_argument("--dwelltime",      type=float, default=None)
    p.add_argument("--k-points",       type=int, default=None)
    p.add_argument("--n-seq-points",   type=int,   default=300)
    p.add_argument("--dim",            type=int,   nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    p.add_argument("--center-freq",    type=float, default=None)
    p.add_argument("--ppm-center",     type=float, default=3.027)
    p.add_argument("--rank",           type=int,   default=20)
    p.add_argument("--brain-threshold",type=float, default=0.08)
    p.add_argument("--brain-erosion",  type=int,   default=3)
    # posterior sampling
    p.add_argument("--n-samples",      type=int,   default=100)
    p.add_argument("--seed",           type=int,   default=0)
    # plot
    p.add_argument("--voxel-x",        type=int,   default=38,
                   help="Column index of the highlighted voxel")
    p.add_argument("--voxel-y",        type=int,   default=20,
                   help="Row index of the highlighted voxel")
    p.add_argument("--threshold",      type=float, default=5e-5,
                   help="vmax for uncertainty map colorbar")
    p.add_argument("--dark-mode",      action="store_true", default=True)
    p.add_argument("--no-dark-mode",   dest="dark_mode", action="store_false")
    p.add_argument("--run-tag",        default="",
                   help="Run identifier from recon_01 (e.g. w5000_l0.0001); "
                        "appended to spice/hessian/lobpcg/uncertainty subdir names")
    return p.parse_args()


# ── shared plot helper ────────────────────────────────────────────────────────

def _save_plots(std_img, mean_spec, im_size, brain_mask, wref_norm, PPM_AXIS, args, out_dir, tag):
    fig = plot_average_variation(
        mean_spec  = mean_spec,
        std_img    = std_img,
        img_shape  = im_size,
        voxel_x    = args.voxel_x,
        voxel_y    = args.voxel_y,
        wref_norm  = wref_norm,
        brain_mask = brain_mask,
        PPM_AXIS   = PPM_AXIS,
        threshold  = args.threshold,
        dark_mode  = args.dark_mode,
    )
    out_path = os.path.join(out_dir, f"fig_09_{tag}_uncert_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[uncert-post] Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    data_dir  = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    _tg       = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    spice_dir = os.path.join(args.out_dir, _tg("spice"))
    out_dir   = os.path.join(args.out_dir, _tg("uncertainty"))
    os.makedirs(out_dir, exist_ok=True)

    Ny, Nx   = args.dim
    N_SEQ    = args.n_seq_points
    K_POINTS = args.k_points
    N_VOXEL  = Ny * Nx
    im_size  = (Ny, Nx, N_SEQ)

    TS         = (K_POINTS / N_SEQ) * args.dwelltime
    sweepwidth = 1.0 / TS
    FREQ_AXIS  = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS   = FREQ_AXIS / args.center_freq + args.ppm_center

    # ── Common loads ─────────────────────────────────────────────────────────
    print(f"[uncert-post] mode={args.mode}  Loading data …")
    _affine_path = data_dir + "affine.npy"
    affine   = np.load(_affine_path) if os.path.exists(_affine_path) else np.eye(4)
    wref_img = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    V_full   = np.load(os.path.join(spice_dir, "V_subspace.npy"))
    V        = V_full[:, :args.rank].astype(D_TYPE)   # (N_seq, rank)

    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    # ── Mode: voxelwise ───────────────────────────────────────────────────────
    if args.mode == "voxelwise":
        hess_dir    = args.hess_dir or os.path.join(args.out_dir, _tg("hessian"))
        sigma_noise = np.load(data_dir + "sigma_noise.npy")
        est_U       = np.load(os.path.join(spice_dir, "U_est.npy"))
        print(f"[uncert-post] V={V.shape}  U={est_U.shape}  sigma_noise={sigma_noise}")

        cov_scale = 2.0 * float(sigma_noise) ** 2  # σ_complex² = 2σ_real²; sigma_noise stores σ_real
        mu_map    = (est_U[:N_VOXEL, :args.rank] @ V.conj().T).astype(D_TYPE)

        print(f"[uncert-post] Drawing {args.n_samples} posterior samples …")
        data, _ = build_dataset_auto(
            mHm_dir    = hess_dir,
            num_voxels = N_VOXEL,
            V          = V,
            mu_map     = mu_map,
            n_samples  = args.n_samples,
            dtype      = D_TYPE,
            seed       = args.seed,
            cov_scale  = cov_scale,
        )
        print(f"[uncert-post] data shape: {data.shape}")

        data_spec = fid_to_spec(data)                        # (n_samples, N_vox, N_seq)
        std_img   = np.std(data_spec, axis=0).reshape(Ny, Nx, N_SEQ)
        mean_spec = np.mean(data_spec, axis=0).reshape(Ny, Nx, N_SEQ)

        np.save(os.path.join(out_dir, "posterior_std.npy"), std_img)
        print(f"[uncert-post] Saved posterior_std.npy  shape={std_img.shape}")
        _save_niftimrs(std_img, mean_spec, out_dir, TS, affine)
        _save_plots(std_img, mean_spec, im_size, brain_mask_inner, wref_norm, PPM_AXIS, args, out_dir, "voxelwise")

    # ── Mode: lobpcg ─────────────────────────────────────────────────────────
    else:
        lobpcg_dir = args.lobpcg_dir or os.path.join(args.out_dir, _tg("lobpcg"))
        Q    = np.load(os.path.join(lobpcg_dir, "lobpcg_Q.npy"))
        vals = np.load(os.path.join(lobpcg_dir, "lobpcg_vals.npy"))
        print(f"[uncert-post] Q={Q.shape}  vals={vals.shape}")

        # The LOBPCG Hessian H = A^H A + λ WW is built WITHOUT the 1/σ² factor,
        # so H⁻¹ is already the correct posterior covariance — no sigma² scaling needed.
        # Override via --sigma2 only if you have a special rescaling reason.
        sigma2 = args.sigma2
        print(f"[uncert-post] sigma2={sigma2} (use --sigma2 to override)")

        print(f"[uncert-post] Drawing {args.n_samples} posterior samples …")
        lac_samples = sample_lowrank(Q, vals, args.n_samples,
                                     sigma2=sigma2, seed=args.seed)
        # lac_samples: (n_samples, N_vox * rank) — perturbations around zero
        allsample_U = lac_samples.reshape(args.n_samples, N_VOXEL, args.rank)
        sim_spice   = allsample_U @ V.conj().T                  # (n_samples, N_vox, N_seq) FID
        sim_spec    = fid_to_spec(sim_spice.reshape(args.n_samples, Ny, Nx, N_SEQ))

        std_img   = np.std(sim_spec, axis=0)                    # (Ny, Nx, N_seq)
        mean_spec = np.mean(sim_spec, axis=0)

        np.save(os.path.join(out_dir, "posterior_std.npy"), std_img)
        print(f"[uncert-post] Saved posterior_std.npy  shape={std_img.shape}")
        _save_niftimrs(std_img, mean_spec, out_dir, TS, affine)
        _save_plots(std_img, mean_spec, im_size, brain_mask_inner, wref_norm, PPM_AXIS, args, out_dir, "lobpcg")

    print("[uncert-post] Done.")


if __name__ == "__main__":
    main()
