#!/usr/bin/env python3
"""
Step 9 — Uncertainty post-processing.

Supports two modes selected by --mode:

  voxelwise (default)
    Loads mHm_*.npy files from the Hessian directory, draws posterior samples
    via Laplace approximation, computes spectral std map, and saves plots.

    Reads  : <data_dir>/wref_o.npy
             <data_dir>/sigma_noise.npy
             <out_dir>/spice/V_subspace.npy
             <out_dir>/spice/U_est.npy
             <hess_dir>/mHm_*.npy

  lobpcg
    Loads pre-computed LOBPCG eigenpairs (from step 10), draws low-rank
    posterior samples of U, maps through V to spectrum domain, and plots.

    Reads  : <data_dir>/wref_o.npy
             <data_dir>/sigma_noise.npy  (optional; fallback to --sigma2)
             <out_dir>/spice/V_subspace.npy
             <lobpcg-dir>/lobpcg_Q.npy
             <lobpcg-dir>/lobpcg_vals.npy

Writes : <out_dir>/uncertainty/fig_09_uncert_map.png
         <out_dir>/uncertainty/fig_09_spice_mean.png
         <out_dir>/uncertainty/posterior_std.npy

Usage:
    python scripts/09_uncertainty_postproc.py \\
        --data-dir  ./data/ \\
        [--mode voxelwise|lobpcg] \\
        [--out-dir  ./output] \\
        [--hess-dir ./output/hessian]        # voxelwise mode \\
        [--lobpcg-dir ./output/lobpcg]       # lobpcg mode \\
        [--n-samples 100] [--voxel-x 38 --voxel-y 20]
"""

import argparse
import os
import sys
from pathlib import Path
from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import binary_erosion
from tqdm import tqdm
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D_TYPE = np.complex64


# ── helpers ───────────────────────────────────────────────────────────────────

def fid_to_spec(fid):
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm="ortho"), axes=-1)


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


# ── plot functions (ported from Run_SPICE_toeplitz.ipynb) ─────────────────────

def plot_average_variation(
    spice_test: np.ndarray,
    img_shape: tuple,
    voxel_x: int,
    voxel_y: int,
    voxel_z: int = 0,
    phi0: float = 0.0,
    brain_mask_inner: Optional[np.ndarray] = None,
    brain_prior_map: Optional[np.ndarray] = None,
    prior_alpha: float = 0.35,
    threshold: Optional[float] = None,
    PPM_AXIS: Optional[np.ndarray] = None,
    cmap: str = "Reds",
    figsize: Tuple[float, float] = (18, 5),
    dark_mode: bool = True,
):
    nx, ny, nt = img_shape
    spice_test = np.asarray(spice_test)
    if spice_test.size != nx * ny * nt:
        if spice_test.ndim == 3 and spice_test.shape == (nx, ny, nt):
            SPICE_img = spice_test
        else:
            raise ValueError(f"spice_test size mismatch: got {spice_test.shape}")
    else:
        SPICE_img = spice_test.reshape(nx, ny, nt)

    Spec = SPICE_img[:, :, np.newaxis, :]
    nx, ny, nz, npts = Spec.shape

    mag_map_2d  = np.mean(np.abs(Spec), axis=-1)[:, :, voxel_z]
    spec_voxel  = Spec[voxel_y, voxel_x, voxel_z, :].copy().astype(np.complex128)
    if phi0 != 0:
        spec_voxel *= np.exp(1j * np.deg2rad(phi0))

    if brain_mask_inner is None:
        mask_2d = np.ones((nx, ny), dtype=bool)
    else:
        bm = np.asarray(brain_mask_inner)
        mask_2d = bm[:, :, voxel_z] if bm.ndim == 3 else bm

    mag_masked = mag_map_2d.copy()
    mag_masked[~mask_2d] = np.nan

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    if dark_mode:
        fig.patch.set_facecolor("black")
        for ax in axs:
            ax.set_facecolor("black")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")

    im = axs[0].imshow(np.abs(mag_map_2d), cmap="viridis", origin="lower")
    axs[0].set_title(f"Avg spectral magnitude (z={voxel_z})")
    plt.colorbar(im, ax=axs[0], fraction=0.046)
    axs[0].add_patch(Rectangle((voxel_x - 0.5, voxel_y - 0.5), 1, 1,
                                linewidth=2, edgecolor="green", facecolor="none"))

    im1 = axs[1].imshow(np.abs(mag_masked), cmap=cmap, origin="lower",
                         vmin=0, vmax=threshold)
    axs[1].set_title("Uncertainty (std, brain mask)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)
    if brain_prior_map is not None:
        prior_2d = np.asarray(brain_prior_map)
        if prior_2d.ndim == 3:
            prior_2d = prior_2d[:, :, voxel_z]
        axs[1].imshow(prior_2d, cmap="gray", origin="lower", alpha=prior_alpha)

    x_axis = PPM_AXIS if PPM_AXIS is not None else np.arange(npts)
    c = "white" if dark_mode else "black"
    axs[2].plot(x_axis, np.real(spec_voxel), label="Real",      color=c)
    axs[2].plot(x_axis, np.abs(spec_voxel),  label="|S|", alpha=0.7)
    axs[2].set_title(f"Spectrum  voxel (row={voxel_y}, col={voxel_x})",
                     color="white" if dark_mode else "black")
    axs[2].set_xlabel("ppm" if PPM_AXIS is not None else "index",
                       color="white" if dark_mode else "black")
    if PPM_AXIS is not None:
        axs[2].invert_xaxis()
    axs[2].set_ylabel("Signal", color="white" if dark_mode else "black")
    axs[2].grid(True, alpha=0.3, color="gray")
    axs[2].legend(labelcolor="white" if dark_mode else "black",
                  facecolor="black" if dark_mode else "white")
    if dark_mode:
        axs[2].set_facecolor("black")
        axs[2].tick_params(colors="white")
        for spine in axs[2].spines.values():
            spine.set_color("white")

    plt.tight_layout()
    return fig


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Uncertainty post-processing — step 9")
    p.add_argument("--mode",           choices=["voxelwise", "lobpcg"],
                   default="voxelwise",
                   help="voxelwise: Laplace/mHm approach; lobpcg: load Q/vals from step 10")
    p.add_argument("--data-dir",       required=True)
    p.add_argument("--out-dir",        default="./output")
    p.add_argument("--hess-dir",       default=None,
                   help="[voxelwise] Directory with mHm_*.npy (default: <out-dir>/hessian)")
    p.add_argument("--lobpcg-dir",     default=None,
                   help="[lobpcg] Directory with lobpcg_Q.npy / lobpcg_vals.npy "
                        "(default: <out-dir>/lobpcg)")
    p.add_argument("--sigma2",         type=float, default=1.0,
                   help="[lobpcg] Posterior variance scale; overridden by sigma_noise.npy if found")
    p.add_argument("--dwelltime",      type=float, default=5e-6)
    p.add_argument("--k-points",       type=int,   default=39762)
    p.add_argument("--n-seq-points",   type=int,   default=300)
    p.add_argument("--dim",            type=int,   nargs=2, default=[64, 64],
                   metavar=("NY", "NX"))
    p.add_argument("--center-freq",    type=float, default=297.219338)
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
    return p.parse_args()


# ── shared plot helper ────────────────────────────────────────────────────────

def _save_plots(std_img, mean_spec, im_size, brain_mask, PPM_AXIS, args, out_dir, tag):
    fig_uncert = plot_average_variation(
        spice_test       = std_img,
        img_shape        = im_size,
        voxel_x          = args.voxel_x,
        voxel_y          = args.voxel_y,
        voxel_z          = 0,
        phi0             = 0,
        brain_mask_inner = brain_mask,
        PPM_AXIS         = PPM_AXIS,
        threshold        = args.threshold,
        dark_mode        = args.dark_mode,
    )
    uncert_path = os.path.join(out_dir, f"fig_09_{tag}_uncert_map.png")
    fig_uncert.savefig(uncert_path, dpi=150, bbox_inches="tight",
                       facecolor=fig_uncert.get_facecolor())
    plt.close(fig_uncert)
    print(f"[uncert-post] Saved {uncert_path}")

    fig_mean = plot_average_variation(
        spice_test       = mean_spec,
        img_shape        = im_size,
        voxel_x          = args.voxel_x,
        voxel_y          = args.voxel_y,
        voxel_z          = 0,
        phi0             = 0,
        brain_mask_inner = brain_mask,
        PPM_AXIS         = PPM_AXIS,
        threshold        = None,
        dark_mode        = args.dark_mode,
        cmap             = "viridis",
    )
    mean_path = os.path.join(out_dir, f"fig_09_{tag}_spice_mean.png")
    fig_mean.savefig(mean_path, dpi=150, bbox_inches="tight",
                     facecolor=fig_mean.get_facecolor())
    plt.close(fig_mean)
    print(f"[uncert-post] Saved {mean_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    data_dir  = args.data_dir.rstrip("/") + "/"
    spice_dir = os.path.join(args.out_dir, "spice")
    out_dir   = os.path.join(args.out_dir, "uncertainty")
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
    wref_img = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    V_full   = np.load(os.path.join(spice_dir, "V_subspace.npy"))
    V        = V_full[:, :args.rank].astype(D_TYPE)   # (N_seq, rank)

    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask = wref_norm > args.brain_threshold
    binary_erosion(brain_mask, iterations=args.brain_erosion)  # inner mask unused here

    # ── Mode: voxelwise ───────────────────────────────────────────────────────
    if args.mode == "voxelwise":
        hess_dir    = args.hess_dir or os.path.join(args.out_dir, "hessian")
        sigma_noise = np.load(data_dir + "sigma_noise.npy")
        est_U       = np.load(os.path.join(spice_dir, "U_est.npy"))
        print(f"[uncert-post] V={V.shape}  U={est_U.shape}  sigma_noise={sigma_noise}")

        cov_scale = float(sigma_noise) ** 2
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
        _save_plots(std_img, mean_spec, im_size, brain_mask, PPM_AXIS, args, out_dir, "voxelwise")

    # ── Mode: lobpcg ─────────────────────────────────────────────────────────
    else:
        lobpcg_dir = args.lobpcg_dir or os.path.join(args.out_dir, "lobpcg")
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
        _save_plots(std_img, mean_spec, im_size, brain_mask, PPM_AXIS, args, out_dir, "lobpcg")

    print("[uncert-post] Done.")


if __name__ == "__main__":
    main()
