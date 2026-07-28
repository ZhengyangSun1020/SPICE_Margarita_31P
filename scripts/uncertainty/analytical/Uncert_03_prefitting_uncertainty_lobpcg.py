#!/usr/bin/env python3
"""
Step 10 — Pre-fitting uncertainty (LOBPCG low-rank) uncertainty via LOBPCG (GPU-accelerated).

Builds the full Hessian operator (same as step 08) and uses LOBPCG to find
the k smallest eigenpairs H v = λ v.  The low-rank covariance H^{-1} ≈ Q Λ^{-1} Q^H
is used to draw posterior samples of U (the SPICE spatial coefficients).

Reads  : <data_dir>/wref_o.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/spice_<run_tag>/V_subspace.npy    (tag e.g. w5000_l0.0001)
         <out_dir>/spice_<run_tag>/U_est.npy
Writes : <out_dir>/lobpcg_<run_tag>_k<k_eig>/lobpcg_Q.npy     (eigenvectors, shape d×k)
         <out_dir>/lobpcg_<run_tag>_k<k_eig>/lobpcg_vals.npy   (eigenvalues, shape k)
         <out_dir>/lobpcg_<run_tag>_k<k_eig>/posterior_std.npy (Ny, Nx, N_seq)
         <out_dir>/lobpcg_<run_tag>_k<k_eig>/fig_10_uncert_map.png

Usage:
    python scripts/uncertainty/analytical/Uncert_03_prefitting_uncertainty_lobpcg.py \
        --data-dir data/processed/invivo_250305_01 \
        --run-tag  w5000_l0.0001 \
        --rank 20 --k-eig 50 --n-samples 100 \
        [--damp 0.0] [--lobpcg-maxiter 200]
"""

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import argparse
import sys
from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn as nn
import torchkbnufft as tkbn
from scipy.ndimage import binary_erosion
from scipy.sparse.linalg import LinearOperator, lobpcg
from tqdm import tqdm
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.scan_params import load_scan_params
from utils.utils import NUFFTOp, calc_Bmatrix, Calc_B0_matrix_mx, build_nufft_ops

D_TYPE   = np.complex64
T_D_TYPE = torch.complex64


# ── Operators (same as 08_uncertainty.py) ────────────────────────────────────

class ToepNUFFTOp(nn.Module):
    def __init__(self, im_size, ktraj, grid_size=None, kernel=None,
                 smaps=None, device=None, norm="ortho"):
        super().__init__()
        self.im_size = tuple(im_size)
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.norm    = norm
        if isinstance(ktraj, np.ndarray):
            ktraj = torch.tensor(ktraj, device=self.device)
        self.register_buffer("ktraj", ktraj.to(self.device))
        if smaps is not None:
            if isinstance(smaps, np.ndarray):
                smaps = torch.tensor(smaps, device=self.device)
            smaps = smaps.to(self.device)
        self.smaps = smaps
        self.toep  = tkbn.ToepNufft().to(self.device)
        if kernel is None:
            self.kernel = tkbn.calc_toeplitz_kernel(
                self.ktraj, im_size=self.im_size, grid_size=grid_size, norm=self.norm
            ).to(self.device)
        else:
            if isinstance(kernel, np.ndarray):
                kernel = torch.tensor(kernel)
            self.kernel = kernel.to(self.device)

    def T_np(self, x_np):
        x_t = torch.tensor(x_np, device=self.device).unsqueeze(0).unsqueeze(0)
        y_t = self.toep(x_t, self.kernel, smaps=self.smaps, norm=self.norm)
        y   = y_t.detach().cpu()
        while y.dim() > 0 and y.shape[0] == 1:
            y = y.squeeze(0)
        return y.numpy()


def make_toep_linop(toep_op):
    n = int(np.prod(toep_op.im_size))
    def mv(x): return toep_op.T_np(x.reshape(toep_op.im_size)).ravel()
    return LinearOperator(shape=(n, n), matvec=mv, rmatvec=mv, dtype=D_TYPE)


def make_fft1d_op(shape, mode="fid2spec", dtype=D_TYPE):
    n_total  = int(np.prod(shape))
    fft_axis = len(shape) - 1
    def fid2spec(x): return np.fft.fftshift(np.fft.fft(x, axis=fft_axis, norm="ortho"), axes=fft_axis)
    def spec2fid(x): return np.fft.ifft(np.fft.ifftshift(x, axes=fft_axis), axis=fft_axis, norm="ortho")
    fw, bw = (fid2spec, spec2fid) if mode == "fid2spec" else (spec2fid, fid2spec)
    def mv(x):  return fw(np.asarray(x).reshape(shape)).ravel().astype(dtype, copy=False)
    def rmv(x): return bw(np.asarray(x).reshape(shape)).ravel().astype(dtype, copy=False)
    return LinearOperator(shape=(n_total, n_total), matvec=mv, rmatvec=rmv, dtype=dtype)


def H_action(deltaU, V, B, FHF, WtW, lam):
    deltaX = deltaU @ V.conj().T
    BX     = (B * deltaX).ravel()
    z      = (FHF @ BX).reshape(B.shape)
    term   = (B.conj() * z) @ V
    reg    = lam * (WtW @ deltaU)
    return (term + reg).astype(D_TYPE)


def make_H_linop(N_vox, rank, V, B, FHF, WtW, lam):
    d = N_vox * rank
    def mv(u_flat):
        dU = u_flat.reshape(N_vox, rank).astype(D_TYPE)
        return H_action(dU, V, B, FHF, WtW, lam).ravel().astype(D_TYPE)
    return LinearOperator((d, d), matvec=mv, dtype=D_TYPE)


def fid_to_spec(fid):
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm="ortho"), axes=-1)


# ── LOBPCG low-rank sampler ───────────────────────────────────────────────────

def build_lowrank_lobpcg(H_linop, d, k=50, maxiter=200, tol=1e-6,
                          damp=0.0, clip_eps=1e-12):
    if damp != 0.0:
        def mv(x): return H_linop.matvec(x) + damp * x
        H_use = LinearOperator(H_linop.shape, matvec=mv, dtype=H_linop.dtype)
        print(f"[lobpcg] damping={damp}")
    else:
        H_use = H_linop

    rng = np.random.default_rng()
    if np.iscomplexobj(np.zeros(1, dtype=H_linop.dtype)):
        X = (rng.standard_normal((d, k)) + 1j * rng.standard_normal((d, k))) / np.sqrt(2)
    else:
        X = rng.standard_normal((d, k))
    X = X.astype(H_linop.dtype, copy=False)

    print(f"[lobpcg] Running LOBPCG  d={d}  k={k}  maxiter={maxiter}  tol={tol} …")
    vals_raw, vecs = lobpcg(H_use, X, largest=False, maxiter=maxiter, tol=tol)

    order    = np.argsort(vals_raw)
    vals_raw = np.asarray(vals_raw)[order]
    Q        = np.asarray(vecs)[:, order].astype(H_linop.dtype, copy=False)

    vals_safe = np.array(vals_raw, dtype=float)
    n_bad = int(np.sum(~np.isfinite(vals_safe)) + np.sum(vals_safe <= 0))
    if n_bad:
        print(f"[lobpcg] WARN {n_bad} non-positive eigenvalues → clipped to {clip_eps}")
    vals_safe[~np.isfinite(vals_safe)] = clip_eps
    vals_safe = np.maximum(vals_safe, clip_eps)
    print(f"[lobpcg] Eigenvalue range: {vals_safe.min():.3e} – {vals_safe.max():.3e}")
    return Q, vals_safe, vals_raw


def sample_lowrank(Q, vals_safe, n_samples, sigma2=1.0, seed=None, batch_size=100):
    """Draw samples from CN(0, sigma2 * Q Λ^{-1} Q^H)."""
    rng = np.random.default_rng(seed)
    d, k = Q.shape
    lam_inv_sqrt = np.sqrt(1.0 / (vals_safe + 1e-20))
    samples = np.zeros((n_samples, d), dtype=D_TYPE)
    for b0 in tqdm(range(0, n_samples, batch_size), desc="Sampling"):
        b1  = min(n_samples, b0 + batch_size)
        bs  = b1 - b0
        Zk  = rng.standard_normal((bs, k)) + 1j * rng.standard_normal((bs, k))
        Zk /= np.sqrt(2)
        low = ((Zk * lam_inv_sqrt[None, :]) @ Q.T).astype(D_TYPE)
        if sigma2 != 1.0:
            low *= np.sqrt(sigma2)
        samples[b0:b1] = low
    return samples


# ── plot (same style as 09) ───────────────────────────────────────────────────

def plot_average_variation(spice_test, img_shape, voxel_x, voxel_y,
                            brain_mask=None, PPM_AXIS=None,
                            threshold=None, dark_mode=True, cmap="Reds"):
    nx, ny, nt = img_shape
    img = np.asarray(spice_test).reshape(nx, ny, nt)
    mag = np.mean(np.abs(img), axis=-1)
    mag_masked = np.where(brain_mask, mag, np.nan) if brain_mask is not None else mag
    spec = img[voxel_y, voxel_x, :].astype(np.complex128)
    x_ax = PPM_AXIS if PPM_AXIS is not None else np.arange(nt)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    if dark_mode:
        fig.patch.set_facecolor("black")
        for ax in axs:
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            for sp in ax.spines.values(): sp.set_color("white")

    im0 = axs[0].imshow(mag_masked, cmap="viridis", origin="lower")
    axs[0].set_title("Avg magnitude")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    axs[0].add_patch(Rectangle((voxel_x-.5, voxel_y-.5), 1, 1,
                                linewidth=2, edgecolor="green", facecolor="none"))
    im1 = axs[1].imshow(np.abs(mag_masked), cmap=cmap, origin="lower",
                         vmin=0, vmax=threshold)
    axs[1].set_title("Uncertainty (brain mask)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)
    c = "white" if dark_mode else "C0"
    axs[2].plot(x_ax, np.real(spec), color=c, label="Real")
    axs[2].plot(x_ax, np.abs(spec), alpha=.7, label="|S|")
    axs[2].set_title(f"Spectrum voxel ({voxel_y},{voxel_x})")
    if PPM_AXIS is not None: axs[2].invert_xaxis()
    axs[2].legend(facecolor="black" if dark_mode else "white",
                  labelcolor="white" if dark_mode else "black")
    axs[2].grid(alpha=.3, color="gray")
    plt.tight_layout()
    return fig


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LOBPCG uncertainty — step 10")
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--out-dir",         default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int, default=None)
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--n-coils",         type=int, default=None)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64])
    p.add_argument("--center-freq",     type=float, default=None)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--rank",            type=int,   default=20)
    p.add_argument("--lambda",          type=float, default=1e-4, dest="lam")
    p.add_argument("--lambda-we-max",   type=float, default=5000.0)
    p.add_argument("--pool-size",       type=int,   default=1)
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--csv-name",        default="SS_training")
    # LOBPCG
    p.add_argument("--k-eig",           type=int,   default=50,
                   help="Number of smallest eigenpairs to compute")
    p.add_argument("--lobpcg-maxiter",  type=int,   default=200)
    p.add_argument("--lobpcg-tol",      type=float, default=1e-3)
    p.add_argument("--damp",            type=float, default=0.0)
    # sampling
    p.add_argument("--n-samples",       type=int,   default=100)
    p.add_argument("--sigma2",          type=float, default=1.0,
                   help="Posterior variance scale (set to sigma_noise^2 if available)")
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--backend",         default="torchnufft",
                   choices=["torchnufft", "finufft"],
                   help="NUFFT backend: torchnufft (default) or finufft")
    # plot
    p.add_argument("--voxel-x",         type=int,   default=38)
    p.add_argument("--voxel-y",         type=int,   default=20)
    p.add_argument("--threshold",       type=float, default=5e-5)
    p.add_argument("--dark-mode",       action="store_true", default=True)
    p.add_argument("--no-dark-mode",    dest="dark_mode", action="store_false")
    p.add_argument("--run-tag",         default="",
                   help="Run identifier from recon_01 (e.g. w5000_l0.0001); "
                        "appended to spice/lobpcg subdir names")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    data_dir  = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    coilmap_dir = os.path.join(args.out_dir, "coilmap")
    b0map_dir   = os.path.join(args.out_dir, "b0map")
    lprm_dir    = os.path.join(args.out_dir, "lipid_removal")
    _tg         = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    spice_dir   = os.path.join(args.out_dir, _tg("spice"))
    out_dir     = os.path.join(args.out_dir, f"{_tg('lobpcg')}_k{args.k_eig}")
    os.makedirs(out_dir, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
    print(f"[lobpcg] device={device_str}")

    Ny, Nx   = args.dim
    N_SEQ    = args.n_seq_points
    K_POINTS = args.k_points
    N_VOXEL  = Ny * Nx
    im_size  = (Ny, Nx, N_SEQ)
    TS        = (K_POINTS / N_SEQ) * args.dwelltime
    sweepwidth = 1.0 / TS
    TIME_AXIS  = np.linspace(TS, TS * N_SEQ, N_SEQ)
    FREQ_AXIS  = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ)
    PPM_AXIS   = FREQ_AXIS / args.center_freq + args.ppm_center

    # ── Load ─────────────────────────────────────────────────────────────────
    print("[lobpcg] Loading data …")
    wref_img        = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir, "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"),     mmap_mode="r")
    B0_map          = np.load(os.path.join(b0map_dir,  "B0_map.npy"))
    V_full          = np.load(os.path.join(spice_dir,  "V_subspace.npy"))
    est_U           = np.load(os.path.join(spice_dir,  "U_est.npy"))

    # auto-load sigma_noise if available
    sigma_noise_path = data_dir + "sigma_noise.npy"
    if os.path.exists(sigma_noise_path):
        sigma_noise = float(np.load(sigma_noise_path))
        sigma2 = 2.0 * sigma_noise ** 2  # σ_complex² = 2σ_real²; sigma_noise stores σ_real
        print(f"[lobpcg] sigma_noise={sigma_noise:.4e}  sigma2={sigma2:.4e}")
    else:
        sigma2 = args.sigma2
        print(f"[lobpcg] sigma_noise.npy not found → using --sigma2={sigma2}")

    V  = V_full[:, :args.rank].astype(D_TYPE)     # (N_seq, rank)
    Vh = V.conj().T                                # (rank, N_seq)

    # ── Brain mask & W ────────────────────────────────────────────────────────
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask       = wref_norm > args.brain_threshold
    brain_mask_inner = binary_erosion(brain_mask, iterations=args.brain_erosion)

    W_edge, _, _W, Nb = calc_Bmatrix(
        wref_norm, wmax=args.lambda_we_max, adj=8,
        pool_size=args.pool_size, minpooling_Handler=True,
        brain_mask=brain_mask, mask_dilate_layers=3,
    )
    WW = (W_edge.conj().T @ W_edge).astype(D_TYPE)

    B0_mat = Calc_B0_matrix_mx(np.nan_to_num(B0_map, nan=0.0), TIME_AXIS)   # (N_vox, N_seq)

    # ── NUFFT operators ───────────────────────────────────────────────────────
    trej = mrsi_ksp_scaled.T.astype(np.float32)
    N_COILS_local = coil_smap_raw.shape[0]
    _, Gram_OP, F1D, _ = build_nufft_ops(
        args.backend, trej, im_size, coil_smap_raw, N_COILS_local, D_TYPE,
        osamp=2.0, ost=2.0, device=device_str,
    )
    fFHFf   = F1D.H @ Gram_OP @ F1D
    Hess_op = make_H_linop(N_VOXEL, args.rank, V, B0_mat, fFHFf, WW, args.lam)
    d       = N_VOXEL * args.rank
    print(f"[lobpcg] Hessian shape={Hess_op.shape}  k_eig={args.k_eig}")

    # ── LOBPCG ────────────────────────────────────────────────────────────────
    Q, vals_safe, vals_raw = build_lowrank_lobpcg(
        Hess_op, d,
        k       = args.k_eig,
        maxiter = args.lobpcg_maxiter,
        tol     = args.lobpcg_tol,
        damp    = args.damp,
    )
    np.save(os.path.join(out_dir, "lobpcg_Q.npy"),    Q)
    np.save(os.path.join(out_dir, "lobpcg_vals.npy"), vals_safe)
    print(f"[lobpcg] Saved Q ({Q.shape}) and vals ({vals_safe.shape})")

    # ── Sample posterior U ────────────────────────────────────────────────────
    print(f"[lobpcg] Drawing {args.n_samples} posterior samples …")
    lac_samples = sample_lowrank(Q, vals_safe, args.n_samples,
                                  sigma2=sigma2, seed=args.seed)
    # lac_samples: (n_samples, N_vox * rank)
    allsample_U  = lac_samples.reshape(args.n_samples, N_VOXEL, args.rank)  # (S, N_vox, rank)
    sim_spice    = allsample_U @ Vh                                           # (S, N_vox, N_seq) FID
    sim_spice_3d = sim_spice.reshape(args.n_samples, Ny, Nx, N_SEQ)

    # ── Std in spectrum domain ────────────────────────────────────────────────
    sim_spec     = fid_to_spec(sim_spice_3d)              # (S, Ny, Nx, N_seq)
    posterior_std = np.std(sim_spec, axis=0)              # (Ny, Nx, N_seq)
    np.save(os.path.join(out_dir, "posterior_std.npy"), posterior_std)
    print(f"[lobpcg] Saved posterior_std.npy  shape={posterior_std.shape}")

    mean_spec = np.mean(sim_spec, axis=0)              # (Ny, Nx, N_seq)
    np.save(os.path.join(out_dir, "mean_spec.npy"), mean_spec)
    print(f"[lobpcg] Saved mean_spec.npy  shape={mean_spec.shape}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plot_average_variation(
        spice_test  = posterior_std,
        img_shape   = im_size,
        voxel_x     = args.voxel_x,
        voxel_y     = args.voxel_y,
        brain_mask  = brain_mask_inner,
        PPM_AXIS    = PPM_AXIS,
        threshold   = args.threshold,
        dark_mode   = args.dark_mode,
    )
    path = os.path.join(out_dir, "fig_10_uncert_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[lobpcg] Saved {path}")

    # mean spectrum plot
    fig2 = plot_average_variation(
        spice_test = mean_spec,
        img_shape  = im_size,
        voxel_x    = args.voxel_x,
        voxel_y    = args.voxel_y,
        brain_mask = brain_mask,
        PPM_AXIS   = PPM_AXIS,
        threshold  = None,
        dark_mode  = args.dark_mode,
        cmap       = "viridis",
    )
    fig2.savefig(os.path.join(out_dir, "fig_10_spice_mean.png"),
                 dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print("[lobpcg] Done.")


if __name__ == "__main__":
    main()
