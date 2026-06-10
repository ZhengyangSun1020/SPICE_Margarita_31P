#!/usr/bin/env python3
"""
Step 8 — Laplacian covariance matrix (per-voxel Hessian) quantification for SPICE.

For each brain voxel v, solves r CG systems H x = e_{v,r} (r = SPICE rank)
and saves the result mHm_{v}.npy = M_v.T @ H^{-1} M_v  (r x r matrix)
where M_v is the selection matrix for voxel v.

Reads  : <data_dir>/wref_o.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/spice/V_subspace.npy
         [<out_dir>/spice/<csv_name>.csv]  (for subspace if V not saved)
Writes : <hess_dir>/mHm_{vox}.npy  for each brain voxel

Usage:
    python scripts/08_Laplacian_Covariance.py \
        --data-dir  ./data/ \
        --out-dir   ./output \
        --hess-dir  ./output/hessian \
        --rank 20 --lambda 1e-4 --max-workers 8 \
        [--vox-start 0 --vox-end 100]    # optional: parallelise over voxel ranges
"""

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from warnings import filterwarnings
filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torchkbnufft as tkbn
from scipy.ndimage import binary_erosion
from scipy.sparse.linalg import LinearOperator, cg
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    NUFFTOp, calc_Bmatrix, read_training_data_from_csv, Calc_B0_matrix_mx,
)

D_TYPE   = np.complex64
T_D_TYPE = torch.complex64


# ── Toeplitz NUFFT wrapper ────────────────────────────────────────────────────

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

        self.toep = tkbn.ToepNufft().to(self.device)
        if kernel is None:
            self.kernel = tkbn.calc_toeplitz_kernel(
                self.ktraj, im_size=self.im_size,
                grid_size=grid_size, norm=self.norm
            ).to(self.device)
        else:
            # kernel may arrive as a numpy array when passed across processes
            if isinstance(kernel, np.ndarray):
                kernel = torch.tensor(kernel)
            self.kernel = kernel.to(self.device)

    def _T_torch(self, x_t):
        return self.toep(x_t, self.kernel, smaps=self.smaps, norm=self.norm)

    def T_np(self, x_np):
        x_t = torch.tensor(x_np, device=self.device).unsqueeze(0).unsqueeze(0)
        y_t = self._T_torch(x_t)
        y   = y_t.detach().cpu()
        while y.dim() > 0 and y.shape[0] == 1:
            y = y.squeeze(0)
        return y.numpy()


def make_toep_linop(toep_op: ToepNUFFTOp) -> LinearOperator:
    n = int(np.prod(toep_op.im_size))
    def mv(x):
        return toep_op.T_np(x.reshape(toep_op.im_size)).ravel()
    return LinearOperator(shape=(n, n), matvec=mv, rmatvec=mv, dtype=D_TYPE)


# ── 1-D FFT operator ─────────────────────────────────────────────────────────

def make_fft1d_op(shape, mode="fid2spec", dtype=D_TYPE):
    n_total  = int(np.prod(shape))
    fft_axis = len(shape) - 1

    def fid2spec(x): return np.fft.fftshift(np.fft.fft(x,  axis=fft_axis, norm="ortho"), axes=fft_axis)
    def spec2fid(x): return np.fft.ifft(np.fft.ifftshift(x, axes=fft_axis), axis=fft_axis, norm="ortho")

    fw, bw = (fid2spec, spec2fid) if mode == "fid2spec" else (spec2fid, fid2spec)

    def mv(x):  return fw(np.asarray(x).reshape(shape)).ravel().astype(dtype, copy=False)
    def rmv(x): return bw(np.asarray(x).reshape(shape)).ravel().astype(dtype, copy=False)
    def mm(X):
        X = np.asarray(X)
        k = X.shape[1] if X.ndim > 1 else 1
        return np.stack([fw(X[:, i].reshape(shape)).ravel() for i in range(k)], axis=1).astype(dtype, copy=False)
    def rmm(X):
        X = np.asarray(X)
        k = X.shape[1] if X.ndim > 1 else 1
        return np.stack([bw(X[:, i].reshape(shape)).ravel() for i in range(k)], axis=1).astype(dtype, copy=False)

    return LinearOperator(shape=(n_total, n_total),
                          matvec=mv, rmatvec=rmv, matmat=mm, rmatmat=rmm, dtype=dtype)


# ── Hessian operator ──────────────────────────────────────────────────────────

def H_action(deltaU, V, B, FHF, WtW, lam):
    """
    H[deltaU]  where deltaU: (N_vox, rank)
    = B*.conj() * FHF * (B * (deltaU @ V.H)) @ V  +  lam * WtW @ deltaU
    """
    deltaX = deltaU @ V.conj().T                   # (N_vox, N_seq)
    BX     = (B * deltaX).ravel()
    z      = (FHF @ BX).reshape(B.shape)
    term   = (B.conj() * z) @ V                    # (N_vox, rank)
    reg    = lam * (WtW @ deltaU)
    return (term + reg).astype(D_TYPE)


def make_H_linop(N_vox, rank, V, B, FHF, WtW, lam) -> LinearOperator:
    d = N_vox * rank
    def mv(u_flat):
        dU = u_flat.reshape(N_vox, rank).astype(D_TYPE)
        return H_action(dU, V, B, FHF, WtW, lam).ravel().astype(D_TYPE)
    return LinearOperator((d, d), matvec=mv, dtype=D_TYPE)


# ── CG solver ─────────────────────────────────────────────────────────────────

def calc_H_inv(b, H_op, d, maxiter=120, rtol=1e-3):
    pbar = tqdm(total=maxiter, leave=False)
    def cb(xk): pbar.update(1)
    x, info = cg(H_op, b, x0=np.zeros(d, dtype=D_TYPE),
                 maxiter=maxiter, rtol=rtol, callback=cb)
    r = H_op @ x - b
    rel = np.linalg.norm(r) / (np.linalg.norm(b) + 1e-16)
    pbar.write(f"  rel_res={rel:.3e}  info={info}")
    pbar.close()
    return x, info


# ── Worker (per-process globals) ──────────────────────────────────────────────

def init_worker(V_, B0_mat_, WW_, ktraj_, im_size_, grid_size_, kernel_,
                rank_, N_vox_, lam_, hess_dir_, device_str_, D_TYPE_, MITER_):
    global V, B0_mat, WW, NUM_SPICE_RANK, N_VOXEL, lam, hess_dir, D_TYPE, MITER
    global Hess_op

    V              = V_
    B0_mat         = B0_mat_
    WW             = WW_
    NUM_SPICE_RANK = rank_
    N_VOXEL        = N_vox_
    lam            = lam_
    hess_dir       = hess_dir_
    D_TYPE         = D_TYPE_
    MITER          = MITER_

    im_size   = tuple(im_size_)
    grid_size = tuple(grid_size_)
    F1D       = make_fft1d_op(im_size, "fid2spec")

    toep_op   = ToepNUFFTOp(im_size=im_size, ktraj=ktraj_, grid_size=grid_size,
                             kernel=kernel_, smaps=None, device=device_str_, norm="ortho")
    Gram_OP   = make_toep_linop(toep_op)
    fFHFf     = F1D.H @ Gram_OP @ F1D
    Hess_op   = make_H_linop(N_VOXEL, NUM_SPICE_RANK, V, B0_mat, fFHFf, WW, lam)


def solve_one_voxel(vox_idx):
    try:
        d    = N_VOXEL * NUM_SPICE_RANK
        cols = []
        for r in range(NUM_SPICE_RANK):
            b = np.zeros(d, dtype=D_TYPE)
            b[vox_idx * NUM_SPICE_RANK + r] = 1.0
            print(f"[uncert] voxel {vox_idx}  rank {r}", flush=True)
            x, _ = calc_H_inv(b, Hess_op, d, maxiter=MITER, rtol=1e-3)
            cols.append(x)
            X = np.array(np.column_stack(cols))              # updated each r (matches cluster)

        X = np.column_stack(cols)                            # (d, rank)
        B = np.zeros((d, NUM_SPICE_RANK), dtype=D_TYPE)
        for r in range(NUM_SPICE_RANK):
            B[vox_idx * NUM_SPICE_RANK + r, r] = 1.0

        os.makedirs(hess_dir, exist_ok=True)
        np.save(os.path.join(hess_dir, f"mHm_{vox_idx}.npy"), B.T @ X)  # (rank, rank)
        print(f"[uncert] saved mHm_{vox_idx}.npy", flush=True)
        return vox_idx
    except Exception:
        print(f"[ERROR] voxel {vox_idx}", flush=True)
        traceback.print_exc()
        raise


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SPICE Hessian uncertainty — step 8")
    p.add_argument("--data-dir",      required=True)
    p.add_argument("--out-dir",       default="./output")
    p.add_argument("--hess-dir",      default=None,
                   help="Output directory for mHm_*.npy (default: <out-dir>/hessian)")
    p.add_argument("--dwelltime",     type=float, default=5e-6)
    p.add_argument("--k-points",      type=int,   default=39762)
    p.add_argument("--n-seq-points",  type=int,   default=300)
    p.add_argument("--n-coils",       type=int,   default=32)
    p.add_argument("--dim",           type=int,   nargs=2, default=[64, 64], metavar=("NY", "NX"))
    p.add_argument("--rank",          type=int,   default=20,
                   help="SPICE subspace rank (must match 04_run_spice.py)")
    p.add_argument("--lambda",        type=float, default=1e-4, dest="lam",
                   help="Regularisation lambda for Hessian (same as SPICE solve)")
    p.add_argument("--lambda-we-max", type=float, default=5000.0,
                   help="Max edge weight W_max for calc_Bmatrix")
    p.add_argument("--pool-size",     type=int,   default=1)
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--csv-name",      default="SS_training",
                   help="CSV filename in spice/ dir (fallback if V_subspace.npy absent)")
    p.add_argument("--max-workers",   type=int,   default=8)
    p.add_argument("--cg-maxiter",    type=int,   default=300)
    p.add_argument("--cg-rtol",       type=float, default=1e-3)
    # voxel range — lets multiple cluster jobs run in parallel on different slices
    p.add_argument("--vox-start",     type=int,   default=None,
                   help="Start index into the brain voxel list (inclusive, default: 0)")
    p.add_argument("--vox-end",       type=int,   default=None,
                   help="End index into the brain voxel list (exclusive, default: all)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    data_dir   = args.data_dir.rstrip("/") + "/"
    coilmap_dir = os.path.join(args.out_dir, "coilmap")
    b0map_dir   = os.path.join(args.out_dir, "b0map")
    lprm_dir    = os.path.join(args.out_dir, "lipid_removal")
    spice_dir   = os.path.join(args.out_dir, "spice")
    hess_dir    = args.hess_dir or os.path.join(args.out_dir, "hessian")
    os.makedirs(hess_dir, exist_ok=True)

    Ny, Nx   = args.dim
    N_SEQ    = args.n_seq_points
    K_POINTS = args.k_points
    N_COILS  = args.n_coils
    N_VOXEL  = Ny * Nx

    TS       = (K_POINTS / N_SEQ) * args.dwelltime
    TIME_AXIS = np.linspace(TS, TS * N_SEQ, N_SEQ)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[uncert] Loading data …")
    wref_img        = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir, "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"),     mmap_mode="r")
    B0_map          = np.load(os.path.join(b0map_dir,  "B0_map.npy"))

    # ── Subspace V ────────────────────────────────────────────────────────────
    v_path = os.path.join(spice_dir, "V_subspace.npy")
    if os.path.exists(v_path):
        V_full = np.load(v_path)                              # (N_SEQ, rank_saved)
        print(f"[uncert] Loaded V_subspace.npy  shape={V_full.shape}")
    else:
        print(f"[uncert] V_subspace.npy not found, building from {args.csv_name}.csv …")
        training = read_training_data_from_csv(spice_dir, args.csv_name).astype(D_TYPE)
        _, _, vh = np.linalg.svd(training, full_matrices=False)
        V_full   = vh.conj().T                                # (N_SEQ, rank)
    V = V_full[:, :args.rank].astype(D_TYPE)                  # (N_SEQ, rank)
    print(f"[uncert] V shape={V.shape}")

    # ── B0 matrix ─────────────────────────────────────────────────────────────
    B0_map_clean = np.nan_to_num(-B0_map, nan=0.0)
    B0_mat = Calc_B0_matrix_mx(B0_map_clean, TIME_AXIS)       # (N_vox, N_SEQ)
    print(f"[uncert] B0_mat shape={B0_mat.shape}")

    # ── Brain mask ─────────────────────────────────────────────────────────────
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask = wref_norm > args.brain_threshold
    print(f"[uncert] Brain voxels: {brain_mask.sum()}")

    # ── Edge-preserving W ─────────────────────────────────────────────────────
    print("[uncert] Building edge-preserving W …")
    W_edge, _, _W, Nb = calc_Bmatrix(
        wref_norm, wmax=args.lambda_we_max, adj=8,
        pool_size=args.pool_size, minpooling_Handler=True,
        brain_mask=brain_mask, mask_dilate_layers=3,
    )
    WW = (W_edge.conj().T @ W_edge).astype(D_TYPE)
    print(f"[uncert] WW shape={WW.shape}")

    # ── Trajectory & coil maps ────────────────────────────────────────────────
    trej  = mrsi_ksp_scaled.T.astype(np.float32)              # (N_shots*K, 3)
    im_size   = (Ny, Nx, N_SEQ)
    osamp, ost = 2.0, 2.0
    grid_size  = (int(np.ceil(osamp * Ny)),
                  int(np.ceil(osamp * Nx)),
                  int(np.ceil(ost   * N_SEQ)))

    ktraj_torch = torch.from_numpy(trej).permute(2, 0, 1).reshape(3, -1).to(device)

    # coil_smap_raw: (C, Ny, Nx) → (1, C, Ny, Nx, N_SEQ)
    coil_smap = np.repeat(
        coil_smap_raw[np.newaxis, :, :, :, np.newaxis], N_SEQ, axis=-1
    ).astype(D_TYPE)
    smap_torch = torch.tensor(coil_smap, device=device).to(T_D_TYPE)

    # ── Toeplitz kernel (computed once, shared with workers) ──────────────────
    print("[uncert] Computing Toeplitz kernel …")
    kernel = tkbn.calc_toeplitz_kernel(
        ktraj_torch, im_size=im_size, grid_size=grid_size, norm="ortho"
    ).to(device)
    kernel_np = kernel.cpu().numpy()
    ktraj_np  = ktraj_torch.cpu().numpy()
    print("[uncert] Kernel computed.")

    # ── Quick in-process check: build once to verify shapes ───────────────────
    F1D     = make_fft1d_op(im_size, "fid2spec")
    toep_op = ToepNUFFTOp(im_size=im_size, ktraj=ktraj_torch, grid_size=grid_size,
                           kernel=kernel, smaps=smap_torch, norm="ortho")
    Gram_OP = make_toep_linop(toep_op)
    fFHFf   = F1D.H @ Gram_OP @ F1D
    Hess_op = make_H_linop(N_VOXEL, args.rank, V, B0_mat, fFHFf, WW, args.lam)
    print(f"[uncert] Hessian shape={Hess_op.shape}  (N_vox*rank={N_VOXEL*args.rank})")
    del F1D, toep_op, Gram_OP, fFHFf, Hess_op   # free; workers rebuild

    # ── Parallel voxel CG ─────────────────────────────────────────────────────
    vox_list = np.flatnonzero(brain_mask.ravel())[args.vox_start:args.vox_end]
    print(f"[uncert] Voxel slice [{args.vox_start}:{args.vox_end}]  "
          f"count={len(vox_list)}  max_workers={args.max_workers} …")

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(
            V, B0_mat, WW,
            ktraj_np, im_size, grid_size, kernel_np,
            args.rank, N_VOXEL, args.lam, hess_dir, "cpu",
            D_TYPE, args.cg_maxiter,
        ),
    ) as ex:
        for _ in ex.map(solve_one_voxel, vox_list):
            pass

    print(f"[uncert] Done. Results in {hess_dir}/")


if __name__ == "__main__":
    main()
