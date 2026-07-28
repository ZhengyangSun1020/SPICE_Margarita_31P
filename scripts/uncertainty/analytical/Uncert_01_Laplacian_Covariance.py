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
         <out_dir>/spice_<run_tag>/V_subspace.npy   (tag e.g. w5000_l0.0001)
         [<out_dir>/spice_<run_tag>/<csv_name>.csv]
Writes : <out_dir>/hessian_<run_tag>/mHm_{vox}.npy  for each brain voxel
         (or explicit --hess-dir to override)

Usage:
    python scripts/uncertainty/analytical/Uncert_01_Laplacian_Covariance.py \
        --data-dir  data/processed/invivo_250305_01 \
        --run-tag   w5000_l0.0001 \
        --rank 20 --lambda 1e-4 --max-workers 8 \
        [--vox-start 0 --vox-end 100]    # optional: parallelise over voxel ranges
"""

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import argparse
import re
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.scan_params import load_scan_params
from utils.pipeline_utils import make_brain_mask
from utils.utils import (
    NUFFTOp, calc_Bmatrix, read_training_data_from_csv, Calc_B0_matrix_mx,
    build_gram_for_worker,
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

def H_action(deltaU, V, B, FHF, WtW, lam, F_loc=None):
    """
    H[deltaU]  where deltaU: (N_vox, rank)
    = B*.conj() * FHF * (B * (deltaU @ V.H)) @ V  +  lam * WtW @ deltaU

    If F_loc is given (finufft backend), FHF is ignored and F_loc.matvec /
    F_loc.rmatvec are called directly, avoiding the Gram wrapper.
    """
    deltaX = deltaU @ V.conj().T                   # (N_vox, N_seq)
    BX     = (B * deltaX).ravel()
    if F_loc is not None:
        z = F_loc.rmatvec(F_loc.matvec(BX.astype(D_TYPE))).reshape(B.shape)
    else:
        z = (FHF @ BX).reshape(B.shape)
    term   = (B.conj() * z) @ V                    # (N_vox, rank)
    reg    = lam * (WtW @ deltaU)
    return (term + reg).astype(D_TYPE)


def make_H_linop(N_vox, rank, V, B, FHF, WtW, lam, F_loc=None) -> LinearOperator:
    d = N_vox * rank
    def mv(u_flat):
        dU = u_flat.reshape(N_vox, rank).astype(D_TYPE)
        return H_action(dU, V, B, FHF, WtW, lam, F_loc=F_loc).ravel().astype(D_TYPE)
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

def init_worker(V_, B0_mat_, WW_, backend_,
                ktraj_np_, im_size_, grid_size_, kernel_np_, device_str_,
                trej_np_, coil_smap_raw_np_, n_coils_,
                rank_, N_vox_, lam_, hess_dir_, D_TYPE_, MITER_):
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
    Gram_OP, F1D, F_loc = build_gram_for_worker(
        backend_, im_size, D_TYPE,
        ktraj_np=ktraj_np_, grid_size=tuple(grid_size_) if grid_size_ is not None else None,
        kernel_np=kernel_np_, device_str=device_str_,
        trej_np=trej_np_, coil_smap_raw_np=coil_smap_raw_np_, n_coils=n_coils_,
    )
    if F_loc is not None:   # finufft: call F.matvec / F.rmatvec directly
        Hess_op = make_H_linop(N_VOXEL, NUM_SPICE_RANK, V, B0_mat, None, WW, lam, F_loc=F_loc)
    else:                   # torchnufft: F1D.H @ Toeplitz_Gram @ F1D
        fFHFf   = F1D.H @ Gram_OP @ F1D
        Hess_op = make_H_linop(N_VOXEL, NUM_SPICE_RANK, V, B0_mat, fFHFf, WW, lam)


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

def parse_run_tag(run_tag):
    """Parse a recon_01_run_spice.py run-tag like 'w5000_l0.0001' -> (wmax, lam).

    Returns (None, None) if run_tag is empty or doesn't match the w<..>_l<..> pattern.
    """
    if not run_tag:
        return None, None
    m = re.match(r"^w([0-9.eE+-]+)_l([0-9.eE+-]+)$", run_tag)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def resolve_lambda_args(args):
    """Fill in --lambda/--lambda-we-max from --run-tag when omitted, and warn
    (without overriding) when an explicitly-passed value disagrees with the tag."""
    tag_wmax, tag_lam = parse_run_tag(args.run_tag)

    if args.lam is None:
        args.lam = tag_lam if tag_lam is not None else 1e-4
    elif tag_lam is not None and not np.isclose(args.lam, tag_lam, rtol=1e-6):
        print(f"[uncert] WARNING: --lambda {args.lam:g} does not match --run-tag "
              f"{args.run_tag!r} (which implies lambda={tag_lam:g}). "
              f"Double-check you're pointing at the right SPICE run — "
              f"proceeding with the explicitly passed --lambda {args.lam:g}.")

    if args.lambda_we_max is None:
        args.lambda_we_max = tag_wmax if tag_wmax is not None else 5000.0
    elif tag_wmax is not None and not np.isclose(args.lambda_we_max, tag_wmax, rtol=1e-6):
        print(f"[uncert] WARNING: --lambda-we-max {args.lambda_we_max:g} does not match "
              f"--run-tag {args.run_tag!r} (which implies wmax={tag_wmax:g}). "
              f"Double-check you're pointing at the right SPICE run — "
              f"proceeding with the explicitly passed --lambda-we-max {args.lambda_we_max:g}.")

    return args


def parse_args():
    p = argparse.ArgumentParser(description="SPICE Hessian uncertainty — step 8")
    p.add_argument("--data-dir",      required=True)
    p.add_argument("--out-dir",       default=None,
                   help="Output directory (default: ./output/<subject_id> derived from --data-dir)")
    p.add_argument("--backend",       default="torchnufft",
                   choices=["torchnufft", "finufft"],
                   help="NUFFT backend: torchnufft (default) or finufft")
    p.add_argument("--hess-dir",      default=None,
                   help="Output directory for mHm_*.npy (default: <out-dir>/hessian)")
    p.add_argument("--dwelltime",     type=float, default=None)
    p.add_argument("--k-points",      type=int, default=None)
    p.add_argument("--n-seq-points",  type=int,   default=300)
    p.add_argument("--n-coils",       type=int, default=None)
    p.add_argument("--dim",           type=int,   nargs=2, default=[64, 64], metavar=("NY", "NX"))
    p.add_argument("--rank",          type=int,   default=20,
                   help="SPICE subspace rank (must match 04_run_spice.py)")
    p.add_argument("--lambda",        type=float, default=None, dest="lam",
                   help="Regularisation lambda for Hessian (same as SPICE solve). "
                        "If omitted, parsed from --run-tag (e.g. w5000_l0.0001); "
                        "if both given and they disagree, a warning is printed and "
                        "the explicit --lambda wins.")
    p.add_argument("--lambda-we-max", type=float, default=None,
                   help="Max edge weight W_max for calc_Bmatrix. "
                        "If omitted, parsed from --run-tag (e.g. w5000_l0.0001); "
                        "if both given and they disagree, a warning is printed and "
                        "the explicit --lambda-we-max wins.")
    p.add_argument("--pool-size",     type=int,   default=1)
    p.add_argument("--minpool",       action="store_true",
                   help="Use directional min-pooling for edge weights; match recon_01 --minpool.")
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component and fill enclosed holes. Default: off. "
                        "Match whatever was passed to recon_01_run_spice.py for this run.")
    p.add_argument("--basis-dir",     default="./basis",
                   help="Shared basis directory (contains SS_training CSV)")
    p.add_argument("--csv-name",      default="SS_training",
                   help="CSV filename in basis/ dir (fallback if V_subspace.npy absent)")
    p.add_argument("--max-workers",   type=int,   default=4)
    p.add_argument("--cg-maxiter",    type=int,   default=300)
    p.add_argument("--cg-rtol",       type=float, default=1e-3)
    # voxel range — lets multiple cluster jobs run in parallel on different slices
    p.add_argument("--vox-start",     type=int,   default=None,
                   help="Start index into the brain voxel list (inclusive, default: 0)")
    p.add_argument("--vox-end",       type=int,   default=None,
                   help="End index into the brain voxel list (exclusive, default: all)")
    p.add_argument("--run-tag",       default="",
                   help="Run identifier from recon_01 (e.g. w5000_l0.0001); "
                        "appended to spice/hessian subdir names")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    args       = resolve_lambda_args(args)
    data_dir   = args.data_dir.rstrip("/") + "/"
    if args.out_dir is None:
        args.out_dir = os.path.join("./output", os.path.basename(args.data_dir.rstrip("/")))
    load_scan_params(args, data_dir, k_key="k_mrsi")
    coilmap_dir = os.path.join(args.out_dir, "coilmap")
    b0map_dir   = os.path.join(args.out_dir, "b0map")
    lprm_dir    = os.path.join(args.out_dir, "lipid_removal")
    _tg         = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    spice_dir   = os.path.join(args.out_dir, _tg("spice"))
    hess_dir    = args.hess_dir or os.path.join(args.out_dir, _tg("hessian"))
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
        print(f"[uncert] V_subspace.npy not found, building from {args.basis_dir}/{args.csv_name}.csv …")
        training = read_training_data_from_csv(args.basis_dir, args.csv_name).astype(D_TYPE)
        _, _, vh = np.linalg.svd(training, full_matrices=False)
        V_full   = vh.conj().T                                # (N_SEQ, rank)
    V = V_full[:, :args.rank].astype(D_TYPE)                  # (N_SEQ, rank)
    print(f"[uncert] V shape={V.shape}")

    # ── B0 matrix ─────────────────────────────────────────────────────────────
    B0_map_clean = np.nan_to_num(B0_map, nan=0.0)
    B0_mat = Calc_B0_matrix_mx(B0_map_clean, TIME_AXIS)       # (N_vox, N_SEQ)
    print(f"[uncert] B0_mat shape={B0_mat.shape}")

    # ── Brain mask ─────────────────────────────────────────────────────────────
    wref_norm, brain_mask, _ = make_brain_mask(
        wref_img, args.brain_threshold, args.brain_erosion,
        cleanup=args.brain_mask_cleanup)
    print(f"[uncert] Brain voxels: {brain_mask.sum()}")

    # ── Edge-preserving W ─────────────────────────────────────────────────────
    print("[uncert] Building edge-preserving W …")
    W_edge, _, _W, Nb = calc_Bmatrix(
        wref_norm, wmax=args.lambda_we_max, adj=8,
        pool_size=args.pool_size, minpooling_Handler=args.minpool,
        brain_mask=brain_mask, mask_dilate_layers=3,
    )
    WW = (W_edge.conj().T @ W_edge).astype(D_TYPE)
    print(f"[uncert] WW shape={WW.shape}")

    # ── Trajectory & coil maps ────────────────────────────────────────────────
    trej_np  = mrsi_ksp_scaled.T.astype(np.float32)              # (N_shots*K, 3)
    im_size  = (Ny, Nx, N_SEQ)
    coil_smap_np = coil_smap_raw                                  # (C, Ny, Nx)
    osamp, ost = 2.0, 2.0
    grid_size  = (int(np.ceil(osamp * Ny)),
                  int(np.ceil(osamp * Nx)),
                  int(np.ceil(ost   * N_SEQ))) if args.backend == "torchnufft" else None

    if args.backend == "torchnufft":
        ktraj_torch = torch.from_numpy(trej_np).permute(2, 0, 1).reshape(3, -1).to(device)
        print("[uncert] Computing Toeplitz kernel …")
        kernel = tkbn.calc_toeplitz_kernel(
            ktraj_torch, im_size=im_size, grid_size=grid_size, norm="ortho"
        ).to(device)
        kernel_np = kernel.cpu().numpy()
        ktraj_np  = ktraj_torch.cpu().numpy()
        print("[uncert] Kernel computed.")
    else:
        kernel_np  = None
        ktraj_np   = None
        device_str = "cpu"
        print("[uncert] Using finufft backend (no Toeplitz kernel needed).")

    # ── Quick in-process check: build once to verify shapes ───────────────────
    _Gram, _F1D, _F_loc = build_gram_for_worker(
        args.backend, im_size, D_TYPE,
        ktraj_np=ktraj_np, grid_size=grid_size, kernel_np=kernel_np, device_str=device_str,
        trej_np=trej_np, coil_smap_raw_np=coil_smap_np, n_coils=N_COILS,
    )
    if _F_loc is not None:
        _Hess = make_H_linop(N_VOXEL, args.rank, V, B0_mat, None, WW, args.lam, F_loc=_F_loc)
    else:
        _fFHFf = _F1D.H @ _Gram @ _F1D
        _Hess  = make_H_linop(N_VOXEL, args.rank, V, B0_mat, _fFHFf, WW, args.lam)
    print(f"[uncert] Hessian shape={_Hess.shape}  (N_vox*rank={N_VOXEL*args.rank})")
    del _Gram, _F1D, _F_loc, _Hess

    # ── Parallel voxel CG ─────────────────────────────────────────────────────
    vox_list = np.flatnonzero(brain_mask.ravel())[args.vox_start:args.vox_end]
    print(f"[uncert] Voxel slice [{args.vox_start}:{args.vox_end}]  "
          f"count={len(vox_list)}  max_workers={args.max_workers} …")
    if len(vox_list) == 0:
        print("[uncert] No voxels in this slice — exiting.")
        return

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(
            V, B0_mat, WW, args.backend,
            ktraj_np, im_size, grid_size, kernel_np, device_str,
            trej_np, coil_smap_np, N_COILS,
            args.rank, N_VOXEL, args.lam, hess_dir,
            D_TYPE, args.cg_maxiter,
        ),
    ) as ex:
        for _ in ex.map(solve_one_voxel, vox_list):
            pass

    print(f"[uncert] Done. Results in {hess_dir}/")


if __name__ == "__main__":
    main()
