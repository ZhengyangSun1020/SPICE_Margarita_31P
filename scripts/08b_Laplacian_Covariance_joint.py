#!/usr/bin/env python3
"""
Step 8b — Per-voxel Hessian covariance for the JOINT lipid+metab refit (04b).

Step 08's mHm_*.npy files are the inverse Hessian of the OLD single-block
(metab-only) normal equations (data term through V_metab alone + lamda1*WW).
04b's U_metab_final is no longer the minimizer of that system — it's the
metab sub-block of the minimizer of the bigger JOINT system [U_lipid,U_metab].
Because V_lipid and V_metab overlap spectrally (the whole reason for doing
the joint refit), the joint Hessian has nonzero lipid<->metab cross-coupling,
so U_lipid can't be treated as a fixed constant when propagating uncertainty
into U_metab_final: the correct per-voxel covariance of U_metab_final is the
metab-metab sub-block of the FULL joint inverse Hessian, not the inverse of
the metab-only sub-block alone.

This script rebuilds the exact normal-equation operator used by
SPICEWithSpatialConstrain_cg_nufft_joint (04b) — same V_joint, same Toeplitz
Gram, same B0 modulation, same block-wise WW_lip/WW + lamda_lip/lambda1 — and,
for each brain voxel, solves R_total = R_lip+R_met CG systems to get the full
(R_total x R_total) per-voxel block of H_joint^-1. It saves:
  - mHm_joint_{vox}.npy  (R_total x R_total)  full lipid+metab covariance block
  - mHm_{vox}.npy        (R_met   x R_met  )  metab-metab sub-block only

mHm_{vox}.npy is drop-in compatible with steps 09/11 (same shape/convention
as old step-08 output): point --hess-dir at this script's output dir and
--rank at R_met, V_subspace.npy is unchanged since 04b warm-starts V_metab
from it without ever updating V. mHm_joint_{vox}.npy is for inspecting the
lipid<->metab cross-covariance terms directly.

IMPORTANT: --lambda1/--lamda-lip/--wmax/--adj/--pool-size/--minpool must
match the actual 04b run, otherwise this Hessian does not correspond to the
normal equations that actually produced U_metab_final.

Reads  : <data_dir>/wref_o.npy
         <out_dir>/coilmap/ecalib_pp.npy
         <out_dir>/b0map/B0_map.npy
         <out_dir>/lipid_removal/mrsi_ksp_scaled.npy
         <out_dir>/lipid_removal/V_lipid.npy
         <out_dir>/spice/V_subspace.npy
         <out_dir>/spice_refit/w_lip_vec.npy           (04b)
Writes : <hess_dir>/mHm_joint_{vox}.npy   (R_total x R_total)
         <hess_dir>/mHm_{vox}.npy         (R_met x R_met, metab-metab sub-block)

Usage:
    python scripts/08b_Laplacian_Covariance_joint.py \
        --data-dir  ./data/ \
        --out-dir   ./output \
        --hess-dir  ./output/hessian_joint \
        --lambda1 1e-4 --lamda-lip 1e-6 --wmax 5e3 --adj 8 --pool-size 1 \
        --max-workers 8 \
        [--vox-start 0 --vox-end 100]
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
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import LinearOperator, cg
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import calc_Bmatrix, Calc_B0_matrix, build_gram_for_worker

D_TYPE   = np.complex64
T_D_TYPE = torch.complex64


# ── Toeplitz NUFFT wrapper (same as step 08, no F1D wrapping — 04b's forward
# model encodes time directly via the NUFFT trajectory, not via a spectral
# Toeplitz + 1D-FFT sandwich) ──────────────────────────────────────────────

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


def make_toep_linop(toep_op: ToepNUFFTOp) -> LinearOperator:
    n = int(np.prod(toep_op.im_size))
    def mv(x):
        return toep_op.T_np(x.reshape(toep_op.im_size)).ravel()
    return LinearOperator(shape=(n, n), matvec=mv, rmatvec=mv, dtype=D_TYPE)


# ── Joint Hessian operator (mirrors SPICEWithSpatialConstrain_cg_nufft_joint's
# mv() in utils/recon.py exactly) ───────────────────────────────────────────

def H_action_joint(deltaU, V, B0_mat, Gram_OP, WW_blocks, lam_blocks, block_slices):
    N_seq = V.shape[0]
    AA = (B0_mat * (deltaU @ V.conj().T)).ravel()
    BB = (Gram_OP @ AA).reshape(-1, N_seq)
    CC = (B0_mat.conj() * BB) @ V
    DD_parts = [lam_i * (WW_i @ deltaU[:, sl])
                for sl, WW_i, lam_i in zip(block_slices, WW_blocks, lam_blocks)]
    DD = np.hstack(DD_parts)
    return (CC + DD).astype(D_TYPE)


def make_H_linop_joint(N_vox, R_total, V, B0_mat, Gram_OP, WW_blocks, lam_blocks, block_slices):
    d = N_vox * R_total
    def mv(u_flat):
        dU = u_flat.reshape(N_vox, R_total).astype(D_TYPE)
        return H_action_joint(dU, V, B0_mat, Gram_OP, WW_blocks, lam_blocks, block_slices).ravel().astype(D_TYPE)
    return LinearOperator((d, d), matvec=mv, dtype=D_TYPE)


# ── CG solver (same as step 08) ──────────────────────────────────────────────

def calc_H_inv(b, H_op, d, maxiter=300, rtol=1e-3):
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

def init_worker(V_, B0_mat_, WW_blocks_, lam_blocks_, block_slices_,
                 backend_,
                 ktraj_np_, im_size_, grid_size_, kernel_np_,
                 trej_np_, coil_smap_raw_np_, n_coils_,
                 R_total_, N_vox_, hess_dir_, MITER_, RTOL_):
    global V, B0_mat, WW_blocks, lam_blocks, block_slices
    global R_TOTAL, N_VOXEL, hess_dir, MITER, RTOL
    global Hess_op

    V             = V_
    B0_mat        = B0_mat_
    WW_blocks     = WW_blocks_
    lam_blocks    = lam_blocks_
    block_slices  = block_slices_
    R_TOTAL       = R_total_
    N_VOXEL       = N_vox_
    hess_dir      = hess_dir_
    MITER         = MITER_
    RTOL          = RTOL_

    im_size = tuple(im_size_)
    Gram_OP, _ = build_gram_for_worker(
        backend_, im_size, D_TYPE,
        ktraj_np=ktraj_np_, grid_size=tuple(grid_size_) if grid_size_ is not None else None,
        kernel_np=kernel_np_, device_str="cpu",
        trej_np=trej_np_, coil_smap_raw_np=coil_smap_raw_np_, n_coils=n_coils_,
    )
    Hess_op = make_H_linop_joint(N_VOXEL, R_TOTAL, V, B0_mat, Gram_OP,
                                 WW_blocks, lam_blocks, block_slices)


def solve_one_voxel(vox_idx):
    try:
        d    = N_VOXEL * R_TOTAL
        cols = []
        for r in range(R_TOTAL):
            b = np.zeros(d, dtype=D_TYPE)
            b[vox_idx * R_TOTAL + r] = 1.0
            print(f"[uncert-joint] voxel {vox_idx}  rank {r}/{R_TOTAL}", flush=True)
            x, _ = calc_H_inv(b, Hess_op, d, maxiter=MITER, rtol=RTOL)
            cols.append(x)

        X = np.column_stack(cols)                               # (d, R_total)
        mHm_joint = X[vox_idx * R_TOTAL:(vox_idx + 1) * R_TOTAL, :]  # (R_total, R_total)

        R_lip = block_slices[0].stop
        mHm_metab = mHm_joint[R_lip:, R_lip:]                    # (R_met, R_met)

        os.makedirs(hess_dir, exist_ok=True)
        np.save(os.path.join(hess_dir, f"mHm_joint_{vox_idx}.npy"), mHm_joint)
        np.save(os.path.join(hess_dir, f"mHm_{vox_idx}.npy"), mHm_metab)
        print(f"[uncert-joint] saved mHm_joint_{vox_idx}.npy + mHm_{vox_idx}.npy", flush=True)
        return vox_idx
    except Exception:
        print(f"[ERROR] voxel {vox_idx}", flush=True)
        traceback.print_exc()
        raise


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Joint-refit Hessian uncertainty — step 8b")
    p.add_argument("--data-dir",      required=True)
    p.add_argument("--out-dir",       default="./output")
    p.add_argument("--hess-dir",      default=None,
                   help="Output directory for mHm_*.npy (default: <out-dir>/hessian_joint)")
    p.add_argument("--backend",       default="torchnufft",
                   choices=["torchnufft", "finufft"],
                   help="NUFFT backend: torchnufft (default) or finufft")
    p.add_argument("--dwelltime",     type=float, default=5e-6)
    p.add_argument("--k-points",      type=int,   default=39762)
    p.add_argument("--n-seq-points",  type=int,   default=300)
    p.add_argument("--n-coils",       type=int,   default=32)
    p.add_argument("--dim",           type=int,   nargs=2, default=[64, 64], metavar=("NY", "NX"))
    # Regularization — MUST match the actual 04b run
    p.add_argument("--lambda1",       type=float, default=1e-4,
                   help="Spatial reg weight for U_metab block (must match 04b)")
    p.add_argument("--lamda-lip",     type=float, default=1e-6,
                   help="Spatial reg weight for U_lipid block (must match 04b)")
    p.add_argument("--wmax",          type=float, default=5e3)
    p.add_argument("--adj",           type=int,   default=8)
    p.add_argument("--pool-size",     type=int,   default=1)
    p.add_argument("--minpool",       action="store_true")
    p.add_argument("--brain-threshold", type=float, default=0.08)
    p.add_argument("--brain-erosion",   type=int,   default=3)
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
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    coilmap_dir = os.path.join(args.out_dir, "coilmap")
    b0map_dir   = os.path.join(args.out_dir, "b0map")
    lprm_dir    = os.path.join(args.out_dir, "lipid_removal")
    spice_dir   = os.path.join(args.out_dir, "spice")
    refit_dir   = os.path.join(args.out_dir, "spice_refit")
    hess_dir    = args.hess_dir or os.path.join(args.out_dir, "hessian_joint")
    os.makedirs(hess_dir, exist_ok=True)

    Ny, Nx   = args.dim
    N_SEQ    = args.n_seq_points
    K_POINTS = args.k_points
    N_VOXEL  = Ny * Nx
    im_size  = (Ny, Nx, N_SEQ)

    TS        = (K_POINTS / N_SEQ) * args.dwelltime
    TIME_AXIS = np.linspace(TS, TS * N_SEQ, N_SEQ)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[uncert-joint] Loading data …")
    wref_img        = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir, "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coilmap_dir, "ecalib_pp.npy"),     mmap_mode="r")
    B0_map          = np.load(os.path.join(b0map_dir,  "B0_map.npy"))

    # ── V_joint = [V_lipid, V_metab] (same as 04b; V is fixed, not re-derived) ──
    V_metab = np.load(os.path.join(spice_dir, "V_subspace.npy")).astype(D_TYPE)
    V_lipid = np.load(os.path.join(lprm_dir, "V_lipid.npy")).astype(D_TYPE)
    R_met   = V_metab.shape[1]
    R_lip   = V_lipid.shape[1]
    R_total = R_lip + R_met
    V_joint = np.hstack([V_lipid, V_metab]).astype(D_TYPE)
    block_slices = [slice(0, R_lip), slice(R_lip, R_total)]
    print(f"[uncert-joint] V_lipid={V_lipid.shape}  V_metab={V_metab.shape}  R_total={R_total}")

    # ── B0 matrix (same sign convention as 04b — NOT negated) ───────────────────
    B0_map_clean = np.nan_to_num(B0_map, nan=0.0)
    B0_mat = Calc_B0_matrix(B0_map_clean, TIME_AXIS).reshape(N_VOXEL, N_SEQ).astype(D_TYPE)
    print(f"[uncert-joint] B0_mat shape={B0_mat.shape}")

    # ── Brain mask ─────────────────────────────────────────────────────────────
    wref_2d   = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask = wref_norm > args.brain_threshold
    print(f"[uncert-joint] Brain voxels: {brain_mask.sum()}")

    # ── Anatomical WW (metab block) ──────────────────────────────────────────
    print("[uncert-joint] Building edge-preserving W …")
    W_edge, _, _W, Nb = calc_Bmatrix(
        wref_norm, wmax=args.wmax, adj=args.adj,
        pool_size=args.pool_size, minpooling_Handler=args.minpool,
        brain_mask=brain_mask, mask_dilate_layers=3,
    )
    WW = (W_edge.conj().T @ W_edge).astype(D_TYPE)

    # ── W_lip (lipid block) — load 04b's saved weight map directly so this
    # Hessian corresponds exactly to the run that produced U_metab_final,
    # rather than re-deriving the GMM classification independently ──────────
    w_lip_path = os.path.join(refit_dir, "w_lip_vec.npy")
    w_lip_vec  = np.load(w_lip_path)
    W_lip   = sp_diags(w_lip_vec)
    WW_lip  = (W_lip.conj().T @ W_lip).astype(D_TYPE)
    print(f"[uncert-joint] Loaded w_lip_vec.npy from {w_lip_path}")

    WW_blocks  = [WW_lip, WW]
    lam_blocks = [args.lamda_lip, args.lambda1]

    # ── Trajectory & coil maps (identical construction to 04b) ──────────────────
    trej_np       = mrsi_ksp_scaled.T.astype(np.float32)
    coil_smap_np  = coil_smap_raw                                  # (C, Ny, Nx)
    N_COILS_local = coil_smap_raw.shape[0]
    osamp, ost    = 2.0, 2.0
    grid_size     = (int(np.ceil(osamp * Ny)),
                     int(np.ceil(osamp * Nx)),
                     int(np.ceil(ost   * N_SEQ))) if args.backend == "torchnufft" else None

    if args.backend == "torchnufft":
        ktraj_torch = torch.from_numpy(trej_np).permute(2, 0, 1).reshape(3, -1).to(device)
        print("[uncert-joint] Computing Toeplitz kernel …")
        kernel = tkbn.calc_toeplitz_kernel(
            ktraj_torch, im_size=im_size, grid_size=grid_size, norm="ortho"
        ).to(device)
        kernel_np = kernel.cpu().numpy()
        ktraj_np  = ktraj_torch.cpu().numpy()
        print("[uncert-joint] Kernel computed.")
    else:
        kernel_np = None
        ktraj_np  = None
        print("[uncert-joint] Using finufft backend (no Toeplitz kernel needed).")

    # ── Quick in-process check: build once to verify shapes ───────────────────
    _Gram, _ = build_gram_for_worker(
        args.backend, im_size, D_TYPE,
        ktraj_np=ktraj_np, grid_size=grid_size, kernel_np=kernel_np, device_str=device_str,
        trej_np=trej_np, coil_smap_raw_np=coil_smap_np, n_coils=N_COILS_local,
    )
    _Hess = make_H_linop_joint(N_VOXEL, R_total, V_joint, B0_mat, _Gram,
                                WW_blocks, lam_blocks, block_slices)
    print(f"[uncert-joint] Hessian shape={_Hess.shape}  (N_vox*R_total={N_VOXEL * R_total})")
    del _Gram, _Hess   # free; workers rebuild on CPU

    # ── Parallel voxel CG ─────────────────────────────────────────────────────
    vox_list = np.flatnonzero(brain_mask.ravel())[args.vox_start:args.vox_end]
    print(f"[uncert-joint] Voxel slice [{args.vox_start}:{args.vox_end}]  "
          f"count={len(vox_list)}  max_workers={args.max_workers} …")

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=init_worker,
        initargs=(
            V_joint, B0_mat, WW_blocks, lam_blocks, block_slices,
            args.backend,
            ktraj_np, im_size, grid_size, kernel_np,
            trej_np, coil_smap_np, N_COILS_local,
            R_total, N_VOXEL, hess_dir, args.cg_maxiter, args.cg_rtol,
        ),
    ) as ex:
        for _ in ex.map(solve_one_voxel, vox_list):
            pass

    print(f"[uncert-joint] Done. Results in {hess_dir}/")


if __name__ == "__main__":
    main()
