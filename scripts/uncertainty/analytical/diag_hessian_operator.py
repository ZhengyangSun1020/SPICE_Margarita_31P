#!/usr/bin/env python3
"""
Diagnostic: compare finufft-recon Hessian vs torchnufft-Uncert_01 Hessian.

The actual SPICE recon uses finufft, where Gram_OP = F^H F (FID-domain).
The Uncert_01 cluster jobs use torchnufft with F1D wrapping:
    H_uncert data term = F1D^H @ Gram_torch @ F1D @ BX

This script checks whether:
    F1D^H @ Gram_torch @ F1D @ BX  ≈  Gram_finufft @ BX
for the same BX (B0-modulated image).

If they agree (< ~1% error), the Uncert_01 torchnufft Hessian is consistent
with the finufft recon → the F1D wrapping is correct for torchnufft.
If they disagree, there is a domain mismatch that could explain the
lambda-dependent slope in analytical vs MC scatter plots.

Also reports:
  - ratio of Rayleigh quotients (data term, full H)
  - relative error at the full-Hessian level
  - Gram_torch @ BX (no F1D) vs Gram_finufft @ BX, to check raw equivalence

Usage (cluster):
    python scripts/uncertainty/analytical/diag_hessian_operator.py \
        --data-dir  data/processed/invivo_260623_01 \
        --out-dir   output/invivo_260623_01 \
        --run-tag   w5000_l0.0001 \
        --rank 20 --n-rand-u 8 --seed 0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from utils.scan_params import load_scan_params
from utils.utils import Calc_B0_matrix_mx, build_gram_for_worker

D_TYPE = np.complex64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",        required=True)
    p.add_argument("--out-dir",         required=True)
    p.add_argument("--run-tag",         default="w5000_l0.0001")
    p.add_argument("--rank",            type=int,   default=20)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64])
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--dwelltime",       type=float, default=None)
    p.add_argument("--k-points",        type=int,   default=None)
    p.add_argument("--n-rand-u",        type=int,   default=8)
    p.add_argument("--seed",            type=int,   default=0)
    return p.parse_args()


def gram_apply(BX, Gram_torch, F1D, F_loc):
    """Apply data term in three ways and return results."""
    # 1. finufft: Gram_finufft @ BX  (reference)
    gram_fi = F_loc.rmatvec(F_loc.matvec(BX.astype(D_TYPE)))

    # 2. torchnufft WITH F1D wrapping  (Uncert_01 current)
    gram_to_f1d = (F1D.H @ Gram_torch @ F1D).matvec(BX.astype(D_TYPE))

    # 3. torchnufft WITHOUT F1D wrapping  (direct Gram)
    gram_to_raw = Gram_torch.matvec(BX.astype(D_TYPE))

    return gram_fi, gram_to_f1d, gram_to_raw


def rayleigh(x, Ax):
    return float(np.real(x.ravel().conj() @ Ax.ravel()))


def rel_err(ref, test):
    return np.linalg.norm(ref - test) / (np.linalg.norm(ref) + 1e-30) * 100


def main():
    args     = parse_args()
    data_dir = args.data_dir.rstrip("/") + "/"
    load_scan_params(args, data_dir, k_key="k_mrsi")

    Ny, Nx    = args.dim
    N_SEQ     = args.n_seq_points
    N_VOXEL   = Ny * Nx
    TS        = (args.k_points / N_SEQ) * args.dwelltime
    TIME_AXIS = np.linspace(TS, TS * N_SEQ, N_SEQ)

    _tg = lambda b: f"{b}_{args.run_tag}" if args.run_tag else b
    lprm_dir  = os.path.join(args.out_dir, "lipid_removal")
    coil_dir  = os.path.join(args.out_dir, "coilmap")
    b0_dir    = os.path.join(args.out_dir, "b0map")
    spice_dir = os.path.join(args.out_dir, _tg("spice"))

    print("[diag] Loading data …")
    mrsi_ksp_scaled = np.load(os.path.join(lprm_dir, "mrsi_ksp_scaled.npy"), mmap_mode="r")
    coil_smap_raw   = np.load(os.path.join(coil_dir, "ecalib_pp.npy"), mmap_mode="r")
    B0_map          = np.load(os.path.join(b0_dir,   "B0_map.npy"))
    V_full          = np.load(os.path.join(spice_dir, "V_subspace.npy"))

    V      = V_full[:, :args.rank].astype(D_TYPE)
    B0_mat = Calc_B0_matrix_mx(np.nan_to_num(B0_map, nan=0.0), TIME_AXIS)

    trej_np  = mrsi_ksp_scaled.T.astype(np.float32)
    im_size  = (Ny, Nx, N_SEQ)
    N_COILS  = coil_smap_raw.shape[0]

    # ── build torchnufft Gram + F1D ──────────────────────────────────────────
    print("[diag] Building torchnufft operators …")
    import torch
    import torchkbnufft as tkbn
    device_str  = "cuda" if torch.cuda.is_available() else "cpu"
    device      = torch.device(device_str)
    ktraj_torch = torch.from_numpy(trej_np).permute(2, 0, 1).reshape(3, -1).to(device)
    grid_size   = (int(np.ceil(2.0 * Ny)), int(np.ceil(2.0 * Nx)), int(np.ceil(2.0 * N_SEQ)))
    kernel      = tkbn.calc_toeplitz_kernel(ktraj_torch, im_size=im_size, grid_size=grid_size, norm="ortho")
    kernel_np   = kernel.cpu().numpy()
    ktraj_np    = ktraj_torch.cpu().numpy()

    Gram_torch, F1D, _ = build_gram_for_worker(
        "torchnufft", im_size, D_TYPE,
        ktraj_np=ktraj_np, grid_size=grid_size, kernel_np=kernel_np, device_str=device_str,
        trej_np=trej_np, coil_smap_raw_np=coil_smap_raw, n_coils=N_COILS,
    )

    # ── build finufft F_loc ───────────────────────────────────────────────────
    print("[diag] Building finufft operators …")
    _, _, F_loc = build_gram_for_worker(
        "finufft", im_size, D_TYPE,
        trej_np=trej_np, coil_smap_raw_np=coil_smap_raw, n_coils=N_COILS,
    )

    # ── test with random δU vectors ───────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    print(f"\n[diag] Testing {args.n_rand_u} random δU   "
          f"(rank={args.rank}, N_vox={N_VOXEL})")
    print(f"  Columns: err% = relative error vs Gram_finufft")
    print(f"  {'err_torch_F1D%':>15}  {'err_torch_raw%':>15}  "
          f"{'RQ_finufft':>12}  {'RQ_torch_F1D':>13}  {'RQ_torch_raw':>13}  "
          f"{'ratio_F1D':>10}  {'ratio_raw':>10}")
    print("  " + "-" * 100)

    results = []
    for i in range(args.n_rand_u):
        dU = (rng.standard_normal((N_VOXEL, args.rank)) +
              1j * rng.standard_normal((N_VOXEL, args.rank))).astype(D_TYPE)

        # Compute BX = B0 * (dU @ V^H)  — the input to the Gram
        deltaX = dU @ V.conj().T              # (N_vox, N_seq)
        BX     = (B0_mat * deltaX).ravel()   # (N_vox * N_seq,)

        # Apply three variants of the data term
        g_fi, g_to_f1d, g_to_raw = gram_apply(BX, Gram_torch, F1D, F_loc)

        # Rayleigh quotients (data term only)
        RQ_fi     = rayleigh(BX, g_fi)
        RQ_to_f1d = rayleigh(BX, g_to_f1d)
        RQ_to_raw = rayleigh(BX, g_to_raw)

        err_f1d = rel_err(g_fi, g_to_f1d)
        err_raw = rel_err(g_fi, g_to_raw)

        ratio_f1d = RQ_to_f1d / (RQ_fi + 1e-30)
        ratio_raw = RQ_to_raw / (RQ_fi + 1e-30)

        print(f"  {err_f1d:>15.4f}  {err_raw:>15.4f}  "
              f"{RQ_fi:>12.4e}  {RQ_to_f1d:>13.4e}  {RQ_to_raw:>13.4e}  "
              f"{ratio_f1d:>10.6f}  {ratio_raw:>10.6f}")

        results.append(dict(
            err_f1d=err_f1d, err_raw=err_raw,
            RQ_fi=RQ_fi, RQ_to_f1d=RQ_to_f1d, RQ_to_raw=RQ_to_raw,
            ratio_f1d=ratio_f1d, ratio_raw=ratio_raw,
        ))

    print("\n[diag] Summary (mean over trials):")
    me_f1d = np.mean([r["err_f1d"]    for r in results])
    me_raw = np.mean([r["err_raw"]    for r in results])
    mr_f1d = np.mean([r["ratio_f1d"] for r in results])
    mr_raw = np.mean([r["ratio_raw"] for r in results])
    print(f"  torch_F1D vs finufft:  err = {me_f1d:.4f}%  RQ ratio = {mr_f1d:.6f}")
    print(f"  torch_raw vs finufft:  err = {me_raw:.4f}%  RQ ratio = {mr_raw:.6f}")
    print()

    # Interpretation
    if me_f1d < 1.0 and abs(mr_f1d - 1.0) < 0.01:
        print("[diag] ✓  F1D^H Gram_torch F1D  ≈  Gram_finufft")
        print("         Uncert_01 torchnufft Hessian is consistent with finufft recon.")
        print("         F1D wrapping is correct for torchnufft.")
    elif me_raw < 1.0 and abs(mr_raw - 1.0) < 0.01:
        print("[diag] ✗  Gram_torch (no F1D) ≈ Gram_finufft  (but Gram_torch WITH F1D differs)")
        print("         The F1D wrapping in Uncert_01 is WRONG for torchnufft.")
        print("         Fix: remove F1D wrapping in init_worker, use Gram_OP directly.")
    else:
        print(f"[diag] ?  Neither variant matches finufft cleanly.")
        print(f"         err_F1D={me_f1d:.2f}%  err_raw={me_raw:.2f}%")
        print(f"         May need to check domain convention (finufft FID vs torchnufft spectral?).")
        if me_f1d < me_raw:
            print("         F1D variant is closer → F1D wrapping is probably correct.")
        else:
            print("         Raw variant is closer → F1D wrapping might be wrong.")


if __name__ == "__main__":
    main()
