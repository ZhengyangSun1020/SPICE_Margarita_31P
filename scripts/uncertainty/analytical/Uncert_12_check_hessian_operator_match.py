#!/usr/bin/env python3
"""
Compare the full SPICE reconstruction Hessian matvec against Uncert_01 H_action.

This is stricter than comparing NUFFT Gram scale alone.  It checks the complete
U-space operator

    deltaU -> deltaU V^H -> B0 -> data Gram -> B0^H -> V

plus the shared spatial regularization term lambda * WW.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)

from utils.graph import calc_Bmatrix
from utils.pipeline_utils import make_brain_mask
from utils.recon import (
    Calc_B0_matrix_mx,
    build_gram_for_worker,
    build_nufft_ops,
)
from utils.scan_params import load_scan_params


D_TYPE = np.complex64


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare full recon matvec and uncertainty Hessian matvec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subject", required=True)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--data-root", default="./data/processed")
    p.add_argument("--out-root", default="./output")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dim", type=int, nargs=2, default=[64, 64], metavar=("NY", "NX"))
    p.add_argument("--n-seq-points", type=int, default=300)
    p.add_argument("--dwelltime", type=float, default=None)
    p.add_argument("--k-points", type=int, default=None)
    p.add_argument("--n-coils", type=int, default=None)
    p.add_argument("--center-freq", type=float, default=None)
    p.add_argument("--rank", type=int, default=20)
    p.add_argument("--lambda", type=float, default=None, dest="lam")
    p.add_argument("--lambda-we-max", type=float, default=5000.0)
    p.add_argument("--adj", type=int, default=8)
    p.add_argument("--pool-size", type=int, default=1)
    p.add_argument("--minpool", action="store_true")
    p.add_argument("--brain-threshold", type=float, default=0.16)
    p.add_argument("--brain-erosion", type=int, default=3)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component and fill enclosed holes. Default: off.")
    p.add_argument("--recon-backend", choices=["finufft", "torchnufft"], default="finufft")
    p.add_argument("--uncert-backend", choices=["finufft", "torchnufft"], default="torchnufft")
    p.add_argument("--device", default="cpu")
    p.add_argument("--osamp", type=float, default=2.0)
    p.add_argument("--ost", type=float, default=2.0)
    p.add_argument("--n-random", type=int, default=1)
    p.add_argument("--local-voxels", type=int, nargs="*", default=[926, 1500, 2590, 3100])
    p.add_argument("--local-ranks", type=int, nargs="*", default=[0, 5, 10, 15])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def elapsed(label, func):
    t0 = time.perf_counter()
    print(f"[hess-match] {label} ...", flush=True)
    out = func()
    dt = time.perf_counter() - t0
    print(f"[hess-match] {label} done in {dt:.2f} s", flush=True)
    return out


def parse_lambda_from_run_tag(run_tag):
    for part in run_tag.split("_"):
        if part.startswith("l") and len(part) > 1:
            try:
                return float(part[1:])
            except ValueError:
                pass
    return None


def norm_u(U):
    return float(np.linalg.norm(np.asarray(U).ravel()))


def inner_u(A, B):
    return np.vdot(np.asarray(A).ravel(), np.asarray(B).ravel())


def rayleigh(U, HU):
    denom = inner_u(U, U).real + 1e-30
    return float(np.real(inner_u(U, HU)) / denom)


def relerr(ref, test):
    return float(np.linalg.norm((np.asarray(test) - np.asarray(ref)).ravel()) /
                 (np.linalg.norm(np.asarray(ref).ravel()) + 1e-30))


def make_random_probe(rng, n_vox, rank, brain_mask=None):
    U = np.zeros((n_vox, rank), dtype=D_TYPE)
    if brain_mask is None:
        sel = np.ones(n_vox, dtype=bool)
    else:
        sel = brain_mask.ravel().astype(bool)
    z = (rng.standard_normal((int(sel.sum()), rank))
         + 1j * rng.standard_normal((int(sel.sum()), rank))) / np.sqrt(2.0)
    U[sel, :] = z.astype(D_TYPE)
    U /= norm_u(U) + 1e-30
    return U


def make_local_probe(vox, r, n_vox, rank):
    U = np.zeros((n_vox, rank), dtype=D_TYPE)
    U[vox, r] = 1.0 + 0.0j
    return U


def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(args.data_root, args.subject)
    out_dir = args.out_dir or os.path.join(args.out_root, args.subject)
    data_dir = data_dir.rstrip("/") + "/"

    if args.lam is None:
        args.lam = parse_lambda_from_run_tag(args.run_tag)
    if args.lam is None:
        raise ValueError("Could not parse lambda from --run-tag; pass --lambda explicitly.")

    Ny, Nx = args.dim
    n_vox = Ny * Nx
    load_scan_params(args, data_dir, k_key="k_mrsi")
    n_seq = args.n_seq_points
    n_coils = args.n_coils
    im_size = (Ny, Nx, n_seq)
    grid_size = (int(np.ceil(args.osamp * Ny)),
                 int(np.ceil(args.osamp * Nx)),
                 int(np.ceil(args.ost * n_seq)))
    ts = (args.k_points / n_seq) * args.dwelltime
    time_axis = np.linspace(ts, ts * n_seq, n_seq)

    print(f"[hess-match] subject={args.subject} run_tag={args.run_tag} lambda={args.lam:g}")
    print(f"[hess-match] recon_backend={args.recon_backend} uncert_backend={args.uncert_backend}")

    out_path = Path(out_dir)
    spice_dir = out_path / f"spice_{args.run_tag}"
    lprm_dir = out_path / "lipid_removal"
    coilmap_dir = out_path / "coilmap"
    b0map_dir = out_path / "b0map"

    V = np.load(spice_dir / "V_subspace.npy")[:, : args.rank].astype(D_TYPE)
    wref = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_norm, brain_mask, _ = make_brain_mask(wref, args.brain_threshold, args.brain_erosion,
                                                cleanup=args.brain_mask_cleanup)
    B0_map = np.load(b0map_dir / "B0_map.npy")
    B0_mat = Calc_B0_matrix_mx(np.nan_to_num(B0_map, nan=0.0), time_axis).reshape(n_vox, n_seq)
    coil_smap = np.load(coilmap_dir / "ecalib_pp.npy", mmap_mode="r")
    mrsi_ksp_scaled = np.load(lprm_dir / "mrsi_ksp_scaled.npy", mmap_mode="r")
    trej = mrsi_ksp_scaled.T.astype(np.float32)

    W_edge, _, _, _ = calc_Bmatrix(
        wref_norm,
        wmax=args.lambda_we_max,
        adj=args.adj,
        pool_size=args.pool_size,
        minpooling_Handler=args.minpool,
        brain_mask=brain_mask,
        mask_dilate_layers=3,
    )
    WW = (W_edge.conj().T @ W_edge).astype(D_TYPE)
    print(f"[hess-match] V={V.shape} WW={WW.shape} brain_voxels={int(brain_mask.sum())}")

    F_recon, Gram_recon, _, _ = elapsed(
        "build recon operators",
        lambda: build_nufft_ops(
            args.recon_backend, trej, im_size, coil_smap, n_coils, D_TYPE,
            osamp=args.osamp, ost=args.ost, device=args.device,
        ),
    )
    del F_recon

    if args.uncert_backend == "torchnufft":
        import torch
        import torchkbnufft as tkbn
        device_str = args.device
        ktraj_torch = torch.from_numpy(trej).permute(2, 0, 1).reshape(3, -1).to(device_str)
        kernel = elapsed(
            "build uncertainty Toeplitz kernel",
            lambda: tkbn.calc_toeplitz_kernel(
                ktraj_torch, im_size=im_size, grid_size=grid_size, norm="ortho"
            ).to(device_str),
        )
        kernel_np = kernel.cpu().numpy()
        ktraj_np = ktraj_torch.cpu().numpy()
    else:
        device_str = "cpu"
        kernel_np = None
        ktraj_np = None

    Gram_uncert, F1D_uncert, F_loc_uncert = elapsed(
        "build uncertainty operators",
        lambda: build_gram_for_worker(
            args.uncert_backend, im_size, D_TYPE,
            ktraj_np=ktraj_np, grid_size=grid_size if args.uncert_backend == "torchnufft" else None,
            kernel_np=kernel_np, device_str=device_str,
            trej_np=trej, coil_smap_raw_np=coil_smap, n_coils=n_coils,
            osamp=args.osamp, ost=args.ost,
        ),
    )
    FHF_uncert = None if F_loc_uncert is not None else (F1D_uncert.H @ Gram_uncert @ F1D_uncert)

    def recon_data(U):
        X = U.reshape(n_vox, args.rank).astype(D_TYPE)
        AA = B0_mat * (X @ V.conj().T)
        z = (Gram_recon @ AA.ravel()).reshape(n_vox, n_seq)
        return ((B0_mat.conj() * z) @ V).astype(D_TYPE)

    def uncert_data(U):
        X = U.reshape(n_vox, args.rank).astype(D_TYPE)
        deltaX = X @ V.conj().T
        BX = (B0_mat * deltaX).ravel()
        if F_loc_uncert is not None:
            z = F_loc_uncert.rmatvec(F_loc_uncert.matvec(BX.astype(D_TYPE))).reshape(n_vox, n_seq)
        else:
            z = (FHF_uncert @ BX).reshape(n_vox, n_seq)
        return ((B0_mat.conj() * z) @ V).astype(D_TYPE)

    def reg_term(U):
        X = U.reshape(n_vox, args.rank).astype(D_TYPE)
        return (args.lam * (WW @ X)).astype(D_TYPE)

    rng = np.random.default_rng(args.seed)
    probes = []
    for i in range(args.n_random):
        probes.append((f"random_brain_{i}", make_random_probe(rng, n_vox, args.rank, brain_mask)))
    for i, vox in enumerate(args.local_voxels):
        r = args.local_ranks[i % len(args.local_ranks)]
        if 0 <= vox < n_vox and 0 <= r < args.rank:
            probes.append((f"local_v{vox}_r{r}", make_local_probe(vox, r, n_vox, args.rank)))

    rows = []
    for name, U in probes:
        print(f"\n[hess-match] Probe {name}", flush=True)
        d_recon = elapsed("  recon data term", lambda U=U: recon_data(U))
        d_uncert = elapsed("  uncert data term", lambda U=U: uncert_data(U))
        reg = reg_term(U)
        h_recon = d_recon + reg
        h_uncert = d_uncert + reg

        qd_recon = rayleigh(U, d_recon)
        qd_uncert = rayleigh(U, d_uncert)
        qr = rayleigh(U, reg)
        qh_recon = rayleigh(U, h_recon)
        qh_uncert = rayleigh(U, h_uncert)
        row = {
            "probe": name,
            "norm_U": norm_u(U),
            "data_relerr_uncert_vs_recon": relerr(d_recon, d_uncert),
            "full_relerr_uncert_vs_recon": relerr(h_recon, h_uncert),
            "q_data_recon": qd_recon,
            "q_data_uncert": qd_uncert,
            "q_data_ratio_uncert_over_recon": qd_uncert / qd_recon if qd_recon != 0 else np.nan,
            "q_reg": qr,
            "q_full_recon": qh_recon,
            "q_full_uncert": qh_uncert,
            "q_full_ratio_uncert_over_recon": qh_uncert / qh_recon if qh_recon != 0 else np.nan,
            "data_over_reg_recon": qd_recon / qr if qr != 0 else np.nan,
            "data_over_reg_uncert": qd_uncert / qr if qr != 0 else np.nan,
            "norm_data_recon": norm_u(d_recon),
            "norm_data_uncert": norm_u(d_uncert),
            "norm_data_ratio_uncert_over_recon": norm_u(d_uncert) / (norm_u(d_recon) + 1e-30),
            "norm_reg": norm_u(reg),
        }
        rows.append(row)
        print(
            "[hess-match] RESULT "
            f"{name}: data_relerr={row['data_relerr_uncert_vs_recon']:.6g} "
            f"full_relerr={row['full_relerr_uncert_vs_recon']:.6g} "
            f"q_data_ratio={row['q_data_ratio_uncert_over_recon']:.6g} "
            f"q_full_ratio={row['q_full_ratio_uncert_over_recon']:.6g} "
            f"data/reg={row['data_over_reg_recon']:.6g}",
            flush=True,
        )

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = str(out_path / f"hessian_operator_match_{args.run_tag}_{args.recon_backend}_vs_{args.uncert_backend}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[hess-match] Saved {out_csv}")

    data_rel = np.array([r["data_relerr_uncert_vs_recon"] for r in rows], dtype=float)
    full_rel = np.array([r["full_relerr_uncert_vs_recon"] for r in rows], dtype=float)
    qdr = np.array([r["q_data_ratio_uncert_over_recon"] for r in rows], dtype=float)
    qfr = np.array([r["q_full_ratio_uncert_over_recon"] for r in rows], dtype=float)
    print("[hess-match] Summary")
    print(f"  data_relerr median/mean/max = {np.nanmedian(data_rel):.6g} / {np.nanmean(data_rel):.6g} / {np.nanmax(data_rel):.6g}")
    print(f"  full_relerr median/mean/max = {np.nanmedian(full_rel):.6g} / {np.nanmean(full_rel):.6g} / {np.nanmax(full_rel):.6g}")
    print(f"  q_data_ratio median/mean    = {np.nanmedian(qdr):.6g} / {np.nanmean(qdr):.6g}")
    print(f"  q_full_ratio median/mean    = {np.nanmedian(qfr):.6g} / {np.nanmean(qfr):.6g}")


if __name__ == "__main__":
    main()
