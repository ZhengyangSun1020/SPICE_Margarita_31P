#!/usr/bin/env python3
"""
Check H_data vs lambda*R along the largest local mHm eigen-directions.

Uncert_15 checks canonical basis probes such as one voxel / one rank.  This
script instead asks a more targeted question:

    In the directions where the saved local posterior block mHm_v has the
    largest variance, are we data-dominated or regularization-dominated?

For each run tag, it loads:

    output/<subject>/spice_<run_tag>/V_subspace.npy
    output/<subject>/hessian_<run_tag>/mHm_*.npy

For selected voxels, it diagonalizes each 20x20 mHm block, takes the largest
eigenvector(s), constructs a local U-space probe, and computes:

    q_data = x^H H_data x / x^H x
    q_reg  = x^H (lambda R) x / x^H x
    reg_over_data = q_reg / q_data

Important: mHm_v is only the local voxel block of H^{-1}; this does not recover
global spatial eigenmodes.  It is still a much better diagnostic than probing a
single canonical rank direction.

Example:

    MPLCONFIGDIR=/tmp /Users/jasonlyu/miniconda3/envs/finufft/bin/python -u \\
      scripts/uncertainty/analytical/Uncert_16_check_mhm_eig_directions.py \\
      --subject invivo_260623_01 \\
      --run-tags w5000_l0.0001 w5000_l1e-05 w5000_l1e-06 \\
      --backend finufft \\
      --wmaxs 5000 \\
      --top-k 20 \\
      --top-eigs 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)

from utils.graph import calc_neighbours, calc_W, directional_min_pool
from utils.pipeline_utils import make_brain_mask
from utils.recon import Calc_B0_matrix_mx, build_gram_for_worker
from utils.scan_params import load_scan_params


D_TYPE = np.complex64


def parse_args():
    p = argparse.ArgumentParser(
        description="Check data/reg balance along largest local mHm eigen-directions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subject", required=True)
    p.add_argument("--run-tags", nargs="+", required=True)
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
    p.add_argument("--backend", choices=["finufft", "torchnufft"], default="finufft")
    p.add_argument("--device", default="cpu")
    p.add_argument("--osamp", type=float, default=2.0)
    p.add_argument("--ost", type=float, default=2.0)

    p.add_argument("--wmaxs", type=float, nargs="+", default=[5000.0])
    p.add_argument("--adj", type=int, default=8)
    p.add_argument("--pool-size", type=int, default=1)
    p.add_argument("--minpool", action="store_true")
    p.add_argument("--mask-dilate-layers", type=int, default=3)
    p.add_argument("--mask-rule", choices=["any", "both"], default="any")
    p.add_argument("--brain-threshold", type=float, default=0.16)
    p.add_argument("--brain-erosion", type=int, default=1)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component and fill enclosed holes. Default: off.")

    p.add_argument("--voxels", type=int, nargs="*", default=None,
                   help="Explicit voxels to probe. If omitted, select top-k by --select-by.")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of voxels to select when --voxels is omitted.")
    p.add_argument("--top-eigs", type=int, default=1,
                   help="Number of largest local mHm eigenvectors per selected voxel.")
    p.add_argument("--select-by", choices=["largest_eig", "trace", "posterior_std"],
                   default="largest_eig",
                   help="How to select voxels when --voxels is omitted.")
    p.add_argument("--out-prefix", default=None)
    return p.parse_args()


def elapsed(label, fn):
    t0 = time.perf_counter()
    print(f"[mhm-eig] {label} ...", flush=True)
    out = fn()
    dt = time.perf_counter() - t0
    print(f"[mhm-eig] {label} done in {dt:.2f} s", flush=True)
    return out


def parse_lambda(run_tag: str):
    for part in run_tag.split("_"):
        if part.startswith("l") and len(part) > 1:
            try:
                return float(part[1:])
            except ValueError:
                pass
    return None


def norm2(U):
    return float(np.real(np.vdot(U.ravel(), U.ravel())))


def make_probe(n_vox, rank, voxel, direction):
    U = np.zeros((n_vox, rank), dtype=D_TYPE)
    d = np.asarray(direction[:rank], dtype=D_TYPE)
    d = d / (np.linalg.norm(d) + 1e-30)
    U[voxel, :] = d
    return U


def graph_edges_and_weights(wref_norm, wmax, args, brain_mask):
    Nb, _ = calc_neighbours(wref_norm, args.adj)
    W = calc_W(wref_norm.ravel(), wmax, Nb)

    if brain_mask is not None:
        mask = np.asarray(brain_mask, dtype=bool)
        dilated_mask = binary_dilation(mask, iterations=args.mask_dilate_layers)
        mask_flat = dilated_mask.ravel()
        if args.mask_rule == "any":
            out_of_mask = ~(mask_flat[Nb[:, 0]] & mask_flat[Nb[:, 1]])
            if np.any(out_of_mask):
                W = W.copy()
                W[out_of_mask] = wmax / 10.0
        else:
            both_out = (~mask_flat[Nb[:, 0]]) & (~mask_flat[Nb[:, 1]])
            if np.any(both_out):
                W = W.copy()
                W[both_out] = wmax / 10.0

    if args.minpool:
        edge_index = [tuple(pair) for pair in Nb]
        W = directional_min_pool(W, edge_index, wref_norm.shape[0], wref_norm.shape[1], args.pool_size)
    return Nb.astype(np.int64), W.astype(np.float64)


def q_reg0_from_edges(U, Nb, W):
    diff = U[Nb[:, 0], :] - U[Nb[:, 1], :]
    edge_energy = np.sum(np.abs(diff) ** 2, axis=1)
    return float(np.sum(W * edge_energy) / (norm2(U) + 1e-30))


def hermitian_eigh(M):
    H = 0.5 * (np.asarray(M) + np.asarray(M).conj().T)
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals.real)[::-1]
    return vals.real[order], vecs[:, order]


def voxel_metric_from_mhm(path):
    M = np.load(path)
    vals, _ = hermitian_eigh(M)
    return float(vals[0]), float(np.sum(np.maximum(vals, 0.0)))


def select_voxels(args, run_tag, out_path, brain_mask):
    hess_dir = out_path / f"hessian_{run_tag}"
    files = sorted(hess_dir.glob("mHm_*.npy"))
    if not files:
        raise FileNotFoundError(f"No mHm_*.npy files in {hess_dir}")

    if args.voxels is not None and len(args.voxels) > 0:
        voxels = [int(v) for v in args.voxels if (hess_dir / f"mHm_{int(v)}.npy").exists()]
        return voxels

    metrics = []
    if args.select_by == "posterior_std":
        std_path = out_path / f"uncertainty_{run_tag}" / "posterior_std.npy"
        std = np.abs(np.load(std_path))
        std_map = np.mean(std, axis=-1).ravel()
        for f in files:
            vox = int(f.stem.split("_")[1])
            if brain_mask.ravel()[vox]:
                metrics.append((float(std_map[vox]), vox))
    else:
        for f in files:
            vox = int(f.stem.split("_")[1])
            if not brain_mask.ravel()[vox]:
                continue
            largest, trace = voxel_metric_from_mhm(f)
            score = largest if args.select_by == "largest_eig" else trace
            metrics.append((score, vox))

    metrics.sort(reverse=True)
    return [vox for _score, vox in metrics[: args.top_k]]


def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(args.data_root, args.subject)
    out_dir = args.out_dir or os.path.join(args.out_root, args.subject)
    data_dir = data_dir.rstrip("/") + "/"
    out_path = Path(out_dir)

    Ny, Nx = args.dim
    n_vox = Ny * Nx
    n_seq = args.n_seq_points
    im_size = (Ny, Nx, n_seq)
    load_scan_params(args, data_dir, k_key="k_mrsi")
    ts = (args.k_points / n_seq) * args.dwelltime
    time_axis = np.linspace(ts, ts * n_seq, n_seq)

    lprm_dir = out_path / "lipid_removal"
    coilmap_dir = out_path / "coilmap"
    b0map_dir = out_path / "b0map"

    wref = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_norm, brain_mask, _ = make_brain_mask(wref, args.brain_threshold, args.brain_erosion,
                                                cleanup=args.brain_mask_cleanup)
    B0_map = np.load(b0map_dir / "B0_map.npy")
    B0_mat = Calc_B0_matrix_mx(np.nan_to_num(B0_map, nan=0.0), time_axis).reshape(n_vox, n_seq)
    coil_smap = np.load(coilmap_dir / "ecalib_pp.npy", mmap_mode="r")
    mrsi_ksp_scaled = np.load(lprm_dir / "mrsi_ksp_scaled.npy", mmap_mode="r")
    trej = mrsi_ksp_scaled.T.astype(np.float32)

    print(f"[mhm-eig] subject={args.subject} run_tags={args.run_tags}")
    print(f"[mhm-eig] backend={args.backend} brain_voxels={int(brain_mask.sum())}")

    if args.backend == "torchnufft":
        import torch
        import torchkbnufft as tkbn

        grid_size = (int(np.ceil(args.osamp * Ny)),
                     int(np.ceil(args.osamp * Nx)),
                     int(np.ceil(args.ost * n_seq)))
        ktraj_torch = torch.from_numpy(trej).permute(2, 0, 1).reshape(3, -1).to(args.device)
        kernel = elapsed(
            "build Toeplitz kernel",
            lambda: tkbn.calc_toeplitz_kernel(
                ktraj_torch, im_size=im_size, grid_size=grid_size, norm="ortho"
            ).to(args.device),
        )
        kernel_np = kernel.cpu().numpy()
        ktraj_np = ktraj_torch.cpu().numpy()
        device_str = args.device
    else:
        grid_size = None
        kernel_np = None
        ktraj_np = None
        device_str = "cpu"

    Gram_OP, F1D, F_loc = elapsed(
        "build data Gram",
        lambda: build_gram_for_worker(
            args.backend,
            im_size,
            D_TYPE,
            ktraj_np=ktraj_np,
            grid_size=grid_size,
            kernel_np=kernel_np,
            device_str=device_str,
            trej_np=trej,
            coil_smap_raw_np=coil_smap,
            n_coils=args.n_coils,
            osamp=args.osamp,
            ost=args.ost,
        ),
    )
    FHF = None if F_loc is not None else (F1D.H @ Gram_OP @ F1D)

    edge_cache = {}
    for wmax in args.wmaxs:
        edge_cache[float(wmax)] = elapsed(
            f"build R edge weights wmax={wmax:g}",
            lambda wmax=wmax: graph_edges_and_weights(wref_norm, wmax, args, brain_mask),
        )

    rows = []
    for run_tag in args.run_tags:
        lam = parse_lambda(run_tag)
        if lam is None:
            raise ValueError(f"Could not parse lambda from run tag {run_tag!r}")

        V = np.load(out_path / f"spice_{run_tag}" / "V_subspace.npy")[:, : args.rank].astype(D_TYPE)
        voxels = select_voxels(args, run_tag, out_path, brain_mask)
        print(f"\n[mhm-eig] run_tag={run_tag} lambda={lam:g} selected_voxels={voxels}")

        def q_data(U):
            delta_x = U @ V.conj().T
            bx = (B0_mat * delta_x).ravel().astype(D_TYPE)
            if F_loc is not None:
                y = F_loc.matvec(bx)
                return float(np.real(np.vdot(y, y)) / (norm2(U) + 1e-30))
            z = FHF @ bx
            return float(np.real(np.vdot(bx, z)) / (norm2(U) + 1e-30))

        for vox in voxels:
            mhm_path = out_path / f"hessian_{run_tag}" / f"mHm_{vox}.npy"
            vals, vecs = hermitian_eigh(np.load(mhm_path))
            for eig_idx in range(min(args.top_eigs, len(vals))):
                direction = vecs[:, eig_idx]
                U = make_probe(n_vox, args.rank, vox, direction)
                qd = elapsed(
                    f"q_data {run_tag} voxel={vox} eig={eig_idx}",
                    lambda U=U: q_data(U),
                )
                for wmax in args.wmaxs:
                    Nb, W = edge_cache[float(wmax)]
                    qreg0 = q_reg0_from_edges(U, Nb, W)
                    qreg = lam * qreg0
                    rows.append({
                        "run_tag": run_tag,
                        "lambda": lam,
                        "voxel": int(vox),
                        "eig_idx": int(eig_idx),
                        "mhm_eigval": float(vals[eig_idx]),
                        "mhm_trace_pos": float(np.sum(np.maximum(vals, 0.0))),
                        "wmax": float(wmax),
                        "q_data": qd,
                        "q_reg0": qreg0,
                        "q_reg": qreg,
                        "reg_over_data": qreg / (qd + 1e-30),
                        "data_over_reg": qd / (qreg + 1e-30),
                        "lambda_cross_reg_eq_data": qd / (qreg0 + 1e-30),
                    })
                    print(
                        f"[mhm-eig]   voxel={vox} eig={eig_idx} "
                        f"mhm_eig={vals[eig_idx]:.6g} "
                        f"wmax={wmax:g} q_data={qd:.6g} "
                        f"reg/data={qreg / (qd + 1e-30):.6g} "
                        f"lambda_cross={qd / (qreg0 + 1e-30):.6g}",
                        flush=True,
                    )

    if args.out_prefix is None:
        safe = "_".join(args.run_tags).replace(".", "p").replace("-", "m")
        out_prefix = out_path / f"mhm_eig_data_reg_{safe}"
    else:
        out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    detail_csv = Path(str(out_prefix) + "_detail.csv")
    with detail_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for run_tag in args.run_tags:
        for wmax in args.wmaxs:
            vals = np.asarray([
                r["reg_over_data"] for r in rows
                if r["run_tag"] == run_tag and r["wmax"] == float(wmax)
            ], dtype=float)
            crosses = np.asarray([
                r["lambda_cross_reg_eq_data"] for r in rows
                if r["run_tag"] == run_tag and r["wmax"] == float(wmax)
            ], dtype=float)
            eigvals = np.asarray([
                r["mhm_eigval"] for r in rows
                if r["run_tag"] == run_tag and r["wmax"] == float(wmax)
            ], dtype=float)
            summary_rows.append({
                "run_tag": run_tag,
                "lambda": parse_lambda(run_tag),
                "wmax": float(wmax),
                "n_probes": int(len(vals)),
                "median_reg_over_data": float(np.nanmedian(vals)),
                "mean_reg_over_data": float(np.nanmean(vals)),
                "min_reg_over_data": float(np.nanmin(vals)),
                "max_reg_over_data": float(np.nanmax(vals)),
                "frac_reg_dominant": float(np.nanmean(vals > 1.0)),
                "median_lambda_cross": float(np.nanmedian(crosses)),
                "median_mhm_eigval": float(np.nanmedian(eigvals)),
                "max_mhm_eigval": float(np.nanmax(eigvals)),
            })

    summary_csv = Path(str(out_prefix) + "_summary.csv")
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[mhm-eig] Saved {detail_csv}")
    print(f"[mhm-eig] Saved {summary_csv}")
    print("\n[mhm-eig] Summary:")
    for row in summary_rows:
        print(
            f"  {row['run_tag']} wmax={row['wmax']:g} "
            f"median reg/data={row['median_reg_over_data']:.4g} "
            f"frac reg>data={row['frac_reg_dominant']:.2f} "
            f"median lambda_cross={row['median_lambda_cross']:.4g} "
            f"median mhm eig={row['median_mhm_eigval']:.4g}"
        )


if __name__ == "__main__":
    main()
