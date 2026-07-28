#!/usr/bin/env python3
"""
Estimate the relative scale of H_data and lambda*R without solving H^{-1}.

This is a fast diagnostic for questions like:

    For this SPICE forward model/subspace, at what lambda does the spatial
    regularizer become comparable to the data curvature?

For selected U-space probe directions x, the script computes

    q_data = x^H H_data x / x^H x
    q_reg0 = x^H R x / x^H x

Then for every requested lambda and wmax:

    q_reg(lambda, wmax) = lambda * q_reg0(wmax)
    reg_over_data       = q_reg / q_data
    lambda_cross        = q_data / q_reg0

If reg_over_data >> 1, that probe direction is regularization dominated.
If reg_over_data << 1, that probe direction is data dominated.
lambda_cross is the approximate transition lambda for that probe.

No CG and no H inverse are used.

Example:

    MPLCONFIGDIR=/tmp /Users/jasonlyu/miniconda3/envs/finufft/bin/python -u \\
      scripts/uncertainty/analytical/Uncert_15_check_data_reg_balance.py \\
      --subject invivo_260623_01 \\
      --run-tag w5000_l0.0001 \\
      --backend finufft \\
      --lambdas 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 \\
      --wmaxs 1000 5000 10000 \\
      --local-voxels 926 1500 2590 3100 \\
      --local-ranks 0 5 10 15 \\
      --probe-mode product
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
        description="Check H_data vs lambda*R Rayleigh scales without H inverse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subject", required=True)
    p.add_argument("--run-tag", required=True,
                   help="Run tag used only to load V_subspace. Lambdas below can be arbitrary.")
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

    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
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

    p.add_argument("--local-voxels", type=int, nargs="*", default=[926, 1500, 2590, 3100])
    p.add_argument("--local-ranks", type=int, nargs="*", default=[0, 5, 10, 15])
    p.add_argument("--all-ranks", action="store_true",
                   help="Use every rank index 0..rank-1, overriding --local-ranks.")
    p.add_argument("--probe-mode", choices=["paired", "product"], default="paired",
                   help="paired: zip voxels/ranks cyclically; product: every voxel x every rank.")
    p.add_argument("--n-random", type=int, default=0,
                   help="Additional normalized random brain-supported U-space probes.")
    p.add_argument("--auto-voxels", type=int, default=0,
                   help="Add this many random brain voxels, paired/product with --local-ranks.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-prefix", default=None)
    return p.parse_args()


def elapsed(label, fn):
    t0 = time.perf_counter()
    print(f"[data-reg] {label} ...", flush=True)
    out = fn()
    dt = time.perf_counter() - t0
    print(f"[data-reg] {label} done in {dt:.2f} s", flush=True)
    return out


def norm2(U):
    return float(np.real(np.vdot(U.ravel(), U.ravel())))


def make_local_probe(n_vox, rank, voxel, rank_idx):
    U = np.zeros((n_vox, rank), dtype=D_TYPE)
    U[voxel, rank_idx] = 1.0 + 0.0j
    return U


def make_random_probe(rng, n_vox, rank, brain_mask):
    U = np.zeros((n_vox, rank), dtype=D_TYPE)
    sel = brain_mask.ravel().astype(bool)
    z = (rng.standard_normal((int(sel.sum()), rank))
         + 1j * rng.standard_normal((int(sel.sum()), rank))) / np.sqrt(2.0)
    U[sel, :] = z.astype(D_TYPE)
    U /= np.sqrt(norm2(U)) + 1e-30
    return U


def build_probes(args, n_vox, brain_mask):
    rng = np.random.default_rng(args.seed)
    voxels = list(args.local_voxels)
    ranks = list(range(args.rank)) if args.all_ranks else list(args.local_ranks)
    if args.auto_voxels > 0:
        brain_vox = np.flatnonzero(brain_mask.ravel())
        take = min(args.auto_voxels, len(brain_vox))
        extra = rng.choice(brain_vox, size=take, replace=False)
        voxels.extend([int(v) for v in extra])

    probes = []
    if args.probe_mode == "product":
        for vox in voxels:
            if 0 <= vox < n_vox and brain_mask.ravel()[vox]:
                for r in ranks:
                    if 0 <= r < args.rank:
                        probes.append((f"local_v{vox}_r{r}", make_local_probe(n_vox, args.rank, vox, r)))
    else:
        for i, vox in enumerate(voxels):
            r = ranks[i % max(1, len(ranks))]
            if 0 <= vox < n_vox and 0 <= r < args.rank and brain_mask.ravel()[vox]:
                probes.append((f"local_v{vox}_r{r}", make_local_probe(n_vox, args.rank, vox, r)))

    for i in range(args.n_random):
        probes.append((f"random_brain_{i}", make_random_probe(rng, n_vox, args.rank, brain_mask)))

    if not probes:
        raise ValueError("No valid probes were created. Check --local-voxels/--local-ranks/brain mask.")
    return probes


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
        dim_y = wref_norm.shape[1]
        edge_index = [tuple(pair) for pair in Nb]
        W = directional_min_pool(W, edge_index, wref_norm.shape[0], dim_y, args.pool_size)

    return Nb.astype(np.int64), W.astype(np.float64)


def q_reg0_from_edges(U, Nb, W):
    """Rayleigh quotient for R using edge form sum_e W_e |x_i - x_j|^2."""
    diff = U[Nb[:, 0], :] - U[Nb[:, 1], :]
    edge_energy = np.sum(np.abs(diff) ** 2, axis=1)
    return float(np.sum(W * edge_energy) / (norm2(U) + 1e-30))


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

    print(f"[data-reg] subject={args.subject} run_tag={args.run_tag}")
    print(f"[data-reg] backend={args.backend} V={V.shape} brain_voxels={int(brain_mask.sum())}")
    print(f"[data-reg] lambdas={args.lambdas}")
    print(f"[data-reg] wmaxs={args.wmaxs}")

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

    def q_data(U):
        delta_x = U @ V.conj().T
        bx = (B0_mat * delta_x).ravel().astype(D_TYPE)
        if F_loc is not None:
            y = F_loc.matvec(bx)
            return float(np.real(np.vdot(y, y)) / (norm2(U) + 1e-30))
        z = FHF @ bx
        return float(np.real(np.vdot(bx, z)) / (norm2(U) + 1e-30))

    probes = build_probes(args, n_vox, brain_mask)
    print(f"[data-reg] probes={len(probes)}")

    data_rows = []
    q_data_by_probe = {}
    for name, U in probes:
        qd = elapsed(f"q_data {name}", lambda U=U: q_data(U))
        q_data_by_probe[name] = qd
        print(f"[data-reg]   {name}: q_data={qd:.6g}")

    qreg0_by_wmax = {}
    for wmax in args.wmaxs:
        Nb, W = elapsed(
            f"build R edge weights wmax={wmax:g}",
            lambda wmax=wmax: graph_edges_and_weights(wref_norm, wmax, args, brain_mask),
        )
        qreg0_by_probe = {}
        for name, U in probes:
            qreg0_by_probe[name] = q_reg0_from_edges(U, Nb, W)
        qreg0_by_wmax[float(wmax)] = qreg0_by_probe

        for lam in args.lambdas:
            for name, _U in probes:
                qd = q_data_by_probe[name]
                qreg0 = qreg0_by_probe[name]
                qreg = lam * qreg0
                row = {
                    "subject": args.subject,
                    "run_tag_for_data_term": args.run_tag,
                    "backend": args.backend,
                    "probe": name,
                    "wmax": float(wmax),
                    "lambda": float(lam),
                    "q_data": qd,
                    "q_reg0": qreg0,
                    "q_reg": qreg,
                    "reg_over_data": qreg / (qd + 1e-30),
                    "data_over_reg": qd / (qreg + 1e-30),
                    "lambda_cross_reg_eq_data": qd / (qreg0 + 1e-30),
                }
                data_rows.append(row)

    summary_rows = []
    for wmax in args.wmaxs:
        lambda_cross = np.asarray([
            q_data_by_probe[name] / (qreg0_by_wmax[float(wmax)][name] + 1e-30)
            for name, _U in probes
        ], dtype=float)
        for lam in args.lambdas:
            vals = np.asarray([
                r["reg_over_data"] for r in data_rows
                if r["wmax"] == float(wmax) and r["lambda"] == float(lam)
            ], dtype=float)
            summary_rows.append({
                "wmax": float(wmax),
                "lambda": float(lam),
                "n_probes": int(len(vals)),
                "median_reg_over_data": float(np.nanmedian(vals)),
                "mean_reg_over_data": float(np.nanmean(vals)),
                "min_reg_over_data": float(np.nanmin(vals)),
                "max_reg_over_data": float(np.nanmax(vals)),
                "frac_reg_dominant": float(np.nanmean(vals > 1.0)),
                "median_lambda_cross": float(np.nanmedian(lambda_cross)),
                "p25_lambda_cross": float(np.nanpercentile(lambda_cross, 25)),
                "p75_lambda_cross": float(np.nanpercentile(lambda_cross, 75)),
            })

    if args.out_prefix is None:
        safe = args.run_tag.replace(".", "p").replace("-", "m")
        out_prefix = out_path / f"data_reg_balance_{safe}"
    else:
        out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    detail_csv = Path(str(out_prefix) + "_detail.csv")
    with detail_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data_rows[0].keys()))
        writer.writeheader()
        writer.writerows(data_rows)

    summary_csv = Path(str(out_prefix) + "_summary.csv")
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[data-reg] Saved {detail_csv}")
    print(f"[data-reg] Saved {summary_csv}")
    print("\n[data-reg] Summary:")
    for row in summary_rows:
        print(
            f"  wmax={row['wmax']:g} lambda={row['lambda']:g} "
            f"median reg/data={row['median_reg_over_data']:.4g} "
            f"frac reg>data={row['frac_reg_dominant']:.2f} "
            f"median lambda_cross={row['median_lambda_cross']:.4g}"
        )


if __name__ == "__main__":
    main()
