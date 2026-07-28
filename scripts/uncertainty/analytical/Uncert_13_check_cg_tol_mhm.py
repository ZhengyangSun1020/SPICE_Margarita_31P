#!/usr/bin/env python3
"""
Check whether CG tolerance changes local Hessian-inverse blocks mHm.

This diagnostic rebuilds the same Hessian operator used by Uncert_01,
solves H x = e_{voxel,rank} for a small set of voxels/ranks and rtols,
and compares the resulting selected mHm block:

    mHm_sel = E_sel^H H^{-1} E_sel

The goal is to test whether rtol=1e-3 is too loose for covariance columns,
especially at small lambda where H is more ill-conditioned.

Example quick smoke test:

    MPLCONFIGDIR=/tmp /Users/jasonlyu/miniconda3/envs/finufft/bin/python -u \\
      scripts/uncertainty/analytical/Uncert_13_check_cg_tol_mhm.py \\
      --subject invivo_260623_01 \\
      --run-tags w5000_l0.0001 w5000_l1e-05 w5000_l1e-06 \\
      --backend finufft \\
      --voxels 1500 \\
      --ranks 0 5 \\
      --rtols 1e-3 1e-5 1e-6
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _root)

from utils.graph import calc_Bmatrix
from utils.pipeline_utils import make_brain_mask
from utils.recon import Calc_B0_matrix_mx, build_gram_for_worker
from utils.scan_params import load_scan_params


D_TYPE = np.complex64


def parse_lambda_from_run_tag(run_tag):
    for part in run_tag.split("_"):
        if part.startswith("l") and len(part) > 1:
            try:
                return float(part[1:])
            except ValueError:
                pass
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Check CG rtol sensitivity for local mHm blocks.",
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
    p.add_argument("--lambda-we-max", type=float, default=5000.0)
    p.add_argument("--adj", type=int, default=8)
    p.add_argument("--pool-size", type=int, default=1)
    p.add_argument("--minpool", action="store_true")
    p.add_argument("--brain-threshold", type=float, default=0.16)
    p.add_argument("--brain-erosion", type=int, default=3)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component and fill enclosed holes. Default: off.")
    p.add_argument("--backend", choices=["finufft", "torchnufft"], default="finufft")
    p.add_argument("--device", default="cpu")
    p.add_argument("--osamp", type=float, default=2.0)
    p.add_argument("--ost", type=float, default=2.0)
    p.add_argument("--voxels", type=int, nargs="+", default=[1500])
    p.add_argument("--ranks", type=int, nargs="+", default=[0, 5])
    p.add_argument("--all-ranks", action="store_true")
    p.add_argument("--rtols", type=float, nargs="+", default=[1e-3, 1e-5, 1e-6])
    p.add_argument("--cg-maxiter", type=int, default=2000)
    p.add_argument("--compare-saved", action="store_true",
                   help="Compare selected trace against saved mHm_<vox>.npy when available.")
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def elapsed(label, fn):
    t0 = time.perf_counter()
    print(f"[cg-tol] {label} ...", flush=True)
    out = fn()
    dt = time.perf_counter() - t0
    print(f"[cg-tol] {label} done in {dt:.2f} s", flush=True)
    return out


def make_h_linop(n_vox, rank, V, B0_mat, WW, lam, F_loc=None, FHF=None):
    d = n_vox * rank

    def mv(u_flat):
        deltaU = np.asarray(u_flat, dtype=D_TYPE).reshape(n_vox, rank)
        deltaX = deltaU @ V.conj().T
        bx = (B0_mat * deltaX).ravel()
        if F_loc is not None:
            z = F_loc.rmatvec(F_loc.matvec(bx.astype(D_TYPE))).reshape(B0_mat.shape)
        else:
            z = (FHF @ bx).reshape(B0_mat.shape)
        data = (B0_mat.conj() * z) @ V
        reg = lam * (WW @ deltaU)
        return (data + reg).ravel().astype(D_TYPE)

    return LinearOperator((d, d), matvec=mv, dtype=D_TYPE)


def solve_column(H_op, b, rtol, maxiter):
    iters = {"n": 0}

    def cb(_xk):
        iters["n"] += 1

    t0 = time.perf_counter()
    x, info = cg(
        H_op,
        b,
        x0=np.zeros_like(b),
        rtol=rtol,
        maxiter=maxiter,
        callback=cb,
    )
    dt = time.perf_counter() - t0
    rel_res = float(np.linalg.norm(H_op @ x - b) / (np.linalg.norm(b) + 1e-30))
    return x.astype(D_TYPE, copy=False), int(info), int(iters["n"]), rel_res, dt


def summarize_block(M):
    Mh = 0.5 * (M + M.conj().T)
    eig = np.linalg.eigvalsh(Mh)
    tr = np.trace(M)
    return {
        "trace_real": float(np.real(tr)),
        "trace_imag_abs": float(abs(np.imag(tr))),
        "min_eig": float(np.min(eig)),
        "max_eig": float(np.max(eig)),
        "cond_eig": float(np.max(eig) / (np.min(eig) + 1e-30)),
        "herm_relerr": float(np.linalg.norm(M - M.conj().T) / (np.linalg.norm(M) + 1e-30)),
    }


def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(args.data_root, args.subject)
    out_dir = args.out_dir or os.path.join(args.out_root, args.subject)
    data_dir = data_dir.rstrip("/") + "/"
    out_path = Path(out_dir)

    load_scan_params(args, data_dir, k_key="k_mrsi")
    Ny, Nx = args.dim
    n_vox = Ny * Nx
    n_seq = args.n_seq_points
    n_coils = args.n_coils
    im_size = (Ny, Nx, n_seq)
    ts = (args.k_points / n_seq) * args.dwelltime
    time_axis = np.linspace(ts, ts * n_seq, n_seq)

    ranks = list(range(args.rank)) if args.all_ranks else sorted(set(args.ranks))
    ranks = [r for r in ranks if 0 <= r < args.rank]
    voxels = sorted(set(args.voxels))
    if not ranks:
        raise ValueError("No valid ranks selected.")
    print(f"[cg-tol] subject={args.subject} backend={args.backend}")
    print(f"[cg-tol] run_tags={args.run_tags}")
    print(f"[cg-tol] voxels={voxels} ranks={ranks} rtols={args.rtols}")

    wref = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_norm, brain_mask, _ = make_brain_mask(wref, args.brain_threshold, args.brain_erosion,
                                                cleanup=args.brain_mask_cleanup)
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
    print(f"[cg-tol] brain_voxels={int(brain_mask.sum())} WW={WW.shape}")

    lprm_dir = out_path / "lipid_removal"
    coilmap_dir = out_path / "coilmap"
    b0map_dir = out_path / "b0map"
    B0_map = np.load(b0map_dir / "B0_map.npy")
    B0_mat = Calc_B0_matrix_mx(np.nan_to_num(B0_map, nan=0.0), time_axis).reshape(n_vox, n_seq)
    coil_smap = np.load(coilmap_dir / "ecalib_pp.npy", mmap_mode="r")
    mrsi_ksp_scaled = np.load(lprm_dir / "mrsi_ksp_scaled.npy", mmap_mode="r")
    trej = mrsi_ksp_scaled.T.astype(np.float32)

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
        "build Hessian Gram",
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
            n_coils=n_coils,
            osamp=args.osamp,
            ost=args.ost,
        ),
    )
    FHF = None if F_loc is not None else (F1D.H @ Gram_OP @ F1D)

    rows = []
    cache = {}

    for run_tag in args.run_tags:
        lam = parse_lambda_from_run_tag(run_tag)
        if lam is None:
            raise ValueError(f"Could not parse lambda from run tag {run_tag!r}")
        spice_dir = out_path / f"spice_{run_tag}"
        V = np.load(spice_dir / "V_subspace.npy")[:, : args.rank].astype(D_TYPE)
        H_op = make_h_linop(n_vox, args.rank, V, B0_mat, WW, lam, F_loc=F_loc, FHF=FHF)
        d = n_vox * args.rank
        hess_dir = out_path / f"hessian_{run_tag}"

        print(f"\n[cg-tol] === run_tag={run_tag} lambda={lam:g} ===", flush=True)
        for vox in voxels:
            if not (0 <= vox < n_vox):
                print(f"[cg-tol] skip invalid voxel {vox}")
                continue
            saved_sel_trace = np.nan
            if args.compare_saved:
                saved_path = hess_dir / f"mHm_{vox}.npy"
                if saved_path.exists():
                    saved = np.load(saved_path)
                    saved_sel_trace = float(np.real(np.trace(saved[np.ix_(ranks, ranks)])))

            for rtol in args.rtols:
                cols = []
                solve_infos = []
                print(f"[cg-tol] voxel={vox} rtol={rtol:g}", flush=True)
                for r in ranks:
                    b = np.zeros(d, dtype=D_TYPE)
                    b[vox * args.rank + r] = 1.0 + 0.0j
                    x, info, n_iter, rel_res, dt = solve_column(H_op, b, rtol, args.cg_maxiter)
                    cols.append(x)
                    solve_infos.append((r, info, n_iter, rel_res, dt))
                    print(
                        f"  rank={r:2d} info={info:4d} iter={n_iter:4d} "
                        f"rel_res={rel_res:.3e} time={dt:.2f}s",
                        flush=True,
                    )

                X = np.column_stack(cols)
                Bsel = np.zeros((d, len(ranks)), dtype=D_TYPE)
                for j, r in enumerate(ranks):
                    Bsel[vox * args.rank + r, j] = 1.0
                M = Bsel.conj().T @ X
                stats = summarize_block(M)
                cache[(run_tag, vox, rtol)] = stats
                max_iter = max(x[2] for x in solve_infos)
                max_rel = max(x[3] for x in solve_infos)
                any_info = max(abs(x[1]) for x in solve_infos)
                print(
                    "[cg-tol] block "
                    f"trace={stats['trace_real']:.6e} "
                    f"sqrt(trace)={np.sqrt(max(stats['trace_real'], 0.0)):.6e} "
                    f"min_eig={stats['min_eig']:.6e} "
                    f"herm={stats['herm_relerr']:.3e}",
                    flush=True,
                )

                rows.append({
                    "run_tag": run_tag,
                    "lambda": lam,
                    "voxel": vox,
                    "ranks": " ".join(str(r) for r in ranks),
                    "rtol": rtol,
                    "cg_maxiter": args.cg_maxiter,
                    "max_iter": max_iter,
                    "max_rel_res": max_rel,
                    "any_info_abs": any_info,
                    "saved_selected_trace": saved_sel_trace,
                    **stats,
                })

            # Within this voxel/run, report relative trace changes vs loosest rtol.
            base_rtol = args.rtols[0]
            base = cache.get((run_tag, vox, base_rtol), {}).get("trace_real", np.nan)
            if np.isfinite(base) and base > 0:
                print(f"[cg-tol] trace ratios vs rtol={base_rtol:g} for voxel={vox}:")
                for rtol in args.rtols:
                    tr = cache[(run_tag, vox, rtol)]["trace_real"]
                    print(f"  rtol={rtol:g}: trace_ratio={tr/base:.6g} std_ratio={np.sqrt(tr/base):.6g}")

    # Across lambda, report trace ratios for each rtol and voxel.
    if len(args.run_tags) >= 2:
        print("\n[cg-tol] === across-lambda selected trace ratios ===")
        for vox in voxels:
            for rtol in args.rtols:
                vals = []
                labels = []
                for run_tag in args.run_tags:
                    key = (run_tag, vox, rtol)
                    if key in cache:
                        vals.append(cache[key]["trace_real"])
                        labels.append(run_tag)
                if len(vals) < 2 or any(v <= 0 for v in vals):
                    continue
                pieces = []
                for i in range(1, len(vals)):
                    ratio = vals[i] / vals[i - 1]
                    pieces.append(
                        f"{labels[i]}/{labels[i-1]} trace={ratio:.4g} std={np.sqrt(ratio):.4g}"
                    )
                print(f"  voxel={vox} rtol={rtol:g}: " + " | ".join(pieces))

    if args.out_csv is None:
        tag = "_".join(args.run_tags).replace(".", "p").replace("-", "m")
        args.out_csv = str(out_path / f"cg_tol_mhm_check_{tag}_{args.backend}.csv")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n[cg-tol] Saved {out_csv}")


if __name__ == "__main__":
    main()
