#!/usr/bin/env python3
"""
Test whether MC/VULCAN mismatch is mostly a global scalar c(lambda), or a
data/regularization relative-balance effect.

Idea in plain terms
-------------------
For each lambda, compute the voxelwise ratio

    ratio(v) = mean_ppm(MC_std(v, ppm)) / mean_ppm(VULCAN_std(v, ppm)).

If the only problem is a global scalar c(lambda), then after dividing by the
median ratio for that lambda,

    ratio_norm(v) = ratio(v) / median_v(ratio(v)),

the residual map should be roughly spatially uniform. In particular it should
not be strongly correlated with the strength of the spatial prior.

If the problem is a data-vs-regularization balance mismatch, then voxels where
lambda*R is stronger relative to H_data should have systematically different
ratio_norm. This script checks that in two ways:

  1. Cheap all-voxel proxy:
       correlate ratio_norm(v) with diag(W^H W)(v).

  2. Optional true local Rayleigh probes:
       for selected voxel/rank basis vectors x, compute

           q_data = x^H H_data x
           q_reg  = x^H (lambda R) x
           balance = q_reg / q_data

       and correlate ratio_norm(voxel) with balance.

Example quick run:

    MPLCONFIGDIR=/tmp /Users/jasonlyu/miniconda3/envs/finufft/bin/python -u \\
      scripts/uncertainty/analytical/Uncert_14_test_scalar_vs_ab.py \\
      --subject invivo_260623_01 \\
      --run-tags w5000_l0.0001 w5000_l1e-05 w5000_l1e-06 \\
      --skip-rayleigh

Example with true Rayleigh probes:

    MPLCONFIGDIR=/tmp /Users/jasonlyu/miniconda3/envs/finufft/bin/python -u \\
      scripts/uncertainty/analytical/Uncert_14_test_scalar_vs_ab.py \\
      --subject invivo_260623_01 \\
      --run-tags w5000_l0.0001 w5000_l1e-05 w5000_l1e-06 \\
      --rayleigh-voxels 926 1500 2590 3100 \\
      --rayleigh-ranks 0 5 10 15
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
from scipy.stats import pearsonr, spearmanr

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
        description="Test global scalar vs data/reg-balance explanation for MC/VULCAN ratio.",
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
    p.add_argument("--brain-erosion", type=int, default=1)
    p.add_argument("--brain-mask-cleanup", action="store_true",
                   help="Extra cleanup pass on the thresholded brain mask: keep only the "
                        "largest connected component and fill enclosed holes. Default: off.")
    p.add_argument("--ppm-range", type=float, nargs=2, default=None)
    p.add_argument("--backend", choices=["finufft", "torchnufft"], default="finufft")
    p.add_argument("--device", default="cpu")
    p.add_argument("--osamp", type=float, default=2.0)
    p.add_argument("--ost", type=float, default=2.0)
    p.add_argument("--skip-rayleigh", action="store_true")
    p.add_argument("--rayleigh-voxels", type=int, nargs="*", default=[926, 1500, 2590, 3100])
    p.add_argument("--rayleigh-ranks", type=int, nargs="*", default=[0, 5, 10, 15])
    p.add_argument("--out-prefix", default=None)
    return p.parse_args()


def elapsed(label, fn):
    t0 = time.perf_counter()
    print(f"[scalar-vs-ab] {label} ...", flush=True)
    out = fn()
    dt = time.perf_counter() - t0
    print(f"[scalar-vs-ab] {label} done in {dt:.2f} s", flush=True)
    return out


def corr_pair(x, y, mask=None):
    xx = np.asarray(x)
    yy = np.asarray(y)
    if mask is not None:
        xx = xx[mask]
        yy = yy[mask]
    xx = xx.ravel()
    yy = yy.ravel()
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if len(xx) < 3 or np.std(xx) == 0.0 or np.std(yy) == 0.0:
        return np.nan, np.nan, int(len(xx))
    return float(pearsonr(xx, yy)[0]), float(spearmanr(xx, yy)[0]), int(len(xx))


def auto_group_path(out_root, subject, run_tag):
    series = "_".join(subject.split("_")[:-1])
    return Path(out_root) / f"group_{series}_{run_tag}" / "prefitting_std.npy"


def load_ratio_map(args, run_tag, brain_mask, ppm_sel):
    out_dir = Path(args.out_dir or (Path(args.out_root) / args.subject))
    analyt_path = out_dir / f"uncertainty_{run_tag}" / "posterior_std.npy"
    group_path = auto_group_path(args.out_root, args.subject, run_tag)
    if not analyt_path.exists():
        raise FileNotFoundError(f"Analytical std not found: {analyt_path}")
    if not group_path.exists():
        raise FileNotFoundError(f"Group std not found: {group_path}")

    analyt = np.abs(np.load(analyt_path))
    mc = np.abs(np.load(group_path))
    if ppm_sel is not None:
        analyt = analyt[:, :, ppm_sel]
        mc = mc[:, :, ppm_sel]
    analyt_mean = analyt.mean(axis=-1)
    mc_mean = mc.mean(axis=-1)
    ratio = mc_mean / (analyt_mean + 1e-30)
    ratio[~brain_mask] = np.nan

    x = analyt_mean[brain_mask].ravel()
    y = mc_mean[brain_mask].ravel()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    origin_slope = float((x[ok] @ y[ok]) / (x[ok] @ x[ok]))
    median_ratio = float(np.nanmedian(ratio[brain_mask]))
    return ratio, median_ratio, origin_slope, str(analyt_path), str(group_path)


def build_rayleigh_operator(args, data_dir, out_dir, n_vox, n_seq, im_size, B0_mat):
    lprm_dir = out_dir / "lipid_removal"
    coilmap_dir = out_dir / "coilmap"
    coil_smap = np.load(coilmap_dir / "ecalib_pp.npy", mmap_mode="r")
    mrsi_ksp_scaled = np.load(lprm_dir / "mrsi_ksp_scaled.npy", mmap_mode="r")
    trej = mrsi_ksp_scaled.T.astype(np.float32)

    if args.backend == "torchnufft":
        import torch
        import torchkbnufft as tkbn

        grid_size = (int(np.ceil(args.osamp * args.dim[0])),
                     int(np.ceil(args.osamp * args.dim[1])),
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
        "build Rayleigh Gram",
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

    def q_data_local(vox, rank_idx, V):
        # Local U basis vector: U[vox, rank_idx] = 1.
        delta_x = np.zeros((n_vox, n_seq), dtype=D_TYPE)
        delta_x[vox, :] = V[:, rank_idx].conj()
        bx = (B0_mat * delta_x).ravel().astype(D_TYPE)
        if F_loc is not None:
            y = F_loc.matvec(bx)
            return float(np.real(np.vdot(y, y)))
        z = FHF @ bx
        return float(np.real(np.vdot(bx, z)))

    return q_data_local


def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(args.data_root, args.subject)
    out_dir = Path(args.out_dir or (Path(args.out_root) / args.subject))
    data_dir = data_dir.rstrip("/") + "/"

    Ny, Nx = args.dim
    n_vox = Ny * Nx
    n_seq = args.n_seq_points
    im_size = (Ny, Nx, n_seq)

    load_scan_params(args, data_dir, k_key="k_mrsi")
    ts = (args.k_points / n_seq) * args.dwelltime
    sweepwidth = 1.0 / ts
    freq_axis = np.linspace(-sweepwidth / 2, sweepwidth / 2, n_seq)
    ppm_axis = freq_axis / args.center_freq + 3.027
    ppm_sel = None
    if args.ppm_range is not None:
        lo, hi = sorted(args.ppm_range)
        ppm_sel = (ppm_axis >= lo) & (ppm_axis <= hi)
        print(f"[scalar-vs-ab] ppm range {lo:g}-{hi:g}: {int(ppm_sel.sum())} bins")

    wref = np.load(data_dir + "wref_o.npy", mmap_mode="r")
    wref_norm, brain_mask, _ = make_brain_mask(wref, args.brain_threshold, args.brain_erosion,
                                                cleanup=args.brain_mask_cleanup)
    print(f"[scalar-vs-ab] brain voxels={int(brain_mask.sum())}")

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
    ww_diag = np.real(WW.diagonal()).reshape(Ny, Nx)
    print(
        "[scalar-vs-ab] WW_diag brain median/iqr/min/max = "
        f"{np.nanmedian(ww_diag[brain_mask]):.6g} / "
        f"{np.nanpercentile(ww_diag[brain_mask], [25, 75])} / "
        f"{np.nanmin(ww_diag[brain_mask]):.6g} / "
        f"{np.nanmax(ww_diag[brain_mask]):.6g}"
    )

    summary_rows = []
    ratio_norm_maps = {}
    ratio_maps = {}
    for run_tag in args.run_tags:
        lam = parse_lambda_from_run_tag(run_tag)
        if lam is None:
            raise ValueError(f"Could not parse lambda from run tag {run_tag!r}")
        ratio, median_ratio, origin_slope, analyt_path, group_path = load_ratio_map(
            args, run_tag, brain_mask, ppm_sel
        )
        ratio_maps[run_tag] = ratio
        ratio_norm = ratio / median_ratio
        ratio_norm_maps[run_tag] = ratio_norm

        p1, s1, n1 = corr_pair(ratio_norm, ww_diag, brain_mask)
        p2, s2, _ = corr_pair(np.log(ratio_norm), np.log(ww_diag + 1e-30), brain_mask)
        iqr_norm = float(np.nanpercentile(ratio_norm[brain_mask], 75)
                         - np.nanpercentile(ratio_norm[brain_mask], 25))
        med_abs = float(np.nanmedian(np.abs(ratio_norm[brain_mask] - 1.0)))
        print(f"\n[scalar-vs-ab] RUN {run_tag} lambda={lam:g}")
        print(f"  analytical: {analyt_path}")
        print(f"  group     : {group_path}")
        print(f"  median_ratio={median_ratio:.6g}  origin_slope={origin_slope:.6g}")
        print(f"  residual spread: IQR(ratio/median)={iqr_norm:.6g}  median|ratio/median-1|={med_abs:.6g}")
        print(f"  corr ratio_norm vs WW_diag: Pearson={p1:+.4f} Spearman={s1:+.4f} n={n1}")
        print(f"  corr log(ratio_norm) vs log(WW_diag): Pearson={p2:+.4f} Spearman={s2:+.4f}")

        summary_rows.append({
            "run_tag": run_tag,
            "lambda": lam,
            "median_ratio": median_ratio,
            "origin_slope": origin_slope,
            "iqr_ratio_norm": iqr_norm,
            "median_abs_ratio_norm_minus_1": med_abs,
            "pearson_ratio_norm_vs_wwdiag": p1,
            "spearman_ratio_norm_vs_wwdiag": s1,
            "pearson_log_ratio_norm_vs_log_wwdiag": p2,
            "spearman_log_ratio_norm_vs_log_wwdiag": s2,
            "n_vox": n1,
        })

    probe_rows = []
    if not args.skip_rayleigh:
        b0map_dir = out_dir / "b0map"
        B0_map = np.load(b0map_dir / "B0_map.npy")
        time_axis = np.linspace(ts, ts * n_seq, n_seq)
        B0_mat = Calc_B0_matrix_mx(np.nan_to_num(B0_map, nan=0.0), time_axis).reshape(n_vox, n_seq)
        q_data_local = build_rayleigh_operator(args, data_dir, out_dir, n_vox, n_seq, im_size, B0_mat)

        probes = []
        for vox in args.rayleigh_voxels:
            if 0 <= vox < n_vox and brain_mask.ravel()[vox]:
                for r in args.rayleigh_ranks:
                    if 0 <= r < args.rank:
                        probes.append((vox, r))
        print(f"\n[scalar-vs-ab] true Rayleigh probes={len(probes)}")

        q_data_cache = {}
        for run_tag in args.run_tags:
            lam = parse_lambda_from_run_tag(run_tag)
            V = np.load(out_dir / f"spice_{run_tag}" / "V_subspace.npy")[:, : args.rank].astype(D_TYPE)
            xs = []
            ys = []
            for vox, r in probes:
                key = (run_tag, vox, r)
                q_data = q_data_cache.get(key)
                if q_data is None:
                    q_data = q_data_local(vox, r, V)
                    q_data_cache[key] = q_data
                q_reg = float(lam * np.real(WW[vox, vox]))
                balance = q_reg / (q_data + 1e-30)
                ratio_norm = float(ratio_norm_maps[run_tag].ravel()[vox])
                if np.isfinite(balance) and np.isfinite(ratio_norm):
                    xs.append(balance)
                    ys.append(ratio_norm)
                probe_rows.append({
                    "run_tag": run_tag,
                    "lambda": lam,
                    "voxel": vox,
                    "rank": r,
                    "ratio_norm": ratio_norm,
                    "q_data": q_data,
                    "q_reg": q_reg,
                    "reg_over_data": balance,
                    "ww_diag": float(np.real(WW[vox, vox])),
                })
                print(
                    f"  {run_tag} voxel={vox} rank={r} "
                    f"ratio_norm={ratio_norm:.4g} q_reg/q_data={balance:.4g} "
                    f"(q_data={q_data:.4g}, q_reg={q_reg:.4g})",
                    flush=True,
                )
            if len(xs) >= 3:
                p, s, n = corr_pair(np.asarray(xs), np.asarray(ys))
                print(
                    f"[scalar-vs-ab] Rayleigh corr {run_tag}: "
                    f"ratio_norm vs q_reg/q_data Pearson={p:+.4f} Spearman={s:+.4f} n={n}"
                )

    if args.out_prefix is None:
        safe_tag = "_".join(args.run_tags).replace(".", "p").replace("-", "m")
        out_prefix = out_dir / f"scalar_vs_ab_{safe_tag}"
    else:
        out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_csv = Path(str(out_prefix) + "_summary.csv")
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[scalar-vs-ab] Saved {summary_csv}")

    if probe_rows:
        probe_csv = Path(str(out_prefix) + "_rayleigh_probes.csv")
        with probe_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
            writer.writeheader()
            writer.writerows(probe_rows)
        print(f"[scalar-vs-ab] Saved {probe_csv}")

    print("\n[scalar-vs-ab] Interpretation:")
    print("  If corr(ratio_norm, WW_diag or q_reg/q_data) is near zero,")
    print("  the mismatch is closer to a per-lambda global scalar c(lambda).")
    print("  If the correlation is strong, then after removing c(lambda),")
    print("  residual error still tracks regularization strength, which supports")
    print("  a data-vs-regularization balance explanation (a,b-style mismatch).")


if __name__ == "__main__":
    main()
