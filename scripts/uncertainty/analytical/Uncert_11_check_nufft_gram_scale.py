#!/usr/bin/env python3
"""
Check torchnufft vs finufft data-Gram scale.

For the same FID-domain perturbation x, this script compares

    q_fin   = || F_finufft x ||^2
    q_torch = || F_torchnufft FFT_t x ||^2

In --mode gramvec it also compares the actual forward-adjoint vectors:

    g_fin   = F_finufft^H F_finufft x
    g_torch = IFFT_t F_torchnufft^H F_torchnufft FFT_t x

or, in --mode toeplitz, the actual torchnufft Toeplitz Gram used by
Uncert_01:

    q_torch = <FFT_t x, G_toep FFT_t x>.

The ratio q_torch / q_fin tells whether the analytical Hessian data term
has a large scale mismatch against the finufft forward model.
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

from utils.recon import NUFFTOp, NUFFTLinearOperator, Calc_B0_matrix_mx
from utils.scan_params import load_scan_params


D_TYPE = np.complex64
T_D_TYPE = None


def fid_to_spec(fid):
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm="ortho"), axes=-1)


def spec_to_fid(spec):
    return np.fft.ifft(np.fft.ifftshift(spec, axes=-1), axis=-1, norm="ortho")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare torchnufft and finufft data-Gram Rayleigh quotients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subject", required=True)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--data-root", default="./data/processed")
    p.add_argument("--out-root", default="./output")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dim", type=int, nargs=2, default=[64, 64], metavar=("NY", "NX"))
    p.add_argument("--dwelltime", type=float, default=None)
    p.add_argument("--k-points", type=int, default=None)
    p.add_argument("--n-coils", type=int, default=None)
    p.add_argument("--center-freq", type=float, default=None)
    p.add_argument("--rank", type=int, default=20)
    p.add_argument("--n-random", type=int, default=1)
    p.add_argument("--local-voxels", type=int, nargs="*", default=[926, 1500, 2590, 3100])
    p.add_argument("--local-ranks", type=int, nargs="*", default=[0, 5, 10, 15])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mode", choices=["forward", "gramvec", "toeplitz", "both", "all"], default="forward")
    p.add_argument("--device", default="cpu", help="torch device for torchnufft, e.g. cpu or cuda")
    p.add_argument("--osamp", type=float, default=2.0)
    p.add_argument("--ost", type=float, default=2.0)
    p.add_argument("--include-b0", action="store_true", default=True)
    p.add_argument("--no-b0", dest="include_b0", action="store_false")
    p.add_argument("--compare-no-smap", action="store_true", default=True)
    p.add_argument("--no-compare-no-smap", dest="compare_no_smap", action="store_false")
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def elapsed(label, func):
    t0 = time.perf_counter()
    print(f"[gram-check] {label} ...", flush=True)
    out = func()
    dt = time.perf_counter() - t0
    print(f"[gram-check] {label} done in {dt:.2f} s", flush=True)
    return out, dt


def normed(x):
    x = np.asarray(x, dtype=D_TYPE).ravel()
    return x / (np.linalg.norm(x) + 1e-30)


def energy(y):
    y = np.asarray(y)
    return float(np.real(np.vdot(y.ravel(), y.ravel())))


def make_probe_random_subspace(rng, n_vox, rank, V, B0_mat):
    U = (rng.standard_normal((n_vox, rank)) + 1j * rng.standard_normal((n_vox, rank))) / np.sqrt(2.0)
    x = U.astype(D_TYPE) @ V.conj().T
    if B0_mat is not None:
        x = B0_mat * x
    return normed(x)


def make_probe_local(vox, r, n_vox, rank, V, B0_mat):
    U = np.zeros((n_vox, rank), dtype=D_TYPE)
    U[vox, r] = 1.0 + 0.0j
    x = U @ V.conj().T
    if B0_mat is not None:
        x = B0_mat * x
    return normed(x)


def main():
    args = parse_args()

    data_dir = args.data_dir or os.path.join(args.data_root, args.subject)
    out_dir = args.out_dir or os.path.join(args.out_root, args.subject)
    data_dir = data_dir.rstrip("/") + "/"

    Ny, Nx = args.dim
    n_vox = Ny * Nx

    print("[gram-check] Loading inputs", flush=True)
    load_scan_params(args, data_dir, k_key="k_mrsi")
    spice_dir = os.path.join(out_dir, f"spice_{args.run_tag}")
    lprm_dir = os.path.join(out_dir, "lipid_removal")
    coil_dir = os.path.join(out_dir, "coilmap")
    b0_dir = os.path.join(out_dir, "b0map")

    trej = np.load(os.path.join(lprm_dir, "mrsi_ksp_scaled.npy"), mmap_mode="r").T.astype(np.float32)
    coil = np.load(os.path.join(coil_dir, "ecalib_pp.npy"), mmap_mode="r").astype(D_TYPE)
    V = np.load(os.path.join(spice_dir, "V_subspace.npy"))[:, : args.rank].astype(D_TYPE)
    n_seq = V.shape[0]
    im_size = (Ny, Nx, n_seq)
    n_coils = int(coil.shape[0])
    n_samples = int(np.prod(trej.shape[:-1]))

    print(f"[gram-check] subject={args.subject} run_tag={args.run_tag}", flush=True)
    print(f"[gram-check] trej={trej.shape} n_samples={n_samples} coil={coil.shape} V={V.shape}", flush=True)
    coil_power = np.sum(np.abs(coil) ** 2, axis=0)
    print(
        "[gram-check] sum|smap|^2 median/std/min/max = "
        f"{np.median(coil_power):.6g} / {np.std(coil_power):.6g} / "
        f"{np.min(coil_power):.6g} / {np.max(coil_power):.6g}",
        flush=True,
    )

    B0_mat = None
    if args.include_b0:
        k_points = int(args.k_points)
        time_axis = np.linspace((k_points / n_seq) * args.dwelltime,
                                (k_points / n_seq) * args.dwelltime * n_seq,
                                n_seq)
        B0_map = np.load(os.path.join(b0_dir, "B0_map.npy"))
        B0_mat = Calc_B0_matrix_mx(np.nan_to_num(B0_map, nan=0.0), time_axis).reshape(n_vox, n_seq)
        print("[gram-check] B0 modulation included", flush=True)
    else:
        print("[gram-check] B0 modulation disabled", flush=True)

    import torch
    import torchkbnufft as tkbn
    import mrinufft

    global T_D_TYPE
    T_D_TYPE = torch.complex64
    device = torch.device(args.device)

    coil_smap = np.repeat(coil[np.newaxis, :, :, :, np.newaxis], n_seq, axis=-1).astype(D_TYPE)
    smap_time = coil_smap.squeeze(0)
    ktraj = torch.from_numpy(trej).permute(2, 0, 1).reshape(3, -1).to(device)
    grid_size = (int(np.ceil(args.osamp * Ny)), int(np.ceil(args.osamp * Nx)), int(np.ceil(args.ost * n_seq)))

    NufftOpCls = mrinufft.get_operator("finufft")
    nufft_raw, _ = elapsed(
        "build finufft operator",
        lambda: NufftOpCls(trej, shape=im_size, n_coils=n_coils,
                           n_batchs=1, squeeze_dims=True, smaps=smap_time),
    )
    fin_op = NUFFTLinearOperator(
        nufft_raw, img_shape=im_size, n_samples=n_samples, n_coils=n_coils, dtype=D_TYPE
    ).to_scipy()

    print(f"[gram-check] build torchnufft forward grid_size={grid_size}", flush=True)
    tnufft = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(device)
    tadjnufft = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, dtype=T_D_TYPE).to(device)
    torch_f = NUFFTOp(
        im_size=im_size, grid_size=grid_size, omega=ktraj,
        smaps=coil_smap, norm="ortho", device=device,
        nufft_ob=tnufft, adjnufft_ob=tadjnufft,
    )
    torch_f_no_smap = None
    if args.compare_no_smap:
        torch_f_no_smap = NUFFTOp(
            im_size=im_size, grid_size=grid_size, omega=ktraj,
            smaps=None, norm="ortho", device=device,
            nufft_ob=tnufft, adjnufft_ob=tadjnufft,
        )

    toep = None
    kernel = None
    smap_t = None
    if args.mode in ("toeplitz", "both", "all"):
        kernel, _ = elapsed(
            "build torchnufft Toeplitz kernel",
            lambda: tkbn.calc_toeplitz_kernel(ktraj, im_size=im_size, grid_size=grid_size, norm="ortho").to(device),
        )
        toep = tkbn.ToepNufft().to(device)
        smap_t = torch.from_numpy(coil_smap).to(device=device, dtype=T_D_TYPE)

    def q_fin(x_fid):
        y = fin_op.matvec(x_fid.astype(D_TYPE))
        return energy(y)

    def q_torch_forward(x_fid, no_smap=False):
        x_spec = fid_to_spec(x_fid.reshape(im_size))
        op = torch_f_no_smap if no_smap else torch_f
        y = op.A_np(x_spec.astype(D_TYPE)).ravel()
        return energy(y)

    def q_torch_toeplitz(x_fid, no_smap=False):
        x_spec = fid_to_spec(x_fid.reshape(im_size)).astype(D_TYPE)
        x_t = torch.from_numpy(x_spec).reshape(1, 1, *im_size).to(device=device, dtype=T_D_TYPE)
        y_t = toep(x_t, kernel, smaps=None if no_smap else smap_t, norm="ortho")
        y = y_t.detach().cpu().numpy().squeeze()
        return float(np.real(np.vdot(x_spec.ravel(), y.ravel())))

    def gram_fin_vec(x_fid):
        x_fid = x_fid.astype(D_TYPE)
        return fin_op.rmatvec(fin_op.matvec(x_fid)).astype(D_TYPE)

    def gram_torch_forward_vec(x_fid, no_smap=False):
        x_spec = fid_to_spec(x_fid.reshape(im_size)).astype(D_TYPE)
        op = torch_f_no_smap if no_smap else torch_f
        y = op.A_np(x_spec)
        g_spec = op.AH_np(y).reshape(im_size).astype(D_TYPE)
        return spec_to_fid(g_spec).ravel().astype(D_TYPE)

    def gram_torch_toeplitz_vec(x_fid, no_smap=False):
        x_spec = fid_to_spec(x_fid.reshape(im_size)).astype(D_TYPE)
        x_t = torch.from_numpy(x_spec).reshape(1, 1, *im_size).to(device=device, dtype=T_D_TYPE)
        y_t = toep(x_t, kernel, smaps=None if no_smap else smap_t, norm="ortho")
        g_spec = y_t.detach().cpu().numpy().squeeze().reshape(im_size).astype(D_TYPE)
        return spec_to_fid(g_spec).ravel().astype(D_TYPE)

    def gram_metrics(x_fid, g_fin, g_torch):
        x_fid = x_fid.astype(D_TYPE).ravel()
        g_fin = g_fin.astype(D_TYPE).ravel()
        g_torch = g_torch.astype(D_TYPE).ravel()
        q_fin = float(np.real(np.vdot(x_fid, g_fin)))
        q_torch = float(np.real(np.vdot(x_fid, g_torch)))
        n_fin = float(np.linalg.norm(g_fin))
        n_torch = float(np.linalg.norm(g_torch))
        diff = g_torch - g_fin
        n_diff = float(np.linalg.norm(diff))
        denom = (n_fin * n_torch) + 1e-30
        cosine = float(np.real(np.vdot(g_fin, g_torch)) / denom)
        return {
            "q_finufft_vec": q_fin,
            "q_torchnufft_vec": q_torch,
            "ratio_torch_over_fin": q_torch / q_fin if q_fin != 0 else np.nan,
            "norm_finufft_gram": n_fin,
            "norm_torchnufft_gram": n_torch,
            "norm_ratio_torch_over_fin": n_torch / n_fin if n_fin != 0 else np.nan,
            "relative_gramvec_error": n_diff / (n_fin + 1e-30),
            "gramvec_cosine": cosine,
            "max_abs_gramvec_diff": float(np.max(np.abs(diff))),
        }

    rng = np.random.default_rng(args.seed)
    probes = []
    for i in range(args.n_random):
        probes.append((f"random_subspace_{i}", make_probe_random_subspace(rng, n_vox, args.rank, V, B0_mat)))

    for i, vox in enumerate(args.local_voxels):
        r = args.local_ranks[i % len(args.local_ranks)]
        if vox < 0 or vox >= n_vox or r < 0 or r >= args.rank:
            print(f"[gram-check] Skipping invalid local probe voxel={vox}, rank={r}", flush=True)
            continue
        probes.append((f"local_v{vox}_r{r}", make_probe_local(vox, r, n_vox, args.rank, V, B0_mat)))

    rows = []
    for name, x in probes:
        print(f"\n[gram-check] Probe {name}", flush=True)
        qf, t_fin = elapsed("  finufft q", lambda x=x: q_fin(x))
        row_base = {"probe": name, "q_finufft": qf, "t_finufft_s": t_fin}

        if args.mode in ("forward", "both", "all"):
            qt, t_torch = elapsed("  torchnufft forward q", lambda x=x: q_torch_forward(x, no_smap=False))
            row = dict(row_base)
            row.update({
                "mode": "forward",
                "q_torchnufft": qt,
                "ratio_torch_over_fin": qt / qf if qf != 0 else np.nan,
                "t_torchnufft_s": t_torch,
            })
            if args.compare_no_smap:
                qt0, t_torch0 = elapsed("  torchnufft forward q no-smap", lambda x=x: q_torch_forward(x, no_smap=True))
                row.update({
                    "q_torchnufft_no_smap": qt0,
                    "ratio_no_smap_over_fin": qt0 / qf if qf != 0 else np.nan,
                    "t_torchnufft_no_smap_s": t_torch0,
                })
            rows.append(row)
            print(
                f"[gram-check] RESULT {name} forward: "
                f"q_fin={qf:.6e} q_torch={qt:.6e} ratio={row['ratio_torch_over_fin']:.6g}",
                flush=True,
            )

        if args.mode in ("gramvec", "all"):
            g_fin, t_fin_g = elapsed("  finufft gramvec F^H F x", lambda x=x: gram_fin_vec(x))
            g_torch, t_torch_g = elapsed(
                "  torchnufft gramvec F^H F x",
                lambda x=x: gram_torch_forward_vec(x, no_smap=False),
            )
            row = dict(row_base)
            row.update({
                "mode": "gramvec",
                "t_finufft_gramvec_s": t_fin_g,
                "t_torchnufft_gramvec_s": t_torch_g,
            })
            row.update(gram_metrics(x, g_fin, g_torch))
            if args.compare_no_smap:
                g_torch0, t_torch0_g = elapsed(
                    "  torchnufft gramvec F^H F x no-smap",
                    lambda x=x: gram_torch_forward_vec(x, no_smap=True),
                )
                m0 = gram_metrics(x, g_fin, g_torch0)
                row.update({
                    "q_torchnufft_no_smap_vec": m0["q_torchnufft_vec"],
                    "ratio_no_smap_over_fin": m0["ratio_torch_over_fin"],
                    "norm_ratio_no_smap_over_fin": m0["norm_ratio_torch_over_fin"],
                    "relative_no_smap_gramvec_error": m0["relative_gramvec_error"],
                    "gramvec_no_smap_cosine": m0["gramvec_cosine"],
                    "t_torchnufft_no_smap_gramvec_s": t_torch0_g,
                })
            rows.append(row)
            print(
                f"[gram-check] RESULT {name} gramvec: "
                f"q_ratio={row['ratio_torch_over_fin']:.6g} "
                f"norm_ratio={row['norm_ratio_torch_over_fin']:.6g} "
                f"relerr={row['relative_gramvec_error']:.6g} "
                f"cosine={row['gramvec_cosine']:.6g}",
                flush=True,
            )

        if args.mode in ("toeplitz", "both", "all"):
            qt, t_torch = elapsed("  torchnufft toeplitz q", lambda x=x: q_torch_toeplitz(x, no_smap=False))
            row = dict(row_base)
            row.update({
                "mode": "toeplitz",
                "q_torchnufft": qt,
                "ratio_torch_over_fin": qt / qf if qf != 0 else np.nan,
                "t_torchnufft_s": t_torch,
            })
            if args.compare_no_smap:
                qt0, t_torch0 = elapsed("  torchnufft toeplitz q no-smap", lambda x=x: q_torch_toeplitz(x, no_smap=True))
                row.update({
                    "q_torchnufft_no_smap": qt0,
                    "ratio_no_smap_over_fin": qt0 / qf if qf != 0 else np.nan,
                    "t_torchnufft_no_smap_s": t_torch0,
                })
            rows.append(row)
            print(
                f"[gram-check] RESULT {name} toeplitz: "
                f"q_fin={qf:.6e} q_torch={qt:.6e} ratio={row['ratio_torch_over_fin']:.6g}",
                flush=True,
            )

    ratios = np.array([r["ratio_torch_over_fin"] for r in rows], dtype=float)
    print("\n[gram-check] Summary ratio_torch_over_fin", flush=True)
    print(
        f"  n={len(ratios)} median={np.nanmedian(ratios):.6g} "
        f"mean={np.nanmean(ratios):.6g} min={np.nanmin(ratios):.6g} max={np.nanmax(ratios):.6g}",
        flush=True,
    )
    print("  If ratio ~0.25, torchnufft data curvature is 4x too small in std terms.", flush=True)
    print("  If ratio ~4.0, torchnufft data curvature is 2x too large in std terms.", flush=True)
    print("  If ratio ~1.0, backend scale is not the 2x explanation.", flush=True)

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = Path(out_dir) / f"gram_scale_{args.run_tag}_{args.mode}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[gram-check] Saved {out_csv}", flush=True)


if __name__ == "__main__":
    main()
