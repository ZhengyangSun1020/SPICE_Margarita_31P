#!/usr/bin/env python3
"""
Real-data check of learned SPICE on an oversampled CRT scan (e.g. the 80-ring
10x10 scan MID00276): reconstruct from a small ring subset and score each
reconstruction by how well it predicts the measured k-space of the held-out
rings (cross-validation; no reference image needed).

Why not an image reference: the rings stop at |k| = k_norm_max (0.44 cycles/voxel
for MID00276), so ~40 % of the N x N grid's k-space is never sampled and any
reference image on that grid needs a regulariser of its own.

For each condition, n_sets disjoint ring subsets are taken (the k-th nearest
ring to each target radius), so the replicates have independent noise:

  5 rings      targets (i + 0.5) / N, i = 0..4   (a regular N x N, N/2-ring scan)
  4 rings      the same with the 4th ring dropped (data_proc_06_undersample)
  3 rings      targets up to 0.25 cycles/voxel  (super-resolution: ~0.6 N acquisition)

Score (subspace, DCF-weighted over the held-out rings R):
  held-out NMSE = || D_R^1/2 (E_R U - Y_R V) ||^2 / || D_R^1/2 Y_R V ||^2
The noise floor sigma^2 * sum(D_R) * rank / ||.||^2 is what a perfect
reconstruction would score; 1.0 is the all-zero reconstruction.

Spectral subspace: SVD of the adjoint FIDs of the held-out rings (the 05c
"adjoint" subspace), rank --rank. Loading follows scripts/05c_spice_recon_31p.py
(06c twix loader, CONCEPT phase correction, FOV shift, CRT DCF x Hamming).

Usage:
    python -m learned_spice.validate_ring_subsets \\
        --dat data/raw/meas_MID00276_FID221481_240FOV_10x10_80rings.dat \\
        --model output/learned_spice/run02_mid276/model.pt --rank 1 \\
        --out-dir output/learned_spice/mid276_validation
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .crt import crt_dcf, encoding_matrix, gram
from .model import EdgeWeightNet, handcrafted_weights, make_features, spice_solve
from .phantom import PCR_PPM, peak_map

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
HC_LAMBDAS = np.logspace(-8, -2, 13)


def _load_05c():
    import sys
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("_05c", SCRIPTS / "05c_spice_recon_31p.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_crt(dat_path):
    """k-t data (n_rings * pts, n_spec), trajectory in cycles/voxel, ring index per sample, info (as 05c)."""
    m = _load_05c()
    info = m.load_concept_twix_3d(dat_path)
    kdata = m.concept_phase_correction(info["kdata"], info["_ti_per_ring"])
    n_spec, pts = info["n_spec"], int(info["traj_pts"].max())
    traj = m.build_trajectory_2d(info["first_circle"], info["n_rings"], pts).astype(np.float64)
    traj *= float(info["k_norm_max_phys"]) / 0.5
    y = kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec).astype(np.complex128)
    d_pe, d_ro = m.compute_fov_shift(info)
    y *= np.exp(-2j * np.pi * (traj[:, 0] * d_pe + traj[:, 1] * d_ro))[:, None]
    ring = np.repeat(np.arange(info["n_rings"]), pts)
    return y, traj, ring, info


def ring_subsets(radii, targets, n_sets):
    """n_sets disjoint subsets: subset k takes the k-th nearest ring to each target radius."""
    order = [np.argsort(np.abs(radii - t)) for t in targets]
    subsets = [np.array([o[k] for o in order]) for k in range(n_sets)]
    used = np.concatenate(subsets)
    assert len(np.unique(used)) == len(used), "ring subsets overlap; lower --n-sets"
    return subsets


def operator(traj, sel, n):
    """E, dcf, G for the k-samples in sel."""
    E = encoding_matrix(traj[sel], n)
    dcf = crt_dcf(traj[sel])
    return E, dcf, gram(E, dcf)


def reconstruct(net, G, B, noise_std_b, hc_lambda):
    """U (n^2, rank) for every method; the hand-crafted lambda sweep is returned for the oracle choice."""
    n2 = G.shape[0]
    n = int(round(np.sqrt(n2)))
    Gt, Bt = torch.as_tensor(G)[None], torch.as_tensor(B)[None]
    tau0 = torch.full((1, n2), 1e-4, dtype=torch.float64)
    out = {"SPICE, lambda=0": spice_solve(Gt, Bt, torch.zeros(1, 4, n, n, dtype=torch.float64), tau0)[0][0].numpy()}

    ref = np.sqrt((np.abs(B)**2).sum(1)).reshape(n, n)
    w_hc = torch.as_tensor(handcrafted_weights(ref / ref.max()))[None]
    sweep = [spice_solve(Gt, Bt, lam * w_hc, tau0)[0][0].numpy() for lam in HC_LAMBDAS]
    out["SPICE, hand-crafted graph"] = sweep[int(np.argmin(np.abs(np.log(HC_LAMBDAS / hc_lambda))))]

    with torch.no_grad():
        log_w, log_tau = net(torch.as_tensor(make_features(G, B, noise_std_b))[None])
    out["Learned SPICE"] = spice_solve(Gt, Bt, log_w.double().exp(), log_tau.double().exp()[:, 0])[0][0].numpy()
    return out, sweep


def main():
    p = argparse.ArgumentParser(description="Ring-subset cross-validation of learned SPICE on real CRT data")
    p.add_argument("--dat", required=True)
    p.add_argument("--model", required=True, help="model.pt from learned_spice.train")
    p.add_argument("--rank", type=int, default=1, help="rank of the data-derived spectral subspace")
    p.add_argument("--hc-lambda", type=float, default=None,
                   help="hand-crafted graph lambda (default: the one tuned in simulation, from metrics.json)")
    p.add_argument("--n-sets", type=int, default=4, help="disjoint replicate subsets per condition")
    p.add_argument("--sigma-k", type=float, default=None,
                   help="k-space noise std (default: FID tail of the outer half of the rings)")
    p.add_argument("--out-dir", default="output/learned_spice/ring_subset_validation")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    net = EdgeWeightNet()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    hc_lambda = args.hc_lambda
    if hc_lambda is None:
        with open(Path(args.model).parent / "metrics.json") as f:
            hc_lambda = json.load(f)["hc_lambda"]

    y, traj, ring, info = load_crt(args.dat)
    n, n_rings = info["matrix_size"], info["n_rings"]
    dwell, larmor = info["dwell_s"] * 2, info["larmor_freq_mhz"]
    radii = np.array([np.sqrt((traj[ring == r]**2).sum(1)).mean() for r in range(n_rings)])
    sigma_k = args.sigma_k or float(np.std(y[ring >= n_rings // 2, -y.shape[1] // 4:]))
    print(f"[validate] {n}x{n}, {n_rings} rings up to {radii.max():.3f} cycles/voxel, "
          f"sigma_k = {sigma_k:.3g}, subspace rank {args.rank}")

    conditions = {
        "5 rings": (np.arange(5) + 0.5) / n,
        "4 rings (4th dropped)": (np.array([0, 1, 2, 4]) + 0.5) / n,
        "3 rings (super-res)": (np.arange(3) + 0.5) / n,
    }
    rows, examples = [], {}
    for cname, targets in conditions.items():
        for k, rs in enumerate(ring_subsets(radii, targets, args.n_sets)):
            test = np.isin(ring, rs)
            Er, dr, Gr = operator(traj, ~test, n)
            _, _, Vh = np.linalg.svd(Er.conj().T @ (dr[:, None] * y[~test]), full_matrices=False)
            V = Vh[:args.rank].conj().T

            E, dcf, G = operator(traj, test, n)
            B = E.conj().T @ (dcf[:, None] * (y[test] @ V))
            Yr = y[~test] @ V
            total = float((dr[:, None] * np.abs(Yr)**2).sum())
            floor = sigma_k**2 * dr.sum() * args.rank / total

            out, sweep = reconstruct(net, G, B, sigma_k * np.sqrt((dcf**2).sum()), hc_lambda)

            def heldout(U):
                return float((dr[:, None] * np.abs(Er @ U - Yr)**2).sum() / total)

            out["SPICE, hand-crafted (oracle lambda)"] = sweep[int(np.argmin([heldout(U) for U in sweep]))]
            for name, U in out.items():
                rows.append({"condition": cname, "replicate": k, "rings": [int(r) for r in rs], "method": name,
                             "heldout_nmse": heldout(U), "noise_floor": floor})
            if k == 0:
                rel_r = np.real(np.diag(Gr)).mean()
                U_vis = np.linalg.solve(Gr + 1e-2 * rel_r * np.eye(n * n), Er.conj().T @ (dr[:, None] * Yr))
                ex = {"Held-out rings (ridge LS, visual only)": U_vis}
                ex.update({m: out[m] for m in ("SPICE, lambda=0", "SPICE, hand-crafted graph",
                                               "SPICE, hand-crafted (oracle lambda)", "Learned SPICE")})
                examples[cname] = {m: peak_map(U @ V.conj().T, dwell, larmor, PCR_PPM).reshape(n, n)
                                   for m, U in ex.items()}
                examples[cname]["_scores"] = {m: heldout(U) for m, U in out.items()}

    summary = {}
    print(f"\n{'condition':24s} {'method':38s} {'held-out NMSE':>16s}   noise floor")
    for cname in conditions:
        for name in dict.fromkeys(r["method"] for r in rows):
            sel = [r for r in rows if r["condition"] == cname and r["method"] == name]
            v = np.array([r["heldout_nmse"] for r in sel])
            fl = float(np.mean([r["noise_floor"] for r in sel]))
            summary.setdefault(cname, {})[name] = {"mean": float(v.mean()), "std": float(v.std()),
                                                   "noise_floor": fl}
            print(f"{cname:24s} {name:38s} {v.mean():8.3f} ± {v.std():.3f}   {fl:.3f}")

    with open(os.path.join(args.out_dir, f"metrics_rank{args.rank}.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows, "sigma_k": sigma_k, "hc_lambda": hc_lambda,
                   "rank": args.rank, "dat": str(args.dat), "model": str(args.model)}, f, indent=2)

    names = [m for m in next(iter(examples.values())) if not m.startswith("_")]
    fig, axes = plt.subplots(len(examples), len(names), figsize=(2.7 * len(names), 2.9 * len(examples)))
    for row, (cname, ex) in zip(np.atleast_2d(axes), examples.items()):
        vmax = ex[names[0]].max()
        for ax, name in zip(row, names):
            ax.imshow(ex[name], vmin=0, vmax=vmax, cmap="magma", origin="lower")
            score = ex["_scores"].get(name)
            ax.set_title(name + ("" if score is None else f"\nheld-out NMSE {score:.3f}"), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(cname, fontsize=8)
    fig.suptitle(f"PCr maps, {Path(args.dat).name}, rank {args.rank} (replicate 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"pcr_maps_rank{args.rank}.png"), dpi=130)
    print(f"[validate] -> {args.out_dir}")


if __name__ == "__main__":
    main()
