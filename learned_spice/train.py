#!/usr/bin/env python3
"""
Train the learned-SPICE weight network on simulated 31P CRT data and compare
it with SPICE baselines on a held-out simulated test set.

Methods compared (all solve the same SPICE normal equation, only the
regulariser differs):
  - SPICE, lambda = 0            : subspace only (tiny ridge for stability)
  - SPICE, hand-crafted graph    : calc_Bmatrix_simple weights from the guidance
                                   image (prior, else the adjoint energy map) and
                                   the CA-SPICE lambda_out rule when the leg mask
                                   is known; lambda tuned on the validation set
  - SPICE, hand-crafted (oracle) : same, with the best lambda per test sample
  - Learned SPICE                : network-predicted edge weights and tau

Usage:
    python -m learned_spice.train --out-dir output/learned_spice/run01
    python -m learned_spice.train --steps 200 --out-dir output/learned_spice/smoke   # quick check
"""

import argparse
import json
import os
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import SimConfig, relative_error, simulate_sample, to_batch
from .model import EdgeWeightNet, handcrafted_weights, lambda_out_rule, learned_solve, spice_solve
from .phantom import PCR_PPM, PI_PPM, peak_map, spectral_subspace, training_fids_pk2pi

HC_LAMBDAS = np.logspace(-8, -2, 13)   # multiplies calc_Bmatrix_simple weights (wmax = 5e4)


def make_set(cfg, V, seed, n_per_size):
    rng = np.random.default_rng(seed)
    return {n: [simulate_sample(cfg, V, rng, n) for _ in range(n_per_size)] for n in cfg.grid_sizes}


def handcrafted_terms(samples):
    """Hand-crafted edge weights and lambda_out per sample (CA-SPICE construction)."""
    ws, taus = [], []
    for s in samples:
        ref = s["prior"] if s["prior"] is not None else s["feats"][0]
        ref = np.abs(ref) / (np.abs(ref).max() + 1e-12)
        ws.append(handcrafted_weights(ref))
        taus.append(lambda_out_rule(s["leg"]).ravel() if s["leg_known"] else np.zeros(ref.size))
    return torch.as_tensor(np.stack(ws)), torch.as_tensor(np.stack(taus))


def solve_methods(net, samples, hc_lambda):
    """U for every method on one list of same-size samples; hc_lambda=None returns the per-lambda sweep."""
    b = to_batch(samples)
    n2 = b["G"].shape[-1]
    k = len(samples)
    out = {}
    zero_w = torch.zeros(k, 4, int(np.sqrt(n2)), int(np.sqrt(n2)), dtype=torch.float64)
    out["SPICE, lambda=0"], _ = spice_solve(b["G"], b["B"], zero_w, torch.full((k, n2), 1e-4, dtype=torch.float64))

    w_hc, tau_hc = handcrafted_terms(samples)
    sweep = []
    for lam in HC_LAMBDAS:
        U, _ = spice_solve(b["G"], b["B"], lam * w_hc, tau_hc + 1e-4)
        sweep.append(U)
    errs = torch.stack([relative_error(U, b["U_true"]) for U in sweep])          # (n_lam, k)
    if hc_lambda is not None:
        out["SPICE, hand-crafted graph"] = sweep[int(np.argmin(np.abs(np.log(HC_LAMBDAS / hc_lambda))))]
        best = errs.argmin(0)
        out["SPICE, hand-crafted (oracle lambda)"] = torch.stack([sweep[best[i]][i] for i in range(k)])

    with torch.no_grad():
        out["Learned SPICE"], _, _ = learned_solve(net, b["feats"], b["G"], b["B"])
    return out, errs, b["U_true"]


def map_nrmse(est, true, mask):
    return float(np.linalg.norm((est - true)[mask]) / (np.linalg.norm(true[mask]) + 1e-12))


def evaluate(net, data, cfg, V, hc_lambda=None):
    """Mean metrics per method; with hc_lambda=None also returns the best global hand-crafted lambda."""
    if hc_lambda is None:
        errs = torch.cat([solve_methods(net, s, None)[1] for s in data.values()], dim=1)
        hc_lambda = float(HC_LAMBDAS[int(errs.mean(1).argmin())])

    rows = []
    for n, samples in data.items():
        out, _, U_true = solve_methods(net, samples, hc_lambda)
        for name, U in out.items():
            err = relative_error(U, U_true).numpy()
            for i, s in enumerate(samples):
                x = U[i].numpy() @ V.conj().T
                info = s["info"]
                cond = ("super-res" if info["n_acq"] < n else
                        "ring-dropped" if info["rings_kept"] < info["n_rings"] else "full")
                rows.append({
                    "method": name, "n": n, "condition": cond, "prior": s["prior"] is not None,
                    "rel_err": float(err[i]),
                    "pcr_nrmse": map_nrmse(peak_map(x, cfg.dwell_s, cfg.larmor_mhz, PCR_PPM).reshape(n, n),
                                           s["pcr_true"], s["leg"]),
                    "pi_nrmse": map_nrmse(peak_map(x, cfg.dwell_s, cfg.larmor_mhz, PI_PPM).reshape(n, n),
                                          s["pi_true"], s["leg"]),
                })
    return rows, hc_lambda


def summarise(rows):
    """{method: {group: {metric: mean}}} for all rows and per condition / prior availability."""
    out = {}
    for m in dict.fromkeys(r["method"] for r in rows):
        mr = [r for r in rows if r["method"] == m]
        groups = {"all": mr}
        for c in ("full", "ring-dropped", "super-res"):
            groups[c] = [r for r in mr if r["condition"] == c]
        groups["with prior"] = [r for r in mr if r["prior"]]
        groups["no prior"] = [r for r in mr if not r["prior"]]
        out[m] = {g: {k: float(np.mean([r[k] for r in v])) for k in ("rel_err", "pcr_nrmse", "pi_nrmse")}
                  | {"count": len(v)} for g, v in groups.items() if v}
    return out


def plot_examples(net, data, cfg, V, hc_lambda, path):
    samples = [data[n][i] for n in data for i in range(2)][:4]
    fig, axes = plt.subplots(len(samples), 5, figsize=(13, 2.7 * len(samples)))
    for row, s in zip(axes, samples):
        out, _, _ = solve_methods(net, [s], hc_lambda)
        n = s["pcr_true"].shape[0]
        maps = [("Truth", s["pcr_true"])]
        for name in ("SPICE, lambda=0", "SPICE, hand-crafted graph", "Learned SPICE"):
            x = out[name][0].numpy() @ V.conj().T
            maps.append((name, peak_map(x, cfg.dwell_s, cfg.larmor_mhz, PCR_PPM).reshape(n, n)))
        vmax = s["pcr_true"].max()
        for ax, (title, img) in zip(row, maps):
            ax.imshow(img, vmin=0, vmax=vmax, cmap="magma", origin="lower")
            ax.set_title(title if title == "Truth" else f"{title}\n{map_nrmse(img, s['pcr_true'], s['leg']):.2f}",
                         fontsize=8)
            ax.axis("off")
        prior = s["prior"] if s["prior"] is not None else np.zeros_like(s["pcr_true"])
        row[4].imshow(prior, cmap="gray", origin="lower")
        info = s["info"]
        row[4].set_title(f"prior {'given' if s['prior'] is not None else 'absent'}\n"
                         f"{n}x{n}, acq {info['n_acq']}, rings {info['rings_kept']}/{info['n_rings']}", fontsize=8)
        row[4].axis("off")
    fig.suptitle("PCr peak-height maps (numbers: NRMSE in the leg)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Train learned SPICE weights on simulated 31P CRT data")
    p.add_argument("--out-dir", default="output/learned_spice/run01")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=12)
    p.add_argument("--n-val", type=int, default=24, help="validation samples per grid size")
    p.add_argument("--n-test", type=int, default=64, help="test samples per grid size")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = SimConfig(rank=args.rank)

    V = spectral_subspace(training_fids_pk2pi(cfg.n_spec, cfg.dwell_s, cfg.larmor_mhz), cfg.rank)
    val = make_set(cfg, V, 1000 + args.seed, args.n_val)
    net = EdgeWeightNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = np.random.default_rng(args.seed)

    best, log = np.inf, []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        n = int(rng.choice(cfg.grid_sizes))
        b = to_batch([simulate_sample(cfg, V, rng, n) for _ in range(args.batch)])
        U, _, _ = learned_solve(net, b["feats"], b["G"], b["B"])
        loss = relative_error(U, b["U_true"]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                v = float(np.mean([relative_error(learned_solve(net, bb["feats"], bb["G"], bb["B"])[0],
                                                  bb["U_true"]).mean() for bb in map(to_batch, val.values())]))
            log.append({"step": step, "train_loss": loss.item(), "val_rel_err": v, "time_s": time.time() - t0})
            print(f"[learned-spice] step {step:5d}  train {loss.item():.4f}  val {v:.4f}  ({time.time() - t0:.0f} s)")
            if v < best:
                best = v
                torch.save({"state_dict": net.state_dict(), "cfg": asdict(cfg), "V": V, "step": step},
                           os.path.join(args.out_dir, "model.pt"))

    net.load_state_dict(torch.load(os.path.join(args.out_dir, "model.pt"), weights_only=False)["state_dict"])
    net.eval()
    _, hc_lambda = evaluate(net, val, cfg, V)
    test = make_set(cfg, V, 2000 + args.seed, args.n_test)
    rows, _ = evaluate(net, test, cfg, V, hc_lambda)
    summary = summarise(rows)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"summary": summary, "hc_lambda": hc_lambda, "train_log": log, "args": vars(args),
                   "cfg": asdict(cfg)}, f, indent=2)
    plot_examples(net, test, cfg, V, hc_lambda, os.path.join(args.out_dir, "pcr_maps.png"))

    print(f"\n[learned-spice] test set ({sum(map(len, test.values()))} samples), hand-crafted lambda = {hc_lambda:.1e}")
    print(f"{'method':38s} {'group':13s} {'rel_err':>8s} {'PCr':>7s} {'Pi':>7s}  n")
    for m, groups in summary.items():
        for g, v in groups.items():
            print(f"{m:38s} {g:13s} {v['rel_err']:8.3f} {v['pcr_nrmse']:7.3f} {v['pi_nrmse']:7.3f}  {v['count']}")


if __name__ == "__main__":
    main()
