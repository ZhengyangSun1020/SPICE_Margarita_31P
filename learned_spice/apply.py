#!/usr/bin/env python3
"""
Apply a trained learned-SPICE model to real 31P CRT data.

Inputs are the arrays the SPICE / CA-SPICE pipeline already produces:
  --system   .npz with the Gram matrix G = F^H D F (key "G"; optional "prior")
  --adj-fid  adjoint voxel FIDs F^H D y, (N*N, n_spec) or dynamic (NR, N*N, n_spec)
  --V        spectral subspace (n_spec, rank), e.g. VD.npy / VD12.npy

Noise: the network needs the noise std of one adjoint-FID entry,
sigma_k * sqrt(sum(dcf^2)) for white k-space noise of std sigma_k.
Give it directly (--noise-std-b), or via --sigma-k and --dcf, or let the script
estimate it from the last part of the FIDs (rough; prints a warning).

Dynamic data: one graph for the whole series (as in CA-SPICE). The structure
features come from the rep-averaged data, the noise feature from a single rep.

Outputs (in --out-dir):
  fid_learned.npy        reconstructed voxel FIDs, same shape as --adj-fid
  learned_system.npz     G, WW, lout_map, lam: (G + lam * WW + diag(lout_map)) U = adj_fid @ V
                         reproduces fid_learned.npy, for the existing SPICE / uncertainty scripts
  learned_weights.png    edge weights and tau maps

Usage:
    python -m learned_spice.apply --model output/learned_spice/run01/model.pt \\
        --system output_study_mid283/system.npz --use-prior \\
        --adj-fid output_study_mid283/adj_fid.npy --V output_study_mid283/VD12.npy \\
        --leg output_study_mid283/reg_TRUE.npy --sigma-k 1.2e-3 --dcf dcf.npy \\
        --out-dir output/learned_spice/mid283
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import EdgeWeightNet, export_system, make_features, spice_solve


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = EdgeWeightNet()
    net.load_state_dict(ckpt["state_dict"])
    return net.eval()


def predict_weights(net, G, B, noise_std_b, prior=None, leg=None):
    """Edge weights (4, n, n) and tau (n*n,) for one data set."""
    feats = torch.as_tensor(make_features(G, B, noise_std_b, prior, leg))[None]
    with torch.no_grad():
        log_w, log_tau = net(feats)
    return log_w[0].double().exp().numpy(), log_tau[0, 0].double().exp().numpy().ravel()


def tail_noise_std(adj_fid, frac=0.2):
    """Rough noise std per adjoint-FID entry from the last `frac` of the FID (signal assumed decayed)."""
    tail = adj_fid[..., -max(4, int(frac * adj_fid.shape[-1])):]
    return float(np.sqrt(np.mean(np.abs(tail - tail.mean(-1, keepdims=True))**2)))


def main():
    p = argparse.ArgumentParser(description="Apply learned SPICE weights to real 31P CRT data")
    p.add_argument("--model", required=True)
    p.add_argument("--system", required=True, help=".npz with G (and optionally prior)")
    p.add_argument("--adj-fid", required=True)
    p.add_argument("--V", required=True)
    p.add_argument("--use-prior", action="store_true", help="use system['prior'] as guidance image")
    p.add_argument("--leg", default=None, help="(N, N) label map; voxels == 1 are leg (reg_TRUE.npy)")
    p.add_argument("--noise-std-b", type=float, default=None)
    p.add_argument("--sigma-k", type=float, default=None, help="k-space noise std")
    p.add_argument("--dcf", default=None, help=".npy DCF used for G and the adjoint")
    p.add_argument("--out-dir", default="output/learned_spice/apply")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    system = np.load(args.system)
    G = system["G"].astype(np.complex128)
    adj = np.load(args.adj_fid).astype(np.complex128)
    V = np.load(args.V).astype(np.complex128)
    n = int(round(np.sqrt(G.shape[0])))
    prior = system["prior"] if args.use_prior else None
    leg = (np.load(args.leg) == 1) if args.leg else None

    if args.noise_std_b is not None:
        noise_std_b = args.noise_std_b
    elif args.sigma_k is not None and args.dcf is not None:
        noise_std_b = args.sigma_k * np.sqrt((np.load(args.dcf)**2).sum())
    else:
        noise_std_b = tail_noise_std(adj)
        print(f"[learned-spice] WARNING: noise std estimated from the FID tail ({noise_std_b:.3g}); "
              "pass --noise-std-b or --sigma-k/--dcf for an exact value.")

    dynamic = adj.ndim == 3
    reps = adj if dynamic else adj[None]
    B_reps = reps @ V                                         # (NR, N*N, rank)
    # the signal level comes from the rep average, the noise feature from the single-rep noise_std_b
    w, tau = predict_weights(load_model(args.model), G, B_reps.mean(0), noise_std_b, prior, leg)

    k = len(B_reps)
    U, _ = spice_solve(torch.as_tensor(G)[None].expand(k, -1, -1), torch.as_tensor(B_reps),
                       torch.as_tensor(w)[None].expand(k, -1, -1, -1), torch.as_tensor(tau)[None].expand(k, -1))
    fid = U.numpy() @ V.conj().T
    np.save(os.path.join(args.out_dir, "fid_learned.npy"), (fid if dynamic else fid[0]).astype(np.complex64))
    export_system(os.path.join(args.out_dir, "learned_system.npz"), G, w, tau, noise_std_b=noise_std_b)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for d, (ax, name) in enumerate(zip(axes, ("E", "S", "SE", "SW"))):
        im = ax.imshow(np.log10(w[d] + 1e-12), origin="lower", cmap="viridis")
        ax.set_title(f"log10 edge weight, {name} neighbour", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
    im = axes[4].imshow(np.log10(tau.reshape(n, n) + 1e-12), origin="lower", cmap="viridis")
    axes[4].set_title("log10 tau (lambda_out)", fontsize=8)
    plt.colorbar(im, ax=axes[4], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "learned_weights.png"), dpi=120)
    print(f"[learned-spice] {k} rep(s), {n}x{n}, rank {V.shape[1]}, noise std {noise_std_b:.3g} -> {args.out_dir}")


if __name__ == "__main__":
    main()
