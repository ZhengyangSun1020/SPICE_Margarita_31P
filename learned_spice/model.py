"""
Learned SPICE weights: a small CNN predicts the edge weights of the SPICE
smoothing graph and a per-voxel Tikhonov map (lambda_out); the reconstruction
stays the convex SPICE normal equation of scripts/05c_spice_recon_31p.py and
the CA-SPICE study (scripts/ca_spice_0724):

    (G + rel * (L(w) + diag(tau))) U = B,        x = U V^H

with G = E^H D E, B = E^H D y V (= adj_fid @ V), rel = mean(real(diag(G))).
Because the problem stays quadratic, the existing Laplacian-covariance
uncertainty pipeline still applies (see voxel_noise_std).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_cdt

# neighbour offsets (row, col): E, S, SE, SW — each 8-connected edge exactly once
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))

FEATURES = ("adjoint", "prior", "has_prior", "leg", "has_leg", "noise", "psf")


# ── graph ───────────────────────────────────────────────────────────────

def edge_index(n: int):
    """(i, j, valid) for each direction: flat voxel indices of both ends, and the (n, n) validity mask."""
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    out = []
    for dr, dc in DIRS:
        r2, c2 = r + dr, c + dc
        valid = (r2 >= 0) & (r2 < n) & (c2 >= 0) & (c2 < n)
        out.append((r * n + c, np.clip(r2, 0, n - 1) * n + np.clip(c2, 0, n - 1), valid))
    return out


def laplacian(w: torch.Tensor) -> torch.Tensor:
    """Graph Laplacian (B, n^2, n^2) from edge weights w (B, 4, n, n); invalid edges are ignored.

    Same matrix as calc_Bmatrix_simple(...)[0] in 05c for weights W on the 8-connected edges.
    """
    b, _, n, _ = w.shape
    L = torch.zeros(b, n * n * n * n, dtype=w.dtype, device=w.device)
    for d, (i, j, valid) in enumerate(edge_index(n)):
        i, j = torch.as_tensor(i[valid]), torch.as_tensor(j[valid])
        wd = w[:, d][:, torch.as_tensor(valid)]
        for a, bb, s in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)):
            L.index_add_(1, a * n * n + bb, s * wd)
    return L.view(b, n * n, n * n)


def handcrafted_weights(ref: np.ndarray, wmax: float = 5e4) -> np.ndarray:
    """calc_Bmatrix_simple edge weights min(1 / (ref_i - ref_j)^2, wmax), as (4, n, n)."""
    n = ref.shape[0]
    flat = ref.ravel()
    w = np.zeros((4, n, n))
    for d, (i, j, valid) in enumerate(edge_index(n)):
        with np.errstate(divide="ignore"):
            wd = np.minimum(1.0 / (flat[i] - flat[j])**2, wmax)
        w[d] = np.where(valid, wd, 0.0)
    return w


def lambda_out_rule(leg: np.ndarray) -> np.ndarray:
    """CA-SPICE lambda_out map (system11.py): min(1, 0.1 d^2), d = chessboard distance to the leg."""
    d = distance_transform_cdt(~leg, metric="chessboard").astype(float)
    return np.minimum(1.0, 0.1 * d**2)


# ── features ────────────────────────────────────────────────────────────

def make_features(G: np.ndarray, B: np.ndarray, noise_std_b: float,
                  prior: np.ndarray | None = None, leg: np.ndarray | None = None) -> np.ndarray:
    """Network input (len(FEATURES), n, n), invariant to the data scale and to the subspace rank.

    G : (n^2, n^2) Gram matrix, B : (n^2, rank) = E^H D y V,
    noise_std_b : noise std of one entry of B (sigma_k * sqrt(diag(E^H D^2 E)), averaged),
    prior : optional guidance image on the CSI grid (e.g. the 80-ring PCr map),
    leg : optional boolean leg mask on the CSI grid.
    """
    n = int(round(np.sqrt(G.shape[0])))
    energy = np.sqrt((np.abs(B)**2).sum(1)).reshape(n, n)
    level = np.percentile(energy, 90) + 1e-12
    feats = np.zeros((len(FEATURES), n, n), dtype=np.float32)
    feats[0] = energy / level
    if prior is not None:
        p = np.abs(prior)
        feats[1] = p / (np.percentile(p, 99) + 1e-12)
        feats[2] = 1.0
    if leg is not None:
        feats[3] = leg
        feats[4] = 1.0
    # noise relative to the signal, per rank component
    feats[5] = np.log10(noise_std_b * np.sqrt(B.shape[1]) / level + 1e-12)
    # PSF concentration of the centre voxel: 1 for a delta, smaller for undersampled / truncated k-space
    c = (n // 2) * n + n // 2
    feats[6] = np.abs(G[c, c])**2 / (np.abs(G[c])**2).sum()
    return feats


# ── network and solver ──────────────────────────────────────────────────

class EdgeWeightNet(nn.Module):
    """Fully convolutional: features (B, F, n, n) -> log edge weights (B, 4, n, n), log tau (B, 1, n, n)."""

    def __init__(self, width: int = 48, depth: int = 5):
        super().__init__()
        layers, c_in = [], len(FEATURES)
        for _ in range(depth):
            layers += [nn.Conv2d(c_in, width, 3, padding=1, padding_mode="replicate"), nn.GELU()]
            c_in = width
        self.body = nn.Sequential(*layers)
        self.edge = nn.Conv2d(width, 4, 3, padding=1, padding_mode="replicate")
        self.tau = nn.Conv2d(width, 1, 3, padding=1, padding_mode="replicate")
        # start near a mildly regularised SPICE: w ~ 0.05, tau ~ 1e-3 (relative to diag(G))
        nn.init.zeros_(self.edge.weight); nn.init.constant_(self.edge.bias, np.log(0.05))
        nn.init.zeros_(self.tau.weight); nn.init.constant_(self.tau.bias, np.log(1e-3))

    def forward(self, feats: torch.Tensor):
        h = self.body(feats)
        return self.edge(h).clamp(-14, 8), self.tau(h).clamp(-14, 4)


def spice_system(G: torch.Tensor, w: torch.Tensor, tau: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """H = G + rel * (L(w) + diag(tau) + eps I) for batched G (B, n^2, n^2) complex."""
    rel = torch.diagonal(G, dim1=-2, dim2=-1).real.mean(-1)[:, None, None]
    reg = laplacian(w) + torch.diag_embed(tau.flatten(1)) + eps * torch.eye(G.shape[-1], dtype=w.dtype)
    return G + rel * reg.to(G.dtype)


def spice_solve(G, B, w, tau, eps=1e-6):
    """U = H^-1 B (differentiable); returns (U, H)."""
    H = spice_system(G, w, tau, eps)
    return torch.linalg.solve(H, B), H


def learned_solve(net: EdgeWeightNet, feats, G, B):
    """Predict the weights for a batch and solve SPICE; returns (U, w, tau)."""
    log_w, log_tau = net(feats)
    w, tau = log_w.double().exp(), log_tau.double().exp()[:, 0]
    U, _ = spice_solve(G, B, w, tau)
    return U, w, tau


# ── uncertainty and export ──────────────────────────────────────────────

def voxel_noise_std(H: np.ndarray, E: np.ndarray, dcf: np.ndarray, sigma_k: float) -> np.ndarray:
    """Per-voxel noise std of each U column for white k-space noise of std sigma_k.

    Cov(U[:, r]) = sigma^2 H^-1 E^H D^2 E H^-1 (sandwich form, since the data term is DCF-weighted);
    the same for every rank component because V has orthonormal columns.
    """
    Hi = np.linalg.inv(H)
    M = Hi @ (E.conj().T * dcf)
    return sigma_k * np.sqrt((np.abs(M)**2).sum(1))


def export_system(path: str, G: np.ndarray, w: np.ndarray, tau: np.ndarray, eps: float = 1e-6, **extra):
    """Save the learned constraints in the CA-SPICE system-file layout.

    The SPICE solve  (G + lam * WW + diag(lout_map)) U = B  then reproduces the learned
    reconstruction exactly (lam = rel, lout_map = rel * (tau + eps)), so the existing SPICE and
    uncertainty scripts can consume WW / lout_map / lam.
    """
    n = w.shape[-1]
    rel = float(np.real(np.diag(G)).mean())
    WW = laplacian(torch.as_tensor(w[None], dtype=torch.float64))[0].numpy()
    np.savez(path, G=G, WW=WW, lout_map=(rel * (tau + eps)).reshape(n, n), lam=rel,
             edge_weights=w, tau=tau.reshape(n, n), **extra)
    return WW
