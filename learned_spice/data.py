"""
On-the-fly training pairs for learned SPICE.

Each sample: a random calf phantom on a fine grid (upsample^2 sub-pixels per
CSI voxel, so the forward model is not the reconstruction model), a random CRT
operator (full / ring-dropped / truncated for super-resolution), and white
k-space noise. Everything is kept in the rank-r subspace: with V orthonormal,
the projected noise y V is still white with the same sigma, so nothing is lost
for the SPICE solve, which only sees y V.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .crt import block_average, crt_dcf, crt_trajectory, encoding_matrix, gram, ring_radii
from .model import make_features
from .phantom import PCR_PPM, PI_PPM, make_calf_phantom, peak_map, phantom_fids


@dataclass
class SimConfig:
    grid_sizes: tuple = (8, 10, 12)       # CSI matrix sizes (CA-SPICE study: 8x8 and 10x10)
    upsample: int = 4                     # sub-pixels per voxel side for the ground truth
    # acquisition defaults from MID00276 (240 mm FOV, 10x10, CONCEPT CRT, 3T)
    n_spec: int = 756
    dwell_s: float = 360e-6               # spectral dwell (twix dwell x 2, as in 05c)
    larmor_mhz: float = 49.894558
    rank: int = 12
    pts_per_ring: int = 60
    p_superres: float = 0.3               # acquire on a coarser grid than the CSI grid
    p_drop: float = 0.4                   # randomly discard rings (data_proc_06_undersample)
    drop_rate: tuple = (0.1, 0.4)
    n_protect: int = 2                    # innermost rings never discarded
    log_noise: tuple = (-2.3, -0.8)       # log10 of k-space sigma / rms(signal)
    p_prior: float = 0.7                  # guidance image (80-ring PCr map) available
    p_leg: float = 0.7                    # leg mask available


def acquisition(cfg: SimConfig, n: int, rng: np.random.Generator):
    """Random CRT operator for an n x n CSI grid; returns (traj, info)."""
    n_acq = n
    if rng.uniform() < cfg.p_superres:
        n_acq = int(rng.choice([max(4, n // 2), max(4, int(round(n / 1.5)))]))
    n_rings = max(2, n_acq // 2)
    keep = np.arange(n_rings)
    if rng.uniform() < cfg.p_drop and n_rings > cfg.n_protect + 1:
        n_drop = int(np.floor(rng.uniform(*cfg.drop_rate) * n_rings))
        pool = np.arange(cfg.n_protect, n_rings)
        drop = rng.choice(pool, size=min(n_drop, len(pool) - 1), replace=False)
        keep = np.setdiff1d(keep, drop)
    radii = ring_radii(n_rings, n_acq)[keep] * n_acq / n      # cycles per CSI voxel
    traj = crt_trajectory(radii, cfg.pts_per_ring, rng)
    return traj, {"n_acq": n_acq, "n_rings": n_rings, "rings_kept": len(keep)}


def simulate_sample(cfg: SimConfig, V: np.ndarray, rng: np.random.Generator, n: int) -> dict:
    up, m = cfg.upsample, n * cfg.upsample
    t = np.arange(cfg.n_spec) * cfg.dwell_s

    ph = make_calf_phantom(n, up, rng)
    fids = phantom_fids(ph, t, cfg.larmor_mhz, rng)                    # (m*m, n_spec)
    Xf = fids @ V                                                      # sub-pixel subspace coefficients
    U_true = block_average(Xf.reshape(m, m, -1), up).reshape(n * n, -1)
    x_true = block_average(fids.reshape(m, m, -1), up).reshape(n * n, -1)

    traj, info = acquisition(cfg, n, rng)
    dcf = crt_dcf(traj)
    E = encoding_matrix(traj, n)
    G = gram(E, dcf)

    y_sig = encoding_matrix(traj, n, up) @ Xf
    sigma = 10 ** rng.uniform(*cfg.log_noise) * np.sqrt(np.mean(np.abs(y_sig)**2))
    noise = sigma * (rng.standard_normal(y_sig.shape) + 1j * rng.standard_normal(y_sig.shape)) / np.sqrt(2)
    B = E.conj().T @ (dcf[:, None] * (y_sig + noise))
    noise_std_b = sigma * np.sqrt((dcf**2).sum())                      # |E_kv| = 1

    leg = block_average(ph["leg"].astype(float), up) > 0.5
    pcr_fine = peak_map(fids, cfg.dwell_s, cfg.larmor_mhz, PCR_PPM).reshape(m, m)
    prior = block_average(gaussian_filter(pcr_fine, 0.5 * up), up)
    prior *= 1 + 0.05 * rng.standard_normal(prior.shape)
    has_prior, has_leg = rng.uniform() < cfg.p_prior, rng.uniform() < cfg.p_leg

    feats = make_features(G, B, noise_std_b, prior if has_prior else None, leg if has_leg else None)
    return {
        "feats": feats, "G": G, "B": B, "U_true": U_true, "leg": leg,
        "prior": prior if has_prior else None, "leg_known": has_leg,
        "E": E, "dcf": dcf, "sigma": sigma,
        "pcr_true": peak_map(x_true, cfg.dwell_s, cfg.larmor_mhz, PCR_PPM).reshape(n, n),
        "pi_true": peak_map(x_true, cfg.dwell_s, cfg.larmor_mhz, PI_PPM).reshape(n, n),
        "info": info,
    }


def to_batch(samples: list[dict]) -> dict:
    """Stack samples of one grid size into torch tensors (complex128 / float32)."""
    return {
        "feats": torch.as_tensor(np.stack([s["feats"] for s in samples])),
        "G": torch.as_tensor(np.stack([s["G"] for s in samples])),
        "B": torch.as_tensor(np.stack([s["B"] for s in samples])),
        "U_true": torch.as_tensor(np.stack([s["U_true"] for s in samples])),
    }


def relative_error(U: torch.Tensor, U_true: torch.Tensor) -> torch.Tensor:
    """Per-sample ||U - U_true|| / ||U_true|| (= FID-domain error inside the subspace)."""
    num = (U - U_true).abs().pow(2).sum((-2, -1))
    return (num / U_true.abs().pow(2).sum((-2, -1))).sqrt()
