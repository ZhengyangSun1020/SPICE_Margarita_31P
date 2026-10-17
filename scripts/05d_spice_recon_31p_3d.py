#!/usr/bin/env python3
"""
31P CONCEPT 3D CRT SPICE reconstruction with optional 3D spatial regularization.

This is a 3D extension of scripts/05c_spice_recon_31p.py:
  - 3D trajectory (kx, ky, kz)
  - 3D NUFFT operator shape (Nx, Ny, Nz)
  - 3D SPICE solve on voxel vector length Nx*Ny*Nz

Notes:
  - Uses the 07b loader for robust Twix parsing and geometry handling.
    The loader auto-detects **interleaved 1H/31P** vs **pure 31P** scans:
      * Interleaved  → data pulled from ``twix.phasecor``; geometry from
        ``sNavigatorArray.asElm[0].sCuboid``; matrix XY from
        ``sWipMemBlock.alFree[7]``.
      * Pure 31P     → data pulled from ``twix.image``; geometry from
        ``sSpecPara.sVoI``; matrix XY from ``sKSpace.lBaseResolution``.
    No user flag is required — the same command works for either type.

Usage examples
--------------
Interleaved 1H/31P scan (e.g. MID00073, liver, 30 rings, with ROI mask):

    conda run -n zs_SPICE_MARGARITA --no-capture-output python \\
      scripts/05d_spice_recon_31p_3d.py \\
      --dat /path/to/meas_MID00073_..._30r.dat \\
      --out-dir output_MID00073_05d \\
      --subspace-src pk-7t-liver-3tpulse \\
      --localizer /path/to/localizer_TRA_29_None \\
      --roi-mask roi_mask_MID00073_liver.npy \\
      --roi-strength strong \\
      --rank 12 --lambda1 0.01 --n-iter 100 --apodization 20

Pure 31P scan (e.g. MID00069, BS_30R_1000Hz):

    conda run -n zs_SPICE_MARGARITA --no-capture-output python \\
      scripts/05d_spice_recon_31p_3d.py \\
      --dat /path/to/meas_MID00069_..._BS_30R_1000Hz.dat \\
      --out-dir output_MID00069_05d \\
      --subspace-src pk-7t-liver-3tpulse \\
      --localizer /path/to/localizer_TRA_29_None \\
      --roi-mask roi_mask_MID00073_liver.npy \\
      --roi-strength strong \\
      --rank 12 --lambda1 0.01 --n-iter 100 --apodization 20

The loader log line ``Scan type: phasecor (interleaved 1H/31P)`` or
``Scan type: image (pure 31P — no phasecor found)`` confirms which branch
was taken.
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mrinufft
import pydicom
from scipy.ndimage import map_coordinates, binary_dilation
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.linalg import lu_factor, lu_solve
from tqdm import tqdm
from warnings import filterwarnings

filterwarnings("ignore")

import importlib.util as _ilu
import pathlib as _pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_concept_twix_31p import concept_phase_correction

_spec = _ilu.spec_from_file_location(
    "_06c", _pl.Path(__file__).parent / "06c_iterative_recon_31p_3d.py")
_06c = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_06c)

build_trajectory_3d = _06c.build_trajectory_3d

_spec_07b = _ilu.spec_from_file_location(
    "_07b", _pl.Path(__file__).parent / "07b_interleaved_recon_31p_3d.py")
_07b = _ilu.module_from_spec(_spec_07b)
_spec_07b.loader.exec_module(_07b)

load_interleaved_31p = _07b.load_interleaved_31p
compute_fov_shift_3d = _07b.compute_fov_shift_3d
_fGSLCalcPRS = _07b._fGSLCalcPRS
_apply_output_xy_orientation = _07b._apply_output_xy_orientation
try:
    from twix_fast import cached_load as _cached_load
except Exception:
    _cached_load = None


def _even_n_roll(vol):
    """Even-matrix NUFFT centering correction (identical to 07b/08): finufft centres
    the isocentre at N//2 while the Fessler/MATLAB convention is symmetric, so for
    EVEN in-plane matrices roll the reconstructed grid by one voxel on both in-plane
    axes. vol is (Nx, Ny, Nz, n_spec). No-op for odd matrices."""
    if vol.shape[0] % 2 == 0 and vol.shape[1] % 2 == 0:
        return np.roll(vol, (-1, -1), axis=(0, 1))
    return vol


# ── Fixed ROI / spatial-prior settings ───────────────────────────────────────
# These were previously CLI flags.  They are pinned here to the values that the
# 7T liver runs converged on, so the only spatial knobs left on the command line
# are --roi-mask (which mask) and --roi-strength (how hard it is enforced).
# Change them here if you need to re-tune; they are deliberately not per-run.
LOCALIZER_PERCENTILES = (2.0, 99.5)   # intensity window for the anatomical prior
LOCALIZER_GAMMA       = 0.7           # contrast gamma applied to the localizer
PRIOR_GAMMA           = 1.0           # extra gamma on the normalized prior
PRIOR_SCALE_PLOT      = 1.0           # display-only scaling of the prior map
PRIOR_SCALE_GRAPH     = 1.0           # scaling of the prior feeding the graph
PRIOR_FLIP_NATIVE_Y   = False         # flip prior y-axis in native MRSI frame
GRAPH_WMAX            = 6000.0        # cap on graph edge weights (1/intensity²)
ROI_RENORM            = True          # renormalize in-ROI prior to [0, 1]
ROI_DILATION          = 1             # dilate mask before deciding edge crossings

# --roi-strength presets → (out_edge_scale, hard_boundary)
# "off" ignores the mask when building the graph (prior still masked).
ROI_STRENGTH_PRESETS = {
    "off":    (None,  False),
    "soft":   (10.0,  False),
    "strong": (100.0, False),
    "strict": (None,  True),
}


def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


def compute_crt_dcf_hamming(traj, hamming_radius=None):
    """CRT DCF x Hamming matching MATLAB op_CalcAndApplyDensComp.m."""
    r = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
    r_ham = np.asarray(hamming_radius, dtype=np.float64) if hamming_radius is not None else r
    if r_ham.shape != r.shape:
        raise ValueError("hamming_radius must match trajectory length")

    unique_r = np.sort(np.unique(np.round(r, 6)))
    n_rings = len(unique_r)
    if n_rings < 2:
        return np.ones_like(r, dtype=np.float32)

    r_norm = unique_r / (float(unique_r.max()) + 1e-12)
    n_per_ring = np.array([(np.abs(r - ri) < 1e-5).sum() for ri in unique_r], dtype=np.float64)
    max_pts = float(n_per_ring.max())

    dcf_ring = np.zeros(n_rings, dtype=np.float64)
    for i in range(n_rings):
        if i == 0:
            area = 0.25 * np.pi * (r_norm[0] + r_norm[1]) ** 2
        elif i == n_rings - 1:
            r_out = 1.5 * r_norm[-1] - 0.5 * r_norm[-2]
            r_in = 0.5 * (r_norm[-1] + r_norm[-2])
            area = np.pi * (r_out ** 2 - r_in ** 2)
        else:
            upper = 0.5 * (r_norm[i] + r_norm[i + 1])
            lower = 0.5 * (r_norm[i - 1] + r_norm[i])
            area = np.pi * (upper ** 2 - lower ** 2)
        dcf_ring[i] = area * (max_pts / n_per_ring[i])

    dcf = np.zeros(len(r), dtype=np.float64)
    for i, ri in enumerate(unique_r):
        dcf[np.abs(r - ri) < 1e-5] = dcf_ring[i]

    ham = 0.54 + 0.46 * np.cos(2.0 * np.pi * r_ham)
    return (dcf * ham).astype(np.float32)


def compute_crt_dcf_perkz(traj, ring_pts):
    """3D CRT DCF for variable-ring-per-kz trajectories (e.g. spherical 3D CRT).

    compute_crt_dcf_hamming pools rings by xy radius across ALL kz levels, so its
    sample-count correction upweights radii that exist on only a few kz levels
    (outer radii of a spherical design get up to ~4x weight -> extra high-k noise),
    and it has no kz-spacing term. Here each ring gets
        annulus area (within its own kz level) x kz Voronoi spacing / pts on ring
    times a spherical 3D Hamming taper on |k| (cycles/pixel, clipped at 0.5).
    For a stack of identical ring sets on uniform kz this reduces to the 2D DCF up
    to a constant. traj is ring-major with ring_pts[r] points per ring.
    """
    ring_pts = np.asarray(ring_pts, dtype=int)
    starts = np.concatenate([[0], np.cumsum(ring_pts)[:-1]])
    r_ring = np.array([np.hypot(traj[s, 0], traj[s, 1]) for s in starts])
    kz_ring = np.array([traj[s, 2] for s in starts])
    levels = np.unique(np.round(kz_ring, 6))
    dkz = np.gradient(levels) if len(levels) > 1 else np.ones(1)

    dcf_ring = np.zeros(len(ring_pts))
    for li, lv in enumerate(levels):
        idx = np.where(np.isclose(kz_ring, lv, atol=1e-6))[0]
        order = np.argsort(r_ring[idx])
        rs = r_ring[idx][order]
        if len(rs) == 1:
            areas = np.array([np.pi * (1.5 * rs[0]) ** 2])
        else:
            edges = np.concatenate([[0.0], 0.5 * (rs[1:] + rs[:-1]), [1.5 * rs[-1] - 0.5 * rs[-2]]])
            areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        dcf_ring[idx[order]] = areas * dkz[li] / ring_pts[idx[order]]

    dcf = np.repeat(dcf_ring, ring_pts)
    r3 = np.sqrt(np.sum(traj.astype(np.float64) ** 2, axis=1))
    ham = 0.54 + 0.46 * np.cos(2.0 * np.pi * np.clip(r3, 0.0, 0.5))
    return (dcf * ham).astype(np.float32)


def generate_training_data(n_spec, dwell_s, larmor_mhz, n_training=5000):
    """Synthetic 31P training FIDs."""
    t = np.arange(n_spec) * dwell_s
    rng = np.random.default_rng(123)
    mets = {
        "PCr": {"shift_ppm": 0.0, "lw_range": (5, 15), "amp_range": (3.5, 4.5)},
        "Pi": {"shift_ppm": 4.85, "lw_range": (5, 20), "amp_range": (0.0, 0.8)},
        "Pi2": {"shift_ppm": 4.25, "lw_range": (5, 20), "amp_range": (0.0, 0.5)},
        "PDE": {"shift_ppm": 3.07, "lw_range": (10, 35), "amp_range": (0.0, 0.3)},
        "gATP": {"shift_ppm": -2.53, "lw_range": (10, 30), "amp_range": (0.8, 1.2)},
        "aATP": {"shift_ppm": -7.58, "lw_range": (10, 30), "amp_range": (0.8, 1.2)},
        "bATP": {"shift_ppm": -15.86, "lw_range": (12, 35), "amp_range": (0.8, 1.2)},
    }

    out = np.zeros((n_training, n_spec), dtype=np.complex64)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        for m in mets.values():
            amp = rng.uniform(*m["amp_range"])
            lw = rng.uniform(*m["lw_range"])
            phase = rng.uniform(0, 2 * np.pi)
            freq_hz = m["shift_ppm"] * larmor_mhz
            sig += amp * np.exp(1j * phase) * np.exp(-lw * np.pi * t) * np.exp(2j * np.pi * freq_hz * t)
        whole_shift = rng.uniform(-0.1, 0.1) * larmor_mhz
        sig *= np.exp(2j * np.pi * whole_shift * t)
        out[i] = sig.astype(np.complex64)
    return out


def generate_training_data_pk2pi(n_spec, dwell_s, larmor_mhz, n_training=5000,
                                 pi_ph_range=True):
    """PK_3T_Skeletal_2Pi-like multiplet synthetic training FIDs.

    Pi is modelled as a SINGLE resonance whose chemical shift is pH-dependent.
    When pi_ph_range=True (default) its center is sampled uniformly across the
    physiological Pi–PCr shift range per FID (cs_bounds below), so the SVD
    subspace spans "one Pi peak that moves with pH" — NOT a fixed Pi/Pi2
    doublet. Set pi_ph_range=False to pin Pi at its resting center (legacy
    two-peak behaviour, kept only for comparison).

    Format: (name, center_ppm, lw_init, amp_range, components, cs_bounds)
    cs_bounds=None → fixed center; otherwise center ~ U(cs_bounds) per FID.
    """
    t = np.arange(n_spec) * dwell_s
    rng = np.random.default_rng(321)
    metab = [
        ("PCr", 0.0, 8.0, (3.5, 4.5), [(0.0, 1.0)], None),
        # single pH-shifting Pi (spans acidic→alkaline, i.e. the old Pi2..Pi span)
        ("Pi", 4.85, 10.0, (0.0, 0.9), [(0.0, 1.0)], (3.9, 5.2)),
        ("gATP", -2.53, 18.0, (0.8, 1.2), [(-0.08, 0.5), (0.08, 0.5)], None),
        ("aATP", -7.58, 18.0, (0.8, 1.2), [(-0.10, 0.5), (0.10, 0.5)], None),
        ("bATP", -15.86, 20.0, (0.8, 1.2), [(-0.10, 0.5), (0.10, 0.5)], None),
    ]

    out = np.zeros((n_training, n_spec), dtype=np.complex64)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        for _, center_ppm, lw_init, amp_range, comps, cs_bounds in metab:
            amp = rng.uniform(*amp_range)
            lw = rng.uniform(lw_init * 0.5, min(50.0, lw_init * 3.0))
            phase = rng.uniform(0, 2 * np.pi)
            center = center_ppm
            if pi_ph_range and cs_bounds is not None:
                center = rng.uniform(*cs_bounds)   # pH degree of freedom (single moving peak)
            for dppm, w in comps:
                freq_hz = (center + dppm) * larmor_mhz
                sig += amp * w * np.exp(1j * phase) * np.exp(-lw * np.pi * t) * np.exp(2j * np.pi * freq_hz * t)
        whole_shift = rng.uniform(-0.1, 0.1) * larmor_mhz
        sig *= np.exp(2j * np.pi * whole_shift * t)
        out[i] = sig.astype(np.complex64)
    return out


def generate_training_data_pk_7t_liver_3tpulse(n_spec, dwell_s, larmor_mhz, n_training=5000):
    """Generate FIDs matching OXSA PK_7T_Liver_3TPulse.m AMARES prior.

    Peaks (chemShift, initial linewidth Hz, initial amplitude), multiplet
    structure taken verbatim from PK_7T_Liver_3TPulse.m. Chemical-shift
    deltas assume a 120.3 MHz 31P Larmor. ATP_BETA is a [1,0,1] multiplet
    (equivalent to two lines at +/-delta with weight 0.5 each). ATP_ALPHA
    and ATP_GAMMA use [0,0] doublets. All other resonances are singlets.
    Amplitudes and linewidths are randomized around the OXSA initial values
    (0.5x - 2x on linewidth, 0.6x - 1.4x on amplitude) to build a diverse
    training bank for the SPICE subspace SVD.
    """
    t = np.arange(n_spec) * dwell_s
    rng = np.random.default_rng(1103)
    beta_delta  = 11.47 / 120.3
    alpha_delta = 19.25 / 120.3
    gamma_delta = 15.86 / 120.3
    #  (center_ppm, lw_init_Hz, amp_init,
    #    (lw_lo, lw_hi) as multipliers of lw_init,
    #    (amp_lo, amp_hi) as multipliers of amp_init,
    #    components = [(delta_ppm, weight), ...])
    metabolites = [
        (-15.996, 70.0, 3.0,  (0.5, 2.0), (0.6, 1.4),
         [(-beta_delta,  0.5), (beta_delta,  0.5)]),                      # ATP_BETA [1,0,1]
        (-7.404,  70.0, 3.7,  (0.5, 2.0), (0.6, 1.4),
         [(-alpha_delta / 2, 0.5), (alpha_delta / 2, 0.5)]),              # ATP_ALPHA [0,0]
        (-2.208,  60.0, 3.0,  (0.5, 2.0), (0.6, 1.4),
         [(-gamma_delta / 2, 0.5), (gamma_delta / 2, 0.5)]),              # ATP_GAMMA [0,0]
        ( 0.000,  25.0, 0.5,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # PCr
        ( 5.366,  80.0, 2.0,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # Pi
        ( 3.1936, 50.0, 2.5,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # GPC
        ( 3.7109, 50.0, 2.5,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # GPE
        ( 6.617,  50.0, 1.5,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # PC
        ( 7.104,  40.0, 0.8,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # PE
        (-8.251,  50.0, 1.0,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # NADH
        (-9.478,  50.0, 0.6,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # UDPG
        ( 2.200,  50.0, 0.6,  (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),      # PEP
    ]
    out = np.zeros((n_training, n_spec), dtype=np.complex64)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        for (center_ppm, lw_init, amp_init,
             (lw_lo, lw_hi), (amp_lo, amp_hi), components) in metabolites:
            amp = amp_init * rng.uniform(amp_lo, amp_hi)
            linewidth = lw_init * rng.uniform(lw_lo, lw_hi)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            for delta_ppm, weight in components:
                freq_hz = (center_ppm + delta_ppm) * larmor_mhz
                sig += (amp * weight * np.exp(1j * phase)
                        * np.exp(-linewidth * np.pi * t)
                        * np.exp(2j * np.pi * freq_hz * t))
        # small global B0 jitter (+/- 0.1 ppm)
        sig *= np.exp(2j * np.pi * rng.uniform(-0.1, 0.1) * larmor_mhz * t)
        out[i] = sig.astype(np.complex64)
    return out


def generate_training_data_pk_7t_skeletal(n_spec, dwell_s, larmor_mhz, n_training=5000):
    """Generate FIDs from PK_7T_Skeletal_voigt.m AMARES initial values.

    Peak table taken verbatim from the OXSA ``values.IVCellArray``:

        PCR         :  0.00 ppm,  LW=10 Hz,  amp=1.0
        Pi          :  4.85 ppm,  LW=30 Hz,  amp=0.5
        PDE_high    :  3.07 ppm,  LW=30 Hz,  amp=0.1
        ATP_BETA*   : -15.86 ppm, LW=10 Hz,  amp=1.0,   multiplet [1,0,1]  Δ=15/120.3
        ATP_ALPHA*  :  -7.58 ppm, LW=10 Hz,  amp=1.0,   multiplet [0,0]    Δ=16/120.3
        ATP_GAMMA*  :  -2.53 ppm, LW=10 Hz,  amp=1.0,   multiplet [0,0]    Δ=15/120.3

    LWs and amplitudes are randomized around these initial values (0.5x-2x on
    LW, 0.6x-1.4x on amplitude) with a ±0.1 ppm global B0 jitter to build a
    diverse subspace training bank, matching the pk-7t-liver-3tpulse policy.
    """
    t = np.arange(n_spec) * dwell_s
    rng = np.random.default_rng(701)
    atp_beta_delta = 15.0 / 120.3
    atp_alpha_delta = 16.0 / 120.3
    atp_gamma_delta = 15.0 / 120.3
    #  (center_ppm, lw_init_Hz, amp_init,
    #    (lw_lo, lw_hi) multipliers of lw_init,
    #    (amp_lo, amp_hi) multipliers of amp_init,
    #    components = [(delta_ppm, weight), ...])
    metabolites = [
        (  0.000, 10.0, 1.0, (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),        # PCR
        (  4.850, 30.0, 0.5, (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),        # Pi
        (  3.070, 30.0, 0.1, (0.5, 2.0), (0.6, 1.4), [(0.0, 1.0)]),        # PDE_high
        (-15.860, 10.0, 1.0, (0.5, 2.0), (0.6, 1.4),                       # ATP_BETA [1,0,1]
         [(-atp_beta_delta,  0.5), (atp_beta_delta,  0.5)]),
        ( -7.580, 10.0, 1.0, (0.5, 2.0), (0.6, 1.4),                       # ATP_ALPHA [0,0]
         [(-atp_alpha_delta / 2, 0.5), (atp_alpha_delta / 2, 0.5)]),
        ( -2.530, 10.0, 1.0, (0.5, 2.0), (0.6, 1.4),                       # ATP_GAMMA [0,0]
         [(-atp_gamma_delta / 2, 0.5), (atp_gamma_delta / 2, 0.5)]),
    ]
    out = np.zeros((n_training, n_spec), dtype=np.complex64)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        for (center_ppm, lw_init, amp_init,
             (lw_lo, lw_hi), (amp_lo, amp_hi), components) in metabolites:
            amp = amp_init * rng.uniform(amp_lo, amp_hi)
            linewidth = lw_init * rng.uniform(lw_lo, lw_hi)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            for delta_ppm, weight in components:
                freq_hz = (center_ppm + delta_ppm) * larmor_mhz
                sig += (amp * weight * np.exp(1j * phase)
                        * np.exp(-linewidth * np.pi * t)
                        * np.exp(2j * np.pi * freq_hz * t))
        sig *= np.exp(2j * np.pi * rng.uniform(-0.1, 0.1) * larmor_mhz * t)
        out[i] = sig.astype(np.complex64)
    return out


def generate_training_data_pk_cardiac(n_spec, dwell_s, larmor_mhz, n_training=5000):
    """Generate 31P cardiac FIDs from OXSA PK_7T_Cardiac.m prior knowledge.

    Peaks, shifts and linewidths taken verbatim from values.IVCellArray /
    values.PKCellArray in
      OXSA/main/+AMARES/+priorKnowledge/PK_7T_Cardiac.m :

        PCR         :   0.00 ppm, LW=33 Hz
        ATP_GAMMA*  :  -2.47 ppm, LW=34 Hz, doublet   J=15 Hz
        ATP_ALPHA*  :  -7.65 ppm, LW=50 Hz, doublet   J=16 Hz
        ATP_BETA*   : -16.33 ppm, LW=56 Hz, [1,0,1]   J=15 Hz, ampRatio 0.5
        PDE         :   2.69 ppm, LW=60 Hz
        2,3-DPG1    :   5.20 ppm, LW=60 Hz            (blood, cardiac-specific)
        2,3-DPG2    :   6.35 ppm, LW=60 Hz            (blood, cardiac-specific)

    NOTE: the 7T cardiac prior does NOT include Pi — in cardiac 31P the Pi
    region is dominated by 2,3-DPG from blood, so OXSA models DPG1/DPG2 only.
    Multiplet splittings (J in Hz, field-independent) are converted to ppm at
    the run's larmor freq: Δppm = J_Hz / larmor_mhz.
    """
    t = np.arange(n_spec) * dwell_s
    rng = np.random.default_rng(731)
    beta_d  = 15.0 / larmor_mhz          # β-ATP triplet adjacent J (outer lines ±J)
    alpha_d = 16.0 / larmor_mhz          # α-ATP doublet J
    gamma_d = 15.0 / larmor_mhz          # γ-ATP doublet J
    metabolites = [
        (  0.000, 33.0, 1.0,  (0.6, 1.6), (0.6, 1.4), [(0.0, 1.0)]),                 # PCr
        ( -2.470, 34.0, 1.0,  (0.6, 1.6), (0.6, 1.4),                                # gamma-ATP doublet
         [(-gamma_d / 2, 0.5), (gamma_d / 2, 0.5)]),
        ( -7.650, 50.0, 1.0,  (0.6, 1.6), (0.6, 1.4),                                # alpha-ATP doublet
         [(-alpha_d / 2, 0.5), (alpha_d / 2, 0.5)]),
        (-16.330, 56.0, 0.5,  (0.6, 1.6), (0.6, 1.4),                                # beta-ATP [1,0,1], ampRatio 0.5
         [(-beta_d, 0.5), (beta_d, 0.5)]),
        (  2.690, 60.0, 0.20, (0.6, 1.6), (0.0, 1.5), [(0.0, 1.0)]),                 # PDE
        (  5.200, 60.0, 0.45, (0.6, 1.6), (0.3, 1.5), [(0.0, 1.0)]),                 # 2,3-DPG1 (blood)
        (  6.350, 60.0, 0.45, (0.6, 1.6), (0.3, 1.5), [(0.0, 1.0)]),                 # 2,3-DPG2 (blood)
    ]
    out = np.zeros((n_training, n_spec), dtype=np.complex64)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        for (center_ppm, lw_init, amp_init,
             (lw_lo, lw_hi), (amp_lo, amp_hi), components) in metabolites:
            amp = amp_init * rng.uniform(amp_lo, amp_hi)
            linewidth = lw_init * rng.uniform(lw_lo, lw_hi)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            for delta_ppm, weight in components:
                freq_hz = (center_ppm + delta_ppm) * larmor_mhz
                sig += (amp * weight * np.exp(1j * phase)
                        * np.exp(-linewidth * np.pi * t)
                        * np.exp(2j * np.pi * freq_hz * t))
        sig *= np.exp(2j * np.pi * rng.uniform(-0.1, 0.1) * larmor_mhz * t)
        out[i] = sig.astype(np.complex64)
    return out


def generate_training_data_pk_3t_brain(n_spec, dwell_s, larmor_mhz, n_training=5000,
                                       phase_mode="random", delay_s=0.0,
                                       delay_jitter_s=1e-4, phase_jitter_deg=10.0,
                                       lw_floor_pcr=None, lw_floor_atp=None):
    """Generate 31P brain FIDs from OXSA PK_3T_Brain_v1.m prior knowledge.

    phase_mode="random": independent random phase per metabolite (subspace also has
        to absorb every phase/first-order combination -> large rank).
    phase_mode="shared": one random zero-order phase per FID shared by all peaks
        (+-phase_jitter_deg per peak) and a common acquisition delay
        delay_s +- delay_jitter_s, i.e. each line is exp(2i*pi*f*(t + delay)).
        delay_s uses the same sign as the spectral first-order correction
        spec * exp(-2i*pi*freq*tau) applied to FIDToSpec(conj(fid)) (tau = delay_s).
    lw_floor_pcr / lw_floor_atp (Hz): raise the lower linewidth bound of PCr and of the
        ATP multiplets + NADH. The OXSA bounds start at 0 Hz; with 4 Hz lines the subspace
        holds features finer than low-SNR data can resolve (sharp J-resolved beta-ATP
        triplets, single-bin notches splitting PCr at 0 ppm). None keeps the defaults.

    Initial values / bounds from
      OXSA/main/+AMARES/+priorKnowledge/PK_3T_Brain_v1.m :

        ATP_BETA*   : -16.74 ppm, LW 10 Hz (0-30), triplet [1,0,1] J=15 Hz,
                      outer/centre ampRatio 0.5  -> 1:2:1
        ATP_ALPHA*  :  -7.88 ppm, LW 10 Hz (0-30), doublet J=16 Hz
        ATP_GAMMA*  :  -2.62 ppm, LW 10 Hz (0-30), doublet J=15 Hz
        PCR         :   0.00 ppm, LW 10 Hz (0-50)
        GPC         :   2.69 ppm, LW 20 Hz (20-50)
        GPE         :   3.19 ppm, LW 20 Hz (20-50)
        Pi          :   5.00 ppm, LW 20 Hz (10-50), shift bound 4-6
        PME         :   6.30 ppm, LW 20 Hz (20-50)
        NADH        :  -8.50 ppm, LW 10 Hz (0-30)

    OXSA starts every amplitude at 1; here amplitudes are drawn around typical
    3T brain ratios so the subspace spans realistic mixtures. Linewidths are
    sampled inside the AMARES bounds, Pi across 4.5-5.5 ppm (pH), small per-peak
    shift jitter, random per-metabolite phase and a global +-0.1 ppm B0 shift.
    Multiplet splittings (J in Hz) are converted with the run's larmor freq.
    """
    t = np.arange(n_spec) * dwell_s
    rng = np.random.default_rng(310)
    beta_d  = 15.0 / larmor_mhz
    alpha_d = 16.0 / larmor_mhz
    gamma_d = 15.0 / larmor_mhz
    # (center_ppm, shift_jitter_ppm, lw_range_Hz, amp_init, amp_range, components)
    metabolites = [
        (  0.00, 0.05, (4.0, 20.0), 1.00, (0.6, 1.4), [(0.0, 1.0)]),                       # PCr
        ( -2.62, 0.10, (4.0, 25.0), 0.70, (0.4, 1.4),
         [(-gamma_d / 2, 0.5), (gamma_d / 2, 0.5)]),                                       # gamma-ATP
        ( -7.88, 0.10, (4.0, 25.0), 0.70, (0.4, 1.4),
         [(-alpha_d / 2, 0.5), (alpha_d / 2, 0.5)]),                                       # alpha-ATP
        (-16.74, 0.15, (4.0, 30.0), 0.60, (0.4, 1.4),
         [(-beta_d, 0.25), (0.0, 0.5), (beta_d, 0.25)]),                                   # beta-ATP 1:2:1
        (  2.69, 0.10, (20.0, 40.0), 0.35, (0.0, 1.5), [(0.0, 1.0)]),                      # GPC
        (  3.19, 0.10, (20.0, 40.0), 0.35, (0.0, 1.5), [(0.0, 1.0)]),                      # GPE
        (  5.00, 0.50, (10.0, 40.0), 0.30, (0.0, 1.6), [(0.0, 1.0)]),                      # Pi (4.5-5.5)
        (  6.30, 0.15, (20.0, 40.0), 0.40, (0.0, 1.5), [(0.0, 1.0)]),                      # PME
        ( -8.50, 0.15, (4.0, 25.0), 0.15, (0.0, 1.5), [(0.0, 1.0)]),                       # NADH
    ]

    def _floor(m, lo_new):
        c, jit, (lo, hi), a0, ar, comps = m
        lo = max(lo, float(lo_new))
        return (c, jit, (lo, max(hi, lo + 5.0)), a0, ar, comps)

    if lw_floor_pcr is not None:
        metabolites[0] = _floor(metabolites[0], lw_floor_pcr)
    if lw_floor_atp is not None:
        for i in (1, 2, 3, 8):                         # gamma, alpha, beta-ATP, NADH
            metabolites[i] = _floor(metabolites[i], lw_floor_atp)
    out = np.zeros((n_training, n_spec), dtype=np.complex64)
    shared = (phase_mode == "shared")
    pj = np.deg2rad(phase_jitter_deg)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        delay = delay_s + rng.uniform(-delay_jitter_s, delay_jitter_s) if shared else 0.0
        b0_hz = rng.uniform(-0.1, 0.1) * larmor_mhz
        for (center_ppm, jit, (lw_lo, lw_hi), amp_init, (amp_lo, amp_hi), components) in metabolites:
            amp = amp_init * rng.uniform(amp_lo, amp_hi)
            linewidth = rng.uniform(lw_lo, lw_hi)
            phase = phase0 + rng.uniform(-pj, pj) if shared else rng.uniform(0.0, 2.0 * np.pi)
            c_ppm = center_ppm + rng.uniform(-jit, jit)
            for delta_ppm, weight in components:
                freq_hz = (c_ppm + delta_ppm) * larmor_mhz + b0_hz
                sig += (amp * weight * np.exp(1j * phase)
                        * np.exp(-linewidth * np.pi * t)
                        * np.exp(2j * np.pi * freq_hz * (t + delay)))
        out[i] = sig.astype(np.complex64)
    return out


def load_localizer_prior_3d(localizer_dir, info, shape_xyz, percentiles, gamma,
                            flip_native_y=True):
    """Sample a DICOM localizer at every MRSI voxel center using world geometry.

    For each MRSI z index the nearest DICOM plane is selected independently,
    then all XY voxel centers are projected into that exact image plane.
    """
    files = []
    for root, _, names in os.walk(localizer_dir):
        files.extend(os.path.join(root, name) for name in names)

    records = []
    for path in files:
        try:
            ds = pydicom.dcmread(path, force=True)
            if not (hasattr(ds, "PixelData") and hasattr(ds, "ImagePositionPatient")
                    and hasattr(ds, "ImageOrientationPatient") and hasattr(ds, "PixelSpacing")):
                continue
            iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
            # DICOM IOP first triplet advances image columns; second advances
            # image rows. PixelSpacing[0] is row spacing and [1] is column.
            col_dir = iop[:3] / (np.linalg.norm(iop[:3]) + 1e-20)
            row_dir = iop[3:] / (np.linalg.norm(iop[3:]) + 1e-20)
            normal = np.cross(col_dir, row_dir)
            normal /= np.linalg.norm(normal) + 1e-20
            records.append({
                "path": path,
                "img": ds.pixel_array.astype(np.float32),
                "ipp": np.asarray(ds.ImagePositionPatient, dtype=np.float64),
                "row_dir": row_dir,
                "col_dir": col_dir,
                "normal": normal,
                "sp_row": float(ds.PixelSpacing[0]),
                "sp_col": float(ds.PixelSpacing[1]),
            })
        except Exception:
            continue
    if not records:
        raise RuntimeError(f"No readable DICOM localizer images in {localizer_dir}")

    nx, ny, nz = shape_xyz
    position = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]], dtype=np.float64)
    normal = np.array([info["snv_x"], info["snv_y"], info["snv_z"]], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-20

    # Keep only localizer planes PARALLEL to the slab (same orientation). Without
    # this, a perpendicular series (e.g. a coronal at the VoI centre, distance 0
    # to every axial MRSI slice) hijacks the nearest-plane pick and the localizer
    # is sampled edge-on -> the 'vertical stripes' artifact. Mirrors 08's
    # parallel-series selection.
    _par = [rec for rec in records if abs(float(np.dot(rec["normal"], normal))) > 0.966]
    if _par:
        records = _par
        print(f"[spice3d] Localizer: {len(records)} plane(s) parallel to the slab kept")
    else:
        print("[spice3d] WARNING: no localizer plane parallel to the slab; using all series")

    ref_normal = normal
    for rec in records:
        if np.dot(rec["normal"], ref_normal) < 0:
            rec["normal"] *= -1
    records.sort(key=lambda rec: float(np.dot(rec["ipp"], ref_normal)))
    phase, read = _fGSLCalcPRS(normal, info["in_plane_rotation"])
    phase /= np.linalg.norm(phase) + 1e-20
    read /= np.linalg.norm(read) + 1e-20
    voxel_xy = float(info["fov_mm"]) / nx
    voxel_z = float(info["fov_m_z"]) * 1000.0 / nz
    cx, cy, cz = (nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2

    prior = np.zeros((nx, ny, nz), dtype=np.float32)
    selected = []
    p_low, p_high = percentiles
    for z in range(nz):
        center = position + (z - cz) * voxel_z * normal
        plane_dists = [abs(float(np.dot(center - rec["ipp"], rec["normal"]))) for rec in records]
        idx = int(np.argmin(plane_dists))
        rec = records[idx]
        img = rec["img"]
        lo, hi = np.percentile(img, [p_low, p_high])
        img = np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0) ** gamma
        rows = np.zeros((nx, ny), dtype=np.float64)
        cols = np.zeros((nx, ny), dtype=np.float64)
        for x in range(nx):
            for y in range(ny):
                world = center + (x - cx) * voxel_xy * phase + (y - cy) * voxel_xy * read
                delta = world - rec["ipp"]
                rows[x, y] = np.dot(delta, rec["row_dir"]) / rec["sp_row"]
                cols[x, y] = np.dot(delta, rec["col_dir"]) / rec["sp_col"]
        prior[:, :, z] = map_coordinates(img, [rows, cols], order=1, mode="constant", cval=0.0)
        image_center = rec["ipp"] + ((img.shape[0] - 1) * rec["sp_row"] / 2) * rec["row_dir"] + ((img.shape[1] - 1) * rec["sp_col"] / 2) * rec["col_dir"]
        selected.append({
            "index": idx,
            "path": rec["path"],
            "distance_mm": plane_dists[idx],
            "mrsi_center": center,
            "localizer_center": image_center,
            "rows": rows,
            "cols": cols,
            "record": rec,
        })

    # The CRT reconstruction y-axis has the opposite direction from the
    # patient-space readout progression used above. Correct it before the graph
    # is built; this is a data correction, not merely an imshow display flip.
    if flip_native_y:
        prior = np.flip(prior, axis=1).copy()
    prior /= prior.max() + 1e-12
    return prior, selected


def build_regularizer_3d(prior_3d, wmax=6000.0, adj=6, mask=None,
                         out_edge_scale=10.0, mask_dilation=1,
                         hard_boundary=False, return_graph=False):
    """Build a 3D graph Laplacian regularizer from a 3D prior map.

    If mask is provided, edges touching outside-mask voxels are attenuated
    (soft boundary) or zeroed (hard boundary), matching the 05c strategy.
    """
    if prior_3d.ndim != 3:
        raise ValueError("prior_3d must be 3D")

    nx, ny, nz = prior_3d.shape
    coords = np.indices((nx, ny, nz)).reshape(3, -1).T.astype(np.float32)
    # 6=faces (d=1), 18=+edges (d<=sqrt2), 26=+corners (d<=sqrt3)
    radius = {6: 1.01, 18: 1.5, 26: 1.75}.get(int(adj), 1.01)
    pairs = KDTree(coords).query_pairs(r=radius, output_type="ndarray")

    ref = prior_3d.reshape(-1)
    c = (np.diff(ref[pairs], axis=1) ** 2).flatten()
    old = np.seterr(divide="ignore")
    w = np.minimum(1.0 / c, wmax)
    np.seterr(**old)

    if mask is not None:
        if mask.shape != prior_3d.shape:
            raise ValueError(f"mask shape {mask.shape} != prior shape {prior_3d.shape}")
        if int(mask_dilation) > 0:
            mask_img = binary_dilation(mask.astype(bool), iterations=int(mask_dilation))
        else:
            mask_img = mask.astype(bool)
        mask_flat = mask_img.reshape(-1)
        out_of_mask = ~(mask_flat[pairs[:, 0]] & mask_flat[pairs[:, 1]])
        w = w.copy()
        if hard_boundary:
            w[out_of_mask] = 0.0
        else:
            w[out_of_mask] = wmax / float(max(out_edge_scale, 1e-6))

    n_vox = ref.size
    a = csr_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(n_vox, n_vox))
    a = 0.5 * (a + a.T)
    deg = np.array(a.sum(axis=1)).reshape(-1)
    l = csr_matrix((deg, (np.arange(n_vox), np.arange(n_vox))), shape=(n_vox, n_vox)) - a
    if return_graph:
        return l.toarray().astype(np.complex64), a.toarray().astype(np.float32)
    return l.toarray().astype(np.complex64)


def _build_gram_3d(nufft_op, shape_xyz):
    """Trajectory-only 3D Gram (F^H W F, density-weighted).

    Depends only on the NUFFT operator and grid — independent of rank, lambda
    and coil — so it is built ONCE and reused across the whole rank sweep (and
    across coils in the multi-coil path). This mirrors the 1H SPICE core, which
    builds a single Gram operator via ``build_nufft_ops`` and reuses it rather
    than rebuilding it per solve.
    """
    nx, ny, nz = shape_xyz
    n_vox = nx * ny * nz
    gram = np.zeros((n_vox, n_vox), dtype=np.complex64)
    for v in tqdm(range(n_vox), desc="Gram columns"):
        delta = np.zeros((nx, ny, nz), dtype=np.complex64)
        delta.flat[v] = 1.0
        gram[:, v] = nufft_op.adj_op(nufft_op.op(delta)).reshape(n_vox)
    return gram


def suggest_spice_rank(s_vals, adj_fid_flat=None, vh=None, max_rank=None,
                       energy_tols=(0.99, 0.999), trunc_tols=(1e-2, 1e-3)):
    """Suggest a SPICE model order (rank) from the training singular values and,
    when available, the projection of the *measured* data onto the subspace.

    Criteria returned in the dict:
      * ``trunc<tol`` : smallest r with relative truncation error
        sqrt(sum(s[r:]^2)/sum(s^2)) < tol.  tol = 1e-3 (0.1%) is the exact rule
        Lam et al. 2020 (learned-subspace SPICE, Fig. 2) used to pick the rank.
      * ``energy>f``  : smallest r whose top-r components hold energy fraction f.
      * ``elbow``     : max-curvature knee (distance-to-chord) of log10(sigma).
      * ``data_knee`` : (needs adj_fid_flat + vh) knee of the fraction of the
        measured adjoint-FID energy captured by the rank-r subspace — the most
        acquisition-relevant metric (a full-curve version of the per-rank
        "energy captured" print already used in 05c).
    ``recommended`` = data_knee if available else the 0.1% truncation rank,
    clamped to [1, max_rank].
    """
    s = np.asarray(s_vals, dtype=np.float64)
    s = s[s > 0]
    n = s.size
    if max_rank is None or max_rank > n:
        max_rank = n

    s2 = s ** 2
    total = float(s2.sum()) + 1e-30
    cum = np.cumsum(s2)
    trunc_err = np.sqrt(np.maximum(total - cum, 0.0) / total)   # error keeping r comps
    cum_energy = cum / total

    out = {}
    for tol in trunc_tols:
        mask = trunc_err < tol
        out[f"trunc<{tol:g}"] = int(np.argmax(mask) + 1) if mask.any() else n
    for f in energy_tols:
        out[f"energy>{f:g}"] = int(min(np.searchsorted(cum_energy, f) + 1, n))

    # elbow: distance from each point to the chord of the log-sigma curve
    y = np.log10(s / s[0] + 1e-30)
    x = np.arange(n, dtype=np.float64)
    denom = np.hypot(y[-1] - y[0], x[-1] - x[0]) + 1e-30
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y
                  + x[-1] * y[0] - y[-1] * x[0]) / denom
    out["elbow"] = int(np.argmax(dist) + 1)

    data_knee = None
    if adj_fid_flat is not None and vh is not None:
        rmax = int(min(max_rank, n, vh.shape[0]))
        data_norm = float(np.linalg.norm(adj_fid_flat)) + 1e-16
        proj = adj_fid_flat @ vh[:rmax].conj().T            # <spectrum, v_k> per voxel
        col_energy = np.sum(np.abs(proj) ** 2, axis=0)
        cap = np.sqrt(np.cumsum(col_energy)) / data_norm    # ||P_r d|| / ||d||
        out["data_energy_curve"] = cap
        xk = np.arange(1, rmax + 1, dtype=np.float64)
        dn = np.hypot(cap[-1] - cap[0], xk[-1] - xk[0]) + 1e-30
        dk = np.abs((cap[-1] - cap[0]) * xk - (xk[-1] - xk[0]) * cap
                    + xk[-1] * cap[0] - cap[-1] * xk[0]) / dn
        data_knee = int(np.argmax(dk) + 1)
        out["data_knee"] = data_knee

    recommended = data_knee if data_knee is not None else out.get("trunc<0.001", out["elbow"])
    out["recommended"] = int(np.clip(recommended, 1, max_rank))
    return out


def spice_reconstruct_3d(adj_fid_flat, nufft_op, V, ww, shape_xyz, lamda,
                         n_iter=0, maxiter=120, gram=None):
    """3D SPICE reconstruction using exact Gram and optional regularization.

    ``gram`` may be a precomputed trajectory-only Gram (see ``_build_gram_3d``);
    pass it in to avoid rebuilding the Gram for every rank in a sweep. If None it
    is built here (backward compatible).
    """
    nx, ny, nz = shape_xyz
    n_vox = nx * ny * nz
    rank = V.shape[1]
    d_type = np.complex64

    if gram is None:
        print("[spice3d] Precomputing 3D Gram matrix ...")
        gram = _build_gram_3d(nufft_op, shape_xyz)

    b_mat = adj_fid_flat @ V
    gram_sys = gram.copy()
    if lamda > 0 and ww is not None:
        gram_sys += lamda * ww

    if n_iter > 0:
        print(f"[spice3d] Iterative CG solve (iters={n_iter}, rank={rank}, lambda={lamda}) ...")
        u = np.zeros((n_vox, rank), dtype=d_type)
        for k in range(rank):
            def mv(x, g=gram_sys):
                return (g @ x.reshape(-1, 1)).ravel().astype(d_type)
            a_op = LinearOperator((n_vox, n_vox), matvec=mv, dtype=d_type)
            u_k, _ = cg(a_op, b_mat[:, k], maxiter=n_iter)
            u[:, k] = u_k
    else:
        print(f"[spice3d] Direct solve ({n_vox}x{n_vox}, rank={rank}, lambda={lamda}) ...")
        u = lu_solve(lu_factor(gram_sys), b_mat).astype(d_type)

    n_spec = V.shape[0]
    spice_fid = (u @ V.conj().T).reshape(nx, ny, nz, n_spec)
    return spice_fid, u


def wsvd_combine(fid_mc, noise_cov=None, spec_window=None):
    """Per-voxel weighted-SVD coil combination in the frequency domain.

    spec_window : bool/index array over the n_spec (fftshifted) frequency axis,
        or None. If given, the per-voxel SVD that estimates the coil weights is
        computed only over those frequencies (spectral windowing); the weights
        are then applied to the full spectrum. None reproduces the original
        full-spectrum WSVD exactly.

    Verbatim port of the routine used by ``scripts/06c_iterative_recon_31p_3d.py``
    (OXSA ``svdReconstructFrequencyDomain`` per voxel).

    Parameters
    ----------
    fid_mc : (n_spec, n_ch, Nx, Ny, Nz) complex128 per-coil FID images.
    noise_cov : (n_ch, n_ch) or None. If None, identity/√0.5 scaling is used.

    Returns
    -------
    combined_fid : (n_spec, Nx, Ny, Nz) complex FID after wSVD combination.
    quality      : (Nx, Ny, Nz) float wSVD singular-value concentration
                   (1 = single dominant coil mode, 0 = flat spectrum).
    """
    n_spec, n_ch = fid_mc.shape[:2]
    spatial_shape = fid_mc.shape[2:]
    n_vox = int(np.prod(spatial_shape))

    spec_mc = np.fft.fftshift(np.fft.fft(fid_mc, axis=0), axes=0)
    spec_flat = spec_mc.reshape(n_spec, n_ch, n_vox)

    if noise_cov is None or n_ch == 1:
        scale_mat = np.eye(n_ch, dtype=np.complex128) * np.sqrt(0.5)
        inv_scale_mat = np.eye(n_ch, dtype=np.complex128) / np.sqrt(0.5)
    else:
        noise_cov = np.asarray(noise_cov, dtype=np.complex128)
        noise_val, noise_vec = np.linalg.eigh(noise_cov)
        noise_val = np.maximum(np.real(noise_val), 1e-30)
        scale_mat = noise_vec @ np.diag(np.sqrt(0.5) / np.sqrt(noise_val))
        inv_scale_mat = np.linalg.inv(scale_mat)

    combined_fd = np.zeros((n_spec, n_vox), dtype=np.complex128)
    quality = np.zeros(n_vox)
    sqrt_nch = np.sqrt(n_ch)

    for i in range(n_vox):
        raw = spec_flat[:, :, i]
        scaled = raw @ scale_mat                       # whitened full spectrum (n_spec, n_ch)
        # coil weights from the SVD; optionally restrict to a metabolite band
        # (spectral windowing, OpTIMUS-style) so weak peaks drive the weights
        # instead of noise-only regions. Weights are then applied to the FULL
        # spectrum: combined = scaled @ v1  (== u0*s0 when the window is all).
        src = scaled if spec_window is None else scaled[spec_window, :]
        u, s, vh = np.linalg.svd(src, full_matrices=False)
        s_norm = np.linalg.norm(s) + 1e-30
        quality[i] = float((s[0] / s_norm * sqrt_nch - 1) / (sqrt_nch - 1 + 1e-30))
        v1 = vh[0, :].conj()
        svd_coil_amp = v1 @ inv_scale_mat
        ref = svd_coil_amp[0]
        svd_rescale = np.linalg.norm(svd_coil_amp) * ref / (abs(ref) + 1e-30)
        combined_fd[:, i] = (scaled @ vh[0, :].conj()) * svd_rescale

    combined_fd = combined_fd.reshape((n_spec,) + spatial_shape)
    quality = quality.reshape(spatial_shape)
    combined_fid = np.fft.ifft(np.fft.ifftshift(combined_fd, axes=0), axis=0)
    return combined_fid, quality


def estimate_noise_cov_fid(fid_mc_flat, tail_frac=0.25):
    """Estimate the (n_ch, n_ch) noise covariance from the FID tail.

    fid_mc_flat : (n_ch, n_vox, n_spec) per-coil FID (image domain). The last
    ``tail_frac`` of each FID is signal-free (metabolites have decayed by the end
    of the readout), so those samples across all voxels estimate the receiver
    coil-coil noise covariance Psi = (1/M) N N^H used by WSVD noise whitening
    (Rodgers & Robson 2010). Whitening is what makes the rank-1 SVD optimal and
    stops heart-blind channels from diluting a dominant surface element.
    """
    n_ch, n_vox, n_spec = fid_mc_flat.shape
    t0 = int(round((1.0 - float(tail_frac)) * n_spec))
    t0 = min(max(t0, 0), n_spec - 1)
    noise = fid_mc_flat[:, :, t0:].reshape(n_ch, -1).astype(np.complex128)
    m = noise.shape[1]
    psi = (noise @ noise.conj().T) / max(m, 1)
    return psi


def spice_reconstruct_3d_multicoil(adj_fid_mc, nufft_op, V, ww, shape_xyz,
                                    lamda, n_iter=0, maxiter=120,
                                    noise_cov=None, gram=None):
    """Per-coil SPICE CG + wSVD coil combination (matches 06c ``iterative_recon_3d``).

    The Gram is trajectory-only (coil-independent) and precomputed once.
    Each coil is solved with its own RHS ``adj_fid_ch @ V``; the resulting
    per-coil FID images are stacked as (n_spec, n_ch, Nx, Ny, Nz) and passed
    through :func:`wsvd_combine` to yield the final single-channel FID volume.

    Parameters
    ----------
    adj_fid_mc : (n_ch, n_vox, n_spec) per-coil DCF-weighted adjoint FID.
    Returns spice_fid (Nx, Ny, Nz, n_spec) complex64, U (n_vox, rank) from
    the first coil (for diagnostic saving), and per-voxel wSVD quality.
    """
    nx, ny, nz = shape_xyz
    n_vox = nx * ny * nz
    n_ch, _, n_spec = adj_fid_mc.shape
    rank = V.shape[1]
    d_type = np.complex64

    if gram is None:
        print(f"[spice3d] Precomputing 3D Gram matrix (shared across {n_ch} coils) ...")
        gram = _build_gram_3d(nufft_op, shape_xyz)

    gram_sys = gram.copy()
    if lamda > 0 and ww is not None:
        gram_sys += lamda * ww

    # The system is coil-independent: factorize it ONCE for the direct path and
    # reuse the factorization for every coil, instead of re-factorizing inside
    # np.linalg.solve n_ch separate times.
    lu_piv = lu_factor(gram_sys) if n_iter <= 0 else None

    fid_mc = np.zeros((n_spec, n_ch, nx, ny, nz), dtype=np.complex128)
    u_first = None
    for ch in range(n_ch):
        b_mat = adj_fid_mc[ch] @ V
        if n_iter > 0:
            print(f"[spice3d] CG coil {ch + 1}/{n_ch} "
                  f"(rank={rank}, iters={n_iter}, lam={lamda}) ...")
            u_ch = np.zeros((n_vox, rank), dtype=d_type)
            for k in range(rank):
                def mv(x, g=gram_sys):
                    return (g @ x.reshape(-1, 1)).ravel().astype(d_type)
                a_op = LinearOperator((n_vox, n_vox), matvec=mv, dtype=d_type)
                u_k, _ = cg(a_op, b_mat[:, k], maxiter=n_iter)
                u_ch[:, k] = u_k
        else:
            print(f"[spice3d] Direct solve coil {ch + 1}/{n_ch} "
                  f"({n_vox}x{n_vox}, rank={rank}, lambda={lamda}, shared factorization) ...")
            u_ch = lu_solve(lu_piv, b_mat).astype(d_type)
        if ch == 0:
            u_first = u_ch
        spice_fid_ch = (u_ch @ V.conj().T).reshape(nx, ny, nz, n_spec)
        fid_mc[:, ch] = spice_fid_ch.transpose(3, 0, 1, 2)

    print(f"[spice3d] wSVD coil combination across {n_ch} coils ...")
    combined_fid, quality = wsvd_combine(fid_mc, noise_cov=noise_cov)
    spice_fid = combined_fid.transpose(1, 2, 3, 0).astype(d_type)
    print(f"[spice3d] wSVD quality: mean={float(np.mean(quality)):.3f}  "
          f"min={float(np.min(quality)):.3f}  max={float(np.max(quality)):.3f}")
    return spice_fid, u_first, quality


def find_best_snr_3d(image_spec):
    nx, ny, nz, n_spec = image_spec.shape
    snr_map = np.zeros((nx, ny, nz), dtype=np.float32)
    tail = slice(int(0.8 * n_spec), n_spec)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                s = np.abs(image_spec[x, y, z, :])
                snr_map[x, y, z] = s.max() / (np.std(s[tail]) + 1e-20)
    idx = np.unravel_index(np.argmax(snr_map), snr_map.shape)
    return idx[0], idx[1], idx[2], snr_map


def plot_results_3d(adj_spec, spice_spec, ppm, ppmlim, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    nx, ny, nz, _ = spice_spec.shape
    vx, vy, vz, snr_map = find_best_snr_3d(spice_spec)
    print(f"[spice3d] Best SNR voxel: [{vx},{vy},{vz}] SNR={snr_map[vx,vy,vz]:.1f}")

    pcr_win = (ppm >= -1.5) & (ppm <= 1.5)
    pi_win = (ppm >= 4.2) & (ppm <= 5.2)
    pcr_adj = np.max(np.abs(adj_spec[..., pcr_win]), axis=-1)
    pcr_sp = np.max(np.abs(spice_spec[..., pcr_win]), axis=-1)
    pi_sp = np.max(np.abs(spice_spec[..., pi_win]), axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].imshow(np.max(pcr_adj, axis=2).T, origin="lower", cmap="magma")
    axes[0].set_title("Adjoint PCr MIP")
    axes[1].imshow(np.max(pcr_sp, axis=2).T, origin="lower", cmap="magma")
    axes[1].set_title("SPICE PCr MIP")
    axes[2].imshow(np.max(pi_sp, axis=2).T, origin="lower", cmap="viridis")
    axes[2].set_title("SPICE Pi MIP")
    fig.savefig(os.path.join(fig_dir, "spice3d_mip_maps.png"), dpi=160)
    plt.close(fig)

    xlim = (float(max(ppmlim)), float(min(ppmlim)))
    fig, ax = plt.subplots(figsize=(10, 4))
    sp = np.abs(spice_spec[vx, vy, vz, :])
    ax.plot(ppm, sp, "k-", lw=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ppm")
    ax.set_ylabel("|spectrum|")
    ax.set_title(f"SPICE3D best voxel [{vx},{vy},{vz}]")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spice3d_best_voxel.png"), dpi=150)
    plt.close(fig)


def _dicom_view_axes(arr, params):
    """Reorient first two axes of ``arr`` into the DICOM row/col frame.

    ``params`` = (a_row, a_col, b_row, b_col) with the same meaning as in the
    prior-map block: a_row/a_col are dot products of the MRSI phase axis (x)
    with row_dir/col_dir; b_row/b_col are the y axis. Works for arrays of any
    dimensionality >= 2 by transposing/flipping only axes 0 and 1.
    """
    if params is None:
        return arr
    a_row, a_col, b_row, b_col = params
    out = arr
    if abs(a_row) >= abs(a_col):
        if a_row < 0:
            out = np.flip(out, axis=0)
        if b_col < 0:
            out = np.flip(out, axis=1)
    else:
        out = np.swapaxes(out, 0, 1)
        if b_row < 0:
            out = np.flip(out, axis=0)
        if a_col < 0:
            out = np.flip(out, axis=1)
    return out


def plot_spectrum_maps_3d(spec_vol, ppm, ppmlim, fig_dir, anatomy=None,
                          dicom_view_params=None, label="SPICE",
                          fname_prefix="spice", apodization_hz=None,
                          best_voxel=None, norm_mode="noise"):
    """Per-slice voxel-spectra grid, EXACTLY matching 06c's
    ``_plot_single_slice_spectra_grid`` layout and normalization.

    Axis convention (identical to 06c):
      * ``spec_vol`` shape is (Nx, Ny, Nz, n_spec)
      * grid cell (Nx-1-xi, yi) shows voxel (xi, yi)  → dim 0 vertical-flipped,
        dim 1 direct horizontal.
      * per-figure y-scale = 99.5th percentile of noise-normalized magnitudes.
      * best voxel (if provided) gets a red border.

    ``dicom_view_params`` is accepted for API compatibility but IGNORED —
    06c does not reorient. When ``anatomy`` is provided (native (Nx,Ny,Nz)
    grid, same axes as ``spec_vol``), an additional ``*_on_anatomy_z*.png``
    figure is saved with spectra overlaid on the anatomy background using
    the identical (Nx-1-xi, yi) convention.
    """
    del dicom_view_params
    os.makedirs(fig_dir, exist_ok=True)
    Nx, Ny, Nz, n_spec = spec_vol.shape
    ppm_min, ppm_max = float(min(ppmlim)), float(max(ppmlim))
    xlim = (ppm_max, ppm_min)
    tail = slice(int(0.8 * n_spec), n_spec)
    # 'global' mode: one raw-amplitude scale across the whole volume so the grid
    # reveals spatial localization (weak voxels stay flat). 'noise'/'peak' are
    # per-voxel and therefore show a peak in every voxel regardless of amplitude.
    global_top = float(np.percentile(np.abs(spec_vol), 99.9) + 1e-20) if norm_mode == "global" else None

    for z in range(Nz):
        fig, axes = plt.subplots(Nx, Ny, figsize=(Ny * 1.6, Nx * 1.25), squeeze=False)
        axes = np.array(axes).reshape(Nx, Ny)
        y_top = 0.0
        # Cache per-voxel normalized spectra so the anatomy overlay can reuse
        # the exact same curves with the exact same layout.
        s_cache = np.zeros((Nx, Ny, n_spec), dtype=np.float32)
        for yi in range(Ny):
            for xi in range(Nx):
                ax = axes[Nx - 1 - xi, yi]
                s = np.nan_to_num(np.abs(spec_vol[xi, yi, z, :]))
                if norm_mode == "global":
                    s_norm = s / global_top          # shared raw-amplitude scale
                elif norm_mode == "peak":
                    s_norm = s / (np.max(s) + 1e-20)
                else:
                    s_norm = s / (np.std(s[tail]) + 1e-20)
                s_cache[xi, yi] = s_norm
                y_top = max(y_top, float(np.percentile(s_norm, 99.5)))
                ax.plot(ppm, s_norm, color='black', lw=0.5)
                ax.set_xlim(xlim)
                ax.axis("off")
                if best_voxel is not None and best_voxel == (xi, yi, z):
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor("red")
                        spine.set_linewidth(1.2)
        if y_top <= 0:
            y_top = 1.0
        for yi in range(Ny):
            for xi in range(Nx):
                axes[Nx - 1 - xi, yi].set_ylim(0.0, y_top * 1.05)

        lb_txt = f" | LB={float(apodization_hz):.0f} Hz" if apodization_hz else ""
        fig.suptitle(
            f"{label} | slice {z + 1}/{Nz} | grid {Nx}x{Ny} | norm={norm_mode}"
            f"{lb_txt} | ppm [{ppm_min:.1f}, {ppm_max:.1f}]",
            fontsize=9,
        )
        fig.subplots_adjust(left=0.03, right=0.995, bottom=0.05, top=0.92,
                            wspace=0.22, hspace=0.22)
        fig.savefig(os.path.join(fig_dir, f"{fname_prefix}_all_spectra_z{z:02d}.png"),
                    dpi=170)
        plt.close(fig)

        # ── on-anatomy overlay: same (Nx-1-xi, yi) grid over localizer ─────
        if anatomy is None:
            continue
        anat_z = np.nan_to_num(anatomy[:, :, z])
        # Anatomy needs both vertical and horizontal flips (180° rotation)
        # relative to the raw MRSI grid to align with the spectra layout.
        anat_disp = anat_z[::-1, ::-1]
        fig = plt.figure(figsize=(Ny * 1.6, Nx * 1.25), facecolor='k')
        ax_bg = fig.add_axes([0, 0, 1, 1])
        ax_bg.imshow(anat_disp, cmap='gray', origin='lower',
                     extent=[0, Ny, 0, Nx],
                     vmin=0, vmax=max(float(anat_disp.max()), 1e-12))
        ax_bg.set_xlim(0, Ny)
        ax_bg.set_ylim(0, Nx)
        ax_bg.axis('off')

        pad = 0.025
        vw, vh = 1.0 / Ny, 1.0 / Nx
        for yi in range(Ny):
            for xi in range(Nx):
                ax = fig.add_axes([
                    yi * vw + pad * vw, xi * vh + pad * vh,
                    vw * (1 - 2 * pad), vh * (1 - 2 * pad),
                ])
                ax.plot(ppm, s_cache[xi, yi], color='white', lw=0.55)
                ax.set_xlim(xlim)
                ax.set_ylim(0.0, y_top * 1.05)
                ax.patch.set_alpha(0.0)
                ax.axis('off')
                if best_voxel is not None and best_voxel == (xi, yi, z):
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor('red')
                        spine.set_linewidth(1.2)

        fig.suptitle(
            f"{label} on anatomy | slice {z + 1}/{Nz} | norm={norm_mode}{lb_txt}",
            color='white', fontsize=10,
        )
        fig.savefig(os.path.join(fig_dir, f"{fname_prefix}_on_anatomy_z{z:02d}.png"),
                    dpi=150, facecolor='k')
        plt.close(fig)


def parse_args():
    _epilog = (
        "Examples\n"
        "--------\n"
        "Interleaved 1H/31P (e.g. MID00073):\n"
        "  python scripts/05d_spice_recon_31p_3d.py \\\n"
        "    --dat meas_MID00073_..._30r.dat \\\n"
        "    --out-dir output_MID00073_05d \\\n"
        "    --subspace-src pk-7t-liver-3tpulse \\\n"
        "    --localizer /path/to/localizer_TRA_29_None \\\n"
        "    --roi-mask roi_mask_MID00073_liver.npy \\\n"
        "    --roi-strength strong \\\n"
        "    --rank 12 --lambda1 0.01 --n-iter 100 --apodization 20\n"
        "\n"
        "Pure 31P (e.g. MID00069, BS_30R_1000Hz):\n"
        "  python scripts/05d_spice_recon_31p_3d.py \\\n"
        "    --dat meas_MID00069_..._BS_30R_1000Hz.dat \\\n"
        "    --out-dir output_MID00069_05d \\\n"
        "    --subspace-src pk-7t-liver-3tpulse \\\n"
        "    --localizer /path/to/localizer_TRA_29_None \\\n"
        "    --roi-mask roi_mask_MID00073_liver.npy \\\n"
        "    --roi-strength strong \\\n"
        "    --rank 12 --lambda1 0.01 --n-iter 100 --apodization 20\n"
        "\n"
        "The loader auto-detects the scan type and prints one of:\n"
        "  'Scan type: phasecor (interleaved 1H/31P)' or\n"
        "  'Scan type: image (pure 31P — no phasecor found)'\n"
    )
    p = argparse.ArgumentParser(
        description="31P CONCEPT 3D CRT SPICE reconstruction "
                    "(auto-detects interleaved vs pure 31P scans)",
        epilog=_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dat", required=True, help="Siemens Twix .dat file")
    p.add_argument("--out-dir", default="./output_spice3d")
    p.add_argument("--cache-dir", default=None,
                   help="Reuse the 07b parsed-twix cache (same as 07b/08) to skip re-parsing")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--extra-shift-ro", type=float, default=0.0)
    p.add_argument("--extra-shift-pe", type=float, default=0.0)
    p.add_argument("--extra-shift-sl", type=float, default=0.0)
    p.add_argument("--apodization", type=float, default=0.0)
    p.add_argument("--rank", type=int, nargs="+", default=[15])
    p.add_argument("--auto-rank", action="store_true",
                   help="Ignore --rank and reconstruct only at the auto-suggested "
                        "model order (0.1 pct singular-value truncation / data-projection knee).")
    p.add_argument("--lambda1", type=float, nargs="+", default=[0.0],
                   help="Graph-regularizer weight(s). Several values -> lambda sweep sharing one "
                        "adjoint + Gram; outputs are then tagged rank<r>_lam<l>")
    p.add_argument("--n-iter", type=int, default=0, help="0=direct solve, >0=iterative")
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--adj", type=int, choices=[6, 18, 26], default=6,
                   help="3D graph neighbourhood: 6=faces, 18=+edges, 26=+corners")
    p.add_argument("--training-size", type=int, default=5000)
    p.add_argument("--subspace-src", default="pk-7t-skeletal",
                   choices=["adjoint", "synthetic", "pk2pi", "pk-7t-skeletal",
                            "pk-7t-liver-3tpulse", "pk-cardiac", "pk-3t-brain", "basis-dat"],
                   help="Spectral subspace prior; pk-7t-skeletal follows PK_7T_Skeletal_voigt.m, "
                        "pk-7t-liver-3tpulse follows PK_7T_Liver_3TPulse.m, pk-cardiac follows "
                        "PK_3T_Cardiac.m (PCr/ATP/PDE/Pi + 2,3-DPG blood), pk-3t-brain follows "
                        "PK_3T_Brain_v1.m (ATP/PCr/GPC/GPE/Pi/PME/NADH), all OXSA AMARES")
    p.add_argument("--pk-phase", choices=["random", "shared"], default="random",
                   help="pk-3t-brain only: 'random' = independent phase per metabolite; "
                        "'shared' = one zero-order phase per FID + common delay --pk-delay-ms")
    p.add_argument("--pk-delay-ms", type=float, default=0.0,
                   help="pk-3t-brain --pk-phase shared: acquisition delay (ms), same sign as the "
                        "first-order correction spec*exp(-2i*pi*f*tau) (MID00456: -0.64)")
    p.add_argument("--pk-lw-floor", type=float, nargs=2, default=None, metavar=("PCR_HZ", "ATP_HZ"),
                   help="pk-3t-brain only: minimum linewidth (Hz) for PCr and for ATP/NADH in the "
                        "training prior (default: OXSA-derived 4 Hz). e.g. 7 12")
    p.add_argument("--dcf", choices=["crt2d", "perkz3d"], default="crt2d",
                   help="crt2d: legacy 2D CRT DCF (rings pooled by radius) x xy Hamming. "
                        "perkz3d: per-kz-level annulus x kz spacing DCF x spherical Hamming, "
                        "rescaled to the crt2d total weight so lambda1 stays comparable "
                        "(use for variable-ring-per-kz 3D CRT)")
    p.add_argument("--prior-npy", default=None,
                   help="Precomputed spatial prior (Nx,Ny,Nz) .npy in the NATIVE recon frame "
                        "(e.g. a resliced anatomical image); replaces the localizer / "
                        "adjoint-magnitude prior")
    p.add_argument("--basis-dat", default=None)
    p.add_argument("--pi-ph-range", action=argparse.BooleanOptionalAction, default=True,
                   help="pk2pi only: model Pi as a SINGLE resonance whose center is sampled "
                        "across the pH-dependent shift range 3.9-5.2 ppm (default). "
                        "Use --no-pi-ph-range for the legacy fixed Pi+Pi2 doublet.")
    p.add_argument("--localizer", default=None,
                   help="DICOM localizer folder used as the 3D spatial prior")
    p.add_argument("--roi-mask", default=None,
                   help="Optional manual ROI mask .npy (bool, shape Nx x Ny x Nz)")
    p.add_argument("--roi-strength", choices=sorted(ROI_STRENGTH_PRESETS), default="strong",
                   help="How hard the ROI is enforced in the graph prior: "
                        "off (mask ignored by graph) | soft (edge_scale=10) | "
                        "strong (edge_scale=100, default) | strict (cross-boundary edges zeroed). "
                        "All other prior settings are fixed at the top of this file.")
    p.add_argument("--roi-display-attenuation", type=float, default=1.0,
                   help="POST-HOC display-only scaling of SPICE voxels outside the ROI, "
                        "applied after reconstruction (1.0 = disabled, the default). "
                        "This is cosmetic masking, NOT a reconstruction constraint — "
                        "do not use it for quantitative localization claims. "
                        "The adjoint is never attenuated.")
    p.add_argument("--out-rot90", type=int, default=0,
                   help="Final in-plane output rotation (default 0 = native/header-derived, matching 08)")
    p.add_argument("--out-flip-lr", action=argparse.BooleanOptionalAction, default=False,
                   help="Final left-right output flip (default off = native, matching 08)")
    p.add_argument("--rep", type=int, default=None)
    p.add_argument("--rep-average", action="store_true",
                   help="Average all reps' k-space before recon (√N SNR). Overrides --rep.")
    p.add_argument("--coil", type=int, default=None,
                   help="Reconstruct a SINGLE receive channel (1-indexed); skips wSVD "
                        "coil combination. For inspecting one coil's data.")
    p.add_argument("--wsvd-whiten", action=argparse.BooleanOptionalAction, default=True,
                   help="WSVD noise whitening (Rodgers & Robson, default ON): estimate the "
                        "coil-coil noise covariance from the FID tail and whiten before the "
                        "rank-1 SVD. Stops heart-blind channels diluting a dominant surface "
                        "coil. Use --no-wsvd-whiten for the legacy unwhitened behaviour.")
    p.add_argument("--wsvd-noise-tail", type=float, default=0.25,
                   help="Fraction of the FID tail treated as signal-free for the WSVD "
                        "noise covariance (default 0.25).")
    p.add_argument("--noise-cov", default=None,
                   help="Path to a precomputed (n_ch x n_ch) complex noise covariance .npy "
                        "(e.g. from the receiver noise prescan). When given, it is used for "
                        "WSVD whitening instead of the FID-tail estimate (--wsvd-noise-tail).")
    p.add_argument("--wsvd-window", type=float, nargs=2, default=None,
                   metavar=("PPM_LO", "PPM_HI"),
                   help="WSVD spectral windowing (OpTIMUS-style): estimate coil weights "
                        "from the SVD over this ppm band only, then apply to the full "
                        "spectrum. Omit for full-spectrum WSVD.")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--spectra-norm", choices=["global", "noise", "peak"], default="global",
                   help="Spectra-grid normalization: 'global' (one amplitude scale, "
                        "shows spatial localization; default), 'noise' (per-voxel SNR), "
                        "'peak' (per-voxel peak).")
    p.add_argument("--adjoint-only", action="store_true")
    p.add_argument("--coil-combine", choices=["wsvd-per-coil", "phase-align-avg"],
                   default="wsvd-per-coil",
                   help=("Coil combination strategy. 'wsvd-per-coil' (default, matches "
                         "06c): run SPICE CG per coil (shared Gram) then per-voxel "
                         "wSVD in the frequency domain. 'phase-align-avg': legacy k-space "
                         "phase-aligned average before recon (single-coil forward model)."))
    p.add_argument("--matrix-size", type=int, default=None,
                   help="Override in-plane matrix size Nx=Ny (default: header lBaseResolution)")
    p.add_argument("--matrix-size-z", type=int, default=None,
                   help="Override slab matrix size Nz (default: header partition/segment count)")
    return p.parse_args()


def _compute_adjoint_flat(info, kdata_flat, traj, n_spec, nxyz, dcf):
    nx, ny, nz = nxyz
    nufft_op = mrinufft.get_operator("finufft", samples=traj, shape=(nx, ny, nz), n_coils=1, density=dcf)
    adj_fid_flat = np.zeros((nx * ny * nz, n_spec), dtype=np.complex64)
    for t in tqdm(range(n_spec), desc="Adjoint", miniters=100):
        adj_fid_flat[:, t] = nufft_op.adj_op(kdata_flat[:, t]).reshape(nx * ny * nz)
    return adj_fid_flat, nufft_op


def _compute_adjoint_multicoil(kdata_flat_mc, traj, n_spec, nxyz, dcf):
    """Per-coil DCF-weighted adjoint (matches 06c ``adj_dcf_mc`` step).

    Parameters
    ----------
    kdata_flat_mc : (n_kpts, n_spec, n_ch) complex k-space data.

    Returns
    -------
    adj_fid_mc : (n_ch, n_vox, n_spec) complex64.
    nufft_op   : single-coil finufft operator with the same DCF applied.
    """
    nx, ny, nz = nxyz
    n_vox = nx * ny * nz
    n_ch = int(kdata_flat_mc.shape[2])
    nufft_op = mrinufft.get_operator("finufft", samples=traj, shape=(nx, ny, nz),
                                     n_coils=1, density=dcf)
    adj_fid_mc = np.zeros((n_ch, n_vox, n_spec), dtype=np.complex64)
    for ch in range(n_ch):
        for t in tqdm(range(n_spec), desc=f"Adjoint ch{ch + 1}/{n_ch}",
                      miniters=100, leave=False):
            adj_fid_mc[ch, :, t] = nufft_op.adj_op(kdata_flat_mc[:, t, ch]).reshape(n_vox)
    return adj_fid_mc, nufft_op


def combine_coils_phase_aligned(kdata):
    """Phase-align all receive coils to the strongest coil and average them.

    The scalar sensitivity estimate is robustly estimated over all acquired
    samples and spectral timepoints. This prevents destructive cancellation
    while retaining all coils for the single-coil SPICE forward model.
    """
    flattened = kdata.transpose(3, 0, 2, 1).reshape(-1, kdata.shape[2], kdata.shape[1])
    powers = np.mean(np.abs(flattened) ** 2, axis=(0, 2))
    ref_idx = int(np.argmax(powers))
    reference = flattened[:, ref_idx, :].reshape(-1)
    denom = np.vdot(reference, reference).real + 1e-20
    gains = np.array([
        np.vdot(reference, flattened[:, ch, :].reshape(-1)) / denom
        for ch in range(kdata.shape[2])
    ])
    gains[np.abs(gains) < 1e-12] = 1.0
    combined = np.mean(flattened / gains[np.newaxis, :, np.newaxis], axis=1)
    return combined.reshape(-1, kdata.shape[1]), ref_idx, gains


def main():
    args = parse_args()
    data_dir = os.path.join(args.out_dir, "data")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    _ldk = dict(rep_override=args.rep, rep_average=args.rep_average)
    if _cached_load is not None and getattr(args, "cache_dir", None):
        _kx = {}
        if args.rep is not None:
            _kx["rep_override"] = args.rep
        if args.rep_average:
            _kx["rep_average"] = True
        info = _cached_load(load_interleaved_31p, args.dat, cache_dir=args.cache_dir,
                            key_extras=_kx, refresh=args.refresh_cache, tag="31p3d", **_ldk)
    else:
        info = load_interleaved_31p(args.dat, rep_override=args.rep, rep_average=args.rep_average)
    kdata = info["kdata"]
    info["kdata"] = None   # free the dict's ref; kdata is corrected in place below
    if args.matrix_size is not None:
        print(f"[spice3d] Overriding matrix_size {info['matrix_size']} -> {args.matrix_size}")
        info["matrix_size"] = int(args.matrix_size)
    if args.matrix_size_z is not None:
        print(f"[spice3d] Overriding Nz {info['Nz']} -> {args.matrix_size_z}")
        info["Nz"] = int(args.matrix_size_z)
    nx = int(info["matrix_size"])
    ny = int(info["matrix_size"])
    nz = int(info["Nz"])
    n_spec = int(info["n_spec"])
    n_ch = int(kdata.shape[2])

    if args.coil is not None:
        _ci = int(args.coil) - 1
        if not (0 <= _ci < n_ch):
            sys.exit(f"--coil {args.coil} out of range (1..{n_ch})")
        kdata = kdata[:, :, _ci:_ci + 1, :].copy()
        n_ch = 1
        print(f"[spice3d] SINGLE-COIL recon: channel {args.coil} (index {_ci}) only — no wSVD")

    if n_ch < 1:
        raise RuntimeError("No channels found in kdata")

    print(f"[spice3d] Grid={nx}x{ny}x{nz}, n_spec={n_spec}, rings={info['n_rings']}, ch={n_ch}")

    if not args.no_phase_corr:
        for ch in range(n_ch):
            kdata[:, :, ch:ch + 1, :] = concept_phase_correction(
                kdata[:, :, ch:ch + 1, :], info["_ti_per_ring"])

    traj = build_trajectory_3d(
        info["first_circle"],
        info["n_rings_total"],
        info["traj_pts"],
        info["kz_cpp_per_ring"],
        info["gamma_2pi"] / 1e9,
        info["fov_m"],
        nx,
    )
    # Reconstruct on a FIXED PHYSICAL FOV (matching MATLAB CONCEPTPerform2DDirectFT):
    # rescale the max-normalized in-plane trajectory (outer ring -> ±0.5) back to
    # physical cycles/pixel (outer -> k_norm_max_phys < 0.5). With max-norm the recon
    # FOV = nominal*(0.5/k_norm_max) is ring-count dependent, magnifying low-ring
    # scans and shifting an off-centre source by ~1 voxel between different ring
    # counts; physical scaling makes the FOV/voxel registration ring-count invariant.
    # kz is already cycles/pixel, so leave it. build_trajectory_3d (imported from 06c)
    # is unchanged; only this recon's local copy is rescaled.
    traj = traj.copy()
    traj[:, 0] *= float(info["k_norm_max_phys"]) / 0.5
    traj[:, 1] *= float(info["k_norm_max_phys"]) / 0.5

    if n_ch == 1:
        kdata_flat = kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
        coil_report = "single receive channel"
        use_per_coil = False
    elif args.coil_combine == "wsvd-per-coil":
        # (n_kpts, n_spec, n_ch): keep coils for per-coil SPICE + wSVD (06c style)
        kdata_flat = kdata.transpose(3, 0, 2, 1).reshape(-1, n_ch, n_spec).transpose(0, 2, 1)
        coil_report = f"per-coil SPICE + wSVD combination ({n_ch} coils, 06c-style)"
        use_per_coil = True
        print(f"[spice3d] Coil combination: {coil_report}")
    else:
        kdata_flat, coil_ref, coil_gains = combine_coils_phase_aligned(kdata)
        coil_report = f"phase-aligned {n_ch}-coil average; reference coil={coil_ref}"
        use_per_coil = False
        print(f"[spice3d] Coil combination: {coil_report}")
        np.save(os.path.join(data_dir, "coil_phase_gains.npy"), coil_gains.astype(np.complex64))

    del kdata   # all branches derived kdata_flat; free the 4D (pts,spec,ch,rings) array

    if not args.no_fov_shift:
        d_ro, d_pe, d_sl = compute_fov_shift_3d(info)
        d_ro += args.extra_shift_ro
        d_pe += args.extra_shift_pe
        d_sl += args.extra_shift_sl
        shift_phase = np.exp(-2j * np.pi * (traj[:, 0] * d_pe + traj[:, 1] * d_ro + traj[:, 2] * (-d_sl)))
        if use_per_coil:
            kdata_flat = kdata_flat * shift_phase[:, np.newaxis, np.newaxis]
        else:
            kdata_flat = kdata_flat * shift_phase[:, np.newaxis]
        print(f"[spice3d] FOV shift: ro={d_ro:.4f}, pe={d_pe:.4f}, kz(sl)={-d_sl:.4f} vox")

    r_xy = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
    ham_radius = r_xy   # traj already physical cyc/pix (rescaled above)
    dcf = compute_crt_dcf_hamming(traj, hamming_radius=ham_radius)
    if args.dcf == "perkz3d":
        _dcf3 = compute_crt_dcf_perkz(traj, np.tile(info["traj_pts"], info["n_rings_total"] // len(info["traj_pts"])))
        dcf = (_dcf3 * (float(dcf.sum()) / float(_dcf3.sum()))).astype(np.float32)
        print("[spice3d] DCF mode: perkz3d (per-kz annulus x kz spacing x spherical Hamming)")
    print(f"[spice3d] DCF weights: min={dcf.min():.4f}, max={dcf.max():.4f}")

    if use_per_coil:
        # Per-coil DCF adjoint (matches 06c). wSVD-combined adjoint is used
        # for downstream saving/plotting so shapes stay identical to the
        # single-coil path.
        adj_fid_mc, nufft_op = _compute_adjoint_multicoil(
            kdata_flat, traj, n_spec, (nx, ny, nz), dcf)
        # WSVD noise whitening: estimate the coil-coil noise covariance from the
        # per-coil FID tail (signal-free) so heart-blind channels don't dilute a
        # dominant surface element. None -> legacy identity (unwhitened) behaviour.
        _wsvd_noise_cov = None
        if args.wsvd_whiten:
            if args.noise_cov is not None:
                _wsvd_noise_cov = np.load(args.noise_cov).astype(np.complex128)
                if _wsvd_noise_cov.shape != (n_ch, n_ch):
                    sys.exit(f"[spice3d] --noise-cov shape {_wsvd_noise_cov.shape} "
                             f"!= (n_ch, n_ch)=({n_ch},{n_ch})")
                _cn = float(np.linalg.cond(_wsvd_noise_cov))
                print(f"[spice3d] WSVD noise whitening ON: noise cov from PRESCAN "
                      f"{args.noise_cov} (cond={_cn:.1f})")
            else:
                _wsvd_noise_cov = estimate_noise_cov_fid(adj_fid_mc, tail_frac=args.wsvd_noise_tail)
                _cn = float(np.linalg.cond(_wsvd_noise_cov))
                print(f"[spice3d] WSVD noise whitening ON: noise cov from last "
                      f"{args.wsvd_noise_tail*100:.0f}% of FID (cond={_cn:.1f})")
        _wsvd_window = None
        if args.wsvd_window is not None:
            _freq = np.fft.fftshift(np.fft.fftfreq(n_spec, d=info["spectral_dwell"]))
            _ppm_axis = -_freq / info["larmor_freq_mhz"] + args.ppm_center
            _wlo, _whi = min(args.wsvd_window), max(args.wsvd_window)
            _wsvd_window = (_ppm_axis >= _wlo) & (_ppm_axis <= _whi)
            print(f"[spice3d] WSVD spectral window: {_wlo:.1f}..{_whi:.1f} ppm "
                  f"({int(_wsvd_window.sum())}/{n_spec} pts)")
        print(f"[spice3d] Per-coil adjoint done. wSVD-combining {n_ch} coils for display ...")
        adj_fid_mc_arr = adj_fid_mc.reshape(n_ch, nx, ny, nz, n_spec).transpose(4, 0, 1, 2, 3)
        adj_fid_combined, _ = wsvd_combine(adj_fid_mc_arr.astype(np.complex128),
                                           noise_cov=_wsvd_noise_cov,
                                           spec_window=_wsvd_window)
        adj_fid_img = adj_fid_combined.transpose(1, 2, 3, 0).astype(np.complex64)
        adj_fid_flat = adj_fid_img.reshape(nx * ny * nz, n_spec)  # used by subspace 'adjoint' src
    else:
        adj_fid_flat, nufft_op = _compute_adjoint_flat(info, kdata_flat, traj, n_spec, (nx, ny, nz), dcf)
        adj_fid_img = adj_fid_flat.reshape(nx, ny, nz, n_spec)
        adj_fid_mc = None

    # Even-N centering correction for the DISPLAY adjoint (matches 07b/08). The SPICE
    # solve keeps adj_fid_flat + the Gram in the native recon frame; only the
    # image-domain outputs (adj_fid_img here, spice_fid later) get the one-voxel roll.
    adj_fid_img = _even_n_roll(adj_fid_img)

    # Apply Gaussian apodization to the adjoint FID too, matching 06c so the
    # adjoint spectra plot is directly comparable with the SPICE spectra plot.
    if args.apodization > 0:
        _dwell = info["spectral_dwell"]
        _t = np.arange(n_spec) * _dwell
        _apo = np.exp(-np.pi ** 2 * args.apodization ** 2 * _t ** 2 / (4 * np.log(2)))
        adj_fid_img = adj_fid_img * _apo
    adj_spec = FIDToSpec(adj_fid_img.conj(), axis=-1)
    adj_fid_output = _apply_output_xy_orientation(adj_fid_img, args.out_rot90, args.out_flip_lr)
    adj_spec_output = _apply_output_xy_orientation(adj_spec, args.out_rot90, args.out_flip_lr)

    np.save(os.path.join(data_dir, "adj_fid.npy"), adj_fid_output.astype(np.complex64))
    np.save(os.path.join(data_dir, "adj_spec.npy"), adj_spec_output.astype(np.complex64))

    localizer_prior = None
    selected_slices = None
    dicom_view_params = None
    if args.localizer:
        localizer_prior, selected_slices = load_localizer_prior_3d(
            args.localizer, info, (nx, ny, nz),
            percentiles=LOCALIZER_PERCENTILES,
            gamma=LOCALIZER_GAMMA,
            flip_native_y=PRIOR_FLIP_NATIVE_Y,
        )
        if PRIOR_GAMMA != 1.0:
            localizer_prior = np.power(np.clip(localizer_prior, 0.0, 1.0), PRIOR_GAMMA).astype(np.float32)
        # Keep native geometry for the reconstruction operator, then orient the
        # saved/displayed prior identically to the saved spectral volumes.
        localizer_prior_output = _apply_output_xy_orientation(
            localizer_prior, args.out_rot90, args.out_flip_lr)
        np.save(os.path.join(data_dir, "localizer_spatial_prior_native_3d.npy"), localizer_prior)
        np.save(os.path.join(data_dir, "localizer_spatial_prior_3d.npy"), localizer_prior_output)
        with open(os.path.join(args.out_dir, "localizer_slice_mapping.txt"), "w", encoding="ascii") as f:
            f.write("MRSI z | localizer sorted index | plane distance mm | MRSI center LPS mm | localizer center LPS mm | DICOM path\n")
            for z, match in enumerate(selected_slices):
                mrs_center = ",".join(f"{v:.3f}" for v in match["mrsi_center"])
                loc_center = ",".join(f"{v:.3f}" for v in match["localizer_center"])
                f.write(f"{z} | {match['index']} | {match['distance_mm']:.4f} | {mrs_center} | {loc_center} | {match['path']}\n")

        # Equivalent to the 05c green voxel-level sum-prior map, now projected
        # across all 3-D MRSI slices. It is the actual spatial prior used in SPICE.
        # We reorient it into the DICOM row/col frame of the reference localizer
        # slice so it visually matches localizer_fov_center_registration.png.
        #
        # Geometry: load_localizer_prior_3d samples at
        #     world = center + (x - cx) * phase + (y - cy) * read
        # and (by default) flips axis y so that in the returned prior, x runs
        # along +phase and y runs along -read. The registration figure uses
        # origin="upper" with vertical=row_dir, horizontal=col_dir. We map:
        #     display rows := axis of (phase, -read) most aligned with row_dir
        #     display cols := axis of (phase, -read) most aligned with col_dir
        # and flip whichever axis has a negative dot product.
        _normal = np.array([info["snv_x"], info["snv_y"], info["snv_z"]], dtype=np.float64)
        _normal /= np.linalg.norm(_normal) + 1e-20
        _phase_vec, _read_vec = _fGSLCalcPRS(_normal, info["in_plane_rotation"])
        _phase_vec = _phase_vec / (np.linalg.norm(_phase_vec) + 1e-20)
        _read_vec = _read_vec / (np.linalg.norm(_read_vec) + 1e-20)
        # In the returned prior array, axis 1 (y) advances along -read after
        # flip_native_y=True, and along +read otherwise.
        _y_axis_vec = (-_read_vec) if PRIOR_FLIP_NATIVE_Y else _read_vec
        _ref_rec = selected_slices[0]["record"]
        _row_dir = _ref_rec["row_dir"]
        _col_dir = _ref_rec["col_dir"]
        _a_row = float(np.dot(_phase_vec, _row_dir))   # phase vs vertical
        _a_col = float(np.dot(_phase_vec, _col_dir))   # phase vs horizontal
        _b_row = float(np.dot(_y_axis_vec, _row_dir))  # y  vs vertical
        _b_col = float(np.dot(_y_axis_vec, _col_dir))  # y  vs horizontal
        dicom_view_params = (_a_row, _a_col, _b_row, _b_col)

        def _prior_to_dicom_view(arr2d):
            """Reorient a (nx, ny) prior slice into the DICOM row/col frame
            of the reference localizer plane so it can be shown next to
            localizer_fov_center_registration.png with origin='upper'.
            """
            if abs(_a_row) >= abs(_a_col):
                # phase axis (x) is closer to row_dir -> keep x on rows.
                out = arr2d                                # rows=x, cols=y
                if _a_row < 0:
                    out = out[::-1, :]
                if _b_col < 0:
                    out = out[:, ::-1]
                return out, ("MRSI x (phase)", "MRSI y (read)")
            else:
                # phase axis (x) is closer to col_dir -> put x on cols, y on rows.
                out = arr2d.T                              # rows=y, cols=x
                if _b_row < 0:
                    out = out[::-1, :]
                if _a_col < 0:
                    out = out[:, ::-1]
                return out, ("MRSI y (read)", "MRSI x (phase)")

        summed_prior = localizer_prior.sum(axis=2)
        summed_view, (ylab, xlab) = _prior_to_dicom_view(summed_prior)
        fig_sum, ax_sum = plt.subplots(figsize=(7, 6))
        im_sum = ax_sum.imshow(summed_view, origin="upper", cmap="Greens")
        ax_sum.set_title("Voxel-level summed spatial prior (DICOM row/col frame)")
        ax_sum.set_xlabel(xlab)
        ax_sum.set_ylabel(ylab)
        fig_sum.colorbar(im_sum, ax=ax_sum, label="sum of normalized 3-D prior")
        fig_sum.tight_layout()
        fig_sum.savefig(os.path.join(fig_dir, "spatial_prior_map.png"), dpi=180)
        plt.close(fig_sum)

        # Green maps for all individual spectral slices, matching the input to
        # the 3-D graph regularizer rather than only showing gray anatomy.
        # Same DICOM row/col reorientation as the summed map.
        fig_green, axes_green = plt.subplots(2, int(np.ceil(nz / 2)), figsize=(18, 6), squeeze=False)
        for z, ax in enumerate(axes_green.ravel()):
            if z < nz:
                view_z, (ylab_z, xlab_z) = _prior_to_dicom_view(localizer_prior[:, :, z])
                im_green = ax.imshow(view_z, origin="upper", cmap="Greens", vmin=0, vmax=1)
                ax.set_title(f"MRSI z={z}: localizer {selected_slices[z]['index']}", fontsize=8)
                fig_slice, ax_slice = plt.subplots(figsize=(6, 5))
                im_slice = ax_slice.imshow(view_z, origin="upper", cmap="Greens", vmin=0, vmax=1)
                ax_slice.set_title(f"Green spatial prior: MRSI z={z}, localizer {selected_slices[z]['index']}")
                ax_slice.set_xlabel(xlab_z)
                ax_slice.set_ylabel(ylab_z)
                fig_slice.colorbar(im_slice, ax=ax_slice, label="normalized prior")
                fig_slice.tight_layout()
                fig_slice.savefig(os.path.join(fig_dir, f"spatial_prior_map_z{z:02d}.png"), dpi=180)
                plt.close(fig_slice)
            ax.axis("off")
        fig_green.colorbar(im_green, ax=axes_green.ravel().tolist(), shrink=0.8, label="normalized prior")
        fig_green.suptitle("Green voxel-level localizer spatial prior used by 3-D SPICE (DICOM row/col frame)")
        fig_green.tight_layout()
        fig_green.savefig(os.path.join(fig_dir, "spatial_prior_map_slices.png"), dpi=180)
        plt.close(fig_green)

        # This is the direct visual registration check in the final MRSI array
        # convention (k=3 + LR flip), unlike the native-DICOM FOV figure below.
        # Gray anatomy and green constraint therefore share precisely the same
        # pixels and cannot appear vertically reversed due to `imshow` origins.
        fig_oriented, axes_oriented = plt.subplots(2, int(np.ceil(nz / 2)), figsize=(18, 6), squeeze=False)
        for z, ax in enumerate(axes_oriented.ravel()):
            if z < nz:
                anatomy = localizer_prior_output[:, :, z].T
                ax.imshow(anatomy, origin="lower", cmap="gray", vmin=0, vmax=1)
                ax.imshow(anatomy, origin="lower", cmap="Greens", vmin=0, vmax=1, alpha=0.65)
                ax.contour(anatomy, levels=[0.20, 0.50, 0.80], colors=["lime"], linewidths=0.55, origin="lower")
                ax.set_title(f"Oriented MRSI z={z}: localizer {selected_slices[z]['index']}", fontsize=8)
            ax.axis("off")
        fig_oriented.suptitle("Final MRSI orientation: gray anatomy + green spatial constraint")
        fig_oriented.tight_layout()
        fig_oriented.savefig(os.path.join(fig_dir, "localizer_prior_registration_oriented.png"), dpi=180)
        plt.close(fig_oriented)

        # Direct registration check: localizer in native pixels, with the
        # sampled MRSI 300 mm FOV polygon and its physical center overlaid.
        fig_reg, axes_reg = plt.subplots(2, int(np.ceil(nz / 2)), figsize=(18, 6), squeeze=False)
        for z, ax in enumerate(axes_reg.ravel()):
            if z >= nz:
                ax.axis("off")
                continue
            match = selected_slices[z]
            rec = match["record"]
            raw = rec["img"]
            lo, hi = np.percentile(raw, LOCALIZER_PERCENTILES)
            ax.imshow(np.clip((raw - lo) / (hi - lo + 1e-12), 0, 1), cmap="gray", origin="upper")
            corner_rows = [match["rows"][0, 0], match["rows"][-1, 0], match["rows"][-1, -1], match["rows"][0, -1], match["rows"][0, 0]]
            corner_cols = [match["cols"][0, 0], match["cols"][-1, 0], match["cols"][-1, -1], match["cols"][0, -1], match["cols"][0, 0]]
            ax.plot(corner_cols, corner_rows, color="deepskyblue", linewidth=1.2, label="300 mm MRSI FOV")
            center_delta = match["mrsi_center"] - rec["ipp"]
            center_row = np.dot(center_delta, rec["row_dir"]) / rec["sp_row"]
            center_col = np.dot(center_delta, rec["col_dir"]) / rec["sp_col"]
            ax.plot(center_col, center_row, marker="+", color="lime", markersize=9, markeredgewidth=1.5, label="MRSI center")
            ax.set_title(f"z={z}, loc={match['index']}, plane d={match['distance_mm']:.1f} mm", fontsize=8)
            ax.axis("off")
        handles, labels = axes_reg[0, 0].get_legend_handles_labels()
        fig_reg.legend(handles, labels, loc="upper center", ncol=2)
        fig_reg.suptitle("Localizer registration check: 300 mm MRSI FOV and center in native DICOM coordinates")
        fig_reg.tight_layout(rect=(0, 0, 1, 0.91))
        fig_reg.savefig(os.path.join(fig_dir, "localizer_fov_center_registration.png"), dpi=180)
        plt.close(fig_reg)

        fig_prior, axes_prior = plt.subplots(2, int(np.ceil(nz / 2)), figsize=(18, 6), squeeze=False)
        for z, ax in enumerate(axes_prior.ravel()):
            if z < nz:
                ax.imshow(localizer_prior_output[:, :, z].T, origin="lower", cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"MRSI z={z}: localizer {selected_slices[z]['index']}", fontsize=8)
            ax.axis("off")
        fig_prior.suptitle("Slice-matched localizer spatial prior (contrast adjusted)")
        fig_prior.tight_layout()
        fig_prior.savefig(os.path.join(fig_dir, "localizer_spatial_prior_slices.png"), dpi=160)
        plt.close(fig_prior)
        print(f"[spice3d] Using slice-matched localizer prior: {args.localizer}")

    if args.adjoint_only:
        with open(os.path.join(args.out_dir, "spice3d_report.txt"), "w", encoding="ascii") as f:
            f.write("05d 3D SPICE adjoint report\n")
            f.write(f"dat={args.dat}\n")
            f.write(f"grid={nx}x{ny}x{nz} n_spec={n_spec}\n")
            f.write(f"coil_combination={coil_report}\n")
            f.write(f"subspace_src={args.subspace_src}\n")
            f.write(f"localizer={args.localizer}\n")
        print("[spice3d] Adjoint-only mode done.")
        return

    if args.subspace_src == "basis-dat":
        if not args.basis_dat:
            sys.exit("--basis-dat is required when --subspace-src basis-dat")
        print(f"[spice3d] Loading basis reference dat: {args.basis_dat}")
        ref_info = load_interleaved_31p(args.basis_dat)
        ref_k = ref_info["kdata"]
        if not args.no_phase_corr:
            for ch in range(ref_k.shape[2]):
                ref_k[:, :, ch:ch + 1, :] = concept_phase_correction(
                    ref_k[:, :, ch:ch + 1, :], ref_info["_ti_per_ring"])
        ref_nx = int(ref_info["matrix_size"])
        ref_ny = int(ref_info["matrix_size"])
        ref_nz = int(ref_info["Nz"])
        ref_n_spec = int(ref_info["n_spec"])
        ref_traj = build_trajectory_3d(
            ref_info["first_circle"],
            ref_info["n_rings_total"],
            ref_info["traj_pts"],
            ref_info["kz_cpp_per_ring"],
            ref_info["gamma_2pi"] / 1e9,
            ref_info["fov_m"],
            ref_nx,
        )
        ref_traj = ref_traj.copy()   # physical FOV rescale (see main traj above)
        ref_traj[:, 0] *= float(ref_info["k_norm_max_phys"]) / 0.5
        ref_traj[:, 1] *= float(ref_info["k_norm_max_phys"]) / 0.5
        if ref_k.shape[2] == 1:
            ref_flat = ref_k[:, :, 0, :].transpose(2, 0, 1).reshape(-1, ref_n_spec)
        else:
            ref_flat, _, _ = combine_coils_phase_aligned(ref_k)
        if not args.no_fov_shift:
            rd_ro, rd_pe, rd_sl = compute_fov_shift_3d(ref_info)
            rd_ro += args.extra_shift_ro
            rd_pe += args.extra_shift_pe
            rd_sl += args.extra_shift_sl
            ref_phase = np.exp(-2j * np.pi * (ref_traj[:, 0] * rd_pe + ref_traj[:, 1] * rd_ro + ref_traj[:, 2] * (-rd_sl)))
            ref_flat = ref_flat * ref_phase[:, np.newaxis]
        ref_rxy = np.sqrt(ref_traj[:, 0] ** 2 + ref_traj[:, 1] ** 2)
        ref_hr = ref_rxy   # ref_traj already physical cyc/pix (rescaled above)
        ref_dcf = compute_crt_dcf_hamming(ref_traj, hamming_radius=ref_hr)
        ref_adj_flat, _ = _compute_adjoint_flat(ref_info, ref_flat, ref_traj, ref_n_spec, (ref_nx, ref_ny, ref_nz), ref_dcf)
        training = ref_adj_flat
    elif args.subspace_src == "adjoint":
        training = adj_fid_flat
    elif args.subspace_src == "pk2pi":
        # spectral_dwell (= 1/spectral_BW from the 07b loader) is the true
        # spectral dwell used everywhere else in 05d and by the data's own ppm
        # axis; use it here too so the pk2pi subspace peaks land at the correct
        # ppm (05c used dwell_s*2 only because its older loader lacked this key).
        training = generate_training_data_pk2pi(n_spec, info["spectral_dwell"], info["larmor_freq_mhz"], args.training_size,
                                                pi_ph_range=args.pi_ph_range)
        print(f"[spice3d] pk2pi Pi model: {'single pH-shifting peak (3.9-5.2 ppm)' if args.pi_ph_range else 'fixed Pi+Pi2 doublet (legacy)'}")
    elif args.subspace_src == "pk-7t-skeletal":
        training = generate_training_data_pk_7t_skeletal(
            n_spec, info["spectral_dwell"], info["larmor_freq_mhz"], args.training_size)
    elif args.subspace_src == "pk-7t-liver-3tpulse":
        training = generate_training_data_pk_7t_liver_3tpulse(
            n_spec, info["spectral_dwell"], info["larmor_freq_mhz"], args.training_size)
    elif args.subspace_src == "pk-cardiac":
        training = generate_training_data_pk_cardiac(
            n_spec, info["spectral_dwell"], info["larmor_freq_mhz"], args.training_size)
    elif args.subspace_src == "pk-3t-brain":
        training = generate_training_data_pk_3t_brain(
            n_spec, info["spectral_dwell"], info["larmor_freq_mhz"], args.training_size,
            phase_mode=args.pk_phase, delay_s=args.pk_delay_ms * 1e-3,
            lw_floor_pcr=(args.pk_lw_floor[0] if args.pk_lw_floor else None),
            lw_floor_atp=(args.pk_lw_floor[1] if args.pk_lw_floor else None))
        print(f"[spice3d] pk-3t-brain phases: {args.pk_phase}"
              + (f", delay {args.pk_delay_ms:+.3f} ms" if args.pk_phase == "shared" else "")
              + (f", LW floor PCr {args.pk_lw_floor[0]:g} Hz / ATP {args.pk_lw_floor[1]:g} Hz"
                 if args.pk_lw_floor else ""))
    else:
        training = generate_training_data(n_spec, info["spectral_dwell"], info["larmor_freq_mhz"], args.training_size)

    print("[spice3d] Subspace SVD ...")
    _, s_vals, vh = np.linalg.svd(training, full_matrices=False)

    # ── Model-order (rank) suggestion ────────────────────────────────────────
    rank_info = suggest_spice_rank(
        s_vals, adj_fid_flat=adj_fid_flat, vh=vh,
        max_rank=min(40, len(s_vals)))
    print("[spice3d] Suggested SPICE rank (subspace + data-driven):")
    for _k in ("trunc<0.01", "trunc<0.001", "energy>0.99", "energy>0.999",
               "elbow", "data_knee"):
        if _k in rank_info:
            print(f"           {_k:>13s} -> rank {rank_info[_k]}")
    print(f"           {'RECOMMENDED':>13s} -> rank {rank_info['recommended']}")
    _r001 = rank_info.get("trunc<0.001")
    if _r001 is not None and max(args.rank) < _r001:
        print(f"[spice3d] NOTE: requested max rank {max(args.rank)} < 0.1% truncation "
              f"rank {_r001}; weak/broad resonances (Pi, PDE, PE) may be under-represented.")
    if args.auto_rank:
        args.rank = [rank_info["recommended"]]
        print(f"[spice3d] --auto-rank: reconstructing only at rank {rank_info['recommended']}")

    # ── Diagnostic figure: singular-value decay + data-capture curve ─────────
    fig_svd, (ax_sv, ax_cap) = plt.subplots(1, 2, figsize=(12, 4.2))
    kmax = min(40, len(s_vals))
    ax_sv.semilogy(np.arange(1, kmax + 1), s_vals[:kmax], "x-", color="0.3")
    _vlines = [("0.1% trunc", "trunc<0.001", "tab:orange"),
               ("elbow", "elbow", "tab:green"),
               ("data knee", "data_knee", "tab:red"),
               ("recommended", "recommended", "k")]
    for _lbl, _key, _c in _vlines:
        if _key in rank_info:
            ax_sv.axvline(rank_info[_key], ls="--", lw=1.1, color=_c,
                          label=f"{_lbl} (r={rank_info[_key]})")
    ax_sv.set_title("Subspace singular values")
    ax_sv.set_xlabel("rank")
    ax_sv.set_ylabel("sigma")
    ax_sv.legend(fontsize=7, loc="upper right")

    if "data_energy_curve" in rank_info:
        cap = rank_info["data_energy_curve"]
        rr = np.arange(1, len(cap) + 1)
        ax_cap.plot(rr, 100.0 * cap, "-o", ms=3, color="tab:blue")
        for _lbl, _key, _c in _vlines:
            if _key in rank_info and rank_info[_key] <= len(cap):
                ax_cap.axvline(rank_info[_key], ls="--", lw=1.1, color=_c)
        ax_cap.set_ylim(top=100.5)
        ax_cap.set_title("Measured data energy captured by rank-r subspace")
        ax_cap.set_xlabel("rank")
        ax_cap.set_ylabel("captured energy [%]")
    else:
        ax_cap.axis("off")
    fig_svd.tight_layout()
    fig_svd.savefig(os.path.join(fig_dir, "spice3d_subspace_svals.png"), dpi=140)
    plt.close(fig_svd)

    # Top-K basis vectors of the spectral subspace (right singular vectors of
    # the training matrix). Useful sanity check that the chosen rank captures
    # the phosphorus resonances rather than noise.
    top_k = int(min(max(args.rank) if args.rank else 8, len(s_vals), 12))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_spec, d=info["spectral_dwell"]))
    ppm_axis_sub = -freq_axis / info["larmor_freq_mhz"] + args.ppm_center
    xlim_sub = (float(max(args.ppmlim)), float(min(args.ppmlim)))
    fig_v, axes_v = plt.subplots(top_k, 1, figsize=(9, 1.4 * top_k), sharex=True)
    if top_k == 1:
        axes_v = [axes_v]
    for k in range(top_k):
        # vh rows are basis FIDs (time-domain). Show real & imag magnitude
        # spectrum of each (conjugate to match reconstruction sign convention).
        basis_fid = vh[k, :]
        basis_spec = np.fft.fftshift(np.fft.fft(np.conj(basis_fid)))
        axes_v[k].plot(ppm_axis_sub, np.abs(basis_spec), "k-", lw=0.7)
        axes_v[k].set_xlim(xlim_sub)
        axes_v[k].set_ylabel(f"v{k + 1}", fontsize=8)
        axes_v[k].tick_params(labelsize=7)
    axes_v[-1].set_xlabel("ppm")
    fig_v.suptitle(f"Top {top_k} spectral subspace vectors  (source: {args.subspace_src})",
                   fontsize=10)
    fig_v.tight_layout()
    fig_v.savefig(os.path.join(fig_dir, "spice3d_subspace_top_vectors.png"), dpi=150)
    plt.close(fig_v)

    if args.prior_npy:
        prior_native = np.load(args.prior_npy).astype(np.float32)
        if prior_native.shape != (nx, ny, nz):
            sys.exit(f"[spice3d] --prior-npy shape {prior_native.shape} != ({nx},{ny},{nz})")
        prior_native = prior_native / (float(prior_native.max()) + 1e-12)
        prior_label = f"precomputed prior {os.path.basename(args.prior_npy)}"
        print(f"[spice3d] Spatial prior source: {args.prior_npy}")
    elif localizer_prior is not None:
        prior_native = localizer_prior.astype(np.float32)
        prior_label = "registered localizer (unsegmented)"
        print("[spice3d] Spatial prior source: registered localizer (no automatic segmentation)")
    else:
        prior_native = np.mean(np.abs(adj_spec), axis=-1).astype(np.float32)
        prior_native = prior_native / (prior_native.max() + 1e-12)
        prior_label = "31P adjoint magnitude"

    roi = None
    if args.roi_mask:
        roi = np.load(args.roi_mask).astype(bool)
        if roi.ndim == 2 and roi.shape == (nx, ny):   # 2D mask from roi_selector → apply to all slices
            roi = np.broadcast_to(roi[:, :, None], (nx, ny, nz)).copy()
        if roi.shape != (nx, ny, nz):
            sys.exit(f"[spice3d] ROI mask shape {roi.shape} != ({nx},{ny},{nz})")
        prior_native = prior_native * roi.astype(np.float32)
        prior_label += " + ROI masked"
        print(f"[spice3d] Applied manual ROI mask to prior: {int(roi.sum())}/{roi.size} voxels kept")

        if ROI_RENORM:
            vals = prior_native[roi]
            if vals.size > 0:
                vmin = float(vals.min())
                vmax = float(vals.max())
                denom = vmax - vmin
                if denom > 1e-12:
                    tmp = np.zeros_like(prior_native, dtype=np.float32)
                    tmp[roi] = ((vals - vmin) / denom).astype(np.float32)
                    prior_native = tmp
                    prior_label += ", roi_renorm"
                    print(f"[spice3d] ROI renorm: in-ROI prior normalized to [0,1] (min={vmin:.4g}, max={vmax:.4g})")
                else:
                    print("[spice3d] ROI renorm skipped: in-ROI prior is constant")

    prior_graph = prior_native.astype(np.float32)
    prior_plot = prior_native.astype(np.float32)

    if PRIOR_SCALE_PLOT != 1.0:
        prior_plot = (prior_plot * float(PRIOR_SCALE_PLOT)).astype(np.float32)
        prior_label += f", plot_scale={PRIOR_SCALE_PLOT:g}"

    if PRIOR_SCALE_GRAPH != 1.0:
        prior_graph = (prior_graph * float(PRIOR_SCALE_GRAPH)).astype(np.float32)
    print(f"[spice3d] Graph prior scale factor: {PRIOR_SCALE_GRAPH:g}")

    # Save a 3D analogue of 05c prior map. z-summed then reoriented into the
    # DICOM row/col frame so it aligns with the per-slice
    # ``spatial_prior_map_z*.png`` and the graph-weight overlays below.
    prior_sum2d = _dicom_view_axes(prior_plot.sum(axis=2), dicom_view_params)
    fig_prior_sum, ax_prior_sum = plt.subplots(figsize=(7, 6))
    im_prior_sum = ax_prior_sum.imshow(prior_sum2d, origin="upper", cmap="Greens",
                                       aspect="equal")
    ax_prior_sum.set_title(f"Voxel-level summed spatial prior ({prior_label})")
    ax_prior_sum.set_xlabel("DICOM col"); ax_prior_sum.set_ylabel("DICOM row")
    fig_prior_sum.colorbar(im_prior_sum, ax=ax_prior_sum, label="sum prior")
    fig_prior_sum.tight_layout()
    fig_prior_sum.savefig(os.path.join(fig_dir, "spatial_prior_map.png"), dpi=180)
    plt.close(fig_prior_sum)

    # Resolve the single --roi-strength knob into graph-boundary settings.
    _roi_edge_scale, _roi_hard = ROI_STRENGTH_PRESETS[args.roi_strength]
    _graph_mask = None if args.roi_strength == "off" else roi
    if roi is not None and args.roi_strength == "off":
        print("[spice3d] ROI strength 'off': mask applied to the prior but ignored by the graph")

    lambdas = [float(l) for l in args.lambda1]
    if max(lambdas) > 0:
        ww, a_graph = build_regularizer_3d(
            prior_graph,
            wmax=GRAPH_WMAX,
            adj=args.adj,
            mask=_graph_mask,
            out_edge_scale=(_roi_edge_scale if _roi_edge_scale is not None else 1.0),
            mask_dilation=ROI_DILATION,
            hard_boundary=_roi_hard,
            return_graph=True,
        )
        if _graph_mask is not None:
            if _roi_hard:
                print(f"[spice3d] ROI graph boundary: strict (dilation={ROI_DILATION}, cross-boundary edges=0)")
            else:
                print(f"[spice3d] ROI graph boundary: {args.roi_strength} (dilation={ROI_DILATION}, edge_scale={_roi_edge_scale})")

        # 3D graph-weight diagnostic, in the exact style of 05c's
        # ``spatial_prior_voxel_sum_map.png``. For each voxel the per-voxel
        # sum of edge weights is computed from A_graph and reoriented into
        # the DICOM row/col frame so it aligns with ``spatial_prior_map.png``.
        # adj=6 -> up to 6 neighbours per interior voxel.  build_regularizer_3d
        # symmetrizes with a = 0.5 * (a + a.T), so each edge contributes
        # wmax/2 (not wmax) to each of its two endpoints.  The saturation cap
        # is therefore wmax * 6 / 2, and omitting the 0.5 would make the
        # normalized figure top out at 0.5 instead of 1.0.
        # NOTE: 1H SPICE (utils/plotting.py plot_voxel_sum_map) accumulates the
        # full w per endpoint, so its weight sums are 2x these values.
        voxel_sum = np.sum(a_graph, axis=0).reshape(nx, ny, nz).astype(np.float32)
        n_neighbors_max = 6.0
        edge_split = 0.5   # from the 0.5 * (a + a.T) symmetrization
        expected_max = float(GRAPH_WMAX) * n_neighbors_max * edge_split
        vmax_actual = float(voxel_sum.max())
        print(f"[spice3d] Graph weight-sum max={vmax_actual:.3g} "
              f"(expected scale <= {expected_max:.3g} for adj=6)")

        # (a) Aggregate sum-across-z map (single 2-D figure like 05c).
        voxel_sum_agg = _dicom_view_axes(voxel_sum.sum(axis=2), dicom_view_params)
        fig_wsum, ax_wsum = plt.subplots(figsize=(6, 5))
        im_wsum = ax_wsum.imshow(voxel_sum_agg, origin="upper", cmap="Greens", aspect="equal")
        ax_wsum.set_title("Voxel-level sum prior map (graph weights, z-summed)", fontsize=11)
        ax_wsum.set_xlabel("DICOM col"); ax_wsum.set_ylabel("DICOM row")
        fig_wsum.colorbar(im_wsum, ax=ax_wsum, label="sum graph weights")
        fig_wsum.tight_layout()
        fig_wsum.savefig(os.path.join(fig_dir, "spatial_prior_voxel_sum_map.png"),
                         dpi=150, bbox_inches="tight")
        plt.close(fig_wsum)

        # (b) Per-slice grid, styled to match the 1H SPICE
        # ``plot_voxel_sum_map`` (utils/plotting.py) used by
        # ``scripts/recon_method/recon_01_run_spice.py``:
        # gray anatomical prior underneath + Greens graph-weight-sum overlay
        # with ``set_under='black'`` and alpha=0.8. One tile per MRSI slice.
        ncols = min(nz, 6)
        nrows = int(np.ceil(nz / ncols))
        fig_grid, axes_grid = plt.subplots(nrows, ncols,
                                           figsize=(ncols * 3.0, nrows * 3.0),
                                           squeeze=False,
                                           facecolor="black")
        vmax_slice = float(voxel_sum.max()) or 1.0
        cmap_overlay = plt.cm.Greens.copy()
        cmap_overlay.set_under("black")
        im = None
        for z in range(nrows * ncols):
            ax = axes_grid[z // ncols, z % ncols]
            ax.set_facecolor("black")
            if z < nz:
                anat_view = _dicom_view_axes(prior_plot[:, :, z], dicom_view_params)
                weight_view = _dicom_view_axes(voxel_sum[:, :, z], dicom_view_params)
                ax.imshow(anat_view, origin="upper", cmap="gray", aspect="equal",
                          alpha=1.0)
                im = ax.imshow(weight_view, origin="upper", cmap=cmap_overlay,
                               vmin=1e-6, vmax=vmax_slice, aspect="equal", alpha=0.8)
                ax.set_title(f"z={z + 1}/{nz}  max={float(voxel_sum[:, :, z].max()):.3g}",
                             fontsize=8, color="white")
            ax.axis("off")
        fig_grid.suptitle("Voxel-level sum prior (graph weights) overlay per slice",
                          fontsize=11, color="white")
        if im is not None:
            cbar = fig_grid.colorbar(im, ax=axes_grid.ravel().tolist(), shrink=0.7,
                                     label="sum graph weights")
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.get_yticklabels(), color="white")
            cbar.ax.yaxis.label.set_color("white")
        fig_grid.savefig(os.path.join(fig_dir, "spatial_prior_voxel_sum_map_slices.png"),
                         dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig_grid)

        # (c) Normalized per-slice grid: divide by wmax * neighbours (05c style).
        voxel_sum_norm = np.clip(voxel_sum / (expected_max + 1e-12), 0.0, 1.0)
        fig_gn, axes_gn = plt.subplots(nrows, ncols,
                                       figsize=(ncols * 3.0, nrows * 3.0),
                                       squeeze=False)
        for z in range(nrows * ncols):
            ax = axes_gn[z // ncols, z % ncols]
            if z < nz:
                view = _dicom_view_axes(voxel_sum_norm[:, :, z], dicom_view_params)
                im_n = ax.imshow(view, origin="upper", cmap="Greens",
                                 vmin=0.0, vmax=1.0, aspect="equal")
                ax.set_title(f"z={z + 1}/{nz}", fontsize=8)
            ax.axis("off")
        fig_gn.suptitle("Voxel-level sum prior (graph weights, normalized)", fontsize=11)
        fig_gn.colorbar(im_n, ax=axes_gn.ravel().tolist(), shrink=0.7,
                        label="normalized weight sum")
        fig_gn.savefig(os.path.join(fig_dir, "spatial_prior_voxel_sum_map_normalized.png"),
                       dpi=150, bbox_inches="tight")
        plt.close(fig_gn)

        # (d) Prior + graph-weight contour overlay per slice (orientation
        # check: high-graph-weight region should sit on the bright prior).
        fig_ov, axes_ov = plt.subplots(nrows, ncols,
                                       figsize=(ncols * 3.0, nrows * 3.0),
                                       squeeze=False)
        for z in range(nrows * ncols):
            ax = axes_ov[z // ncols, z % ncols]
            if z < nz:
                prior_view = _dicom_view_axes(prior_plot[:, :, z], dicom_view_params)
                norm_view = _dicom_view_axes(voxel_sum_norm[:, :, z], dicom_view_params)
                ax.imshow(prior_view, origin="upper", cmap="gray", aspect="equal")
                ax.contour(norm_view, levels=[0.2, 0.4, 0.6, 0.8],
                           colors="lime", linewidths=0.9, origin="upper")
                ax.set_title(f"z={z + 1}/{nz}", fontsize=8)
            ax.axis("off")
        fig_ov.suptitle("Spatial prior + graph-weight contours (orientation check)",
                        fontsize=11)
        fig_ov.savefig(os.path.join(fig_dir, "spatial_prior_graph_overlay.png"),
                       dpi=150, bbox_inches="tight")
        plt.close(fig_ov)
    else:
        ww = None

    dwell_spec = info["spectral_dwell"]
    freq = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_spec))
    ppm = -freq / info["larmor_freq_mhz"] + args.ppm_center

    # Build the trajectory-only Gram ONCE and reuse it for every rank (and every
    # coil) below — it depends only on the NUFFT operator and grid, not on rank,
    # lambda or coil. Mirrors the 1H SPICE core, which constructs a single Gram
    # operator and reuses it rather than rebuilding it per solve.
    print("[spice3d] Precomputing shared 3D Gram matrix (reused across ranks/coils) ...")
    shared_gram = _build_gram_3d(nufft_op, (nx, ny, nz))

    for lam, rank in [(l, r) for l in lambdas for r in args.rank]:
        v = vh[:rank, :].conj().T
        t0 = time.time()
        if use_per_coil:
            spice_fid, u, _wsvd_quality = spice_reconstruct_3d_multicoil(
                adj_fid_mc,
                nufft_op,
                v,
                ww,
                (nx, ny, nz),
                lamda=lam,
                n_iter=args.n_iter,
                maxiter=args.maxiter,
                noise_cov=_wsvd_noise_cov,
                gram=shared_gram,
            )
        else:
            spice_fid, u = spice_reconstruct_3d(
                adj_fid_flat,
                nufft_op,
                v,
                ww,
                (nx, ny, nz),
                lamda=lam,
                n_iter=args.n_iter,
                maxiter=args.maxiter,
                gram=shared_gram,
            )
        print(f"[spice3d] rank={rank} lambda1={lam:g} done in {time.time()-t0:.1f}s")

        # Even-N centering correction on the SPICE output (matches the adjoint above / 07b/08).
        spice_fid = _even_n_roll(spice_fid)

        if args.apodization > 0:
            t_apo = np.arange(n_spec) * dwell_spec
            apo = np.exp(-np.pi ** 2 * args.apodization ** 2 * t_apo ** 2 / (4 * np.log(2)))
            spice_fid = spice_fid * apo

        spice_spec = FIDToSpec(spice_fid.conj(), axis=-1)

        # Attenuate SPICE out-of-ROI voxels (not zero) so they remain visible
        # but suppressed. The graph-prior mask only softens coupling; the
        # data-fidelity term still lets out-of-ROI voxels carry signal.
        # The adjoint is left completely untouched.
        if roi is not None and args.roi_display_attenuation != 1.0:
            _atten = float(args.roi_display_attenuation)
            _mask4 = np.where(roi, 1.0, _atten).astype(np.float32)[..., None]
            spice_fid = spice_fid * _mask4
            spice_spec = spice_spec * _mask4
            if rank == args.rank[0] and lam == lambdas[0]:
                print(f"[spice3d] SPICE out-of-ROI attenuation: {_atten:g} "
                      f"({int(roi.sum())}/{roi.size} in-ROI voxels unchanged, "
                      f"use --roi-display-attenuation 1.0 to disable)")

        spice_fid_output = _apply_output_xy_orientation(spice_fid, args.out_rot90, args.out_flip_lr)
        spice_spec_output = _apply_output_xy_orientation(spice_spec, args.out_rot90, args.out_flip_lr)

        tag = f"rank{rank}" if len(lambdas) == 1 else f"rank{rank}_lam{lam:g}"
        np.save(os.path.join(data_dir, f"spice3d_fid_{tag}.npy"), spice_fid_output.astype(np.complex64))
        np.save(os.path.join(data_dir, f"spice3d_spec_{tag}.npy"), spice_spec_output.astype(np.complex64))
        np.save(os.path.join(data_dir, f"spice3d_U_{tag}.npy"), u.astype(np.complex64))
        np.save(os.path.join(data_dir, f"spice3d_V_{tag}.npy"), v.astype(np.complex64))

        rank_fig = os.path.join(fig_dir, tag)
        # PCr/Pi MIP maps intentionally skipped: metabolite-specific maps are
        # meaningless for phantoms that may contain neither PCr nor Pi. Only the
        # per-voxel spectra grids are plotted below.
        os.makedirs(rank_fig, exist_ok=True)
        # Per-voxel spectra maps for both the ADJOINT (ground-truth-orientation
        # sanity check) and the SPICE recon, using 06c's exact grid layout and
        # noise normalization so the adjoint plot is directly comparable to
        # 06c/07b outputs.
        _vx, _vy, _vz, _ = find_best_snr_3d(adj_spec)
        plot_spectrum_maps_3d(
            adj_spec,
            ppm,
            args.ppmlim,
            rank_fig,
            anatomy=localizer_prior if localizer_prior is not None else None,
            label="Adjoint (DCF)",
            fname_prefix="adjoint",
            apodization_hz=args.apodization,
            best_voxel=(_vx, _vy, _vz),
            norm_mode=args.spectra_norm,
        )
        _sx, _sy, _sz, _ = find_best_snr_3d(spice_spec)
        plot_spectrum_maps_3d(
            spice_spec,
            ppm,
            args.ppmlim,
            rank_fig,
            anatomy=localizer_prior if localizer_prior is not None else None,
            label=f"SPICE rank={rank}",
            fname_prefix="spice",
            apodization_hz=args.apodization,
            best_voxel=(_sx, _sy, _sz),
            norm_mode=args.spectra_norm,
        )

    with open(os.path.join(args.out_dir, "spice3d_report.txt"), "w", encoding="ascii") as f:
        f.write("05d 3D SPICE report\n")
        f.write(f"dat={args.dat}\n")
        f.write(f"grid={nx}x{ny}x{nz} n_spec={n_spec}\n")
        f.write(f"coil_combination={coil_report}\n")
        f.write(f"rank={args.rank}\n")
        f.write(f"lambda1={lambdas} n_iter={args.n_iter} maxiter={args.maxiter}\n")
        f.write(f"apodization_hz={args.apodization}\n")
        f.write(f"subspace_src={args.subspace_src}\n")
        f.write(f"localizer={args.localizer}\n")
        f.write(f"localizer_percentiles={LOCALIZER_PERCENTILES}\n")
        f.write(f"localizer_contrast_gamma={LOCALIZER_GAMMA}\n")
        f.write("localizer_auto_segmentation=disabled\n")
        f.write(f"roi_mask={args.roi_mask}\n")
        f.write(f"prior_gamma={PRIOR_GAMMA} prior_scale={PRIOR_SCALE_PLOT} prior_scale_graph={PRIOR_SCALE_GRAPH} wmax={GRAPH_WMAX}\n")
        f.write(f"roi_strength={args.roi_strength} roi_edge_scale={_roi_edge_scale} roi_dilation={ROI_DILATION} roi_hard_boundary={_roi_hard} roi_renorm={ROI_RENORM} roi_display_attenuation={args.roi_display_attenuation}\n")
        f.write(f"prior_flip_native_y={PRIOR_FLIP_NATIVE_Y}\n")
        f.write(f"output_orientation=rot90_k={args.out_rot90 % 4},flip_lr={args.out_flip_lr}\n")
        if selected_slices:
            distances = [match["distance_mm"] for match in selected_slices]
            f.write(f"localizer_plane_distance_mm_min_max={min(distances):.4f},{max(distances):.4f}\n")

    print("[spice3d] Done.")


if __name__ == "__main__":
    main()
