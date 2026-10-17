"""
31P calf phantom and spectral model for the learned-SPICE simulator.

The spectral model is the PK_3T_Skeletal_2Pi multiplet table used by
generate_training_data_pk2pi in scripts/05c_spice_recon_31p.py, copied here
so this package does not import the 05c script (which pulls in mri-nufft and
the twix reader at import time). Keep the two tables in sync.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

_DELTA_BETA = 15.0 / 51.0
_DELTA_ALPHA = 16.0 / 51.0
_DELTA_GAMMA = 15.0 / 51.0

# (name, center_ppm, lw_init_hz, amp_range, components, cs_bounds), as in 05c
PK2PI_METABOLITES = [
    ("PCr",  0.0,    5.0,  (1.5, 2.5),  [(0.0, 1.0)],  None),
    ("Pi",   4.85,   5.0,  (0.0, 0.9),  [(0.0, 1.0)],  (4.3, 5.1)),
    ("PDE",  3.07,  30.0,  (0.0, 0.15), [(0.0, 1.0)],  None),
    ("gATP", -2.53,  10.0, (0.5, 1.5),
     [(-_DELTA_GAMMA / 2, 1.0), (_DELTA_GAMMA / 2, 1.0)],  None),
    ("aATP", -7.58,  10.0, (0.5, 1.5),
     [(-_DELTA_ALPHA / 2, 1.0), (_DELTA_ALPHA / 2, 1.0)],  None),
    ("bATP", -15.86, 10.0, (0.5, 1.5),
     [(-_DELTA_BETA, 0.5), (0.0, 1.0), (_DELTA_BETA, 0.5)],  None),
]

# ppm windows for the PCr / Pi peak-height maps (Pi window as in the CA-SPICE study)
PCR_PPM = (-0.5, 0.5)
PI_PPM = (3.8, 5.4)


def training_fids_pk2pi(n_spec, dwell_s, larmor_mhz, n_training=5000, seed=42):
    """Random 31P FIDs for the spectral subspace: generate_training_data_pk2pi(pi_ph_range=True) of 05c."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_spec) * dwell_s
    training = np.zeros((n_training, n_spec), dtype=np.complex128)
    for i in range(n_training):
        sig = np.zeros(n_spec, dtype=np.complex128)
        for _, center_ppm, lw_init, amp_range, components, cs_bounds in PK2PI_METABOLITES:
            amp = rng.uniform(*amp_range)
            lw = rng.uniform(lw_init * 0.5, min(50.0, lw_init * 3.0))
            phase = rng.uniform(0, 2 * np.pi)
            center = rng.uniform(*cs_bounds) if cs_bounds is not None else center_ppm
            for delta_ppm, weight in components:
                freq_hz = (center + delta_ppm) * larmor_mhz
                sig += (amp * weight * np.exp(1j * phase) * np.exp(-lw * np.pi * t)
                        * np.exp(2j * np.pi * freq_hz * t))
        sig *= np.exp(2j * np.pi * rng.uniform(-0.1, 0.1) * larmor_mhz * t)
        training[i] = sig
    return training


def spectral_subspace(training: np.ndarray, rank: int) -> np.ndarray:
    """V (n_spec, rank) with orthonormal columns; FIDs are x = U V^H, as in 05c."""
    _, _, Vh = np.linalg.svd(training, full_matrices=False)
    return Vh[:rank].conj().T


def compartment_fid(rng, t, larmor_mhz, state="rest"):
    """One compartment spectrum (PK2PI table); 'exercise' depletes PCr and raises / acidifies Pi."""
    sig = np.zeros(len(t), dtype=np.complex128)
    depletion = rng.uniform(0.2, 0.7) if state == "exercise" else 0.0
    for name, center_ppm, lw_init, amp_range, components, cs_bounds in PK2PI_METABOLITES:
        amp = rng.uniform(*amp_range)
        lw = rng.uniform(lw_init * 0.8, lw_init * 2.0)
        center = center_ppm
        if name == "PCr":
            amp *= 1.0 - depletion
        if name == "Pi":
            amp = amp * 0.3 + depletion * rng.uniform(1.0, 2.0)
            center = rng.uniform(*cs_bounds) if state == "exercise" else rng.uniform(4.8, 5.0)
        for delta_ppm, weight in components:
            sig += (amp * weight * np.exp(-lw * np.pi * t)
                    * np.exp(2j * np.pi * (center + delta_ppm) * larmor_mhz * t))
    return sig


def _ellipse(x, y, cx, cy, ax, ay, rot=0.0):
    c, s = np.cos(rot), np.sin(rot)
    u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
    return (u / ax)**2 + (v / ay)**2 <= 1.0


def make_calf_phantom(n: int, upsample: int, rng: np.random.Generator) -> dict:
    """Random calf cross-section on an (n*upsample)^2 grid, in units of the FOV ([-0.5, 0.5]).

    Returns
    -------
    comp : (m, m) int    muscle compartment per sub-pixel, -1 = no 31P (air, fat, bone)
    leg  : (m, m) bool   leg outline (skin), for the leg-fraction mask
    scale: (m, m) float  smooth amplitude x surface-coil receive profile
    b0_hz, lw_hz : (m, m) extra frequency shift / line broadening per sub-pixel
    n_comp : number of compartments (the last one is a lesion if present)
    """
    m = n * upsample
    c = (np.arange(m) + 0.5) / m - 0.5
    y, x = np.meshgrid(c, c, indexing="ij")
    rot = rng.uniform(-0.4, 0.4)
    cx, cy = rng.uniform(-0.06, 0.06, 2)
    ax, ay = rng.uniform(0.28, 0.40), rng.uniform(0.24, 0.36)

    leg = _ellipse(x, y, cx, cy, ax, ay, rot)
    fat = rng.uniform(0.02, 0.05)
    muscle = _ellipse(x, y, cx, cy, ax - fat, ay - fat, rot)
    # tibia (anterior-medial) and fibula (lateral): no 31P
    tib = _ellipse(x, y, cx + rng.uniform(-0.1, 0.0), cy - 0.5 * ay, rng.uniform(0.05, 0.08),
                   rng.uniform(0.05, 0.08))
    fib = _ellipse(x, y, cx + 0.55 * ax, cy - rng.uniform(0.0, 0.15), 0.035, 0.035)
    muscle &= ~(tib | fib)

    # compartments: Voronoi cells of random seeds inside the muscle
    n_cells = int(rng.integers(2, 6))
    seeds = np.stack([cx + rng.uniform(-ax, ax, n_cells) * 0.7,
                      cy + rng.uniform(-ay, ay, n_cells) * 0.7], axis=-1)
    d2 = (x[..., None] - seeds[:, 0])**2 + (y[..., None] - seeds[:, 1])**2
    comp = np.where(muscle, d2.argmin(-1), -1)
    n_comp = n_cells

    # optional lesion / focal change that the anatomy does not show
    if rng.uniform() < 0.3:
        les = muscle & _ellipse(x, y, *(seeds[rng.integers(n_cells)] + rng.uniform(-0.05, 0.05, 2)),
                                rng.uniform(0.04, 0.10), rng.uniform(0.04, 0.10), rng.uniform(0, np.pi))
        comp[les] = n_comp
        n_comp += 1

    # smooth amplitude variation x surface coil under the posterior calf
    smooth = 1.0 + 0.15 * gaussian_filter(rng.standard_normal((m, m)), m / 6) * (m / 6)
    coil_angle = rng.uniform(np.pi / 4, 3 * np.pi / 4)
    coil = np.array([0.6 * np.cos(coil_angle), 0.6 * np.sin(coil_angle)])
    d = np.sqrt((x - coil[0])**2 + (y - coil[1])**2)
    d0 = rng.uniform(0.25, 0.6)
    sens = (1 + (d / d0)**2) ** -1.5
    scale = np.clip(smooth, 0.5, 1.5) * sens / sens[muscle].max()

    b0 = gaussian_filter(rng.standard_normal((m, m)), m / 4)
    b0_hz = rng.uniform(0, 6) * b0 / (np.abs(b0).max() + 1e-12)
    lw_hz = rng.uniform(0, 4) * np.abs(gaussian_filter(rng.standard_normal((m, m)), m / 4))

    return {"comp": comp, "leg": leg, "scale": scale, "b0_hz": b0_hz, "lw_hz": lw_hz, "n_comp": n_comp}


def phantom_fids(ph: dict, t: np.ndarray, larmor_mhz: float, rng: np.random.Generator) -> np.ndarray:
    """Sub-pixel FIDs (m*m, n_spec): compartment spectrum x amplitude x local B0 / broadening."""
    states = rng.choice(["rest", "exercise"], size=ph["n_comp"], p=[0.6, 0.4])
    spectra = np.stack([compartment_fid(rng, t, larmor_mhz, s) for s in states])
    spectra *= np.exp(1j * rng.uniform(-0.3, 0.3))   # small residual zero-order phase

    comp = ph["comp"].ravel()
    on = comp >= 0
    fids = np.zeros((comp.size, len(t)), dtype=np.complex128)
    local = np.exp((2j * np.pi * ph["b0_hz"].ravel()[on, None] - np.pi * ph["lw_hz"].ravel()[on, None]) * t)
    fids[on] = spectra[comp[on]] * ph["scale"].ravel()[on, None] * local
    return fids


def peak_map(fids: np.ndarray, dwell_s: float, larmor_mhz: float, ppm_window) -> np.ndarray:
    """Peak height of |spectrum| in a ppm window, per voxel (FIDs use exp(+2 pi i f t))."""
    spec = np.fft.fftshift(np.fft.fft(fids, axis=-1), axes=-1)
    ppm = np.fft.fftshift(np.fft.fftfreq(fids.shape[-1], d=dwell_s)) / larmor_mhz
    sel = (ppm >= ppm_window[0]) & (ppm <= ppm_window[1])
    return np.abs(spec[..., sel]).max(axis=-1)
