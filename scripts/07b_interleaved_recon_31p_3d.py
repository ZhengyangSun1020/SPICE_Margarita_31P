#!/usr/bin/env python3
"""
Interleaved 1H/31P CONCEPT 3D CRT reconstruction.

In the interleaved flash_3dcrt sequence:
  - twix.image    → 1H FLASH k-space (not used here)
  - twix.phasecor → 31P CRT ring-by-ring FID data (16 receive channels)

This script extracts the 31P phasecor data and runs per-coil adjoint +
iterative CG reconstruction, then applies wSVD coil combination in the
frequency domain (OXSA svdReconstructFrequencyDomain).

Pipeline (mirrors reconUsingNUFFT_3D_splitCoils_VE.m):
  1. Reconstruct each coil independently through the 3D exact-Gram CG
  2. FFT each per-coil image → frequency domain
  3. Per-voxel SVD-based optimal combination (wSVD) → single combined spectrum

Reconstruction settings are kept identical to 06c_iterative_recon_31p_3d.py:
  - CRT Hamming DCF evaluated on the physical in-plane ring radius
  - Exact Gram matrix (E^H E) precomputed column-by-column
  - lam_max from max Gram eigenvalue (eigvalsh), so --lam means the same thing
    in both scripts
  - Gaussian apodization, conjugated FID + negated PPM axis

Note: the exact Gram is N_vox² complex128 — memory grows quickly with grid size
(e.g. 16×16×12 = 3072 vox → 151 MB; 24×24×12 = 6912 vox → 764 MB).

Usage:
    python 07b_interleaved_recon_31p_3d.py \\
        --dat "F:/AliceConceptB1/.../flash_3dcrt_8x8x8_20rings_cor_iso.dat" \\
        --out-dir ./output_interleaved --lam 0.001 --maxiter 100
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
from scipy.sparse.linalg import LinearOperator, cg
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

try:
    import mapvbvd
except ImportError:
    sys.exit("pymapvbvd required: pip install pymapvbvd")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_concept_twix_31p import concept_phase_correction
from twix_fast import accelerate_mdh_read, cached_load

# The stock pymapvbvd MDH scan is a per-ADC seek+fread loop and dominates the
# load time on multi-GB interleaved files; swap in the mmap reader.
accelerate_mdh_read()


def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


def compute_crt_dcf_hamming(traj, hamming_radius=None):
    """CRT DCF × Hamming following MATLAB op_CalcAndApplyDensComp.m.

    MATLAB path used for reference:
      Method = 'ConcentricRingTrajectory_Theoretical'
      DCF ring weights from annulus areas + per-ring sample-count correction
      Hamming term: 0.54 + 0.46 * cos(2*pi*r), with r in [0, 0.5] after max-norm
    """
    r = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
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

    # Ring-wise DCF from annulus areas (MATLAB DCFPreG2) with sample-count correction.
    dcf_ring = np.zeros(n_rings, dtype=np.float64)
    for i in range(n_rings):
        if i == 0:
            area = 0.25 * np.pi * (r_norm[0] + r_norm[1])**2
        elif i == n_rings - 1:
            r_out = 1.5 * r_norm[-1] - 0.5 * r_norm[-2]
            r_in = 0.5 * (r_norm[-1] + r_norm[-2])
            area = np.pi * (r_out**2 - r_in**2)
        else:
            upper = 0.5 * (r_norm[i] + r_norm[i + 1])
            lower = 0.5 * (r_norm[i - 1] + r_norm[i])
            area = np.pi * (upper**2 - lower**2)
        dcf_ring[i] = area * (max_pts / n_per_ring[i])

    dcf = np.zeros(len(r), dtype=np.float64)
    for i, ri in enumerate(unique_r):
        dcf[np.abs(r - ri) < 1e-5] = dcf_ring[i]

    ham = 0.54 + 0.46 * np.cos(2.0 * np.pi * r_ham)
    return (dcf * ham).astype(np.float32)


# ── Utilities shared with 06c ────────────────────────────────────────────────

def _ice_to_signed(val):
    if val > 0.5 * 2**16:
        return val - 2**16
    return val


def _hdr_float(hdr, section, *keys, default=None):
    d = hdr.get(section, {})
    for k in keys:
        val = d.get(k if isinstance(k, tuple) else (k,), None)
        if val is not None:
            return float(val)
    return default


def _read_slice_normal(hdr, base):
    """Read a Siemens slice normal (dSag,dCor,dTra), returning (snv_x,snv_y,snv_z).

    Siemens OMITS zero components, so a partially-set normal (e.g. only dCor=1 for
    a coronal slab) means the other components are 0 — NOT the transversal default.
    Only when ALL three components are absent does it default to transversal [0,0,1].
    (Defaulting dTra=1 unconditionally turned a coronal [0,1,0] into an oblique
    [0,1,1] and broke the geometry — this fixes that.)
    """
    sx = _hdr_float(hdr, 'MeasYaps', base + ('sNormal', 'dSag'), default=0.0)
    sy = _hdr_float(hdr, 'MeasYaps', base + ('sNormal', 'dCor'), default=0.0)
    sz = _hdr_float(hdr, 'MeasYaps', base + ('sNormal', 'dTra'), default=0.0)
    if sx == 0.0 and sy == 0.0 and sz == 0.0:
        sz = 1.0                       # fully-unset normal -> transversal
    return sx, sy, sz


def _fGSLClassOri(dGs):
    s, c, t = abs(dGs[0]), abs(dGs[1]), abs(dGs[2])
    eq_sc, eq_st, eq_ct = abs(s-c) <= 1e-6, abs(s-t) <= 1e-6, abs(c-t) <= 1e-6
    if ((eq_sc and eq_st) or (eq_sc and s < t) or (eq_st and s > c) or
            (eq_ct and c > s) or (s > c and s < t) or (s < c and c < t) or
            (s < t and t > c) or (c < t and t > s)):
        return 2
    elif ((eq_sc and s > t) or (eq_st and s < c) or (s < c and c > t) or
            (s > t and s < c) or (s < t and t < c)):
        return 1
    elif ((eq_ct and c < s) or (s > c and s > t) or (c > t and c < s) or
            (c < t and t < s)):
        return 0
    raise ValueError("Invalid slice orientation")


def _fGSLCalcPRS(dGs, dPhi):
    ori = _fGSLClassOri(dGs)
    if ori == 2:
        d = np.sqrt(dGs[1]**2 + dGs[2]**2)
        dGp = np.array([0.0, dGs[2]/d, -dGs[1]/d])
    elif ori == 1:
        d = np.sqrt(dGs[0]**2 + dGs[1]**2)
        dGp = np.array([dGs[1]/d, -dGs[0]/d, 0.0])
    else:
        d = np.sqrt(dGs[0]**2 + dGs[1]**2)
        dGp = np.array([-dGs[1]/d, dGs[0]/d, 0.0])
    dGr = np.cross(dGs, dGp)
    if abs(dPhi) > 1e-10:
        co, si = np.cos(dPhi), np.sin(dPhi)
        dGp = co * dGp - si * dGr
        dGr = np.cross(dGs, dGp)
    return dGp, dGr


# ── kz detection (identical to 06c) ─────────────────────────────────────────

def _detect_kz_per_ring(ice, lin_vals, par_vals, n_rings, n_par, Nz_override,
                         gamma_mhz_per_mt, fov_m_z, idb_vals):
    if Nz_override is not None:
        Nz = Nz_override
        return np.zeros(n_rings * n_par), Nz, "forced 2D (Nz_override)"

    n_scans = len(lin_vals)

    # Strategy 1: n_par > 1 → Cartesian kz from Par counter
    if n_par > 1:
        kz_cpp = np.zeros(n_rings * n_par)
        for p_idx in range(n_par):
            kz_val = (p_idx - n_par / 2) / n_par
            kz_cpp[p_idx * n_rings: (p_idx + 1) * n_rings] = kz_val
        return kz_cpp, n_par, f"Par counter  Nz={n_par}"

    # Strategy 2: ICE[21..23] → physical kz via MATLAB CONCEPTCalculateKSpacePos3D convention
    # ICE[21] = sign bit (nonzero → negate), ICE[22] = integer amplitude (mT/m),
    # ICE[23] = fractional amplitude × 10000.  kz_phys = z_mT × gamma × 700µs
    _DUR_US = 700.0
    kz_phys = np.zeros(n_rings)
    for r in range(n_rings):
        idx = np.where(lin_vals == r)[0]
        if len(idx) == 0:
            continue
        s0 = idx[0]
        sign_bit = int(ice[s0, 21])
        z_mT = float(ice[s0, 22]) + float(ice[s0, 23]) / 10000.0
        if sign_bit != 0:
            z_mT = -z_mT
        kz_phys[r] = z_mT * gamma_mhz_per_mt * _DUR_US   # m^-1

    unique_phys = sorted(set(round(v, 2) for v in kz_phys.tolist()))
    kz_phys_max = max(abs(v) for v in unique_phys) if unique_phys else 0.0
    if len(unique_phys) > 1 and kz_phys_max > 0:
        kz_cpp = kz_phys * 0.5 / kz_phys_max
        Nz = len(unique_phys)
        return kz_cpp, Nz, (
            f"ICE[21..23] MATLAB-conv  {len(unique_phys)} kz levels  "
            f"kz_phys=[{min(unique_phys):.2f},{max(unique_phys):.2f}] m^-1  "
            f"kz_norm=[{kz_cpp.min():.3f},{kz_cpp.max():.3f}]  Nz={Nz}")

    # Strategy 3: ICE[4..5] gradient-moment
    kz_gm = np.zeros(n_rings)
    for r in range(n_rings):
        idx = np.where(lin_vals == r)[0]
        if len(idx) == 0:
            continue
        s0 = idx[0]
        gm_int  = _ice_to_signed(float(ice[s0, 4]))
        gm_frac = _ice_to_signed(float(ice[s0, 5]))
        kz_gm[r] = gm_int + gm_frac / 10000.0

    unique_gm = sorted(set(round(g, 4) for g in kz_gm.tolist()))
    if len(unique_gm) > 1 and max(abs(g) for g in unique_gm) > 0:
        delta_gm = 1e9 / (fov_m_z * gamma_mhz_per_mt)
        kz_cpp = kz_gm / delta_gm
        Nz = len(unique_gm)
        return kz_cpp, Nz, f"ICE[4..5] gradient-moment  Nz={Nz}"

    return np.zeros(n_rings), 1, "2D (no kz encoding detected)"


# ── FOV shift ────────────────────────────────────────────────────────────────

def compute_fov_shift_3d(info, z_resolution_for_shift=None):
    # pos from header is in mm; convert to m to match fov_m
    pos = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]]) / 1000.0
    snv = np.array([info["snv_x"],   info["snv_y"],   info["snv_z"]])
    ipr = info["in_plane_rotation"]
    fov_m = info["fov_m"]
    N = info["matrix_size"]
    fov_m_z = info["fov_m_z"]
    Nz = int(z_resolution_for_shift) if z_resolution_for_shift is not None else int(info["Nz"])
    Nz = max(Nz, 1)
    dGp, dGr = _fGSLCalcPRS(snv, ipr)
    d_ro = float(np.dot(pos, dGr)) / (fov_m / N)
    d_pe = float(np.dot(pos, dGp)) / (fov_m / N)
    d_sl = float(np.dot(pos, snv)) / (fov_m_z / Nz)
    return d_ro, d_pe, d_sl


# ── Data loading (phasecor edition) ─────────────────────────────────────────

def load_interleaved_31p(dat_path, pts_per_ring_override=None,
                          n_spec_override=None,
                          matrix_size_override=None,
                          matrix_size_z_override=None,
                          rep_override=None,
                          rep_average=False):
    """Load 31P CRT data.

    For interleaved 1H/31P files uses twix.phasecor; for pure 31P files
    falls back to twix.image.  rep_override selects a single dynamic
    measurement (0-indexed Rep counter); None uses all Reps.

    Returns kdata of shape (pts_per_ring, n_spec, n_ch, n_rings_total).
    """
    print(f"\n{'='*70}")
    print(f"Loading 31P CRT: {dat_path}")
    print(f"{'='*70}")

    twix = mapvbvd.mapVBVD(dat_path)
    if isinstance(twix, list):
        twix = twix[-1]

    # Prefer phasecor (interleaved 1H/31P); fall back to image (pure 31P).
    # Some Twix objects expose ``phasecor`` even without data — check size too.
    phasecor_obj = getattr(twix, 'phasecor', None)
    is_interleaved = False
    if phasecor_obj is not None:
        try:
            _n_pc = int(getattr(phasecor_obj, 'NAcq', 0)) or int(getattr(phasecor_obj, 'sqzSize', [0])[0])
        except Exception:
            _n_pc = 1
        if _n_pc > 0:
            is_interleaved = True
    if is_interleaved:
        img = twix['phasecor']
        print("  Scan type: phasecor (interleaved 1H/31P)")
    elif hasattr(twix, 'image'):
        img = twix['image']
        print("  Scan type: image (pure 31P — no phasecor found)")
    else:
        raise RuntimeError("No 'phasecor' or 'image' scan found in TWIX file.")
    hdr = twix['hdr']

    # Optional noise scans (used for MATLAB-like noise-prewhitened wSVD).
    noise_cov = None
    noise_obj = twix.get('noise', None)
    if noise_obj is not None:
        try:
            noise_u = noise_obj.unsorted()  # typically (col_noise, n_cha, n_noise_tr)
            if noise_u.ndim == 3 and noise_u.shape[1] > 1:
                noise_mat = np.transpose(noise_u, (0, 2, 1)).reshape(-1, noise_u.shape[1])
                noise_cov = np.cov(noise_mat, rowvar=False)
                print(f"  noise scans:      {noise_u.shape}  -> noise_cov {noise_cov.shape}")
        except Exception as e:
            print(f"  noise scans:      unavailable ({type(e).__name__}: {e})")

    # ── Header ──────────────────────────────────────────────────────────────
    # NOTE: The header Larmor frequency may refer to 1H (the primary nucleus).
    # We print it and also compute the expected 31P frequency for reference.
    larmor_freq_hz = _hdr_float(
        hdr, 'MeasYaps', ('sTXSPEC', 'asNucleusInfo', '0', 'lFrequency'),
        default=49894611)
    larmor_freq_mhz = larmor_freq_hz / 1e6

    # Secondary nucleus frequency (31P channel)
    larmor_31p_hz = _hdr_float(
        hdr, 'MeasYaps', ('sTXSPEC', 'asNucleusInfo', '1', 'lFrequency'),
        default=None)

    # For interleaved 1H/31P scans: idx 0 = 1H (high), idx 1 = 31P (low).
    # For pure 31P scans: idx 0 = 31P (low), idx 1 = 1H (high).
    # Pick the smaller of the two as the 31P Larmor frequency.
    if larmor_31p_hz is not None and larmor_31p_hz > 0:
        larmor_31p = min(larmor_freq_hz, larmor_31p_hz)
        larmor_1h  = max(larmor_freq_hz, larmor_31p_hz)
        print(f"  1H Larmor: {larmor_1h:.0f} Hz ({larmor_1h/1e6:.3f} MHz)")
        larmor_freq_hz  = larmor_31p
        larmor_freq_mhz = larmor_freq_hz / 1e6
        print(f"  31P Larmor: {larmor_freq_hz:.0f} Hz ({larmor_freq_mhz:.3f} MHz)")
    else:
        print(f"  Larmor (primary): {larmor_freq_hz:.0f} Hz ({larmor_freq_mhz:.3f} MHz)")

    # alDwellTime[0] is the 31P phasecor dwell; in interleaved 1H/31P scans
    # alDwellTime[1] is the 1H FLASH dwell (~3-4 µs) — do NOT use for 31P.
    dwell_ns = _hdr_float(hdr, 'MeasYaps', ('sRXSPEC', 'alDwellTime', '0'),
                           default=180000)
    print(f"  Dwell time (31P phasecor): {dwell_ns:.0f} ns")
    dwell_s = dwell_ns * 1e-9

    # ── 31P geometry ─────────────────────────────────────────────────────────
    # For interleaved 1H/31P scans sSpecPara.sVoI is the 1H FLASH volume, so
    # the 31P MRSI cuboid lives in sNavigatorArray.asElm[0].sCuboid and the
    # 31P matrix size in sWipMemBlock.alFree[7]. For pure 31P scans the
    # cuboid IS the primary spec volume and matrix size sits in
    # sKSpace.lBaseResolution as usual.
    base_res     = int(_hdr_float(hdr, 'MeasYaps', ('sKSpace', 'lBaseResolution'), default=8))
    l_partitions = int(_hdr_float(hdr, 'MeasYaps', ('sKSpace', 'lPartitions'), default=0))

    if is_interleaved:
        _nav = ('sNavigatorArray', 'asElm', '0', 'sCuboid')
        fov_mm   = _hdr_float(hdr, 'MeasYaps', _nav + ('dReadoutFOV',), default=250)
        fov_z_mm = _hdr_float(hdr, 'MeasYaps', _nav + ('dThickness',),  default=fov_mm)
        pos_sag  = _hdr_float(hdr, 'MeasYaps', _nav + ('sPosition', 'dSag'), default=0)
        pos_cor  = _hdr_float(hdr, 'MeasYaps', _nav + ('sPosition', 'dCor'), default=0)
        pos_tra  = _hdr_float(hdr, 'MeasYaps', _nav + ('sPosition', 'dTra'), default=0)
        snv_x, snv_y, snv_z = _read_slice_normal(hdr, _nav)
        ipr      = _hdr_float(hdr, 'MeasYaps', _nav + ('dInPlaneRot',),      default=0)
        # 31P matrix XY from sWipMemBlock.alFree[7] (MATLAB idx 8, 1-based).
        n31p_xy  = _hdr_float(hdr, 'MeasYaps', ('sWipMemBlock', 'alFree', '7'), default=None)
        matrix_size = matrix_size_override or (int(n31p_xy) if n31p_xy else base_res)
    else:
        _voi = ('sSpecPara', 'sVoI')
        fov_mm   = _hdr_float(hdr, 'MeasYaps', _voi + ('dReadoutFOV',), default=250)
        fov_z_mm = _hdr_float(hdr, 'MeasYaps', _voi + ('dThickness',),  default=fov_mm)
        pos_sag  = _hdr_float(hdr, 'MeasYaps', _voi + ('sPosition', 'dSag'), default=0)
        pos_cor  = _hdr_float(hdr, 'MeasYaps', _voi + ('sPosition', 'dCor'), default=0)
        pos_tra  = _hdr_float(hdr, 'MeasYaps', _voi + ('sPosition', 'dTra'), default=0)
        snv_x, snv_y, snv_z = _read_slice_normal(hdr, _voi)
        ipr      = _hdr_float(hdr, 'MeasYaps', _voi + ('dInPlaneRot',),      default=0)
        matrix_size = matrix_size_override or base_res

    fov_m = fov_mm / 1e3

    gamma_2pi = 17.235e6   # 31P gyromagnetic ratio [Hz/T]

    print(f"\n--- Header ---")
    print(f"  Larmor:      {larmor_freq_hz:.0f} Hz  ({larmor_freq_mhz:.3f} MHz)")
    print(f"  Dwell:       {dwell_ns:.0f} ns")
    print(f"  FOV_xy:      {fov_mm:.0f} mm")
    print(f"  FOV_z:       {fov_z_mm} mm")
    print(f"  Matrix XY:   {matrix_size}×{matrix_size}")
    print(f"  lPartitions: {l_partitions}")

    # ── MDH counters ─────────────────────────────────────────────────────────
    lin_vals = img.Lin.astype(int)
    idb_vals = img.Idb.astype(int)
    idc_vals = img.Idc.astype(int)
    idd_vals = img.Idd.astype(int)
    ida_vals = img.Ida.astype(int)
    n_scans  = len(lin_vals)
    n_rings  = int(img.NLin)
    n_ch     = int(img.NCha)

    try:
        par_vals = img.Par.astype(int)
        n_par = int(par_vals.max()) + 1
    except AttributeError:
        par_vals = np.zeros(n_scans, dtype=int)
        n_par = 1

    ice = img.iceParam   # (n_scans, 24)

    # Nz from ushSeg: MATLAB segRescaledFrom1 = ushSeg_signed – min + 1; max = Nz
    try:
        seg_vals   = img.Seg.astype(int)
        seg_signed = np.where(seg_vals > 1000, seg_vals - 65536, seg_vals)
        seg_from1  = seg_signed - seg_signed.min() + 1
        nz_from_seg = int(seg_from1.max())
    except AttributeError:
        nz_from_seg = None

    print(f"\n--- MDH counters ---")
    print(f"  n_rings (NLin): {n_rings}")
    print(f"  n_ch (NCha):    {n_ch}")
    print(f"  Lin:  {lin_vals.min()}..{lin_vals.max()}")
    print(f"  Idc:  {np.unique(idc_vals).tolist()}  (pts_per_ring = 2×Idc)")
    print(f"  Idb:  {np.unique(idb_vals).tolist()}  (TI/azimuth direction)")
    print(f"  Idd:  {np.unique(idd_vals).tolist()}")
    print(f"  Ida:  {np.unique(ida_vals).tolist()}  (ADC repetitions per ring)")
    print(f"  Nz (ushSeg → 3rd dim): {nz_from_seg}")

    print(f"\n--- ICE params (first 5 scans) ---")
    for s in range(min(5, n_scans)):
        print(f"  scan {s:2d}  Lin={lin_vals[s]:3d}  "
              f"ice[0..9]={ice[s, :10].astype(int).tolist()}")
    print(f"  ice[20..23] sample: {ice[0, 20:24].astype(int).tolist()}")

    print(f"\n  Nz will be auto-detected from kz encoding")

    # ── Repetition detection (MATLAB onlineLoops style) ──────────────────────
    # Reps are cumulative ring-loops that reset each rep; the last ring of each
    # rep is where lin == max(lin). Cluster consecutive maxLin scans → rep ends.
    # (img.Rep is often all-zero for CONCEPT, so we detect from the ring loop.)
    _maxlin  = int(lin_vals.max())
    _maxidx  = np.where(lin_vals == _maxlin)[0]
    if len(_maxidx):
        _breaks      = np.where(np.diff(_maxidx) != 1)[0]
        _cluster_end = np.append(_maxidx[_breaks], _maxidx[-1])
    else:
        _cluster_end = np.array([n_scans - 1])
    n_reps = int(len(_cluster_end))
    print(f"\n--- Repetitions ---")
    _ends = _cluster_end.tolist()
    print(f"  {n_reps} rep(s) detected (ring-loop maxLin={_maxlin}); rep-end scan idx "
          f"{_ends if n_reps <= 10 else str(_ends[:10]) + ' ...'}")

    # ── Scan selection: rep-average, chosen rep, else first-complete-k-space ──
    _do_rep_avg = False
    _rep_ranges = None
    if rep_average and n_reps > 1:
        _rep_ranges = [(0 if k == 0 else int(_cluster_end[k - 1]) + 1,
                        int(_cluster_end[k]) + 1) for k in range(n_reps)]
        _rep_lens = [e - s for s, e in _rep_ranges]
        sel = np.arange(_rep_ranges[0][0], _rep_ranges[0][1])
        _r0 = _rep_ranges[0]
        _ok = (len(set(_rep_lens)) == 1) and all(
            np.array_equal(lin_vals[_r0[0]:_r0[1]], lin_vals[s:e]) and
            np.array_equal(par_vals[_r0[0]:_r0[1]], par_vals[s:e]) and
            np.array_equal(idb_vals[_r0[0]:_r0[1]], idb_vals[s:e])
            for s, e in _rep_ranges)
        if _ok:
            _do_rep_avg = True
            print(f"  Rep-average: averaging {n_reps} reps (len {_rep_lens[0]} each, "
                  f"identical (Lin,Par,Idb) ordering) BEFORE recon")
        else:
            print(f"  Rep-average requested but reps differ (lens {_rep_lens} or ordering) "
                  f"→ falling back to rep 1 only")
    elif rep_override is not None and n_reps >= 1:
        rep_sel   = min(max(int(rep_override), 1), n_reps)          # 1-indexed
        rep_start = 0 if rep_sel == 1 else int(_cluster_end[rep_sel - 2]) + 1
        rep_end   = int(_cluster_end[rep_sel - 1]) + 1              # exclusive
        sel = np.arange(rep_start, rep_end)
        print(f"  Rep selection: reconstructing rep {rep_sel}/{n_reps} "
              f"(scan idx {rep_start}..{rep_end - 1}, {len(sel)} ADC events)")
    else:
        # Find the first scan at which every unique ring has been seen at Ida==max.
        max_ida = int(ida_vals.max())
        if max_ida == 0:
            n_scans_use = n_scans
            print(f"  First-k-space filter: Ida_max=0 (single average) → all {n_scans} scans")
        else:
            all_lin_set = set(lin_vals.tolist())
            covered     = set()
            n_scans_use = n_scans
            for s in range(n_scans):
                if int(ida_vals[s]) == max_ida:
                    covered.add(int(lin_vals[s]))
                    if covered >= all_lin_set:
                        n_scans_use = s + 1
                        break
            print(f"  First-k-space filter: Ida_max={max_ida} → using {n_scans_use} of "
                  f"{n_scans} scans ({len(covered)}/{len(all_lin_set)} rings at Ida=max)")
        sel = np.arange(0, n_scans_use)

    # ── Raw data (all channels) ──────────────────────────────────────────────
    img.squeeze      = True
    img.flagRemoveOS = False
    u = img.unsorted()   # (Col_raw, n_ch, n_scans_total)
    if u.ndim == 2:                    # single-coil: squeeze=True dropped the channel axis
        u = u[:, np.newaxis, :]        # restore → (Col_raw, 1, n_scans_total)  (mirrors 06c)
    # Average matching k-space samples across reps BEFORE recon (linear √N SNR
    # gain with correctly-estimated coil weights downstream). Writes the mean
    # into the rep-1 slots that `sel` then selects.
    if _do_rep_avg:
        _stack = np.stack([u[:, :, s:e] for (s, e) in _rep_ranges], axis=0)
        u[:, :, _rep_ranges[0][0]:_rep_ranges[0][1]] = _stack.mean(axis=0)
    # Trim all scan-indexed arrays and raw data to the selected scans
    # (`sel` = chosen rep range, or first-complete-k-space prefix)
    lin_vals = lin_vals[sel]
    idb_vals = idb_vals[sel]
    ida_vals = ida_vals[sel]
    idc_vals = idc_vals[sel]
    idd_vals = idd_vals[sel]
    par_vals = par_vals[sel]
    ice      = ice[sel]
    u        = u[:, :, sel]
    n_scans  = len(sel)
    col_per_scan = u.shape[0]
    print(f"\n--- Data ---")
    print(f"  Col (raw, no OS removal): {col_per_scan}")
    print(f"  unsorted shape: {u.shape}")

    u[-4:, :, :] = 0          # blank the last 4 ADC points (was per-scan before)

    idc_val      = int(idc_vals[0])
    pts_per_ring = pts_per_ring_override or (2 * idc_val)
    print(f"  pts_per_ring = 2×Idc = 2×{idc_val} = {pts_per_ring}")

    unique_par = sorted(np.unique(par_vals).tolist())

    # Group scans by (ring, par, idb) — same logic as 06c
    # Store all channels: scans_by_key[(ring, par, idb)] = list of (Col_raw, n_ch) arrays
    scans_by_key = {}
    for s in range(n_scans):
        key = (lin_vals[s], par_vals[s], idb_vals[s])
        if key not in scans_by_key:
            scans_by_key[key] = []
        scans_by_key[key].append(u[:, :, s])   # view; concatenate copies later

    # Determine vecSize (spectral points per ring)
    ring_vec_sizes = []
    for r in range(n_rings):
        for p in unique_par:
            mask = (lin_vals == r) & (par_vals == p)
            if not np.any(mask):
                continue
            n_ti  = int(idb_vals[mask].max()) + 1
            n_adc = int(np.sum(mask & (idb_vals == 0)))
            if n_adc == 0:
                n_adc = int(np.sum(mask))
            vec = round(col_per_scan * n_adc / pts_per_ring * n_ti - 0.5)
            ring_vec_sizes.append(vec)

    default_vec = int(_hdr_float(
        hdr, 'MeasYaps', ('sSpecPara', 'lVectorSize'), default=512))
    pointsInFID = (min(ring_vec_sizes) if ring_vec_sizes else default_vec)
    if n_spec_override is not None:
        pointsInFID = n_spec_override
    print(f"  pointsInFID = {pointsInFID}")

    # ── Build multi-channel kdata ────────────────────────────────────────────
    # Shape: (pts_per_ring, n_spec, n_ch, n_rings_total)
    n_rings_total = n_rings * n_par
    kdata = np.zeros((pts_per_ring, pointsInFID, n_ch, n_rings_total),
                     dtype=np.complex64)

    for r in tqdm(range(n_rings), desc="read rings", miniters=20):
        for p_idx, p in enumerate(unique_par):
            rg   = p_idx * n_rings + r
            mask = (lin_vals == r) & (par_vals == p)
            if not np.any(mask):
                continue
            n_ti      = int(idb_vals[mask].max()) + 1
            vec_per_ti = pointsInFID // n_ti

            # All channels are reshaped in one go: for a (raw_len, n_ch) block
            # the F-order reshape to (pts, vec, n_ch) is exactly the per-channel
            # (pts, vec) F-order reshape stacked along the last axis.
            if n_ti == 1:
                raw_len = pts_per_ring * pointsInFID
                blocks  = scans_by_key.get((r, p, 0),
                                           [np.zeros((raw_len, n_ch), dtype=np.complex64)])
                raw_mc  = np.concatenate(blocks, axis=0)[:raw_len]   # (raw_len, n_ch)
                kdata[:, :, :, rg] = raw_mc.reshape(
                    pts_per_ring, pointsInFID, n_ch, order='F')
            else:
                useful = vec_per_ti * pts_per_ring
                ti_blocks = np.zeros((pts_per_ring, vec_per_ti, n_ti, n_ch),
                                     dtype=np.complex64)
                for ti in range(n_ti):
                    blocks = scans_by_key.get((r, p, ti),
                                              [np.zeros((useful, n_ch), dtype=np.complex64)])
                    raw_ti = np.concatenate(blocks, axis=0)[:useful]   # (useful, n_ch)
                    ti_blocks[:, :, ti, :] = raw_ti.reshape(
                        pts_per_ring, vec_per_ti, n_ch, order='F')
                # axes 1,2 are adjacent -> C-order merge keeps the ti-fastest order
                merged = ti_blocks.reshape(pts_per_ring, vec_per_ti * n_ti, n_ch)
                eff = min(vec_per_ti * n_ti, pointsInFID)
                kdata[:, :eff, :, rg] = merged[:, :eff, :]

    print(f"\n  kdata shape: {kdata.shape}   max={np.abs(kdata).max():.4e}")

    # ── Phase correction params ──────────────────────────────────────────────
    ti_per_ring_base = np.zeros(n_rings, dtype=int)
    for r in range(n_rings):
        mask_r = lin_vals == r
        if np.any(mask_r):
            ti_per_ring_base[r] = int(idd_vals[mask_r][0])
    ti_per_ring_3d = np.tile(ti_per_ring_base, n_par)
    print(f"  TI per ring (Idd): {ti_per_ring_base}")

    # ── Trajectory parameters ────────────────────────────────────────────────
    delta_gm = 1e9 / (fov_m * gamma_2pi)

    first_circle_kx = np.zeros(n_rings)
    first_circle_ky = np.zeros(n_rings)
    for r in range(n_rings):
        idx = np.where(lin_vals == r)[0]
        if len(idx) == 0:
            continue
        s0 = idx[0]
        kx_int  = _ice_to_signed(float(ice[s0, 0]))
        kx_frac = _ice_to_signed(float(ice[s0, 1]))
        ky_int  = _ice_to_signed(float(ice[s0, 2]))
        ky_frac = _ice_to_signed(float(ice[s0, 3]))
        first_circle_kx[r] = kx_int + kx_frac / 10000.0
        first_circle_ky[r] = ky_int + ky_frac / 10000.0

    gamma_mhz_per_mt = gamma_2pi / 1e9
    fov_m_z = (fov_z_mm / 1e3) if fov_z_mm else fov_m

    kz_cpp_per_ring, Nz, kz_source = _detect_kz_per_ring(
        ice, lin_vals, par_vals, n_rings, n_par,
        matrix_size_z_override, gamma_mhz_per_mt, fov_m_z, idb_vals)

    # Override Nz: ushSeg is authoritative for interleaved 31P (lPartitions is 1H)
    if matrix_size_z_override is not None:
        Nz = matrix_size_z_override
    elif nz_from_seg is not None and nz_from_seg > 1:
        Nz = nz_from_seg

    r_max = np.sqrt(first_circle_kx**2 + first_circle_ky**2).max()
    print(f"\n  kz source:   {kz_source}")
    print(f"  Nz:          {Nz}  (from ushSeg={nz_from_seg}, lPartitions={l_partitions})")
    print(f"  kz_cpp/ring: {kz_cpp_per_ring.round(4).tolist()}")
    print(f"  Ring radii:  {np.sqrt(first_circle_kx**2 + first_circle_ky**2).round(1)}")
    print(f"  delta_gm:    {delta_gm:.2f}")
    k_norm_max_phys = float(r_max * gamma_mhz_per_mt * (fov_m / matrix_size))
    print(f"  k_norm_max (physical): "
          f"{k_norm_max_phys:.4f}  "
          f"(MATLAB max-norm: 0.5000  → max_radius={r_max:.1f})")

    # 31P spectral bandwidth is fixed at 2778 Hz (hard-coded, sequence-specific)
    SPECTRAL_BW_HZ = 2778.0
    spectral_dwell  = 1.0 / SPECTRAL_BW_HZ
    print(f"  Spectral BW: {SPECTRAL_BW_HZ:.0f} Hz (hard-coded) = "
          f"{SPECTRAL_BW_HZ/larmor_freq_mhz:.1f} ppm  "
          f"(spectral_dwell={spectral_dwell*1e6:.1f}µs)")
    freq_hz_axis   = np.fft.fftshift(np.fft.fftfreq(pointsInFID, d=spectral_dwell))
    ppm_axis       = freq_hz_axis / larmor_freq_mhz

    return dict(
        kdata=kdata,           # (pts_per_ring, n_spec, n_ch, n_rings_total)
        n_ch=n_ch,
        n_reps=n_reps,         # repetitions detected in the raw data
        rep_used=(rep_override if rep_override is not None else None),
        n_rings=n_rings,
        n_rings_total=n_rings_total,
        n_par=n_par,
        Nz=Nz,
        traj_pts=np.full(n_rings, pts_per_ring, dtype=int),
        n_spec=pointsInFID,
        matrix_size=matrix_size,
        dwell_s=dwell_s,
        larmor_freq_mhz=larmor_freq_mhz,
        fov_mm=fov_mm,
        fov_m=fov_m,
        fov_m_z=fov_m_z,
        gamma_2pi=gamma_2pi,
        delta_gm=delta_gm,
        k_norm_max_phys=k_norm_max_phys,
        noise_cov=noise_cov,
        first_circle=np.array([first_circle_kx, first_circle_ky]),
        kz_cpp_per_ring=kz_cpp_per_ring,
        ppm_axis=ppm_axis,
        spectral_dwell=spectral_dwell,
        pos_sag=pos_sag, pos_cor=pos_cor, pos_tra=pos_tra,
        snv_x=snv_x, snv_y=snv_y, snv_z=snv_z,
        in_plane_rotation=ipr,
        _ti_per_ring=ti_per_ring_3d,
        unique_par=unique_par,
    )


# ── wSVD coil combination ─────────────────────────────────────────────────────

def wsvd_combine(fid_mc, noise_cov=None):
    """Per-voxel weighted SVD coil combination in frequency domain.

    Implements OXSA svdReconstructFrequencyDomain per voxel.
    When noise_cov is None, uses identity (no noise prewhitening).

    Parameters
    ----------
    fid_mc    : (n_spec, n_ch, Nx, Ny[, Nz])  per-coil FID images (complex128)
    noise_cov : (n_ch, n_ch) noise covariance matrix, or None

    Returns
    -------
    combined_fid : (n_spec, Nx, Ny[, Nz])  wSVD-combined FID
    quality      : (Nx, Ny[, Nz])          SVD quality per voxel [0..1]
    """
    n_spec, n_ch = fid_mc.shape[:2]
    spatial_shape = fid_mc.shape[2:]
    n_vox = int(np.prod(spatial_shape))

    # FID → frequency domain
    spec_mc = np.fft.fftshift(np.fft.fft(fid_mc, axis=0), axes=0)   # (n_freq, n_ch, ...)
    spec_flat = spec_mc.reshape(n_spec, n_ch, n_vox)                  # (n_freq, n_ch, n_vox)

    # Noise prewhitening matrix (MATLAB scaleMatrixFft)
    if noise_cov is None or n_ch == 1:
        scale_mat     = np.eye(n_ch, dtype=np.complex128) * np.sqrt(0.5)
        inv_scale_mat = np.eye(n_ch, dtype=np.complex128) / np.sqrt(0.5)
    else:
        noiseCov = np.asarray(noise_cov, dtype=np.complex128)
        noiseVal, noiseVec = np.linalg.eigh(noiseCov)
        noiseVal = np.maximum(np.real(noiseVal), 1e-30)
        scale_mat     = noiseVec @ np.diag(np.sqrt(0.5) / np.sqrt(noiseVal))
        inv_scale_mat = np.linalg.inv(scale_mat)

    combined_fd = np.zeros((n_spec, n_vox), dtype=np.complex128)
    quality     = np.zeros(n_vox)
    sqrt_nch    = np.sqrt(n_ch)

    for i in range(n_vox):
        raw    = spec_flat[:, :, i]       # (n_freq, n_ch)
        scaled = raw @ scale_mat          # (n_freq, n_ch)

        u, s, vh = np.linalg.svd(scaled, full_matrices=False)
        # u: (n_freq, K)  s: (K,)  vh: (K, n_ch)

        # SVD quality (OXSA formula)
        s_norm = np.linalg.norm(s) + 1e-30
        quality[i] = float((s[0] / s_norm * sqrt_nch - 1) / (sqrt_nch - 1 + 1e-30))

        # Coil amplitudes: MATLAB v(:,1)' * invScaleMatrixFft
        # NumPy: v(:,1) = vh[0,:].conj()
        v1 = vh[0, :].conj()
        svdCoilAmplitudes = v1 @ inv_scale_mat        # (n_ch,)

        # Phase reference: normalise by first coil
        ref        = svdCoilAmplitudes[0]
        svdRescale = np.linalg.norm(svdCoilAmplitudes) * ref / (abs(ref) + 1e-30)

        combined_fd[:, i] = u[:, 0] * s[0] * svdRescale

    combined_fd  = combined_fd.reshape((n_spec,) + spatial_shape)
    quality      = quality.reshape(spatial_shape)
    combined_fid = np.fft.ifft(np.fft.ifftshift(combined_fd, axes=0), axis=0)

    return combined_fid, quality


# ── Trajectory (same as 06c) ─────────────────────────────────────────────────

def build_trajectory_3d(first_circle, n_rings_total, traj_pts_arr,
                         kz_cpp_per_ring, gamma_mhz_per_mt, fov_m, N):
    n_rings_base = first_circle.shape[1]
    radii  = np.sqrt(first_circle[0]**2 + first_circle[1]**2)
    phi0   = np.arctan2(first_circle[1], first_circle[0])
    max_radius = float(radii.max())

    all_kx, all_ky, all_kz = [], [], []
    for rg in range(n_rings_total):
        r  = rg % n_rings_base
        ns = int(traj_pts_arr[r])
        angles = 2 * np.pi * np.arange(ns) / ns + phi0[r]
        all_kx.append(radii[r] * np.cos(angles) * 0.5 / max_radius)
        all_ky.append(radii[r] * np.sin(angles) * 0.5 / max_radius)
        all_kz.append(np.full(ns, kz_cpp_per_ring[rg]))

    kx = np.concatenate(all_kx)
    ky = np.concatenate(all_ky)
    kz = np.concatenate(all_kz)
    print(f"[traj3d] {len(kx)} k-space points  "
          f"kx=[{kx.min():.3f},{kx.max():.3f}]  "
          f"ky=[{ky.min():.3f},{ky.max():.3f}]  "
          f"kz=[{kz.min():.3f},{kz.max():.3f}]")
    return np.stack([kx, ky, kz], axis=-1).astype(np.float32)


# ── Reconstruction: per-coil adjoint + exact-Gram CG → wSVD ─────────────────

def iterative_recon_3d(kdata_mc, traj, Nx, Ny, Nz, n_spec, n_ch,
                        noise_cov=None, dcf_hamming_radius=None,
                        maxiter=50, lam=0.0, rtol=1e-3):
    """Per-coil 3D adjoint (+optional CG) then wSVD combination.

    E^H E is computed exactly via column-by-column NUFFT (same operator as the RHS),
    matching MATLAB CONCEPTPerform3DNUFFT_Iterative.m and 06c.

    kdata_mc : (n_kpts, n_spec, n_ch)
    maxiter=0 → DCF adjoint only (fast sanity check), no Gram computed.
    Returns (recon_fid, adj_raw_fid, adj_dcf_fid) each shape (Nx, Ny, Nz, n_spec).
      recon_fid   = CG result (or DCF adjoint when maxiter=0)
      adj_raw_fid = unweighted adjoint (simple gridding)
      adj_dcf_fid = DCF-weighted adjoint (analytical annular weights)
    """
    N_vox  = Nx * Ny * Nz
    D_TYPE = np.complex64
    G_TYPE = np.complex128

    adjoint_only = (maxiter <= 0)
    if adjoint_only:
        print("[iter3d] Adjoint-only mode (maxiter=0) — skipping Gram matrix and CG")

    # NUFFT operator for adjoint (batched over spectral points)
    nufft_op_1 = mrinufft.get_operator(
        "finufft", samples=traj, shape=(Nx, Ny, Nz),
        n_coils=n_spec, density=False)

    # CRT DCF with Hamming filter (matches MATLAB op_CalcAndApplyDensComp.m)
    dcf_weights = compute_crt_dcf_hamming(traj, hamming_radius=dcf_hamming_radius)
    print(f"[iter3d] DCF weights: min={dcf_weights.min():.4f}  max={dcf_weights.max():.4f}")

    if not adjoint_only:
        # Single-coil NUFFT for exact Gram computation (same traj/shape as nufft_op_1)
        nufft_single = mrinufft.get_operator(
            "finufft", samples=traj, shape=(Nx, Ny, Nz), n_coils=1, density=False)

        # Exact Gram G = E^H E (column-by-column).  Consistent with b_flat from nufft_op_1.
        print(f"[iter3d] Precomputing exact Gram matrix (E^H E), "
              f"{N_vox}×{N_vox} complex128 = {N_vox**2 * 16 / 1e9:.2f} GB ...")
        Gram = np.zeros((N_vox, N_vox), dtype=G_TYPE)
        for v in tqdm(range(N_vox), desc="Gram cols"):
            delta = np.zeros((Nx, Ny, Nz), dtype=D_TYPE)
            delta.flat[v] = 1.0
            Gram[:, v] = nufft_single.adj_op(nufft_single.op(delta)).reshape(N_vox)

        if lam > 0:
            lam_max  = float(np.real(np.linalg.eigvalsh(Gram)).max())
            lam_eff  = lam * lam_max
            print(f"[iter3d] lam_max={lam_max:.4g}, lam_rel={lam:.4g} -> lam_eff={lam_eff:.4g}")
            Gram_reg = Gram + lam_eff * np.eye(N_vox, dtype=G_TYPE)
        else:
            lam_eff  = 0.0
            Gram_reg = Gram

        total_size = N_vox * n_spec

        def mv(x_flat):
            return (Gram_reg @ x_flat.reshape(N_vox, n_spec)).ravel()

        A_op = LinearOperator((total_size, total_size), matvec=mv, dtype=G_TYPE)

    # Per-coil images: (n_spec, n_ch, Nx, Ny, Nz)
    X_mc       = np.zeros((n_spec, n_ch, Nx, Ny, Nz), dtype=G_TYPE)
    adj_raw_mc = np.zeros((n_spec, n_ch, Nx, Ny, Nz), dtype=G_TYPE)
    adj_dcf_mc = np.zeros((n_spec, n_ch, Nx, Ny, Nz), dtype=G_TYPE)

    for ch in range(n_ch):
        print(f"[iter3d] Coil {ch+1}/{n_ch} — adjoint ...")
        kdata_ch = kdata_mc[:, :, ch].astype(D_TYPE)   # (n_kpts, n_spec)

        # Batched adjoint: adj_op expects (n_coils, n_kpts) → transpose
        b_ch = nufft_op_1.adj_op(kdata_ch.T).astype(G_TYPE)  # (n_spec, Nx, Ny, Nz)
        adj_raw_mc[:, ch] = b_ch                             # unweighted adjoint

        # DCF-weighted adjoint (apply analytical weights in k-space before adj_op)
        kdata_dcf = kdata_ch * dcf_weights[:, np.newaxis]    # (n_kpts, n_spec)
        adj_dcf_mc[:, ch] = nufft_op_1.adj_op(kdata_dcf.T).astype(G_TYPE)

        if adjoint_only:
            X_mc[:, ch] = adj_dcf_mc[:, ch]   # DCF adjoint as primary output
        else:
            print(f"[iter3d] Coil {ch+1}/{n_ch} — CG ...")
            # CG RHS = unweighted E^H y (consistent with Gram = E^H E)
            b_flat = b_ch.reshape(n_spec, N_vox).T.ravel()
            pbar = tqdm(total=maxiter, desc=f"CG ch{ch}", leave=(n_ch == 1))
            def _cb(xk): pbar.update(1)
            x_flat, info_code = cg(A_op, b_flat,
                                    x0=np.zeros(total_size, dtype=G_TYPE),
                                    maxiter=maxiter, rtol=rtol, callback=_cb)
            pbar.close()
            if info_code == 0:
                print(f"[iter3d]   CG status: converged (rtol={rtol:g}, maxiter={maxiter})")
            elif info_code > 0:
                if info_code >= maxiter:
                    print(f"[iter3d]   CG status: did not reach tolerance (hit maxiter={maxiter})")
                else:
                    print(f"[iter3d]   CG status: stopped after {info_code} iterations before tolerance")
            else:
                print(f"[iter3d]   CG status: solver breakdown/error (info={info_code})")
            X_mc[:, ch] = x_flat.reshape(N_vox, n_spec).T.reshape(n_spec, Nx, Ny, Nz)

    # wSVD combination: (n_spec, n_ch, Nx, Ny, Nz) → (n_spec, Nx, Ny, Nz)
    print("[iter3d] Applying wSVD coil combination ...")
    recon_fid,   quality = wsvd_combine(X_mc,       noise_cov=noise_cov)
    adj_raw_fid, _       = wsvd_combine(adj_raw_mc, noise_cov=noise_cov)
    adj_dcf_fid, _       = wsvd_combine(adj_dcf_mc, noise_cov=noise_cov)

    if n_ch > 1:
        print(f"[iter3d] wSVD quality  mean={float(np.mean(quality)):.3f}  "
              f"min={float(np.min(quality)):.3f}  max={float(np.max(quality)):.3f}")

    # (n_spec, Nx, Ny, Nz) → (Nx, Ny, Nz, n_spec)
    return (recon_fid.transpose(1, 2, 3, 0).astype(D_TYPE),
            adj_raw_fid.transpose(1, 2, 3, 0).astype(D_TYPE),
            adj_dcf_fid.transpose(1, 2, 3, 0).astype(D_TYPE))


# ── Plotting ─────────────────────────────────────────────────────────────────

def find_best_snr_3d(image_spec):
    Nx, Ny, Nz, n_spec = image_spec.shape
    snr_map = np.zeros((Nx, Ny, Nz))
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                s = np.nan_to_num(np.abs(image_spec[x, y, z, :]))
                snr_map[x, y, z] = s.max() / (np.std(s[int(0.8*n_spec):]) + 1e-20)
    idx = np.unravel_index(snr_map.argmax(), snr_map.shape)
    return idx[0], idx[1], idx[2], snr_map


def _plot_single_slice_spectra_grid(spec_arr, PPM, z, fig_dir, fname_prefix,
                                    method_name, tag="", ppmlim=None,
                                    norm_mode="noise", lb_hz=None,
                                    best_voxel=None):
    """Plot one spaced voxel-spectra grid for a single reconstruction method."""
    Nx, Ny, Nz, n_spec = spec_arr.shape
    if z < 0 or z >= Nz:
        return

    fig, axes = plt.subplots(Nx, Ny, figsize=(Ny * 1.6, Nx * 1.25), squeeze=False)
    axes = np.array(axes).reshape(Nx, Ny)

    if ppmlim is None:
        ppm_min, ppm_max = float(np.min(PPM)), float(np.max(PPM))
    else:
        ppm_min, ppm_max = float(min(ppmlim)), float(max(ppmlim))
    xlim = (ppm_max, ppm_min)

    tail = slice(int(0.8 * n_spec), n_spec)
    y_top = 0.0

    for yi in range(Ny):
        for xi in range(Nx):
            ax = axes[Nx - 1 - xi, yi]
            s = np.nan_to_num(np.abs(spec_arr[xi, yi, z, :]))

            if norm_mode == "peak":
                s_norm = s / (np.max(s) + 1e-20)
            else:
                s_norm = s / (np.std(s[tail]) + 1e-20)

            y_top = max(y_top, float(np.percentile(s_norm, 99.5)))
            ax.plot(PPM, s_norm, color="black", lw=0.5)
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

    lb_txt = f" | LB={float(lb_hz):.0f} Hz" if lb_hz is not None else ""
    fig.suptitle(
        f"{method_name} | slice {z + 1}/{Nz} | grid {Nx}x{Ny} | norm={norm_mode}{lb_txt} | ppm [{ppm_min:.1f}, {ppm_max:.1f}]",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.05, top=0.92,
                        wspace=0.22, hspace=0.22)
    fig.savefig(os.path.join(fig_dir, f"{fname_prefix}_z{z:02d}{tag}.png"), dpi=170)
    plt.close(fig)


def _apply_output_xy_orientation(arr, rot90_k=0, flip_lr=False):
    """Apply XY orientation transform to a 4D spectra volume.

    arr shape is (Nx, Ny, Nz, n_spec). Rotation uses numpy.rot90 semantics on
    axes (0, 1); optional left-right flip is applied in display columns (axis 1).
    """
    out = np.rot90(arr, k=int(rot90_k) % 4, axes=(0, 1))
    if flip_lr:
        out = np.flip(out, axis=1)
    return out


def _interp_z_plus_one(arr):
    """Linear interpolation along z: (Nx,Ny,Nz,n_spec) -> (Nx,Ny,Nz+1,n_spec)."""
    Nx, Ny, Nz, n_spec = arr.shape
    if Nz < 1:
        return arr

    z_old = np.arange(Nz, dtype=np.float64)
    z_new = np.linspace(0.0, float(Nz - 1), Nz + 1, dtype=np.float64)

    out = np.empty((Nx, Ny, Nz + 1, n_spec), dtype=arr.dtype)
    is_complex = np.iscomplexobj(arr)

    for xi in range(Nx):
        for yi in range(Ny):
            block = arr[xi, yi, :, :]  # (Nz, n_spec)
            if is_complex:
                re = np.real(block)
                im = np.imag(block)
                for si in range(n_spec):
                    out[xi, yi, :, si] = (
                        np.interp(z_new, z_old, re[:, si])
                        + 1j * np.interp(z_new, z_old, im[:, si])
                    )
            else:
                for si in range(n_spec):
                    out[xi, yi, :, si] = np.interp(z_new, z_old, block[:, si])

    return out


def plot_results(adj_dcf_spec, iter_spec, PPM, ppmlim, fig_dir, tag="",
                 adj_raw_spec=None, apodization_hz=None):
    Nx, Ny, Nz, n_spec = iter_spec.shape
    # Use DCF adjoint as reference for robust peak localization (same convention as 06c).
    ref_spec = adj_dcf_spec
    vx, vy, vz, snr_map = find_best_snr_3d(ref_spec)
    print(f"[plot] Best SNR voxel: [{vx},{vy},{vz}]  SNR={snr_map[vx,vy,vz]:.1f}")

    ppm_min, ppm_max = float(min(ppmlim)), float(max(ppmlim))
    ppm_mask = (PPM >= ppm_min) & (PPM <= ppm_max)
    if not np.any(ppm_mask):
        ppm_mask = np.ones_like(PPM, dtype=bool)
    xlim = (ppm_max, ppm_min)

    peaks = [(0, 'PCr'), (5, 'Pi'), (-2.5, 'g-ATP'), (-7.5, 'a-ATP'), (-16, 'b-ATP')]

    rows_spec = []
    if adj_raw_spec is not None:
        rows_spec.append((adj_raw_spec, "Adj (raw)"))
    rows_spec.append((adj_dcf_spec, "Adj (DCF)"))
    rows_spec.append((iter_spec, "Iterative CG"))
    n_recon = len(rows_spec)

    # Spatial maps
    ncols = min(Nz, 6)
    nrows = int(np.ceil(Nz / ncols))
    fig, axes = plt.subplots(nrows * n_recon, ncols,
                             figsize=(ncols * 3.0, nrows * n_recon * 2.5 + 0.8))
    axes = np.array(axes).reshape(nrows * n_recon, ncols)

    map_cache = []
    for spec_arr, _ in rows_spec:
        mag3d = np.mean(np.nan_to_num(np.abs(spec_arr[..., ppm_mask])), axis=-1)
        vmax = float(np.percentile(mag3d, 99.5))
        if vmax <= 0:
            vmax = float(np.max(mag3d) + 1e-20)
        map_cache.append((mag3d, 0.0, vmax))

    row_images = [None] * (nrows * n_recon)
    for z in range(Nz):
        block = z // ncols
        col = z % ncols
        for ri, (_spec_arr, rlabel) in enumerate(rows_spec):
            row = block * n_recon + ri
            mag3d, vmin, vmax = map_cache[ri]
            ax = axes[row, col]
            im = ax.imshow(np.flipud(mag3d[:, :, z]), origin="upper", cmap="viridis",
                           vmin=vmin, vmax=vmax)
            if row_images[row] is None:
                row_images[row] = im
            ax.set_title(f"{rlabel} | slice {z + 1}/{Nz}", fontsize=8)
            ax.axis("off")
            if ri == n_recon - 1 and vz == z:
                ax.plot(vy, Nx - 1 - vx, "r+", ms=8, mew=1.5)
    # Hide unused axes
    for z in range(Nz, nrows * ncols):
        for ri in range(n_recon):
            axes[z // ncols * n_recon + ri, z % ncols].axis("off")

    for row in range(nrows * n_recon):
        if row_images[row] is not None:
            cbar = fig.colorbar(row_images[row], ax=axes[row, :], fraction=0.015, pad=0.01)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("mean |spectrum|", fontsize=7)

    title_parts = " | ".join(rl for _, rl in rows_spec)

    fig.suptitle(
        f"Spatial maps | methods: {title_parts} | ppm [{ppm_min:.2f}, {ppm_max:.2f}]",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"spatial_maps{tag}.png"), dpi=150)
    plt.close(fig)

    # Best-SNR spectrum
    fig, ax = plt.subplots(figsize=(10, 4))
    sp = np.nan_to_num(np.abs(iter_spec[vx, vy, vz, :]))
    ax.fill_between(PPM, sp, alpha=0.25)
    ax.plot(PPM, sp, lw=0.8)
    ax.set_xlim(xlim); ax.set_xlabel("ppm"); ax.set_ylabel("|spectrum|")
    ax.set_title(
        f"Best-SNR voxel spectrum | Iterative CG | voxel [{vx},{vy},{vz}] | slice {vz + 1}/{Nz} | SNR={snr_map[vx,vy,vz]:.1f}"
    )
    for ppm_ref, name in peaks:
        ax.axvline(ppm_ref, color="red", lw=0.6, ls="--")
        ax.text(ppm_ref, ax.get_ylim()[1]*0.95, name, color="red",
                fontsize=7, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"spectrum_best_snr{tag}.png"), dpi=150)
    plt.close(fig)

    # All-voxel spectra for EVERY z-slice and method (one PNG per method per slice)
    for spec_arr, method_name, fname_prefix in [
        (adj_raw_spec if adj_raw_spec is not None else None,
         "Adjoint (raw)", "adj_raw_all_spectra"),
        (adj_dcf_spec, "Adjoint (DCF)", "adj_dcf_all_spectra"),
        (iter_spec, "Iterative CG", "iter3d_all_spectra"),
    ]:
        if spec_arr is None:
            continue
        for z in range(Nz):
            _plot_single_slice_spectra_grid(
                spec_arr, PPM, z, fig_dir,
                fname_prefix=fname_prefix,
                method_name=method_name,
                tag=tag,
                ppmlim=ppmlim,
                norm_mode="noise",
                lb_hz=apodization_hz,
                best_voxel=(vx, vy, vz),
            )

    print(f"[plot] Figures saved to {fig_dir}  ({Nz} slice maps)")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Interleaved 1H/31P CONCEPT 3D CRT — 31P reconstruction")
    p.add_argument("--dat", required=True,
                   help="Siemens Twix .dat file (interleaved flash_3dcrt)")
    p.add_argument("--out-dir", default="./output_interleaved")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--pad-z-shift", choices=["auto", "always", "never"], default="auto",
                   help="MATLAB-style zResolution handling for FOV shift: "
                        "auto=use Nz+1 on 7T, always=always Nz+1, never=use Nz.")
    p.add_argument("--pad-z-output", choices=["auto", "always", "never"], default="auto",
                   help="MATLAB-like z interpolation for saved/plot outputs: "
                        "auto=interpolate Nz->Nz+1 on 7T, always=always +1, never=no interpolation.")
    p.add_argument("--pad-z-dim", action="store_true",
                   help="MATLAB padZDim: reconstruct DIRECTLY on an Nz+1 z-grid "
                        "(zResolution = Nz+1 into the NUFFT and FOV shift), instead of "
                        "the post-hoc z interpolation. Overrides --pad-z-shift/--pad-z-output.")
    p.add_argument("--maxiter", type=int, default=0,
                   help="CG iterations (0 = DCF adjoint only [default], >0 = iterative CG)")
    p.add_argument("--lam", type=float, default=1.0,
                   help="Scale-invariant Tikhonov weight (fraction of lam_max)")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--matrix-size", type=int, default=None,
                   help="Override XY matrix size (default: lBaseResolution from header)")
    p.add_argument("--matrix-size-z", type=int, default=None)
    p.add_argument("--no-wsvd-noise-cov", action="store_true",
                   help="Disable noise-covariance prewhitening in wSVD (use identity)")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--apodization", type=float, default=0.0,
                   help="Gaussian apodization FWHM linewidth in Hz (0 = none, e.g. 30)")
    p.add_argument("--out-rot90", type=int, default=0,
                   help="Rotate reconstructed spectra volume in XY by k*90deg before saving/plotting")
    p.add_argument("--out-flip-lr", action="store_true",
                   help="Flip reconstructed spectra volume left-right in XY before saving/plotting")
    p.add_argument("--cache-dir", default=None,
                   help="Directory for the parsed-twix cache. When set, the twix "
                        "parse is done once per .dat and reused by later runs "
                        "(cache files are roughly the size of kdata).")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Ignore any existing cache entry and re-parse the .dat")
    p.add_argument("--rep", type=int, default=None,
                   help="Reconstruct a single repetition (1-indexed, MATLAB repToRecon "
                        "style; reps are detected from the ring loop, since img.Rep is "
                        "often all-zero). Omit to use the first complete k-space. The "
                        "number of reps found is printed during loading / --inspect-only.")
    p.add_argument("--rep-average", action="store_true",
                   help="Average all reps' k-space BEFORE recon (√N SNR). Overrides --rep.")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(args.out_dir, "data")
    fig_dir  = os.path.join(args.out_dir, "figures")
    for d in [data_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load phasecor (31P) ──────────────────────────────────────────────
    load_kwargs = dict(
        matrix_size_override=args.matrix_size,
        matrix_size_z_override=args.matrix_size_z,
        rep_override=args.rep,
        rep_average=args.rep_average,
    )
    # Keep the default (rep=None) cache key stable; give each rep its own entry.
    key_extras = dict(load_kwargs)
    if args.rep is None:
        key_extras.pop("rep_override", None)
    if not args.rep_average:
        key_extras.pop("rep_average", None)
    info = cached_load(
        load_interleaved_31p, args.dat,
        cache_dir=args.cache_dir, key_extras=key_extras,
        refresh=args.refresh_cache, tag="31p3d", **load_kwargs)

    _nreps = info.get("n_reps", 1)
    if args.rep is not None:
        print(f"\n[recon] Repetitions: {_nreps} found; reconstructing rep {args.rep}")
    else:
        print(f"\n[recon] Repetitions: {_nreps} found; using first complete k-space "
              f"(pass --rep 1..{_nreps} to pick one)")

    if args.inspect_only:
        print("\nInspection complete. Run without --inspect-only to reconstruct.")
        return

    kdata_mc = info["kdata"]   # (pts_per_ring, n_spec, n_ch, n_rings_total)
    n_ch     = info["n_ch"]
    N        = info["matrix_size"]
    Nz       = info["Nz"]
    n_spec   = info["n_spec"]
    n_rings  = info["n_rings"]
    n_par    = info["n_par"]
    is_7t    = float(info["larmor_freq_mhz"]) > 80.0

    # MATLAB padZDim: reconstruct directly on an Nz+1 grid (zResolution = Nz+1)
    Nz_recon = Nz + 1 if args.pad_z_dim else Nz
    if args.pad_z_dim:
        print(f"[recon] padZDim=True: reconstructing directly on Nz+1={Nz_recon} "
              f"z-grid (MATLAB-style), post-hoc z interpolation disabled")

    print(f"\n[recon] Grid: {N}×{N}×{Nz_recon},  n_spec={n_spec},  "
          f"rings={n_rings},  par={n_par},  n_ch={n_ch}")

    # ── 2. Phase correction (per-channel) ───────────────────────────────────
    if not args.no_phase_corr:
        print("[recon] Applying phase correction per channel ...")
        ti_per_ring = info["_ti_per_ring"]
        for ch in range(n_ch):
            kdata_mc[:, :, ch:ch+1, :] = concept_phase_correction(
                kdata_mc[:, :, ch:ch+1, :], ti_per_ring)
        print(f"[recon] kdata after phase corr: {kdata_mc.shape}")

    # ── 3. Trajectory ───────────────────────────────────────────────────────
    gamma_mhz_per_mt = info["gamma_2pi"] / 1e9
    n_rings_total    = info["n_rings_total"]
    traj = build_trajectory_3d(
        info["first_circle"], n_rings_total, info["traj_pts"],
        info["kz_cpp_per_ring"], gamma_mhz_per_mt, info["fov_m"], N)
    # Reconstruct on a FIXED PHYSICAL FOV (matching MATLAB CONCEPTPerform2DDirectFT):
    # rescale the max-normalized in-plane trajectory (outer ring -> ±0.5) back to
    # physical cycles/pixel (outer -> k_norm_max_phys < 0.5). max-norm ties the recon
    # FOV to the outermost ring, so it is ring-count dependent and shifts an off-centre
    # source by ~1 voxel between different ring counts; physical scaling makes the
    # FOV/voxel registration ring-count invariant. kz is already cycles/pixel.
    traj = traj.copy()
    traj[:, 0] *= float(info["k_norm_max_phys"]) / 0.5
    traj[:, 1] *= float(info["k_norm_max_phys"]) / 0.5

    # ── 4. Flatten k-space: (n_kpts, n_spec, n_ch) ──────────────────────────
    # kdata_mc: (pts_per_ring, n_spec, n_ch, n_rings_total)
    # → transpose to (n_rings_total, pts_per_ring, n_spec, n_ch) → reshape
    kdata_flat_mc = kdata_mc.transpose(3, 0, 1, 2).reshape(-1, n_spec, n_ch)
    print(f"[recon] kdata_flat_mc: {kdata_flat_mc.shape}  (n_kpts, n_spec, n_ch)")

    # ── 5. FOV shift ────────────────────────────────────────────────────────
    if not args.no_fov_shift:
        Nz_base = int(info["Nz"])
        if args.pad_z_dim:
            # MATLAB passes zResolution (=Nz+1) to calculateInPlaneShiftBeforeRecon.
            z_res_shift = Nz_recon
            z_mode_msg = "padZDim(Nz+1)"
        elif args.pad_z_shift == "always":
            z_res_shift = Nz_base + 1
            z_mode_msg = "always"
        elif args.pad_z_shift == "never":
            z_res_shift = Nz_base
            z_mode_msg = "never"
        else:
            # MATLAB interleaved 7T workflow pads zResolution by +1 for shift calc.
            z_res_shift = Nz_base + 1 if is_7t else Nz_base
            z_mode_msg = "auto(7T:+1)" if is_7t else "auto(no pad)"

        d_ro, d_pe, d_sl = compute_fov_shift_3d(info, z_resolution_for_shift=z_res_shift)
        # CRT ICE trajectory physical directions (all orientations):
        #   kx = -dGp,  ky = -dGr,  kz = +snv
        # Because kx/ky are NEGATED physical directions, d_pe and d_ro already have
        # the correct sign for centering (the negation cancels in the formula).
        # kz = +snv (not negated), so d_sl must be negated for correct centering.
        shift_phase = np.exp(
            -2j * np.pi * (traj[:, 0] * d_pe + traj[:, 1] * d_ro + traj[:, 2] * (-d_sl)))
        kdata_flat_mc = kdata_flat_mc * shift_phase[:, np.newaxis, np.newaxis]
        print(f"[recon] Z shift resolution: Nz_base={Nz_base}, z_res_shift={z_res_shift} ({z_mode_msg})")
        print(f"[recon] FOV shift: kx={d_pe:.4f}, ky={d_ro:.4f}, kz(sl)={-d_sl:.4f} vox")

    # ── 6. Per-coil adjoint + iterative CG → wSVD ─────────────────────────
    t0 = time.time()
    # MATLAB applies the Hamming term on physical in-plane radius. traj is now already
    # physical cyc/pix (rescaled above), so r_xy IS that radius (no extra factor).
    r_xy       = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    ham_radius = r_xy

    noise_cov = None if args.no_wsvd_noise_cov else info.get("noise_cov", None)
    if args.no_wsvd_noise_cov:
        print("[recon] wSVD noise prewhitening disabled (--no-wsvd-noise-cov)")
    elif noise_cov is None:
        print("[recon] wSVD noise prewhitening: no noise scans found, using identity")

    recon_fid, adj_raw_fid, adj_dcf_fid = iterative_recon_3d(
        kdata_flat_mc, traj,
        N, N, Nz_recon, n_spec, n_ch,
        noise_cov=noise_cov,
        dcf_hamming_radius=ham_radius,
        maxiter=args.maxiter, lam=args.lam, rtol=args.rtol)
    print(f"[recon] Reconstruction done in {time.time()-t0:.1f}s")

    # Even-matrix NUFFT centering correction. finufft centres the isocentre at index
    # N//2, whereas the Fessler/MATLAB convention (n_shift = ceil(N/2)-0.5) is symmetric
    # about the VoI centre; for EVEN matrices this differs by one in-plane voxel. Roll
    # the reconstructed grid by one voxel on both in-plane axes so the recon matches the
    # MATLAB pipeline (empirically the "left + down one voxel" offset). No-op for odd N.
    if N % 2 == 0:
        recon_fid   = np.roll(recon_fid,   (-1, -1), axis=(0, 1))
        adj_raw_fid = np.roll(adj_raw_fid, (-1, -1), axis=(0, 1))
        adj_dcf_fid = np.roll(adj_dcf_fid, (-1, -1), axis=(0, 1))
        print("[recon] Even-N centering: rolled in-plane grid by (-1,-1) to match MATLAB")

    # Optional MATLAB-like z interpolation to match Siemens CSI grid (e.g., 11 -> 12 at 7T).
    # Skipped when --pad-z-dim already reconstructed on the Nz+1 grid.
    Nz_base_out = recon_fid.shape[2]
    if args.pad_z_dim:
        z_res_output = Nz_base_out
        z_out_mode = "padZDim(recon grid, no interp)"
    elif args.pad_z_output == "always":
        z_res_output = Nz_base_out + 1
        z_out_mode = "always"
    elif args.pad_z_output == "never":
        z_res_output = Nz_base_out
        z_out_mode = "never"
    else:
        z_res_output = Nz_base_out + 1 if is_7t else Nz_base_out
        z_out_mode = "auto(7T:+1)" if is_7t else "auto(no pad)"

    if z_res_output == Nz_base_out + 1:
        recon_fid   = _interp_z_plus_one(recon_fid)
        adj_raw_fid = _interp_z_plus_one(adj_raw_fid)
        adj_dcf_fid = _interp_z_plus_one(adj_dcf_fid)
        print(f"[recon] Output Z interpolation: Nz={Nz_base_out} -> {z_res_output} ({z_out_mode})")
    else:
        print(f"[recon] Output Z interpolation: Nz={Nz_base_out} (no change, {z_out_mode})")

    # Optional output orientation correction applied consistently to all outputs.
    if (int(args.out_rot90) % 4) != 0 or args.out_flip_lr:
        recon_fid = _apply_output_xy_orientation(
            recon_fid, rot90_k=args.out_rot90, flip_lr=args.out_flip_lr)
        adj_raw_fid = _apply_output_xy_orientation(
            adj_raw_fid, rot90_k=args.out_rot90, flip_lr=args.out_flip_lr)
        adj_dcf_fid = _apply_output_xy_orientation(
            adj_dcf_fid, rot90_k=args.out_rot90, flip_lr=args.out_flip_lr)
        print(f"[recon] Output XY orientation: rot90_k={int(args.out_rot90) % 4}, "
              f"flip_lr={bool(args.out_flip_lr)}")

    # ── 9. FID → spectrum (with optional apodization) ────────────────────────
    # Gaussian apodization before FFT: exp(-π²·lb²·t²/(4·ln2))
    # lb is FWHM linewidth in Hz, matches MATLAB: exp(-pi^2*lb^2*t.^2/(4*log(2)))
    apo_tag = ""
    if args.apodization > 0:
        t_apo = np.arange(n_spec) * info["spectral_dwell"]
        apod  = np.exp(-np.pi**2 * args.apodization**2 * t_apo**2 / (4 * np.log(2)))
        recon_fid   = recon_fid   * apod
        adj_raw_fid = adj_raw_fid * apod
        adj_dcf_fid = adj_dcf_fid * apod
        apo_tag     = f"_apo{args.apodization:.0f}Hz"
        print(f"[recon] Applied {args.apodization:.0f} Hz Gaussian apodization")

    # Conjugate FID before FFT: mirrors spectrum around f=0 to match scanner sign
    # convention (paired with the negated PPM axis below, as in 06c).
    recon_spec   = FIDToSpec(recon_fid.conj(),   axis=-1)
    adj_raw_spec = FIDToSpec(adj_raw_fid.conj(), axis=-1)
    adj_dcf_spec = FIDToSpec(adj_dcf_fid.conj(), axis=-1)

    # ── 10. Save ─────────────────────────────────────────────────────────────
    np.save(os.path.join(data_dir, "recon_fid.npy"),      recon_fid)
    np.save(os.path.join(data_dir, "recon_spec.npy"),     recon_spec)
    np.save(os.path.join(data_dir, "adjoint_raw_fid.npy"), adj_raw_fid)
    np.save(os.path.join(data_dir, "adjoint_raw_spec.npy"), adj_raw_spec)
    np.save(os.path.join(data_dir, "adjoint_dcf_fid.npy"), adj_dcf_fid)
    np.save(os.path.join(data_dir, "adjoint_dcf_spec.npy"), adj_dcf_spec)
    with open(os.path.join(args.out_dir, "recon_report.txt"), "w", encoding="ascii") as f:
        f.write("07b interleaved 31P reconstruction report\n")
        f.write(f"dat={args.dat}\n")
        f.write(f"out_dir={args.out_dir}\n")
        f.write(f"matrix_xy={N} nz_raw={Nz} nz_output={recon_fid.shape[2]}\n")
        f.write(f"pad_z_shift={args.pad_z_shift}\n")
        f.write(f"pad_z_output={args.pad_z_output}\n")
        f.write(f"pad_z_dim={bool(args.pad_z_dim)}\n")
        f.write(f"rep={args.rep}\n")
        f.write(f"out_rot90_k={int(args.out_rot90) % 4}\n")
        f.write(f"out_flip_lr={bool(args.out_flip_lr)}\n")
        f.write(f"maxiter={args.maxiter} lam={args.lam} rtol={args.rtol}\n")
        f.write(f"apodization_hz={args.apodization}\n")
    print(f"[recon] Saved to {data_dir}")

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    FREQ   = np.fft.fftshift(np.fft.fftfreq(n_spec, d=info["spectral_dwell"]))
    PPM    = -FREQ / info["larmor_freq_mhz"] + args.ppm_center
    ppmlim = tuple(sorted(args.ppmlim, reverse=True))
    plot_results(adj_dcf_spec, recon_spec, PPM, ppmlim, fig_dir, tag=apo_tag,
                 adj_raw_spec=adj_raw_spec, apodization_hz=args.apodization)
    print("[recon] Done.")


if __name__ == "__main__":
    main()
