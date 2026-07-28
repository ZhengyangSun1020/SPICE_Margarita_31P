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
  1. Reconstruct each coil independently through the 3D Toeplitz CG
  2. FFT each per-coil image → frequency domain
  3. Per-voxel SVD-based optimal combination (wSVD) → single combined spectrum

Usage:
    python 07b_interleaved_recon_31p_3d.py \\
        --dat "F:/AliceConceptB1/.../flash_3dcrt_8x8x8_20rings_cor_iso.dat" \\
        --out-dir ./output_interleaved [--lam 0.01] [--maxiter 50]
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


def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


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

def compute_fov_shift_3d(info):
    # pos from header is in mm; convert to m to match fov_m
    pos = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]]) / 1000.0
    snv = np.array([info["snv_x"],   info["snv_y"],   info["snv_z"]])
    ipr = info["in_plane_rotation"]
    fov_m = info["fov_m"]
    N = info["matrix_size"]
    fov_m_z = info["fov_m_z"]
    Nz = info["Nz"]
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
                          rep_override=None):
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

    # Prefer phasecor (interleaved 1H/31P); fall back to image (pure 31P)
    if hasattr(twix, 'phasecor'):
        img = twix['phasecor']
        print("  Scan type: phasecor (interleaved 1H/31P)")
    elif hasattr(twix, 'image'):
        img = twix['image']
        print("  Scan type: image (pure 31P — no phasecor found)")
    else:
        raise RuntimeError("No 'phasecor' or 'image' scan found in TWIX file.")
    hdr = twix['hdr']

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

    # ── 31P geometry from navigator array (sSpecPara.sVoI is 1H, not 31P) ──────
    # sNavigatorArray.asElm[0].sCuboid holds the 31P MRSI volume geometry.
    _nav = ('sNavigatorArray', 'asElm', '0', 'sCuboid')
    fov_mm   = _hdr_float(hdr, 'MeasYaps', _nav + ('dReadoutFOV',), default=250)
    fov_m    = fov_mm / 1e3
    fov_z_mm = _hdr_float(hdr, 'MeasYaps', _nav + ('dThickness',),  default=fov_mm)

    # 31P matrix XY: sWipMemBlock.alFree[7] (MATLAB index 8, 1-based)
    n31p_xy   = _hdr_float(hdr, 'MeasYaps', ('sWipMemBlock', 'alFree', '7'), default=None)
    base_res  = int(_hdr_float(hdr, 'MeasYaps', ('sKSpace', 'lBaseResolution'), default=8))
    matrix_size = matrix_size_override or (int(n31p_xy) if n31p_xy else base_res)

    l_partitions = int(_hdr_float(hdr, 'MeasYaps', ('sKSpace', 'lPartitions'), default=0))

    gamma_2pi = 17.235e6   # 31P gyromagnetic ratio [Hz/T]

    pos_sag = _hdr_float(hdr, 'MeasYaps', _nav + ('sPosition', 'dSag'), default=0)
    pos_cor = _hdr_float(hdr, 'MeasYaps', _nav + ('sPosition', 'dCor'), default=0)
    pos_tra = _hdr_float(hdr, 'MeasYaps', _nav + ('sPosition', 'dTra'), default=0)
    snv_x   = _hdr_float(hdr, 'MeasYaps', _nav + ('sNormal',   'dSag'), default=0)
    snv_y   = _hdr_float(hdr, 'MeasYaps', _nav + ('sNormal',   'dCor'), default=0)
    snv_z   = _hdr_float(hdr, 'MeasYaps', _nav + ('sNormal',   'dTra'), default=1)
    ipr     = _hdr_float(hdr, 'MeasYaps', _nav + ('dInPlaneRot',),      default=0)

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

    # ── First-complete-k-space selection (mirrors MATLAB) ────────────────────
    # Find the first scan at which every unique ring (Lin value) has been seen
    # at Ida == max_ida.  Works for both ring-first and Ida-first scan orderings.
    max_ida = int(ida_vals.max())
    if max_ida == 0:
        n_scans_use = n_scans
        print(f"  First-k-space filter: Ida_max=0 (single average) → using all {n_scans} scans")
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
        print(f"  First-k-space filter: Ida_max={max_ida}  "
              f"→ using {n_scans_use} of {n_scans} scans "
              f"({len(covered)}/{len(all_lin_set)} rings covered at Ida=max)")

    # ── Raw data (all channels) ──────────────────────────────────────────────
    img.squeeze      = True
    img.flagRemoveOS = False
    u = img.unsorted()   # (Col_raw, n_ch, n_scans_total)
    # Trim all scan-indexed arrays and raw data to first complete k-space
    lin_vals = lin_vals[:n_scans_use]
    idb_vals = idb_vals[:n_scans_use]
    ida_vals = ida_vals[:n_scans_use]
    idc_vals = idc_vals[:n_scans_use]
    idd_vals = idd_vals[:n_scans_use]
    par_vals = par_vals[:n_scans_use]
    ice      = ice[:n_scans_use]
    u        = u[:, :, :n_scans_use]
    n_scans  = n_scans_use
    col_per_scan = u.shape[0]
    print(f"\n--- Data ---")
    print(f"  Col (raw, no OS removal): {col_per_scan}")
    print(f"  unsorted shape: {u.shape}")

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
        adc = u[:, :, s].copy()   # (Col_raw, n_ch)
        adc[-4:, :] = 0
        scans_by_key[key].append(adc)

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

            if n_ti == 1:
                raw_len = pts_per_ring * pointsInFID
                blocks  = scans_by_key.get((r, p, 0),
                                           [np.zeros((raw_len, n_ch), dtype=np.complex64)])
                raw_mc  = np.concatenate(blocks, axis=0)[:raw_len]   # (raw_len, n_ch)
                for ch in range(n_ch):
                    kdata[:, :, ch, rg] = raw_mc[:, ch].reshape(
                        pts_per_ring, pointsInFID, order='F')
            else:
                useful = vec_per_ti * pts_per_ring
                for ch in range(n_ch):
                    ti_blocks = np.zeros((pts_per_ring, vec_per_ti, n_ti), dtype=np.complex64)
                    for ti in range(n_ti):
                        blocks = scans_by_key.get((r, p, ti),
                                                  [np.zeros((useful, n_ch), dtype=np.complex64)])
                        raw_ti = np.concatenate(blocks, axis=0)[:useful, ch]
                        ti_blocks[:, :, ti] = raw_ti.reshape(
                            pts_per_ring, vec_per_ti, order='F')
                    merged = ti_blocks.reshape(pts_per_ring, vec_per_ti * n_ti)
                    eff = min(vec_per_ti * n_ti, pointsInFID)
                    kdata[:, :eff, ch, rg] = merged[:, :eff]

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
    print(f"  k_norm_max (physical): "
          f"{r_max * gamma_mhz_per_mt * (fov_m / matrix_size):.4f}  "
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


# ── Reconstruction: per-coil adjoint + Toeplitz CG → wSVD ───────────────────

def iterative_recon_3d(kdata_mc, traj, Nx, Ny, Nz, n_spec, n_ch,
                        noise_cov=None, maxiter=50, lam=0.0, rtol=1e-3):
    """Per-coil 3D adjoint (+optional CG) then wSVD combination.

    kdata_mc : (n_kpts, n_spec, n_ch)
    maxiter=0 → adjoint only (fast sanity check), no Toeplitz PSF computed.
    Returns (recon_fid, adj_fid) each shape (Nx, Ny, Nz, n_spec) — wSVD combined.
    """
    N_vox  = Nx * Ny * Nz
    D_TYPE = np.complex64
    G_TYPE = np.complex128

    adjoint_only = (maxiter <= 0)
    if adjoint_only:
        print("[iter3d] Adjoint-only mode (maxiter=0) — skipping Toeplitz PSF and CG")

    # NUFFT operator for adjoint (batched over spectral points)
    nufft_op_1 = mrinufft.get_operator(
        "finufft", samples=traj, shape=(Nx, Ny, Nz),
        n_coils=n_spec, density=False)

    if not adjoint_only:
        Nx2, Ny2, Nz2 = 2 * Nx, 2 * Ny, 2 * Nz
        nufft_op_2x = mrinufft.get_operator(
            "finufft", samples=traj * 0.5, shape=(Nx2, Ny2, Nz2),
            n_coils=1, density=False)

        print("[iter3d] Computing Toeplitz PSF on 2× zero-padded grid ...")
        delta_c = np.zeros((Nx2, Ny2, Nz2), dtype=D_TYPE)
        delta_c[Nx, Ny, Nz] = 1.0
        psf   = nufft_op_2x.adj_op(nufft_op_2x.op(delta_c)).reshape(Nx2, Ny2, Nz2)
        H_toe = np.fft.fftn(np.fft.ifftshift(psf.astype(G_TYPE)))

        if lam > 0:
            lam_max = float(np.real(H_toe).max())
            lam_eff = lam * lam_max
            print(f"[iter3d] lam_max={lam_max:.4g}, lam_rel={lam:.4g} -> lam_eff={lam_eff:.4g}")
        else:
            lam_eff = 0.0

        total_size = N_vox * n_spec

        def mv(x_flat):
            X3D   = x_flat.reshape(N_vox, n_spec).astype(G_TYPE).T.reshape(n_spec, Nx, Ny, Nz)
            X_pad = np.zeros((n_spec, Nx2, Ny2, Nz2), dtype=G_TYPE)
            X_pad[:, :Nx, :Ny, :Nz] = X3D
            Z_pad = np.fft.ifftn(
                np.fft.fftn(X_pad, axes=(1, 2, 3)) * H_toe[np.newaxis], axes=(1, 2, 3))
            Z = Z_pad[:, :Nx, :Ny, :Nz].reshape(n_spec, N_vox).T.astype(G_TYPE)
            if lam_eff > 0:
                Z = Z + lam_eff * x_flat.reshape(N_vox, n_spec).astype(G_TYPE)
            return Z.ravel()

        A_op = LinearOperator((total_size, total_size), matvec=mv, dtype=G_TYPE)

    # Per-coil images: (n_spec, n_ch, Nx, Ny, Nz)
    X_mc   = np.zeros((n_spec, n_ch, Nx, Ny, Nz), dtype=G_TYPE)
    adj_mc = np.zeros((n_spec, n_ch, Nx, Ny, Nz), dtype=G_TYPE)

    for ch in range(n_ch):
        print(f"[iter3d] Coil {ch+1}/{n_ch} — adjoint{'' if adjoint_only else ' + CG'} ...")
        kdata_ch = kdata_mc[:, :, ch].astype(D_TYPE)   # (n_kpts, n_spec)

        # Batched adjoint: adj_op expects (n_coils, n_kpts) → transpose
        b_ch = nufft_op_1.adj_op(kdata_ch.T).astype(G_TYPE)  # (n_spec, Nx, Ny, Nz)
        adj_mc[:, ch] = b_ch
        if adjoint_only:
            X_mc[:, ch] = b_ch
        else:
            b_flat = b_ch.reshape(n_spec, N_vox).T.ravel()
            pbar = tqdm(total=maxiter, desc=f"CG ch{ch}", leave=False)
            def _cb(xk): pbar.update(1)
            x_flat, info_code = cg(A_op, b_flat,
                                    x0=np.zeros(total_size, dtype=G_TYPE),
                                    maxiter=maxiter, rtol=rtol, callback=_cb)
            pbar.close()
            print(f"[iter3d]   CG info={info_code}")
            X_mc[:, ch] = x_flat.reshape(N_vox, n_spec).T.reshape(n_spec, Nx, Ny, Nz)

    # wSVD combination: (n_spec, n_ch, Nx, Ny, Nz) → (n_spec, Nx, Ny, Nz)
    print("[iter3d] Applying wSVD coil combination ...")
    recon_fid, quality = wsvd_combine(X_mc.astype(G_TYPE),   noise_cov=noise_cov)
    adj_fid,   _       = wsvd_combine(adj_mc.astype(G_TYPE), noise_cov=noise_cov)

    print(f"[iter3d] wSVD quality  mean={float(np.mean(quality)):.3f}  "
          f"min={float(np.min(quality)):.3f}  max={float(np.max(quality)):.3f}")

    # (n_spec, Nx, Ny, Nz) → (Nx, Ny, Nz, n_spec)
    return (recon_fid.transpose(1, 2, 3, 0).astype(np.complex64),
            adj_fid.transpose(1, 2, 3, 0).astype(np.complex64))


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


def plot_results(adj_spec, iter_spec, PPM, ppmlim, fig_dir, tag=""):
    Nx, Ny, Nz, n_spec = iter_spec.shape
    vx, vy, vz, snr_map = find_best_snr_3d(iter_spec)
    print(f"[plot] Best SNR voxel: [{vx},{vy},{vz}]  SNR={snr_map[vx,vy,vz]:.1f}")

    # x-axis spans the full ppm range (high ppm left, low ppm right — MRS convention)
    xlim = (float(PPM.max()), float(PPM.min()))

    peaks = [(0, 'PCr'), (5, 'Pi'), (-2.5, 'g-ATP'), (-7.5, 'a-ATP'), (-16, 'b-ATP')]

    # Spatial maps
    ncols = min(Nz, 6)
    nrows = int(np.ceil(Nz / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 3, nrows * 5 + 0.5))
    axes = np.array(axes).reshape(nrows * 2, ncols)
    for z in range(Nz):
        row_adj  = 2 * (z // ncols)
        row_iter = row_adj + 1
        col = z % ncols
        ax  = axes[row_adj, col]
        ax.imshow(np.mean(np.abs(adj_spec[:, :, z, :]), axis=-1).T,
                  origin="lower", cmap="viridis")
        ax.set_title(f"Adj z={z}", fontsize=8); ax.axis("off")
        ax2 = axes[row_iter, col]
        ax2.imshow(np.mean(np.abs(iter_spec[:, :, z, :]), axis=-1).T,
                   origin="lower", cmap="viridis")
        ax2.set_title(f"Iter z={z}", fontsize=8); ax2.axis("off")
        if vz == z:
            ax2.plot(vx, vy, "r+", ms=8, mew=1.5)
    # Hide unused axes
    for z in range(Nz, nrows * ncols):
        axes[2*(z//ncols), z%ncols].axis("off")
        axes[2*(z//ncols)+1, z%ncols].axis("off")
    fig.suptitle("Adjoint (top) vs. Iterative CG (bottom) — mean |spectrum|", fontsize=10)
    fig.savefig(os.path.join(fig_dir, f"spatial_maps{tag}.png"), dpi=150)
    plt.close(fig)

    # Best-SNR spectrum
    fig, ax = plt.subplots(figsize=(10, 4))
    sp = np.nan_to_num(np.abs(iter_spec[vx, vy, vz, :]))
    ax.fill_between(PPM, sp, alpha=0.25)
    ax.plot(PPM, sp, lw=0.8)
    ax.set_xlim(xlim); ax.set_xlabel("ppm"); ax.set_ylabel("|spectrum|")
    ax.set_title(f"Iterative CG — Voxel [{vx},{vy},{vz}]  (SNR={snr_map[vx,vy,vz]:.1f})")
    for ppm_ref, name in peaks:
        ax.axvline(ppm_ref, color="red", lw=0.6, ls="--")
        ax.text(ppm_ref, ax.get_ylim()[1]*0.95, name, color="red",
                fontsize=7, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"spectrum_best_snr{tag}.png"), dpi=150)
    plt.close(fig)

    # All-voxel spectra for EVERY z-slice (one PNG per slice)
    # sharey=True lets matplotlib auto-scale y to the tallest peak in the slice
    for z in range(Nz):
        fig_z, axes_z = plt.subplots(Ny, Nx, figsize=(Nx * 2.5, Ny * 2),
                                      sharex=True, sharey=True)
        axes_z = np.array(axes_z).reshape(Ny, Nx)
        for yi in range(Ny):
            for xi in range(Nx):
                ax = axes_z[Ny - 1 - yi, xi]
                ax.plot(PPM, np.nan_to_num(np.abs(iter_spec[xi, yi, z, :])), lw=0.6)
                ax.set_xlim(xlim)
                ax.tick_params(labelsize=5)
                ax.set_title(f"({xi},{yi})", fontsize=5, pad=1)
                ax.axvline(0.0, color="red", lw=0.4, ls="--")
                if z == vz and xi == vx and yi == vy:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(1.5)
        # Set explicit ylim per slice to avoid degenerate [0,0] range with sharey=True
        vmax_slice = float(np.nanmax(np.nan_to_num(np.abs(iter_spec[:, :, z, :]))))
        for row in axes_z:
            for a in row:
                a.set_ylim(0, max(vmax_slice, 1e-10) * 1.05)
        label = f" [best SNR slice]" if z == vz else ""
        fig_z.suptitle(f"Iterative CG — z={z}{label}", fontsize=10)
        fig_z.tight_layout()
        fig_z.savefig(os.path.join(fig_dir, f"all_spectra_z{z:02d}{tag}.png"), dpi=150)
        plt.close(fig_z)

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
    p.add_argument("--maxiter", type=int, default=50)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--matrix-size", type=int, default=None,
                   help="Override XY matrix size (default: lBaseResolution from header)")
    p.add_argument("--matrix-size-z", type=int, default=None)
    p.add_argument("--no-wsvd-noise-cov", action="store_true",
                   help="Disable noise-covariance prewhitening in wSVD (use identity)")
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--apodization", type=float, default=0.0,
                   help="Lorentzian apodization linewidth (Hz) applied to FID before FFT")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(args.out_dir, "data")
    fig_dir  = os.path.join(args.out_dir, "figures")
    for d in [data_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load phasecor (31P) ──────────────────────────────────────────────
    info = load_interleaved_31p(
        args.dat,
        matrix_size_override=args.matrix_size,
        matrix_size_z_override=args.matrix_size_z,
    )

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

    print(f"\n[recon] Grid: {N}×{N}×{Nz},  n_spec={n_spec},  "
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

    # ── 4. Flatten k-space: (n_kpts, n_spec, n_ch) ──────────────────────────
    # kdata_mc: (pts_per_ring, n_spec, n_ch, n_rings_total)
    # → transpose to (n_rings_total, pts_per_ring, n_spec, n_ch) → reshape
    kdata_flat_mc = kdata_mc.transpose(3, 0, 1, 2).reshape(-1, n_spec, n_ch)
    print(f"[recon] kdata_flat_mc: {kdata_flat_mc.shape}  (n_kpts, n_spec, n_ch)")

    # ── 5. FOV shift ────────────────────────────────────────────────────────
    if not args.no_fov_shift:
        d_ro, d_pe, d_sl = compute_fov_shift_3d(info)
        # CRT ICE trajectory physical directions (all orientations):
        #   kx = -dGp,  ky = -dGr,  kz = +snv
        # Because kx/ky are NEGATED physical directions, d_pe and d_ro already have
        # the correct sign for centering (the negation cancels in the formula).
        # kz = +snv (not negated), so d_sl must be negated for correct centering.
        shift_phase = np.exp(
            -2j * np.pi * (traj[:, 0] * d_pe + traj[:, 1] * d_ro + traj[:, 2] * (-d_sl)))
        kdata_flat_mc = kdata_flat_mc * shift_phase[:, np.newaxis, np.newaxis]
        print(f"[recon] FOV shift: kx={d_pe:.4f}, ky={d_ro:.4f}, kz(sl)={-d_sl:.4f} vox")

    # ── 6. Per-coil adjoint + iterative CG → wSVD ─────────────────────────
    t0 = time.time()
    recon_fid, adj_fid = iterative_recon_3d(
        kdata_flat_mc, traj,
        N, N, Nz, n_spec, n_ch,
        noise_cov=None,
        maxiter=args.maxiter, lam=args.lam, rtol=args.rtol)
    print(f"[recon] Reconstruction done in {time.time()-t0:.1f}s")

    # ── 9. FID → spectrum (with optional apodization) ────────────────────────
    apo_tag = ""
    if args.apodization > 0:
        t_apo = np.arange(n_spec) * info["spectral_dwell"]
        apod  = np.exp(-args.apodization * np.pi * t_apo).astype(np.float32)
        recon_fid = recon_fid * apod
        adj_fid   = adj_fid   * apod
        apo_tag   = f"_apo{args.apodization:.0f}Hz"
        print(f"[recon] Applied Lorentzian apodization: {args.apodization} Hz")
    recon_spec = FIDToSpec(recon_fid, axis=-1)
    adj_spec   = FIDToSpec(adj_fid,   axis=-1)

    # ── 10. Save ─────────────────────────────────────────────────────────────
    np.save(os.path.join(data_dir, "recon_fid.npy"),    recon_fid)
    np.save(os.path.join(data_dir, "recon_spec.npy"),   recon_spec)
    np.save(os.path.join(data_dir, "adjoint_fid.npy"),  adj_fid)
    print(f"[recon] Saved to {data_dir}")

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    PPM    = info["ppm_axis"]
    ppmlim = tuple(sorted(args.ppmlim, reverse=True))
    plot_results(adj_spec, recon_spec, PPM, ppmlim, fig_dir, tag=apo_tag)
    print("[recon] Done.")


if __name__ == "__main__":
    main()
