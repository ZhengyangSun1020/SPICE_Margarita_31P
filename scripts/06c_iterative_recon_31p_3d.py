#!/usr/bin/env python3
"""
31P CONCEPT 3D CRT iterative NUFFT reconstruction (CG solver).

Unified CRT iterative NUFFT reconstruction for 2D/3D use.

Trajectory model — stack-of-rings (most common for 3D CONCEPT CRT):
  - kx, ky: same 2D CRT rings at every kz level
  - kz: Cartesian phase encoding from Par counter (uniform spacing)
         OR from ICE parameters [4..5] if non-zero (non-Cartesian kz)

Key differences from 2D:
  - Exact Gram matrix (E^H E) precomputed column-by-column, consistent with MATLAB
  - lam_max from max Gram eigenvalue (eigvalsh), same normalization as Gram columns
  - 3D output shape: (Nx, Ny, Nz, n_spec)

Usage:
    python scripts/06c_iterative_recon_31p_3d.py \
        --dat "/mnt/f/AliceConceptB1/in_vivo_3T/phantom/For_Susan/meas_MID02472_FID207000_2D_CRT_12X12_NA1_6rings.dat" \
        --out-dir output_phantom_2D_12x12_6rings --lam 0.001 --maxiter 100
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


# ── GSL slice orientation helpers (shared with 06b) ─────────────────────────

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


# ── Data loading ─────────────────────────────────────────────────────────────

def _detect_kz_per_ring(ice, lin_vals, par_vals, n_rings, n_par, Nz_override,
                         gamma_mhz_per_mt, fov_m_z, idb_vals):
    """Determine kz (in normalized mri-nufft units) for each ring.

    Priority:
    1. n_par > 1 → uniform Cartesian kz from Par counter (tiled per ring)
    2. ICE[21..22] vary across rings → 3D CRT per-ring kz (inner rings at diff kz)
    3. ICE[4..5] vary significantly → legacy kz gradient-moment encoding
    4. All rings at kz=0 (effectively 2D)

    Returns (kz_cpp_per_ring, Nz, kz_source)
    where kz_cpp_per_ring has shape (n_rings_total,).
    """
    def first_scan(r):
        mask = (lin_vals == r) & (idb_vals == 0)
        return int(np.where(mask)[0][0]) if np.any(mask) else None

    # ── Strategy 1: Par counter ─────────────────────────────────────
    if n_par > 1:
        Nz = Nz_override or n_par
        # kdata layout: ring_global = p_idx * n_rings + r
        n_rings_total = n_rings * n_par
        kz_per_ring = np.zeros(n_rings_total)
        for p_idx in range(n_par):
            kz_val = (p_idx - Nz // 2) / Nz
            kz_per_ring[p_idx * n_rings:(p_idx + 1) * n_rings] = kz_val
        return kz_per_ring, Nz, f"Par counter (Nz={Nz}, uniform Cartesian)"

    # ── Strategy 2: ICE[21..23] → kz amplitude (MATLAB freePara encoding) ──
    # MATLAB CONCEPTCalculateKSpacePos3D.m:
    #   freePara(2,...) = ice[:,21]  → sign flag (nonzero = negate)
    #   freePara(3,...) = ice[:,22]  → integer part of z-gradient amplitude
    #   freePara(4,...) = ice[:,23]  → fractional part × 10000
    # Physical kz [cycles/m] = kz_amp [mT/m] * gamma [MHz/mT] * T_z [µs]
    # Normalized (mri-nufft): kz_norm = kz_phys * 0.5 / max(|kz_phys|)
    #   (matches MATLAB: pi*KZ/max(KZ))  T_z cancels in the ratio.
    kz_amp = np.zeros(n_rings)
    for r in range(n_rings):
        s0 = first_scan(r)
        if s0 is None:
            continue
        sign_flag = int(ice[s0, 21])
        int_part  = float(int(ice[s0, 22]))
        frac_part = float(int(ice[s0, 23]))
        amp = int_part + frac_part / 10000.0
        if sign_flag:
            amp = -amp
        kz_amp[r] = amp

    unique_amps = sorted(set(round(a, 4) for a in kz_amp.tolist()))
    kz_amp_max = max(abs(a) for a in unique_amps) if unique_amps else 0.0
    if len(unique_amps) > 1 and kz_amp_max > 0:
        Nz = Nz_override or len(unique_amps)
        kz_cpp = kz_amp * 0.5 / kz_amp_max
        return kz_cpp, Nz, (
            f"ICE[21..23] kz-amp  unique_amp={unique_amps}  "
            f"kz_amp_max={kz_amp_max:.4f} mT/m  Nz={Nz}  "
            f"kz_norm/ring={sorted({round(float(v),4) for v in kz_cpp})}")

    # ── Strategy 3: ICE[4..5] gradient-moment ──────────────────────
    kz_raw = np.zeros(n_rings)
    for r in range(n_rings):
        s0 = first_scan(r)
        if s0 is None:
            continue
        kz_raw[r] = (_ice_to_signed(int(ice[s0, 4]))
                     + _ice_to_signed(int(ice[s0, 5])) / 10000.0)
    kz_range = kz_raw.max() - kz_raw.min()
    if kz_range > 1.0:
        Nz = Nz_override or max(3, int(round(kz_range / max(1, abs(kz_raw[kz_raw != 0]).min()))))
        kz_cpp = kz_raw * gamma_mhz_per_mt * (fov_m_z / Nz)
        return kz_cpp, Nz, f"ICE[4..5] gradient-moment  Nz={Nz}"

    # ── Strategy 4: 2D (all kz=0) ──────────────────────────────────
    return np.zeros(n_rings), 1, "all kz=0 (2D / single slab)"


def load_concept_twix_3d(dat_path, pts_per_ring_override=None,
                          n_spec_override=None,
                          matrix_size_override=None,
                          matrix_size_z_override=None,
                          rep_sel=None):
    """Load 3D CRT CONCEPT data from Siemens Twix .dat file.

    Auto-detects kz encoding:
    - n_par > 1: uniform Cartesian kz from Par counter (stack-of-rings)
    - ICE[21..22] vary: inner-ring 3D CRT (each ring has its own kz level)
    - ICE[4..5] vary: legacy kz gradient-moment encoding
    - otherwise: 2D (all kz=0)
    """
    print(f"\n{'='*70}")
    print(f"Loading: {dat_path}")
    print(f"{'='*70}")

    twix = mapvbvd.mapVBVD(dat_path)
    if isinstance(twix, list):
        twix = twix[-1]

    img = twix['image']
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

    # ── Header parameters ──────────────────────────────────────────────────
    larmor_freq_hz = _hdr_float(
        hdr, 'MeasYaps', ('sTXSPEC', 'asNucleusInfo', '0', 'lFrequency'), default=49894611)
    larmor_freq_mhz = larmor_freq_hz / 1e6
    dwell_ns = _hdr_float(hdr, 'MeasYaps', ('sRXSPEC', 'alDwellTime', '0'), default=180000)
    dwell_s = dwell_ns * 1e-9
    fov_mm = _hdr_float(
        hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'dReadoutFOV'), default=240)
    fov_m = fov_mm / 1e3
    base_res = int(_hdr_float(
        hdr, 'MeasYaps', ('sKSpace', 'lBaseResolution'), default=12))
    matrix_size = matrix_size_override or base_res

    # z resolution from header (MATLAB: fullResolutionZ = sKSpace.lPartitions)
    l_partitions = int(_hdr_float(hdr, 'MeasYaps', ('sKSpace', 'lPartitions'), default=0))

    # z slab thickness (full 3D z-FOV)
    fov_z_mm = _hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'dThickness'))
    if fov_z_mm is None:
        fov_z_mm = _hdr_float(hdr, 'MeasYaps', ('sSliceArray', 'asSlice', '0', 'dThickness'))

    nucleus = hdr.get('MeasYaps', {}).get(
        ('sRXSPEC', 'asNucleusInfo', '0', 'tNucleus'), '"31P"')
    gamma_2pi_hdr = _hdr_float(hdr, 'Dicom', ('lGyroMagnRatioOverTwoPi',))
    if gamma_2pi_hdr is not None:
        gamma_2pi = gamma_2pi_hdr
    elif '31P' in str(nucleus):
        gamma_2pi = 17.235e6
    else:
        gamma_2pi = 42.577e6

    pos_sag = _hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dSag'), default=0)
    pos_cor = _hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dCor'), default=0)
    pos_tra = _hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dTra'), default=0)
    snv_x = _hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dSag'), default=0)
    snv_y = _hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dCor'), default=0)
    snv_z = _hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dTra'), default=1)
    ipr = _hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'sVoI', 'dInPlaneRot'), default=0)

    # ── MDH counters ───────────────────────────────────────────────────────
    lin_vals = img.Lin.astype(int)
    idb_vals = img.Idb.astype(int)
    idc_vals = img.Idc.astype(int)
    idd_vals = img.Idd.astype(int)
    ida_vals = img.Ida.astype(int)
    rep_vals = img.Rep.astype(int)
    n_scans = len(lin_vals)
    n_rings = int(img.NLin)

    try:
        par_vals = img.Par.astype(int)
        n_par = int(par_vals.max()) + 1
    except AttributeError:
        par_vals = np.zeros(n_scans, dtype=int)
        n_par = 1

    ice = img.iceParam  # (n_scans, 24)

    print(f"\n--- Header ---")
    print(f"  Nucleus:        {nucleus}")
    print(f"  Larmor:         {larmor_freq_hz:.0f} Hz  ({larmor_freq_mhz:.3f} MHz)")
    print(f"  Dwell:          {dwell_ns:.0f} ns")
    print(f"  FOV_xy:         {fov_mm:.0f} mm")
    print(f"  FOV_z (slab):   {fov_z_mm} mm")
    print(f"  Matrix XY:      {matrix_size}x{matrix_size}")
    print(f"  lPartitions:    {l_partitions}")
    print(f"  n_rings (NLin): {n_rings}")
    print(f"  n_par  (NPar):  {n_par}")

    ice_kz_nonzero = np.any(ice[:, 4:6] != 0)
    print(f"\n--- ICE params (first 5 scans) ---")
    for s in range(min(5, n_scans)):
        print(f"  scan {s:2d}  Lin={lin_vals[s]:2d}  Par={par_vals[s]:2d}  "
              f"ice[0..7]={ice[s, :8].astype(int).tolist()}")
    print(f"  ice[4..5] (kz?) non-zero: {ice_kz_nonzero}")

    print(f"\n--- MDH counters ---")
    print(f"  Lin: {lin_vals.min()}..{lin_vals.max()}")
    print(f"  Par: {par_vals.min()}..{par_vals.max()}")
    print(f"  Idb: {idb_vals.min()}..{idb_vals.max()}")
    print(f"  Idc: {idc_vals.min()}..{idc_vals.max()}")
    print(f"  Idd: {idd_vals.min()}..{idd_vals.max()}")
    print(f"  Ida: {ida_vals.min()}..{ida_vals.max()}")
    n_reps = len(np.unique(rep_vals))
    print(f"  Rep: {rep_vals.min()}..{rep_vals.max()}  ({n_reps} dynamic measurements)")

    # ── Rep filter (dynamic data) ──────────────────────────────────────────
    if rep_sel is not None and n_reps > 1:
        avail = sorted(np.unique(rep_vals).tolist())
        if rep_sel not in avail:
            raise ValueError(f"--rep {rep_sel} not found; available reps: {avail}")
        rep_mask = (rep_vals == rep_sel)
        lin_vals = lin_vals[rep_mask]
        idb_vals = idb_vals[rep_mask]
        idc_vals = idc_vals[rep_mask]
        idd_vals = idd_vals[rep_mask]
        ida_vals = ida_vals[rep_mask]
        par_vals = par_vals[rep_mask]
        ice      = ice[rep_mask]
        _rep_scan_indices = np.where(rep_mask)[0]
        n_scans = len(lin_vals)
        print(f"  Rep filter: Rep={rep_sel}  → {n_scans} scans selected")
    else:
        _rep_scan_indices = None
        if rep_sel is not None:
            print(f"  Rep filter: only 1 Rep present → using all scans")

    # Nz is determined later by _detect_kz_per_ring; placeholder here for printing
    print(f"\n  Nz will be auto-detected from kz encoding")

    # ── Raw data ───────────────────────────────────────────────────────────
    img.squeeze = True
    img.flagRemoveOS = False
    u = img.unsorted()   # (Col_raw, n_cha, n_tr)
    if _rep_scan_indices is not None:
        u = u[..., _rep_scan_indices]   # filter to selected Rep
    col_per_scan = u.shape[0]
    n_cha = u.shape[1] if u.ndim == 3 else 1
    print(f"\n--- Data ---")
    print(f"  Col (raw, no OS removal): {col_per_scan}")
    print(f"  n_coils:        {n_cha}")

    idc_val = int(idc_vals[0])
    pts_per_ring = pts_per_ring_override or (2 * idc_val)
    print(f"  pts_per_ring = 2×Idc = 2×{idc_val} = {pts_per_ring}")

    unique_par = sorted(np.unique(par_vals).tolist())

    # Group scans by (ring, par, temporal-interleave); store all coils per ADC
    scans_by_key = {}
    for s in range(n_scans):
        key = (lin_vals[s], par_vals[s], idb_vals[s])
        if key not in scans_by_key:
            scans_by_key[key] = []
        adc = (u[:, :, s] if u.ndim == 3 else u[:, np.newaxis, s]).copy()  # (col, n_cha)
        adc[-4:] = 0  # zero trailing noise samples
        scans_by_key[key].append(adc)

    # Determine vecSize = spectral points per ring
    ring_vec_sizes = []
    for r in range(n_rings):
        for p in unique_par:
            mask = (lin_vals == r) & (par_vals == p)
            if not np.any(mask):
                continue
            n_ti = int(idb_vals[mask].max()) + 1
            n_adc = int(np.sum(mask & (idb_vals == 0)))
            vec = round(col_per_scan * n_adc / pts_per_ring * n_ti - 0.5)
            ring_vec_sizes.append(vec)

    default_vec = int(_hdr_float(hdr, 'MeasYaps', ('sSpecPara', 'lVectorSize'), default=512))
    pointsInFID = (min(ring_vec_sizes) if ring_vec_sizes else default_vec)
    if n_spec_override is not None:
        pointsInFID = n_spec_override
    print(f"  pointsInFID = {pointsInFID}")

    # ── Build kdata (pts_per_ring, n_spec, n_cha, n_rings_total) ─────────────
    # Index: ring_global = p_idx * n_rings + r
    n_rings_total = n_rings * n_par
    _zeros_adc = np.zeros((col_per_scan, n_cha), dtype=np.complex64)
    kdata = np.zeros((pts_per_ring, pointsInFID, n_cha, n_rings_total), dtype=np.complex64)

    for r in range(n_rings):
        for p_idx, p in enumerate(unique_par):
            rg = p_idx * n_rings + r
            mask = (lin_vals == r) & (par_vals == p)
            if not np.any(mask):
                continue
            n_ti = int(idb_vals[mask].max()) + 1
            vec_per_ti = pointsInFID // n_ti

            if n_ti == 1:
                raw_len = pts_per_ring * pointsInFID
                raw_mc = np.concatenate(
                    scans_by_key.get((r, p, 0), [_zeros_adc]),
                    axis=0)[:raw_len]            # (raw_len, n_cha)
                for cha in range(n_cha):
                    kdata[:, :, cha, rg] = raw_mc[:, cha].reshape(
                        pts_per_ring, pointsInFID, order='F')
            else:
                useful = vec_per_ti * pts_per_ring
                _zeros_ti = np.zeros((useful, n_cha), dtype=np.complex64)
                ti_blocks = np.zeros((pts_per_ring, vec_per_ti, n_ti, n_cha), dtype=np.complex64)
                for ti in range(n_ti):
                    raw_ti_mc = np.concatenate(
                        scans_by_key.get((r, p, ti), [_zeros_ti]),
                        axis=0)[:useful]         # (useful, n_cha)
                    for cha in range(n_cha):
                        ti_blocks[:, :, ti, cha] = raw_ti_mc[:, cha].reshape(
                            pts_per_ring, vec_per_ti, order='F')
                # merge TI blocks per coil: (pts, vec*n_ti, n_cha)
                merged = np.zeros((pts_per_ring, vec_per_ti * n_ti, n_cha), dtype=np.complex64)
                for cha in range(n_cha):
                    merged[:, :, cha] = ti_blocks[:, :, :, cha].reshape(
                        pts_per_ring, vec_per_ti * n_ti)
                eff = min(vec_per_ti * n_ti, pointsInFID)
                kdata[:, :eff, :, rg] = merged[:, :eff, :]

    print(f"\n  kdata shape: {kdata.shape}   max={np.abs(kdata).max():.4e}")

    # ── Phase correction params ────────────────────────────────────────────
    # Idd = TI count per ring (same for all partitions of that ring)
    ti_per_ring_base = np.zeros(n_rings, dtype=int)
    for r in range(n_rings):
        ti_per_ring_base[r] = int(idd_vals[lin_vals == r][0])
    ti_per_ring_3d = np.tile(ti_per_ring_base, n_par)  # (n_rings_total,)
    print(f"  TI per ring (Idd): {ti_per_ring_base}")

    # ── Trajectory parameters ──────────────────────────────────────────────
    delta_gm = 1e9 / (fov_m * gamma_2pi)

    first_circle_kx = np.zeros(n_rings)
    first_circle_ky = np.zeros(n_rings)
    for r in range(n_rings):
        s0 = np.where(lin_vals == r)[0][0]
        kx_int  = _ice_to_signed(float(ice[s0, 0]))
        kx_frac = _ice_to_signed(float(ice[s0, 1]))
        ky_int  = _ice_to_signed(float(ice[s0, 2]))
        ky_frac = _ice_to_signed(float(ice[s0, 3]))
        first_circle_kx[r] = kx_int + kx_frac / 10000.0
        first_circle_ky[r] = ky_int + ky_frac / 10000.0

    # ── kz per ring: auto-detected ────────────────────────────────────────
    gamma_mhz_per_mt = gamma_2pi / 1e9
    fov_m_z = (fov_z_mm / 1e3) if fov_z_mm else fov_m

    kz_cpp_per_ring, Nz, kz_source = _detect_kz_per_ring(
        ice, lin_vals, par_vals, n_rings, n_par,
        matrix_size_z_override, gamma_mhz_per_mt, fov_m_z, idb_vals)

    # Use lPartitions from header as default Nz (MATLAB: fullResolutionZ = sKSpace.lPartitions)
    # kz_cpp_per_ring values use max-normalization and are independent of Nz grid size.
    if matrix_size_z_override is None and l_partitions > 0:
        Nz = l_partitions

    print(f"\n  kz source:    {kz_source}")
    print(f"  Nz:           {Nz}  (from lPartitions={l_partitions})")
    print(f"  kz_cpp/ring:  {kz_cpp_per_ring.round(4).tolist()}")
    print(f"  Ring radii:   {np.sqrt(first_circle_kx**2 + first_circle_ky**2).round(1)}")
    print(f"  delta_gm:     {delta_gm:.2f}")
    r_max = np.sqrt(first_circle_kx**2 + first_circle_ky**2).max()
    k_norm_max_phys = float(r_max * gamma_mhz_per_mt * (fov_m / matrix_size))
    print(f"  k_norm_max (physical): "
          f"{k_norm_max_phys:.4f}  "
          f"(MATLAB max-norm: 0.5000  → max_radius={r_max:.1f})")

    spectral_dwell = dwell_s * 2   # ×2 oversampling correction
    freq_hz_axis = np.fft.fftshift(np.fft.fftfreq(pointsInFID, d=spectral_dwell))
    ppm_axis = freq_hz_axis / larmor_freq_mhz

    return dict(
        kdata=kdata,
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
        kz_cpp_per_ring=kz_cpp_per_ring,   # (n_rings_total,)
        k_norm_max_phys=k_norm_max_phys,
        ppm_axis=ppm_axis,
        pos_sag=pos_sag, pos_cor=pos_cor, pos_tra=pos_tra,
        snv_x=snv_x, snv_y=snv_y, snv_z=snv_z,
        in_plane_rotation=ipr,
        _ti_per_ring=ti_per_ring_3d,
        unique_par=unique_par,
        noise_cov=noise_cov,
    )


# ── Trajectory ───────────────────────────────────────────────────────────────

def build_trajectory_3d(first_circle, n_rings_total, traj_pts_arr,
                         kz_cpp_per_ring,
                         gamma_mhz_per_mt, fov_m, N):
    """Build (n_kpts, 3) 3D CRT trajectory in mri-nufft convention.

    kz_cpp_per_ring : (n_rings_total,) normalized kz per ring ∈ [-0.5, 0.5]
    Returns traj with kx, ky, kz ∈ [-0.5, 0.5] (cycles/pixel).

    For stack-of-rings (n_par > 1): ring_global = p_idx * n_rings_per_par + r,
      matches kdata ordering; kz_cpp_per_ring is already tiled across partitions.
    For per-ring kz (n_par == 1): ring_global == ring index.
    """
    # first_circle holds the per-ring starting point for n_rings (unique rings).
    # When n_par > 1, n_rings_total = n_par * n_rings; the xy part repeats.
    n_rings_base = first_circle.shape[1]  # unique rings (without par repetition)
    radii = np.sqrt(first_circle[0]**2 + first_circle[1]**2)
    phi0 = np.arctan2(first_circle[1], first_circle[0])

    # MATLAB normalization: KX = pi*KX/max(KX), KY = pi*KY/max(KY)
    # maps the outermost ring to ±0.5 (Nyquist), filling the grid.
    # Physical normalization (gamma*fov/N) would only reach ~0.41 for this dataset.
    # Using max-norm matches MATLAB's CONCEPTPerform3DNUFFT_Iterative.m exactly,
    # and ensures the phase ramp for FOV shift is consistent with the NUFFT grid.
    max_radius = float(radii.max())

    all_kx, all_ky, all_kz = [], [], []
    for rg in range(n_rings_total):
        r = rg % n_rings_base  # ring index within the base set
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


# ── FOV shift ────────────────────────────────────────────────────────────────

def compute_fov_shift_3d(info):
    """Return (d_ro, d_pe, d_sl) in voxels for 3D FOV shift correction."""
    pos = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]])
    snv = np.array([info["snv_x"], info["snv_y"], info["snv_z"]])
    snv = snv / (np.linalg.norm(snv) + 1e-30)
    phase_vec, read_vec = _fGSLCalcPRS(snv, info["in_plane_rotation"])
    rot_mat = np.column_stack([phase_vec, read_vec, snv])
    base_pos = rot_mat.T @ pos
    vox_xy = info["fov_mm"] / info["matrix_size"]
    vox_z  = (info["fov_m_z"] * 1e3) / info["Nz"]
    d_pe = base_pos[0] / vox_xy
    d_ro = base_pos[1] / vox_xy
    d_sl = base_pos[2] / vox_z
    return d_ro, d_pe, d_sl


# ── wSVD coil combination ─────────────────────────────────────────────────────

def wsvd_combine(fid_mc, noise_cov=None):
    """Per-voxel weighted SVD coil combination in frequency domain.

    Implements OXSA svdReconstructFrequencyDomain per voxel.

    Parameters
    ----------
    fid_mc    : (n_spec, n_ch, Nx, Ny[, Nz])  per-coil FID images (complex128)
    noise_cov : (n_ch, n_ch) noise covariance matrix, or None

    Returns
    -------
    combined_fid : (n_spec, Nx, Ny[, Nz])
    quality      : (Nx, Ny[, Nz])
    """
    n_spec, n_ch = fid_mc.shape[:2]
    spatial_shape = fid_mc.shape[2:]
    n_vox = int(np.prod(spatial_shape))

    spec_mc   = np.fft.fftshift(np.fft.fft(fid_mc, axis=0), axes=0)
    spec_flat = spec_mc.reshape(n_spec, n_ch, n_vox)

    if noise_cov is None or n_ch == 1:
        scale_mat     = np.eye(n_ch, dtype=np.complex128) * np.sqrt(0.5)
        inv_scale_mat = np.eye(n_ch, dtype=np.complex128) / np.sqrt(0.5)
    else:
        noiseCov = np.asarray(noise_cov, dtype=np.complex128)
        noiseVal, noiseVec = np.linalg.eigh(noiseCov)
        noiseVal      = np.maximum(np.real(noiseVal), 1e-30)
        scale_mat     = noiseVec @ np.diag(np.sqrt(0.5) / np.sqrt(noiseVal))
        inv_scale_mat = np.linalg.inv(scale_mat)

    combined_fd = np.zeros((n_spec, n_vox), dtype=np.complex128)
    quality     = np.zeros(n_vox)
    sqrt_nch    = np.sqrt(n_ch)

    for i in range(n_vox):
        raw    = spec_flat[:, :, i]
        scaled = raw @ scale_mat
        u, s, vh = np.linalg.svd(scaled, full_matrices=False)
        s_norm     = np.linalg.norm(s) + 1e-30
        quality[i] = float((s[0] / s_norm * sqrt_nch - 1) / (sqrt_nch - 1 + 1e-30))
        v1 = vh[0, :].conj()
        svdCoilAmplitudes = v1 @ inv_scale_mat
        ref        = svdCoilAmplitudes[0]
        svdRescale = np.linalg.norm(svdCoilAmplitudes) * ref / (abs(ref) + 1e-30)
        combined_fd[:, i] = u[:, 0] * s[0] * svdRescale

    combined_fd  = combined_fd.reshape((n_spec,) + spatial_shape)
    quality      = quality.reshape(spatial_shape)
    combined_fid = np.fft.ifft(np.fft.ifftshift(combined_fd, axes=0), axis=0)
    return combined_fid, quality


# ── Iterative reconstruction ─────────────────────────────────────────────────

def iterative_recon_3d(kdata_mc, traj, Nx, Ny, Nz, n_spec, n_ch=1,
                        noise_cov=None, nufft_op_batch=None,
                        dcf_hamming_radius=None,
                        maxiter=100, lam=0.0, rtol=1e-3):
    """3D iterative CG reconstruction.

    E^H E is computed exactly via column-by-column NUFFT (same operator as the RHS),
    matching MATLAB CONCEPTPerform3DNUFFT_Iterative.m and the 1H recon_03 reference.

    For multi-coil data (n_ch > 1): runs CG per coil (shared Gram matrix)
    then applies wSVD coil combination in the frequency domain.
    For n_ch=1: single-coil path, no wSVD needed.

    kdata_mc : (n_kpts, n_spec, n_ch)
    traj     : (n_kpts, 3) trajectory in [-0.5, 0.5] (cycles/voxel)
    Returns (recon_fid, adj_raw_fid, adj_dcf_fid) each shape (Nx, Ny, Nz, n_spec).
    """
    N_vox  = Nx * Ny * Nz
    D_TYPE = np.complex64
    G_TYPE = np.complex128

    adjoint_only = (maxiter <= 0)
    if adjoint_only:
        print("[iter3d] Adjoint-only mode (maxiter=0) — skipping Toeplitz PSF and CG")

    # Returns (recon_fid, adj_raw_fid, adj_dcf_fid) each shape (Nx, Ny, Nz, n_spec).
    # recon_fid = CG result (or raw adjoint when maxiter=0)
    # adj_raw_fid = unweighted adjoint (simple gridding)
    # adj_dcf_fid = DCF-weighted adjoint (analytical annular weights)

    # NUFFT on original grid (batched over spectral for adjoint)
    nufft_op_1 = mrinufft.get_operator(
        "finufft", samples=traj, shape=(Nx, Ny, Nz),
        n_coils=n_spec, density=False)

    # CRT DCF with Hamming filter (matches MATLAB op_CalcAndApplyDensComp.m)
    dcf_weights = compute_crt_dcf_hamming(traj, hamming_radius=dcf_hamming_radius)   # (n_kpts,)
    print(f"[iter3d] DCF weights: min={dcf_weights.min():.4f}  max={dcf_weights.max():.4f}")

    if not adjoint_only:
        # Single-coil NUFFT for exact Gram computation (same traj/shape as nufft_op_1)
        nufft_single = mrinufft.get_operator(
            "finufft", samples=traj, shape=(Nx, Ny, Nz), n_coils=1, density=False)

        # Exact Gram G = E^H E (column-by-column).  Consistent with b_flat from nufft_op_1.
        print("[iter3d] Precomputing exact Gram matrix (E^H E) ...")
        Gram = np.zeros((N_vox, N_vox), dtype=G_TYPE)
        for v in tqdm(range(N_vox), desc="Gram cols"):
            delta = np.zeros((Nx, Ny, Nz), dtype=D_TYPE)
            delta.flat[v] = 1.0
            Gram[:, v] = nufft_single.adj_op(nufft_single.op(delta)).reshape(N_vox)

        if lam > 0:
            lam_max = float(np.real(np.linalg.eigvalsh(Gram)).max())
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

    # Per-coil images
    X_mc       = np.zeros((n_spec, n_ch, Nx, Ny, Nz), dtype=G_TYPE)
    adj_raw_mc = np.zeros((n_spec, n_ch, Nx, Ny, Nz), dtype=G_TYPE)
    adj_dcf_mc = np.zeros((n_spec, n_ch, Nx, Ny, Nz), dtype=G_TYPE)

    for ch in range(n_ch):
        label = f"Coil {ch+1}/{n_ch}" if n_ch > 1 else "Single coil"
        print(f"[iter3d] {label} — adjoint ...")
        kdata_ch = kdata_mc[:, :, ch].astype(D_TYPE)     # (n_kpts, n_spec)

        # Batched adjoint: adj_op expects (n_coils, n_kpts) → transpose
        b_ch = nufft_op_1.adj_op(kdata_ch.T).astype(G_TYPE)  # (n_spec, Nx, Ny, Nz)
        adj_raw_mc[:, ch] = b_ch                               # unweighted adjoint

        # DCF-weighted adjoint (apply analytical weights in k-space before adj_op)
        kdata_dcf = kdata_ch * dcf_weights[:, np.newaxis]      # (n_kpts, n_spec)
        adj_dcf_mc[:, ch] = nufft_op_1.adj_op(kdata_dcf.T).astype(G_TYPE)

        if adjoint_only:
            X_mc[:, ch] = adj_dcf_mc[:, ch]   # DCF adjoint as primary output
        else:
            print(f"[iter3d] {label} — CG ...")
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

    # wSVD combination (trivial for n_ch=1: phase-normalised by first coil = identity)
    print("[iter3d] Applying wSVD coil combination ...")
    recon_fid,   quality = wsvd_combine(X_mc,       noise_cov=noise_cov)
    adj_raw_fid, _       = wsvd_combine(adj_raw_mc, noise_cov=noise_cov)
    adj_dcf_fid, _       = wsvd_combine(adj_dcf_mc, noise_cov=noise_cov)

    if n_ch > 1:
        print(f"[iter3d] wSVD quality  mean={float(np.mean(quality)):.3f}  "
              f"min={float(np.min(quality)):.3f}  max={float(np.max(quality)):.3f}")

    return (recon_fid.transpose(1, 2, 3, 0).astype(D_TYPE),
            adj_raw_fid.transpose(1, 2, 3, 0).astype(D_TYPE),
            adj_dcf_fid.transpose(1, 2, 3, 0).astype(D_TYPE))


# ── Plotting ─────────────────────────────────────────────────────────────────

def find_best_snr_3d(image_spec):
    """Return (vy, vx, vz, snr_map) for 3D+spec array (Nx, Ny, Nz, n_spec)."""
    Nx, Ny, Nz, n_spec = image_spec.shape
    snr_map = np.zeros((Nx, Ny, Nz))
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                s = np.abs(image_spec[x, y, z, :])
                snr_map[x, y, z] = s.max() / (np.std(s[int(0.8*n_spec):]) + 1e-20)
    idx = np.unravel_index(snr_map.argmax(), snr_map.shape)
    return idx[0], idx[1], idx[2], snr_map


def plot_results_3d(adj_dcf_spec, iter_spec, PPM, ppmlim, fig_dir, tag="",
                    adj_raw_spec=None):
    """Plot spatial maps and spectra for up to 3 reconstructions.

    adj_dcf_spec : DCF-weighted adjoint  (Nx, Ny, Nz, n_spec)
    iter_spec    : CG iterative or raw adjoint when maxiter=0  (Nx, Ny, Nz, n_spec)
    adj_raw_spec : unweighted adjoint (optional)  (Nx, Ny, Nz, n_spec)
    """
    Nx, Ny, Nz, n_spec = iter_spec.shape
    # Use DCF adjoint for best-SNR detection (most reliable signal peak)
    ref_spec = adj_dcf_spec
    vx, vy, vz, snr_map = find_best_snr_3d(ref_spec)
    print(f"[iter3d] Best SNR voxel: [{vx},{vy},{vz}]  SNR={snr_map[vx,vy,vz]:.1f}")

    peaks = [(0, 'PCr'), (5, 'Pi'), (-2.5, 'g-ATP'), (-7.5, 'a-ATP'), (-16, 'b-ATP')]
    ppm_min, ppm_max = float(min(ppmlim)), float(max(ppmlim))
    ppm_mask = (PPM >= ppm_min) & (PPM <= ppm_max)
    if not np.any(ppm_mask):
        ppm_mask = np.ones_like(PPM, dtype=bool)
    xlim = (ppm_max, ppm_min)

    # Build the list of (spec, row_label) for spatial maps
    rows_spec = []
    if adj_raw_spec is not None:
        rows_spec.append((adj_raw_spec, "Adj (raw)"))
    rows_spec.append((adj_dcf_spec, "Adj (DCF)"))
    rows_spec.append((iter_spec,    "Iter CG"))
    n_recon = len(rows_spec)

    # ── Spatial maps: mean|spectrum| for each z slice ─────────────────
    ncols = min(Nz, 6)
    nrows = int(np.ceil(Nz / ncols))
    fig, axes = plt.subplots(nrows * n_recon, ncols,
                             figsize=(ncols * 3, nrows * n_recon * 2.5 + 0.5))
    axes = np.array(axes).reshape(nrows * n_recon, ncols)

    # Precompute windowed mean-|spectrum| maps and explicit method-specific display ranges.
    # This avoids accidental cross-method visual coupling and makes value scales clear.
    map_cache = []
    for spec_arr, _ in rows_spec:
        mag3d = np.mean(np.nan_to_num(np.abs(spec_arr[..., ppm_mask])), axis=-1)  # (Nx, Ny, Nz)
        vmax = float(np.percentile(mag3d, 99.5))
        if vmax <= 0:
            vmax = float(np.max(mag3d) + 1e-20)
        map_cache.append((mag3d, 0.0, vmax))

    row_images = [None] * (nrows * n_recon)

    for z in range(Nz):
        block = z // ncols
        col   = z % ncols
        for ri, (spec_arr, rlabel) in enumerate(rows_spec):
            row = block * n_recon + ri
            mag3d, vmin, vmax = map_cache[ri]
            mag = mag3d[:, :, z]
            ax  = axes[row, col]
            im = ax.imshow(np.flipud(mag), origin="upper", cmap="viridis",
                           vmin=vmin, vmax=vmax)
            if row_images[row] is None:
                row_images[row] = im
            ax.set_title(f"{rlabel} z={z} [{vmin:.2e},{vmax:.2e}]", fontsize=7)
            ax.axis("off")
            if ri == n_recon - 1 and vz == z:   # mark best voxel on bottom row
                ax.plot(vy, Nx - 1 - vx, "r+", ms=8, mew=1.5)

    for z in range(Nz, nrows * ncols):
        for ri in range(n_recon):
            axes[z // ncols * n_recon + ri, z % ncols].axis("off")

    # Add one colorbar per displayed row so each method row has its own value scale.
    for row in range(nrows * n_recon):
        if row_images[row] is not None:
            cbar = fig.colorbar(row_images[row], ax=axes[row, :], fraction=0.015, pad=0.01)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("mean |spectrum|", fontsize=7)

    title_parts = " | ".join(rl for _, rl in rows_spec)
    fig.suptitle(f"{title_parts} — mean |spectrum| in [{ppm_min:.2f}, {ppm_max:.2f}] ppm", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"iter3d_spatial_maps{tag}.png"), dpi=150)
    plt.close(fig)

    # ── Best SNR spectrum — overlay all 3 ─────────────────────────────
    colors  = ['gray', 'steelblue', 'darkgreen']
    linests = ['--', '-', '-']
    fig, ax = plt.subplots(figsize=(10, 4))
    for ri, (spec_arr, rlabel) in enumerate(rows_spec):
        sp = np.nan_to_num(np.abs(spec_arr[vx, vy, vz, :]))
        ax.plot(PPM, sp, color=colors[ri % len(colors)],
                ls=linests[ri % len(linests)], lw=0.9, label=rlabel)
    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ppm")
    ax.set_ylabel("|spectrum|")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Best SNR voxel [{vx},{vy},{vz}]  (SNR={snr_map[vx,vy,vz]:.1f}){tag}")
    for pos, name in peaks:
        ax.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.text(pos, ax.get_ylim()[1] * 0.92, name, fontsize=8, ha='center', color='red')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"iter3d_spectrum_best_snr{tag}.png"), dpi=150)
    plt.close(fig)

    # ── All spectra grid — one PNG per z slice per reconstruction ────────
    for spec_arr, label_prefix, fname_prefix in [
        (adj_raw_spec if adj_raw_spec is not None else None,
         "Adjoint (raw)",  "adj_raw_all_spectra"),
        (adj_dcf_spec,  "Adjoint (DCF)",  "adj_dcf_all_spectra"),
        (iter_spec,     "Iterative CG",   "iter3d_all_spectra"),
    ]:
        if spec_arr is None:
            continue
        for z in range(Nz):
            fig_z, axes_z = plt.subplots(Nx, Ny, figsize=(Ny * 1.35, Nx * 1.15))
            axes_z = np.array(axes_z).reshape(Nx, Ny)
            for yi in range(Ny):
                for xi in range(Nx):
                    ax = axes_z[Nx - 1 - xi, yi]  # 90° CCW
                    sp_v = np.nan_to_num(np.abs(spec_arr[xi, yi, z, :]))
                    sp_v = sp_v / (sp_v.max() + 1e-20)
                    ax.plot(sp_v, 'k', lw=0.6)
                    ax.set_xlim(0, len(sp_v) - 1)
                    ymax = float(np.max(sp_v)) + 1e-12
                    ymin = float(np.min(sp_v))
                    if ymax - ymin < 1e-6:
                        ymin, ymax = 0.0, 1.0
                    ax.set_ylim(ymin, ymax)
                    ax.axis("off")
                    if z == vz and xi == vx and yi == vy:
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_edgecolor("red")
                            spine.set_linewidth(1.5)
                        ax.axis("on")
                        ax.tick_params(left=False, bottom=False,
                                       labelleft=False, labelbottom=False)
            slice_label = " [best SNR slice]" if z == vz else ""
            fig_z.suptitle(f"{label_prefix} — z={z}{slice_label}{tag}", fontsize=10)
            fig_z.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92,
                                  wspace=0.02, hspace=0.02)
            fig_z.savefig(os.path.join(fig_dir, f"{fname_prefix}_z{z:02d}{tag}.png"), dpi=150)
            plt.close(fig_z)

    print(f"[iter3d] Figures saved to {fig_dir}")


def plot_localizer_grid_with_voxel_spectra(recon_spec, fig_dir, localizer_path,
                                           fov_mm, larmor_freq_mhz, dwell_s,
                                           tag="", ppm_window=(-20.0, 20.0)):
    """Plot localizer background with fixed MRSI grid and voxel spectra.

    This reproduces the previously validated workflow:
    - localizer as grayscale background
    - centered MRSI grid (FOV-sized)
    - per-voxel mini spectra inside each grid cell
    """
    Nx, Ny, Nz, n_spec = recon_spec.shape
    if Nz != 1:
        print("[iter3d] Localizer overlay skipped: only Nz=1 supported")
        return

    ds = pydicom.dcmread(localizer_path)
    loc = ds.pixel_array.astype(np.float64)
    loc = (loc - np.percentile(loc, 2.0)) / (np.percentile(loc, 99.5) - np.percentile(loc, 2.0) + 1e-12)
    loc = np.clip(loc, 0.0, 1.0)
    H, W = loc.shape

    ps_row = float(ds.PixelSpacing[0])
    ps_col = float(ds.PixelSpacing[1])
    box_h = float(fov_mm) / ps_row
    box_w = float(fov_mm) / ps_col
    cell_w = box_w / Ny
    cell_h = box_h / Nx

    center_row = (H - 1) / 2.0
    center_col = (W - 1) / 2.0
    x0 = center_col - box_w / 2.0
    y0 = center_row - box_h / 2.0

    ppm = -np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s * 2.0)) / float(larmor_freq_mhz)
    ppm_min, ppm_max = float(min(ppm_window)), float(max(ppm_window))
    idx = np.where((ppm >= ppm_min) & (ppm <= ppm_max))[0]
    if idx.size < 8:
        idx = np.arange(n_spec)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(loc, cmap='gray', origin='upper')

    for c in range(Ny + 1):
        xc = x0 + c * cell_w
        ax.plot([xc, xc], [y0, y0 + box_h], color='red', lw=0.6)
    for r in range(Nx + 1):
        yr = y0 + r * cell_h
        ax.plot([x0, x0 + box_w], [yr, yr], color='red', lw=0.6)

    for i in range(Nx):
        for j in range(Ny):
            s = np.abs(recon_spec[i, j, 0, idx])
            s = np.nan_to_num(s)
            s = s / (s.max() + 1e-12)

            # Same orientation convention used in the validated manual overlay.
            r = Nx - 1 - i
            c = j
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = cx0 + cell_w
            cy1 = cy0 + cell_h

            px = 0.08 * cell_w
            py = 0.12 * cell_h
            xs = np.linspace(cx0 + px, cx1 - px, s.size)
            ys = (cy1 - py) - s * (cell_h - 2 * py)
            ax.plot(xs, ys, color='cyan', lw=0.32, alpha=0.82)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    ax.set_title('Localizer + voxel spectra overlay', fontsize=10)

    out_png = os.path.join(fig_dir, f"iter3d_localizer_grid_with_voxel_spectra{tag}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    out_txt = os.path.join(fig_dir, f"iter3d_localizer_grid_with_voxel_spectra{tag}.txt")
    with open(out_txt, 'w') as f:
        f.write(f"localizer_used: {localizer_path}\n")
        f.write(f"grid_anchor: centered image\n")
        f.write(f"grid_box_xywh_pix: [{x0:.6f}, {y0:.6f}, {box_w:.6f}, {box_h:.6f}]\n")
        f.write(f"voxel_size_pix: [{cell_w:.6f}, {cell_h:.6f}]\n")
        f.write(f"mapping: r=Nx-1-i, c=j\n")
        f.write(f"ppm_window: [{ppm_min}, {ppm_max}]\n")

    print(f"[iter3d] Localizer overlay saved: {out_png}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="31P CONCEPT 3D CRT iterative NUFFT reconstruction")
    p.add_argument("--dat", required=True, help="Siemens Twix .dat file")
    p.add_argument("--out-dir", default="./output_3d")
    p.add_argument("--inspect-only", action="store_true",
                   help="Print header/MDH info and exit (no reconstruction)")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--maxiter", type=int, default=0,
                   help="CG iterations (0 = DCF adjoint only [default], >0 = iterative CG)")
    p.add_argument("--lam", type=float, default=1.0,
                   help="Scale-invariant Tikhonov weight (fraction of lam_max)")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--matrix-size", type=int, default=None)
    p.add_argument("--matrix-size-z", type=int, default=None,
                   help="Override Nz (default: n_par from MDH)")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--rep", type=int, default=None,
                   help="Dynamic measurement index (0-based Rep counter). "
                        "Omit to use all scans; use 0 for the first measurement.")
    p.add_argument("--apodization", type=float, default=0.0,
                   help="Gaussian apodization FWHM linewidth in Hz (0 = none, e.g. 30)")
    p.add_argument("--localizer", type=str, default=None,
                   help="Optional localizer DICOM .IMA path for grid + voxel-spectra overlay")
    p.add_argument("--extra-shift-ro", type=float, default=0.0,
                   help="Additional readout shift in voxels applied in k-space phase (after header FOV shift)")
    p.add_argument("--extra-shift-pe", type=float, default=0.0,
                   help="Additional phase-encode shift in voxels applied in k-space phase (after header FOV shift)")
    p.add_argument("--extra-shift-sl", type=float, default=0.0,
                   help="Additional slice shift in voxels applied in k-space phase (after header FOV shift)")
    return p.parse_args()


def main():
    args = parse_args()

    rep_tag = f"_rep{args.rep:02d}" if args.rep is not None else ""
    data_dir = os.path.join(args.out_dir, "data")
    fig_dir  = os.path.join(args.out_dir, f"figures{rep_tag}")
    for d in [data_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────
    info = load_concept_twix_3d(
        args.dat,
        matrix_size_override=args.matrix_size,
        matrix_size_z_override=args.matrix_size_z,
        rep_sel=args.rep,
    )

    if args.inspect_only:
        print(f"\n{'='*70}")
        print("Inspection complete. Run without --inspect-only to reconstruct.")
        print(f"{'='*70}")
        return

    kdata   = info["kdata"]
    N       = info["matrix_size"]
    Nz      = info["Nz"]
    n_spec  = info["n_spec"]
    n_rings = info["n_rings"]
    n_par   = info["n_par"]
    dwell_s = info["dwell_s"]

    print(f"\n[iter3d] Grid: {N}×{N}×{Nz},  n_spec={n_spec},  rings={n_rings},  par={n_par}")

    # ── 2. Phase correction ────────────────────────────────────────────────
    if not args.no_phase_corr:
        print("[iter3d] Applying phase correction ...")
        kdata = concept_phase_correction(kdata, info["_ti_per_ring"])
        print(f"[iter3d] kdata after phase corr: {kdata.shape}")

    # ── 3. Trajectory ──────────────────────────────────────────────────────
    gamma_mhz_per_mt = info["gamma_2pi"] / 1e9
    n_rings_total = info["n_rings_total"]
    traj = build_trajectory_3d(
        info["first_circle"], n_rings_total, info["traj_pts"],
        info["kz_cpp_per_ring"],
        gamma_mhz_per_mt, info["fov_m"], N)

    # ── 4. Flatten kdata: (n_kpts, n_spec, n_ch) ──────────────────────────
    # kdata shape: (pts_per_ring, n_spec, n_ch, n_rings_total)
    n_ch = kdata.shape[2]
    kdata_flat_mc = kdata.transpose(3, 0, 1, 2).reshape(-1, n_spec, n_ch)
    print(f"[iter3d] kdata_flat_mc: {kdata_flat_mc.shape}  (n_kpts, n_spec, n_ch={n_ch})")

    # ── 5. FOV shift ───────────────────────────────────────────────────────
    if not args.no_fov_shift:
        d_ro, d_pe, d_sl = compute_fov_shift_3d(info)
        d_ro += args.extra_shift_ro
        d_pe += args.extra_shift_pe
        d_sl += args.extra_shift_sl
        # CRT ICE trajectory physical directions (all orientations):
        #   kx = -dGp,  ky = -dGr,  kz = +snv
        # kx/ky are NEGATED physical directions, so d_pe/d_ro signs are already correct.
        # kz = +snv (not negated), so d_sl must be negated for correct centering.
        shift_phase = np.exp(
            -2j * np.pi * (traj[:, 0] * d_pe + traj[:, 1] * d_ro + traj[:, 2] * (-d_sl)))
        kdata_flat_mc = kdata_flat_mc * shift_phase[:, np.newaxis, np.newaxis]
        print(f"[iter3d] FOV shift: ro={d_ro:.4f}, pe={d_pe:.4f}, kz(sl)={-d_sl:.4f} vox")

    # ── 6. Reconstruct (per-coil CG + wSVD) ───────────────────────────────
    t0 = time.time()
    # MATLAB applies the Hamming term on physical in-plane radius (not max-normalized 0.5 radius).
    # Build that radius from the NUFFT trajectory scale factor.
    r_xy = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    ham_radius = r_xy * (float(info["k_norm_max_phys"]) / 0.5)

    recon_fid, adj_raw_fid, adj_dcf_fid = iterative_recon_3d(
        kdata_flat_mc, traj,
        N, N, Nz, n_spec, n_ch=n_ch,
        noise_cov=info.get("noise_cov", None),
        dcf_hamming_radius=ham_radius,
        maxiter=args.maxiter, lam=args.lam, rtol=args.rtol)
    print(f"[iter3d] Reconstruction done in {time.time()-t0:.1f}s")

    # Gaussian apodization before FFT: exp(-π²·lb²·t²/(4·ln2))
    # lb is FWHM linewidth in Hz, matches MATLAB: exp(-pi^2*lb^2*t.^2/(4*log(2)))
    dwell_spec = dwell_s * 2   # spectral dwell (2× ADC dwell after OS removal)
    if args.apodization > 0:
        t_apo = np.arange(n_spec) * dwell_spec
        apo   = np.exp(-np.pi**2 * args.apodization**2 * t_apo**2 / (4 * np.log(2)))
        recon_fid    = recon_fid    * apo
        adj_raw_fid  = adj_raw_fid  * apo
        adj_dcf_fid  = adj_dcf_fid  * apo
        print(f"[iter3d] Applied {args.apodization:.0f} Hz Gaussian apodization")

    # Conjugate FID before FFT: mirrors spectrum around f=0 to match scanner sign convention
    recon_spec   = FIDToSpec(recon_fid.conj(),   axis=-1)   # (Nx, Ny, Nz, n_spec)
    adj_raw_spec = FIDToSpec(adj_raw_fid.conj(), axis=-1)
    adj_dcf_spec = FIDToSpec(adj_dcf_fid.conj(), axis=-1)

    # ── 8. Save ────────────────────────────────────────────────────────────
    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_spec))
    PPM  = -FREQ / info["larmor_freq_mhz"] + args.ppm_center

    np.save(os.path.join(data_dir, f"iter3d_recon_fid{rep_tag}.npy"),      recon_fid)
    np.save(os.path.join(data_dir, f"iter3d_recon_spec{rep_tag}.npy"),     recon_spec)
    np.save(os.path.join(data_dir, f"iter3d_adj_raw_fid{rep_tag}.npy"),    adj_raw_fid)
    np.save(os.path.join(data_dir, f"iter3d_adj_dcf_fid{rep_tag}.npy"),    adj_dcf_fid)
    print(f"[iter3d] Saved to {data_dir}")

    # ── 9. Figures ─────────────────────────────────────────────────────────
    plot_results_3d(adj_dcf_spec, recon_spec, PPM, args.ppmlim, fig_dir,
                    tag=rep_tag, adj_raw_spec=adj_raw_spec)

    if args.localizer:
        try:
            plot_localizer_grid_with_voxel_spectra(
                recon_spec, fig_dir, args.localizer,
                fov_mm=info["fov_mm"],
                larmor_freq_mhz=info["larmor_freq_mhz"],
                dwell_s=dwell_s,
                tag=rep_tag,
                ppm_window=(-20.0, 20.0),
            )
        except Exception as e:
            print(f"[iter3d] Localizer overlay skipped due to error: {type(e).__name__}: {e}")

    print("[iter3d] Done.")


if __name__ == "__main__":
    main()
