"""
Reconstruct 31P CONCEPT CRT MRSI directly from Siemens Twix (.dat) file.

Reads the raw data via pymapVBVD, extracts acquisition parameters from
the TWIX header (ICE program parameters for k-space positions), rearranges
the data into (kpts_per_ring, n_spec, n_coils, n_rings), and reconstructs
using mri-nufft (finufft backend) with wSVD coil combination.

Usage:
    python recon_concept_twix_31p.py \
        --dat "F:/path/to/meas_*.dat" \
        --out-dir ./output_31p \
        [--inspect-only] \
        [--pts-per-ring 6] \
        [--ppm-center 0.0] \
        [--ppmlim -20 10] \
        [--plot-voxel VY VX]
"""

from __future__ import annotations
import argparse, os, sys, time
from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mapvbvd
except ImportError:
    sys.exit("pymapvbvd required: pip install pymapvbvd")

try:
    import mrinufft
except ImportError:
    sys.exit("mri-nufft required: pip install mri-nufft[finufft]")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_crt_mrinufft import (
    build_crt_trajectory,
    compute_crt_dcf,
    compute_fov_shift,
    reconstruct_adjoint,
    reconstruct_all_coils_adjoint,
    coil_combine_wsvd,
)

try:
    from fsl_mrs.utils.misc import FIDToSpec
except ImportError:
    def FIDToSpec(x, axis=-1):
        return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


# =====================================================================
# CONCEPT temporal-interleave phase correction
# (Python port of CONCEPTPhaseCorrection_LowRAM.m)
# =====================================================================
def concept_phase_correction(kdata, temporal_interleaves):
    """Apply CONCEPT temporal-interleave phase correction.

    Translates CONCEPTPhaseCorrection_LowRAM.m:
      1. Zero-pad spectral dim 2× (prepend zeros)
      2. iFFT → multiply by conj(phasecorr) → FFT
      3. Keep second half (discard zero-padded region)

    Parameters
    ----------
    kdata : (ns, pointsInFID, n_coils, n_rings)
        Reshaped k-space after MATLAB-style rearrangement.
    temporal_interleaves : int or (n_rings,) array
        Number of temporal interleaves per ring (from ushIdd + 1).

    Returns
    -------
    corrected : same shape as input, phase-corrected.
    """
    ns, pointsInFID, _, n_rings = kdata.shape
    vs = pointsInFID * 2

    if np.isscalar(temporal_interleaves):
        ti_arr = np.full(n_rings, temporal_interleaves)
    else:
        ti_arr = np.asarray(temporal_interleaves)

    # Build per-ring phase correction: (ns, vs, n_rings)
    # MATLAB:  timeoffset = ti * ((1:ns)-1) / ns          → (ns,)
    #          phasecorr  = exp(-2πi * timeoffset' * ((0:vs-1)/vs - 0.5))
    sample_idx = np.arange(ns, dtype=np.float64)       # 0 … ns-1
    freq_axis  = np.arange(vs, dtype=np.float64) / vs - 0.5  # (vs,)

    phasecorr = np.zeros((ns, vs, n_rings), dtype=np.complex128)
    for r in range(n_rings):
        timeoffset = ti_arr[r] * sample_idx / ns        # (ns,)
        phasecorr[:, :, r] = np.exp(-2j * np.pi * np.outer(timeoffset, freq_axis))

    # Zero-pad: prepend pointsInFID zeros along spectral axis
    pad = np.zeros_like(kdata)
    kdata_padded = np.concatenate([pad, kdata], axis=1)  # (ns, vs, C, R)

    # iFFT along spectral dim (axis 1)
    out = np.fft.fftshift(
        np.fft.ifft(np.fft.fftshift(kdata_padded, axes=1), axis=1),
        axes=1,
    )

    # Apply conj(phasecorr)  — broadcast over coils
    out = out * np.conj(phasecorr[:, :, np.newaxis, :])

    # FFT back
    out = np.fft.fftshift(
        np.fft.fft(np.fft.fftshift(out, axes=1), axis=1),
        axes=1,
    )

    # Keep second half: MATLAB  in(:, vs/2+1 : end, …)  →  0-indexed [vs//2 :]
    out = out[:, vs // 2:, :, :]

    return out.astype(np.complex64)


def parse_args():
    p = argparse.ArgumentParser(description="31P CONCEPT CRT recon from Twix")
    p.add_argument("--dat", required=True, help="Path to Siemens Twix .dat file")
    p.add_argument("--out-dir", default="./output_concept_31p")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--pts-per-ring", type=int, default=None,
                   help="Override pts per ring (spatial samples on each circle)")
    p.add_argument("--n-spec", type=int, default=None,
                   help="Override number of spectral points")
    p.add_argument("--matrix-size", type=int, default=None)
    p.add_argument("--coil", type=int, default=None, help="Single coil index (0-based)")
    p.add_argument("--no-all-coils", action="store_true", help="Skip wSVD, use single coil")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--no-phase-corr", action="store_true", help="Skip temporal interleave phase correction")
    p.add_argument("--no-dcf", action="store_true", help="Skip density compensation")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 10])
    p.add_argument("--plot-voxel", type=int, nargs=2, default=None)
    return p.parse_args()


def load_concept_twix(dat_path, pts_per_ring_override=None, n_spec_override=None,
                      matrix_size_override=None):
    """Read CONCEPT CRT data from a Siemens Twix .dat file."""
    print(f"\n{'='*70}")
    print(f"Loading: {dat_path}")
    print(f"{'='*70}")

    twix = mapvbvd.mapVBVD(dat_path)
    if isinstance(twix, list):
        twix = twix[-1]

    img = twix['image']
    hdr = twix['hdr']

    # ── Extract header parameters ───────────────────────────────────────
    def _hdr_get(section, *keys, default=None):
        d = hdr.get(section, {})
        for k in keys:
            if isinstance(k, tuple):
                val = d.get(k, None)
            else:
                val = d.get((k,), None)
            if val is not None:
                return val
        return default

    larmor_freq_hz = float(_hdr_get('MeasYaps', ('sTXSPEC', 'asNucleusInfo', '0', 'lFrequency'), default=49894611))
    larmor_freq_mhz = larmor_freq_hz / 1e6
    dwell_ns = float(_hdr_get('MeasYaps', ('sRXSPEC', 'alDwellTime', '0'), default=180000))
    dwell_s = dwell_ns * 1e-9
    fov_mm = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'dReadoutFOV'), default=240))
    fov_m = fov_mm / 1e3
    base_res = int(float(_hdr_get('MeasYaps', ('sKSpace', 'lBaseResolution'), default=8)))
    matrix_size = matrix_size_override or base_res
    vector_size = int(float(_hdr_get('MeasYaps', ('sSpecPara', 'lVectorSize'), default=1024)))

    nucleus = _hdr_get('MeasYaps', ('sRXSPEC', 'asNucleusInfo', '0', 'tNucleus'),
                       ('sCoilSelectMeas', 'aRxCoilSelectData', '0', 'tNucleus'),
                       default='"31P"')
    # Read GyroMagnRatioOverTwoPi from header (Hz/T), fall back to standard values
    gamma_2pi_hdr = _hdr_get('Dicom', ('lGyroMagnRatioOverTwoPi',), default=None)
    if gamma_2pi_hdr is not None:
        gamma_2pi = float(gamma_2pi_hdr)
    elif '31P' in str(nucleus):
        gamma_2pi = 17.235e6
    elif '1H' in str(nucleus):
        gamma_2pi = 42.577e6
    else:
        gamma_2pi = 17.235e6

    # Slice position
    pos_sag = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dSag'), default=0))
    pos_cor = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dCor'), default=0))
    pos_tra = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dTra'), default=0))
    snv_x = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dSag'), default=0))
    snv_y = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dCor'), default=0))
    snv_z = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dTra'), default=1))
    ipr = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'dInPlaneRot'), default=0))

    # ── Per-scan MDH counters ───────────────────────────────────────────
    lin_vals = img.Lin.astype(int)  # ring index
    ida_vals = img.Ida.astype(int)  # coil index
    n_scans = len(lin_vals)
    n_rings = int(img.NLin)
    n_coils = len(np.unique(ida_vals))

    # ICE parameters contain k-space position info
    ice = img.iceParam  # (n_scans, 24)

    print(f"\n--- Header ---")
    print(f"  Nucleus:     {nucleus}")
    print(f"  Larmor:      {larmor_freq_hz:.0f} Hz ({larmor_freq_mhz:.3f} MHz)")
    print(f"  Dwell:       {dwell_ns:.0f} ns")
    print(f"  FOV:         {fov_mm:.0f} mm")
    print(f"  Matrix:      {matrix_size}x{matrix_size}")
    print(f"  VectorSize:  {vector_size}")
    print(f"  n_rings:     {n_rings}")
    print(f"  n_coils:     {n_coils}")
    print(f"  n_scans:     {n_scans}")
    print(f"  Pos (S,C,T): ({pos_sag:.1f}, {pos_cor:.1f}, {pos_tra:.1f}) mm")

    # ── Read k-space data ───────────────────────────────────────────────
    img.squeeze = True
    img.flagRemoveOS = False
    data_sorted = img['']

    # Find where actual data lives in the sorted array
    # Idc and Idd might be constant MDH fields creating phantom dimensions
    sqz_dims = img.sqzDims
    sqz_size = img.sqzSize
    print(f"\n--- Data ---")
    print(f"  sqzDims: {sqz_dims}")
    print(f"  sqzSize: {sqz_size}")

    # Extract per-ring, per-coil data from unsorted: (Col_raw, 1, n_tr)
    img.flagRemoveOS = False
    u = img.unsorted()  # (Col_raw, n_cha_unsorted, n_tr)
    n_col_raw = u.shape[0]

    col_per_scan = n_col_raw
    print(f"  Col (raw, no OS removal): {col_per_scan}")
    print(f"  Col (with OS removal):    {col_per_scan // 2}")

    # ── Determine pts_per_ring and n_spec ───────────────────────────────
    idc_val = int(img.Idc[0])
    numPointsCircle = 2 * idc_val

    # Override if requested
    if pts_per_ring_override is not None:
        numPointsCircle = pts_per_ring_override

    pts_per_ring = numPointsCircle

    print(f"\n--- MATLAB-style reshape ---")
    print(f"  numPointsCircle (raw) = 2 * {idc_val} = {2 * idc_val}")
    print(f"  pts_per_ring = {pts_per_ring}")

    # ── Rearrange k-space data ──────────────────────────────────────────
    # Group scans by (ring, temporal_interleave) using unsorted data directly
    idb_vals = img.Idb.astype(int)
    scans_by_ring_ti = {}
    for scan_idx in range(n_scans):
        r_idx = lin_vals[scan_idx]
        ti_idx = idb_vals[scan_idx]
        key = (r_idx, ti_idx)
        if key not in scans_by_ring_ti:
            scans_by_ring_ti[key] = []
        adc = (u[:, 0, scan_idx] if u.ndim == 3 else u[:, scan_idx]).copy()
        adc[-4:] = 0
        scans_by_ring_ti[key].append(adc)

    # Compute vecSize per ring, then use the minimum across rings
    ring_vec_sizes = []
    for r in range(n_rings):
        n_ti = max(idb_vals[lin_vals == r]) + 1
        n_adc_per_ti = sum(1 for s in range(n_scans)
                          if lin_vals[s] == r and idb_vals[s] == 0)
        vec_r = round(col_per_scan * n_adc_per_ti / pts_per_ring * n_ti - 0.5)
        ring_vec_sizes.append(vec_r)
        print(f"  Ring {r}: {n_adc_per_ti} ADCs/TI, nTI={n_ti}, vecSize={vec_r}")

    pointsInFID = min(ring_vec_sizes)
    if n_spec_override is not None:
        pointsInFID = n_spec_override
    print(f"  Global vecSize (min across rings) = {pointsInFID}")

    kdata = np.zeros((pts_per_ring, pointsInFID, 1, n_rings), dtype=np.complex64)

    for r in range(n_rings):
        n_ti = max(idb_vals[lin_vals == r]) + 1
        vec_per_ti = pointsInFID // n_ti
        useful_per_ti = vec_per_ti * pts_per_ring
        effective_n_spec = vec_per_ti * n_ti

        if n_ti == 1:
            useful_total = pointsInFID * pts_per_ring
            raw = np.concatenate(scans_by_ring_ti[(r, 0)])[:useful_total]
            kdata[:, :pointsInFID, 0, r] = raw.reshape(
                pts_per_ring, pointsInFID, order='F')
        else:
            ti_blocks = np.zeros((pts_per_ring, vec_per_ti, n_ti), dtype=np.complex64)
            for ti in range(n_ti):
                raw_ti = np.concatenate(scans_by_ring_ti[(r, ti)])[:useful_per_ti]
                ti_blocks[:, :, ti] = raw_ti.reshape(
                    pts_per_ring, vec_per_ti, order='F')
            merged = ti_blocks.reshape(pts_per_ring, effective_n_spec)
            kdata[:, :effective_n_spec, 0, r] = merged

    n_coils = 1  # now single-coil after concatenation

    print(f"\n  kdata shape (before phase corr): {kdata.shape}")
    print(f"  |kdata| max: {np.abs(kdata).max():.4e}")

    # ── CONCEPT phase correction (temporal interleave) ──────────────────
    # MATLAB: temporalInterleaves = ushIdd + 1 (0-indexed counter → count)
    # Idd can vary per ring (e.g., outer rings have more TI than inner rings)

    # Phase correction params — WTC uses Idd+1 (NOT max(Idb)+1)
    # Idd = temporal interleave COUNT (0-based), Idb = temporal interleave INDEX
    idd_vals = img.Idd.astype(int)
    ti_per_ring = np.zeros(n_rings, dtype=int)
    for r in range(n_rings):
        ti_per_ring[r] = int(idd_vals[lin_vals == r][0])  # Idd directly (MATLAB uses ushIdd, not +1)
    print(f"  Phase corr TI per ring (Idd): {ti_per_ring}")

    # ── K-space trajectory from ICE parameters ──────────────────────────
    # From MATLAB CONCEPTCalculateKSpacePos.m and the C++ sequence code:
    #   ice[0] = int(dGradFirstCoord[ring][0])           (kx integer part)
    #   ice[1] = int(10000 * fractional(kx))             (kx fractional * 10000)
    #   ice[2] = int(dGradFirstCoord[ring][1])           (ky integer part)
    #   ice[3] = int(10000 * fractional(ky))             (ky fractional * 10000)
    # Reconstruct: kx = ice[0] + ice[1]/10000, ky = ice[2] + ice[3]/10000
    # With unsigned-to-signed correction for values > 2^15

    delta_gm = 1e9 / (fov_m * gamma_2pi)

    def _ice_to_signed(val):
        """Convert unsigned 16-bit ICE param to signed."""
        if val > 0.5 * 2**16:
            return val - 2**16
        return val

    first_circle_kx = np.zeros(n_rings)
    first_circle_ky = np.zeros(n_rings)
    for r in range(n_rings):
        scan_for_ring = np.where(lin_vals == r)[0][0]
        kx_int  = _ice_to_signed(float(ice[scan_for_ring, 0]))
        kx_frac = _ice_to_signed(float(ice[scan_for_ring, 1]))
        ky_int  = _ice_to_signed(float(ice[scan_for_ring, 2]))
        ky_frac = _ice_to_signed(float(ice[scan_for_ring, 3]))
        first_circle_kx[r] = kx_int + kx_frac / 10000.0
        first_circle_ky[r] = ky_int + ky_frac / 10000.0

    first_circle = np.array([first_circle_kx, first_circle_ky])
    radii = np.sqrt(first_circle_kx**2 + first_circle_ky**2)
    print(f"\n  Ring radii (from ICE): {radii}")
    print(f"  delta_gm = {delta_gm:.2f}")
    print(f"  k_norm_max = {radii.max() / (delta_gm * matrix_size):.4f}")

    # ── Spectral axes ───────────────────────────────────────────────────
    # From MATLAB prepConceptForSpectro.m line 30:
    #   dwellTime = alDwellTime(1) * 2 / 1e9   (×2 for 2x ADC oversampling)
    spectral_dwell = dwell_s * 2  # ×2 OS correction
    spectral_bw_hz = 1.0 / spectral_dwell
    freq_hz_axis = np.fft.fftshift(np.fft.fftfreq(pointsInFID, d=spectral_dwell))
    ppm_axis = freq_hz_axis / larmor_freq_mhz

    print(f"\n  Spectral dwell: {spectral_dwell*1e6:.1f} µs")
    print(f"  Spectral BW: {spectral_bw_hz:.1f} Hz ({spectral_bw_hz/larmor_freq_mhz:.1f} ppm)")

    traj_pts = np.full(n_rings, pts_per_ring, dtype=int)

    info = dict(
        kdata=kdata,
        n_rings=n_rings,
        traj_pts=traj_pts,
        n_spec=pointsInFID,
        n_coils=n_coils,
        matrix_size=matrix_size,
        n_freq_enc=matrix_size,
        n_phas_enc=matrix_size,
        dwell_ns=dwell_ns,
        dwell_s=dwell_s,
        spectral_bw_hz=spectral_bw_hz,
        larmor_freq_hz=larmor_freq_hz,
        larmor_freq_mhz=larmor_freq_mhz,
        fov_mm=fov_mm,
        fov_m=fov_m,
        gamma_2pi=gamma_2pi,
        delta_gm=delta_gm,
        first_circle=first_circle,
        freq_hz=freq_hz_axis,
        ppm_axis=ppm_axis,
        pos_cor=pos_cor, pos_sag=pos_sag, pos_tra=pos_tra,
        snv_x=snv_x, snv_y=snv_y, snv_z=snv_z,
        in_plane_rotation=ipr,
        _ti_per_ring=ti_per_ring,
    )

    return info


def concept_reconstruct(kdata, kx_raw, ky_raw, N, fov_m, gamma_mhz_per_mt,
                        fov_shift=None, dcf_flag=False, hamming_flag=False):
    """CONCEPT 2D NUFFT reconstruction (port of CONCEPTPerform2DDirectFT.m).

    Uses mri-nufft (finufft) instead of explicit SFT for speed.

    Parameters
    ----------
    kdata    : (total_kpts, n_spec) complex k-space
    kx_raw, ky_raw : (total_kpts,) trajectory in gradient-moment units (mT·µs/m)
    N        : reconstruction matrix size
    fov_m    : FOV in metres
    gamma_mhz_per_mt : gyromagnetic ratio in MHz/mT (31P: 17.235e-3)
    fov_shift : (shift_ro, shift_pe) in voxels, or None
    dcf_flag : apply radial density compensation
    hamming_flag : apply Hamming apodization

    Returns
    -------
    recon : (n_spec, N, N) complex
    """
    total_kpts, n_spec = kdata.shape

    # 1. Physical scaling: gradient-moment -> cycles/pixel
    kx_phys = kx_raw * gamma_mhz_per_mt  # 1/m  (MHz/mT * mT·µs/m = µs·MHz/m = 1/m)
    ky_phys = ky_raw * gamma_mhz_per_mt
    kx_cpp = kx_phys * (fov_m / N)  # cycles per pixel
    ky_cpp = ky_phys * (fov_m / N)

    print(f"  k range: kx=[{kx_cpp.min():.4f}, {kx_cpp.max():.4f}], "
          f"ky=[{ky_cpp.min():.4f}, {ky_cpp.max():.4f}] cyc/pix")

    # 2. FOV shift via phase ramp (MATLAB line 35-36)
    data = kdata.copy()
    if fov_shift is not None and (fov_shift[0] != 0 or fov_shift[1] != 0):
        shift_phase = np.exp(-2j * np.pi * (kx_cpp * fov_shift[0] + ky_cpp * fov_shift[1]))
        data = data * shift_phase[:, np.newaxis]
        print(f"  FOV shift: [{fov_shift[0]:.2f}, {fov_shift[1]:.2f}] voxels")

    # 3. Density compensation / apodization weights
    w = np.ones(total_kpts, dtype=np.float32)
    if dcf_flag:
        r_per_pt = np.sqrt(kx_cpp**2 + ky_cpp**2)
        w = r_per_pt / (r_per_pt.max() + 1e-12)
        w = np.maximum(w, 0.01)
        print(f"  DCF: radial weighting applied")
    if hamming_flag:
        r_norm = np.sqrt(kx_cpp**2 + ky_cpp**2) * 2
        ham = 0.54 + 0.46 * np.cos(np.pi * r_norm)
        ham = np.maximum(ham, 0)
        w *= ham
        print(f"  Hamming apodization applied")

    # 4. NUFFT via mri-nufft (finufft)
    # Convert trajectory to mri-nufft convention: [-0.5, 0.5]
    traj = np.stack([kx_cpp, ky_cpp], axis=-1).astype(np.float32)

    nufft_op = mrinufft.get_operator(
        "finufft",
        samples=traj,
        shape=(N, N),
        n_coils=1,
        density=w if (dcf_flag or hamming_flag) else False,
    )

    recon = np.zeros((n_spec, N, N), dtype=np.complex64)
    t0 = time.time()
    for s in range(n_spec):
        if s % 50 == 0:
            print(f"    spectral {s}/{n_spec}")
        recon[s] = nufft_op.adj_op(data[:, s]).astype(np.complex64)
    print(f"  NUFFT recon done in {time.time() - t0:.1f}s")
    return recon


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    info = load_concept_twix(
        args.dat,
        pts_per_ring_override=args.pts_per_ring,
        n_spec_override=args.n_spec,
        matrix_size_override=args.matrix_size,
    )

    if args.inspect_only:
        print(f"\n{'='*70}")
        print("Inspect complete. Run without --inspect-only to reconstruct.")
        print(f"{'='*70}")
        return

    kdata = info["kdata"]  # (TrajPts, vecSize, 1, n_rings)
    N = info["matrix_size"]
    n_spec = info["n_spec"]
    n_rings = info["n_rings"]
    first_circle = info["first_circle"]
    fov_m = info["fov_m"]
    traj_pts_per_ring = int(info["traj_pts"][0])
    ppm_axis = info["ppm_axis"] + args.ppm_center

    # 31P: 17.235e-3 MHz/mT;  1H: 42.577e-3 MHz/mT
    gamma_mhz_per_mt = info["gamma_2pi"] / 1e9  # Hz/T -> MHz/mT

    # ── 1. Phase correction (CONCEPTPhaseCorrection_LowRAM.m) ───────────
    if not args.no_phase_corr:
        ti_per_ring = info["_ti_per_ring"]  # Idd per ring
        print(f"\n  Phase correction: TI per ring (Idd) = {ti_per_ring}")
        kdata = concept_phase_correction(kdata, ti_per_ring)
        print(f"  kdata after phase corr: {kdata.shape}")

    # ── 2. Build trajectory (CONCEPTCalculateKSpacePos.m) ───────────────
    # Compute ring radii and starting angles from FirstCirclekSpacePoint
    radii = np.sqrt(first_circle[0]**2 + first_circle[1]**2)
    phi0 = np.arctan2(first_circle[1], first_circle[0])

    # Build all k-space positions: R * cos(2π*(n-1)/ns + phi) for n=1..ns
    all_kx, all_ky = [], []
    for r in range(n_rings):
        ns = traj_pts_per_ring
        angles = 2 * np.pi * np.arange(ns) / ns + phi0[r]
        all_kx.append(radii[r] * np.cos(angles))
        all_ky.append(radii[r] * np.sin(angles))
    kx_raw = np.concatenate(all_kx)  # gradient-moment units (mT·µs/m)
    ky_raw = np.concatenate(all_ky)

    print(f"\n  Trajectory: {len(kx_raw)} pts, {n_rings} rings")
    print(f"  Ring radii: {radii.round(1)}")

    # ── 3. FOV shift (calculateInPlaneShiftBeforeRecon.m) ───────────────
    fov_shift = None
    if not args.no_fov_shift:
        pos = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]])
        snv = np.array([info["snv_x"], info["snv_y"], info["snv_z"]])
        ipr = info["in_plane_rotation"]
        fov_mm = info["fov_mm"]

        # Rotation matrix from slice normal (GSL.fGSLCalcPRS equivalent)
        normal = snv / (np.linalg.norm(snv) + 1e-30)
        target = np.array([0.0, 0.0, 1.0])
        cross_n = np.cross(normal, target)
        dot_n = np.dot(normal, target)
        if np.linalg.norm(cross_n) < 1e-10:
            Rot = np.eye(3)
        else:
            K = np.array([[0, -cross_n[2], cross_n[1]],
                          [cross_n[2], 0, -cross_n[0]],
                          [-cross_n[1], cross_n[0], 0]])
            Rot = np.eye(3) + K + K @ K / (1 + dot_n)

        if abs(ipr) > 1e-10:
            c, s = np.cos(ipr), np.sin(ipr)
            Rot_ipr = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            Rot = Rot_ipr @ Rot

        rot_pos = Rot.T @ pos  # basePositionRot = rotMat' * TwixSlicePosition
        shift_pe = rot_pos[0] / (fov_mm / N)  # PE direction
        shift_ro = rot_pos[1] / (fov_mm / N)  # RO direction
        fov_shift = (shift_ro, shift_pe)
        print(f"  FOV shift: RO={shift_ro:.2f}, PE={shift_pe:.2f} voxels")

    # ── 4. Flatten kdata for NUFFT: (total_kpts, n_spec) ────────────────
    kdata_flat = kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
    print(f"  kdata_flat: {kdata_flat.shape}")

    # ── 5. Reconstruct ──────────────────────────────────────────────────
    print(f"\n  Reconstructing {N}x{N} with {n_spec} spectral points...")
    recon_fid = concept_reconstruct(
        kdata_flat, kx_raw, ky_raw, N, fov_m, gamma_mhz_per_mt,
        fov_shift=fov_shift,
        dcf_flag=not args.no_dcf,
    )
    recon_spec = FIDToSpec(recon_fid, axis=0)
    label = "CONCEPT"

    # ── 6. Plot ─────────────────────────────────────────────────────────
    mag_map = np.mean(np.abs(recon_spec), axis=0)

    if args.plot_voxel is not None:
        vy, vx = args.plot_voxel
    else:
        vy, vx = N // 2, N // 2

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].scatter(kx_raw * gamma_mhz_per_mt * fov_m / N,
                    ky_raw * gamma_mhz_per_mt * fov_m / N,
                    s=2, alpha=0.5)
    axes[0].set_aspect("equal")
    axes[0].set_title("CRT Trajectory (cyc/pix)")

    im = axes[1].imshow(mag_map, origin="lower", cmap="viridis")
    axes[1].plot(vx, vy, "rx", ms=10, mew=2)
    axes[1].set_title(f"Spatial map ({label})")
    plt.colorbar(im, ax=axes[1])

    spec = recon_spec[:, vy, vx]
    axes[2].plot(ppm_axis, np.real(spec), 'b-', alpha=0.7, label='Real')
    axes[2].plot(ppm_axis, np.abs(spec), 'r-', alpha=0.5, label='Mag')
    axes[2].set_xlim(args.ppmlim[1], args.ppmlim[0])
    axes[2].set_xlabel("ppm")
    axes[2].set_ylabel("intensity")
    axes[2].set_title(f"Spectrum [{vy},{vx}]")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "concept_31p_recon.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {fig_path}")

    np.save(os.path.join(args.out_dir, "recon_spec.npy"), recon_spec)
    np.save(os.path.join(args.out_dir, "recon_fid.npy"), recon_fid)
    print(f"Saved recon_spec.npy  shape={recon_spec.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
