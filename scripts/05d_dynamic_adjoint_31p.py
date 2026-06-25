#!/usr/bin/env python3
"""
31P CONCEPT CRT dynamic adjoint NUFFT reconstruction.

Reconstructs each repetition independently, producing a 4D dataset
(Ny, Nx, n_spec, n_reps) for dynamic 31P studies (e.g., exercise/recovery).

Usage:
    python 05d_dynamic_adjoint_31p.py --dat "path/to/dynamic.dat" --out-dir ./output
    python 05d_dynamic_adjoint_31p.py --dat "path/to/dynamic.dat" --lb 10
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
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapvbvd


# ── FFT / helpers ───────────────────────────────────────────────────────

def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


def concept_phase_correction(kdata, temporal_interleaves):
    ns, pointsInFID, _, n_rings = kdata.shape
    vs = pointsInFID * 2
    if np.isscalar(temporal_interleaves):
        ti_arr = np.full(n_rings, temporal_interleaves)
    else:
        ti_arr = np.asarray(temporal_interleaves)
    sample_idx = np.arange(ns, dtype=np.float64)
    freq_axis = np.arange(vs, dtype=np.float64) / vs - 0.5
    phasecorr = np.zeros((ns, vs, n_rings), dtype=np.complex128)
    for r in range(n_rings):
        timeoffset = ti_arr[r] * sample_idx / ns
        phasecorr[:, :, r] = np.exp(-2j * np.pi * np.outer(timeoffset, freq_axis))
    pad = np.zeros_like(kdata)
    kdata_padded = np.concatenate([pad, kdata], axis=1)
    out = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(kdata_padded, axes=1), axis=1), axes=1)
    out = out * np.conj(phasecorr[:, :, np.newaxis, :])
    out = np.fft.fftshift(np.fft.fft(np.fft.fftshift(out, axes=1), axis=1), axes=1)
    out = out[:, vs // 2:, :, :]
    return out.astype(np.complex64)


# ── Trajectory ──────────────────────────────────────────────────────────

def build_trajectory(first_circ, n_rings, traj_pts, gamma_mhz_per_mt, fov_m, N):
    radii = np.sqrt(first_circ[0]**2 + first_circ[1]**2)
    phi0 = np.arctan2(first_circ[1], first_circ[0])
    all_kx, all_ky = [], []
    for r in range(n_rings):
        angles = 2 * np.pi * np.arange(traj_pts) / traj_pts + phi0[r]
        all_kx.append(radii[r] * np.cos(angles))
        all_ky.append(radii[r] * np.sin(angles))
    kx_raw = np.concatenate(all_kx)
    ky_raw = np.concatenate(all_ky)
    kx_cpp = kx_raw * gamma_mhz_per_mt * (fov_m / N)
    ky_cpp = ky_raw * gamma_mhz_per_mt * (fov_m / N)
    return kx_raw, ky_raw, kx_cpp, ky_cpp


def _fGSLClassOri(dGs):
    s, c, t = abs(dGs[0]), abs(dGs[1]), abs(dGs[2])
    eq_sc, eq_st, eq_ct = abs(s-c)<=1e-6, abs(s-t)<=1e-6, abs(c-t)<=1e-6
    if ((eq_sc and eq_st) or (eq_sc and s<t) or (eq_st and s>c) or
            (eq_ct and c>s) or (s>c and s<t) or (s<c and c<t) or
            (s<t and t>c) or (c<t and t>s)):
        return 2
    elif ((eq_sc and s>t) or (eq_st and s<c) or (s<c and c>t) or
            (s>t and s<c) or (s<t and t<c)):
        return 1
    elif ((eq_ct and c<s) or (s>c and s>t) or (c>t and c<s) or (c<t and t<s)):
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


def compute_dcf(kx_cpp, ky_cpp, kx_raw, ky_raw, use_hamming=True):
    r_per_pt = np.sqrt(kx_cpp**2 + ky_cpp**2)
    weights = np.maximum(r_per_pt / (r_per_pt.max() + 1e-12), 0.01)
    if use_hamming:
        kx_norm = np.pi * kx_raw / (np.abs(kx_raw).max() + 1e-30)
        ky_norm = np.pi * ky_raw / (np.abs(ky_raw).max() + 1e-30)
        r_norm = np.sqrt(kx_norm**2 + ky_norm**2)
        weights = weights * (0.54 + 0.46 * np.cos(r_norm))
    return weights.astype(np.float32)


# ── Dynamic data loader ────────────────────────────────────────────────

def load_dynamic_concept_twix(dat_path):
    """Load dynamic CONCEPT CRT data, returning per-repetition k-space."""
    print(f"\n{'='*70}")
    print(f"Loading: {dat_path}")
    print(f"{'='*70}")

    twix = mapvbvd.mapVBVD(dat_path)
    if isinstance(twix, list):
        twix = twix[-1]

    img = twix['image']
    hdr = twix['hdr']

    def _hdr_get(section, key, default=None):
        d = hdr.get(section, {})
        val = d.get(key if isinstance(key, tuple) else (key,), None)
        return val if val is not None else default

    larmor_freq_hz = float(_hdr_get('MeasYaps', ('sTXSPEC', 'asNucleusInfo', '0', 'lFrequency'), default=49894611))
    larmor_freq_mhz = larmor_freq_hz / 1e6
    dwell_ns = float(_hdr_get('MeasYaps', ('sRXSPEC', 'alDwellTime', '0'), default=180000))
    dwell_s = dwell_ns * 1e-9
    fov_mm = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'dReadoutFOV'), default=200))
    fov_m = fov_mm / 1e3
    matrix_size = int(float(_hdr_get('MeasYaps', ('sKSpace', 'lBaseResolution'), default=10)))

    nucleus = _hdr_get('MeasYaps', ('sRXSPEC', 'asNucleusInfo', '0', 'tNucleus'), default='"31P"')
    gamma_2pi_hdr = _hdr_get('Dicom', ('lGyroMagnRatioOverTwoPi',), default=None)
    if gamma_2pi_hdr is not None:
        gamma_2pi = float(gamma_2pi_hdr)
    elif '31P' in str(nucleus):
        gamma_2pi = 17.235e6
    else:
        gamma_2pi = 42.577e6

    pos_sag = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dSag'), default=0))
    pos_cor = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dCor'), default=0))
    pos_tra = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sPosition', 'dTra'), default=0))
    snv_x = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dSag'), default=0))
    snv_y = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dCor'), default=0))
    snv_z = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'sNormal', 'dTra'), default=1))
    ipr = float(_hdr_get('MeasYaps', ('sSpecPara', 'sVoI', 'dInPlaneRot'), default=0))

    lin_vals = img.Lin.astype(int)
    rep_vals = img.Rep.astype(int)
    idb_vals = img.Idb.astype(int)
    idd_vals = img.Idd.astype(int)
    n_scans = len(lin_vals)
    n_rings = int(img.NLin)
    n_reps = int(img.NRep)

    ice = img.iceParam
    img.squeeze = True
    img.flagRemoveOS = False
    u = img.unsorted()
    n_col_raw = u.shape[0]

    idc_val = int(img.Idc[0])
    pts_per_ring = 2 * idc_val

    print(f"  Nucleus: {nucleus}, Larmor: {larmor_freq_mhz:.3f} MHz")
    print(f"  Matrix: {matrix_size}x{matrix_size}, FOV: {fov_mm:.0f}mm")
    print(f"  Rings: {n_rings}, Reps: {n_reps}, pts/ring: {pts_per_ring}")
    print(f"  Total scans: {n_scans}, scans/rep: {n_scans // n_reps}")

    # Rearrange per repetition
    ring_vec_sizes = []
    rep0_mask = rep_vals == 0
    for r in range(n_rings):
        ring_mask = (lin_vals == r) & rep0_mask
        n_ti = max(idb_vals[ring_mask]) + 1
        n_adc = sum(1 for s in range(n_scans) if lin_vals[s] == r and rep_vals[s] == 0 and idb_vals[s] == 0)
        vec_r = round(n_col_raw * n_adc / pts_per_ring * n_ti - 0.5)
        ring_vec_sizes.append(vec_r)
    pointsInFID = min(ring_vec_sizes)

    ti_per_ring = np.zeros(n_rings, dtype=int)
    for r in range(n_rings):
        ring_mask = (lin_vals == r) & rep0_mask
        ti_per_ring[r] = int(idd_vals[ring_mask][0])

    print(f"  vecSize: {pointsInFID}, TI per ring: {ti_per_ring}")

    # ICE trajectory params (same for all reps)
    def _ice_to_signed(val):
        return val - 2**16 if val > 0.5 * 2**16 else val

    first_circle_kx = np.zeros(n_rings)
    first_circle_ky = np.zeros(n_rings)
    for r in range(n_rings):
        scan_for_ring = np.where((lin_vals == r) & rep0_mask)[0][0]
        kx_int = _ice_to_signed(float(ice[scan_for_ring, 0]))
        kx_frac = _ice_to_signed(float(ice[scan_for_ring, 1]))
        ky_int = _ice_to_signed(float(ice[scan_for_ring, 2]))
        ky_frac = _ice_to_signed(float(ice[scan_for_ring, 3]))
        first_circle_kx[r] = kx_int + kx_frac / 10000.0
        first_circle_ky[r] = ky_int + ky_frac / 10000.0

    # Build kdata per rep
    all_kdata = []
    for rep in tqdm(range(n_reps), desc="Loading reps"):
        kdata = np.zeros((pts_per_ring, pointsInFID, 1, n_rings), dtype=np.complex64)
        rep_mask = rep_vals == rep

        scans_by_ring_ti = {}
        for scan_idx in np.where(rep_mask)[0]:
            r_idx = lin_vals[scan_idx]
            ti_idx = idb_vals[scan_idx]
            key = (r_idx, ti_idx)
            if key not in scans_by_ring_ti:
                scans_by_ring_ti[key] = []
            adc = (u[:, 0, scan_idx] if u.ndim == 3 else u[:, scan_idx]).copy()
            adc[-4:] = 0
            scans_by_ring_ti[key].append(adc)

        for r in range(n_rings):
            ring_rep_mask = (lin_vals == r) & rep_mask
            n_ti = max(idb_vals[ring_rep_mask]) + 1
            vec_per_ti = pointsInFID // n_ti
            useful_per_ti = vec_per_ti * pts_per_ring
            effective_n_spec = vec_per_ti * n_ti

            if n_ti == 1:
                useful_total = pointsInFID * pts_per_ring
                raw = np.concatenate(scans_by_ring_ti[(r, 0)])[:useful_total]
                kdata[:, :pointsInFID, 0, r] = raw.reshape(pts_per_ring, pointsInFID, order='F')
            else:
                ti_blocks = np.zeros((pts_per_ring, vec_per_ti, n_ti), dtype=np.complex64)
                for ti in range(n_ti):
                    raw_ti = np.concatenate(scans_by_ring_ti[(r, ti)])[:useful_per_ti]
                    ti_blocks[:, :, ti] = raw_ti.reshape(pts_per_ring, vec_per_ti, order='F')
                merged = ti_blocks.reshape(pts_per_ring, effective_n_spec)
                kdata[:, :effective_n_spec, 0, r] = merged

        all_kdata.append(kdata)

    info = dict(
        n_rings=n_rings, n_reps=n_reps, n_spec=pointsInFID,
        matrix_size=matrix_size, pts_per_ring=pts_per_ring,
        dwell_s=dwell_s, larmor_freq_mhz=larmor_freq_mhz,
        fov_mm=fov_mm, fov_m=fov_m, gamma_2pi=gamma_2pi,
        first_circle=np.array([first_circle_kx, first_circle_ky]),
        traj_pts=np.full(n_rings, pts_per_ring, dtype=int),
        pos_sag=pos_sag, pos_cor=pos_cor, pos_tra=pos_tra,
        snv_x=snv_x, snv_y=snv_y, snv_z=snv_z,
        in_plane_rotation=ipr,
        _ti_per_ring=ti_per_ring,
    )
    return all_kdata, info


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="31P dynamic CRT adjoint reconstruction")
    p.add_argument("--dat", required=True)
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--no-dcf", action="store_true")
    p.add_argument("--no-hamming", action="store_true")
    p.add_argument("--lb", type=float, default=0.0, help="Line broadening (Hz)")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(args.out_dir, "data")
    fig_dir = os.path.join(args.out_dir, "figures")
    for d in [data_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load dynamic data ────────────────────────────────────────
    all_kdata, info = load_dynamic_concept_twix(args.dat)
    N = info["matrix_size"]
    n_spec = info["n_spec"]
    n_reps = info["n_reps"]
    n_rings = info["n_rings"]
    traj_pts = info["pts_per_ring"]
    dwell_s = info["dwell_s"] * 2
    larmor_mhz = info["larmor_freq_mhz"]
    gamma_mhz_per_mt = info["gamma_2pi"] / 1e9

    print(f"\n[dyn] {N}x{N}, {n_spec} spec, {n_rings} rings, {n_reps} reps")

    # ── 2. Trajectory (same for all reps) ───────────────────────────
    kx_raw, ky_raw, kx_cpp, ky_cpp = build_trajectory(
        info["first_circle"], n_rings, traj_pts,
        gamma_mhz_per_mt, info["fov_m"], N)
    traj = np.stack([kx_cpp, ky_cpp], axis=-1).astype(np.float32)

    # FOV shift
    dim1_shift = dim2_shift = 0.0
    if not args.no_fov_shift:
        pos = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]])
        snv = np.array([info["snv_x"], info["snv_y"], info["snv_z"]])
        phase_vec, read_vec = _fGSLCalcPRS(snv, info["in_plane_rotation"])
        rot_mat = np.column_stack([phase_vec, read_vec, snv])
        base_pos = rot_mat.T @ pos
        vox_size = info["fov_mm"] / N
        dim1_shift = base_pos[0] / vox_size
        dim2_shift = base_pos[1] / vox_size
        print(f"[dyn] FOV shift: {dim1_shift:.4f}, {dim2_shift:.4f} vox")

    # DCF
    dcf = None
    if not args.no_dcf:
        dcf = compute_dcf(kx_cpp, ky_cpp, kx_raw, ky_raw,
                          use_hamming=not args.no_hamming)

    # NUFFT operator
    nufft_op = mrinufft.get_operator(
        "finufft", samples=traj, shape=(N, N),
        n_coils=1, density=dcf if dcf is not None else False)

    # Apodization
    t_axis = np.arange(n_spec) * dwell_s
    apod = np.ones(n_spec, dtype=np.float32)
    if args.lb > 0:
        apod = np.exp(-np.pi * args.lb * t_axis).astype(np.float32)
        print(f"[dyn] Line broadening: {args.lb} Hz")

    # PPM axis
    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    PPM = FREQ / larmor_mhz + args.ppm_center

    # ── 3. Reconstruct all reps ─────────────────────────────────────
    print(f"[dyn] Reconstructing {n_reps} repetitions ...")
    recon_all = np.zeros((n_reps, N, N, n_spec), dtype=np.complex64)
    shift_phase = np.exp(-2j * np.pi * (kx_cpp * dim1_shift + ky_cpp * dim2_shift))

    t0 = time.time()
    for rep in tqdm(range(n_reps), desc="Reps"):
        kdata = all_kdata[rep]

        if not args.no_phase_corr:
            kdata = concept_phase_correction(kdata, info["_ti_per_ring"])

        kdata_flat = kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)

        if not args.no_fov_shift:
            kdata_flat = kdata_flat * shift_phase[:, np.newaxis]

        image_fid = np.zeros((N, N, n_spec), dtype=np.complex64)
        for s in range(n_spec):
            image_fid[:, :, s] = nufft_op.adj_op(kdata_flat[:, s]).astype(np.complex64)

        image_fid *= apod[np.newaxis, np.newaxis, :]
        recon_all[rep] = image_fid

    elapsed = time.time() - t0
    print(f"[dyn] All reps done in {elapsed:.1f}s ({elapsed/n_reps:.1f}s/rep)")

    # Convert to spectrum
    recon_spec = FIDToSpec(recon_all, axis=-1)

    # ── 4. Save ─────────────────────────────────────────────────────
    np.save(os.path.join(data_dir, "dynamic_fid.npy"), recon_all)
    np.save(os.path.join(data_dir, "dynamic_spec.npy"), recon_spec)
    print(f"[dyn] Saved dynamic_fid.npy {recon_all.shape} and dynamic_spec.npy")

    # ── 5. Find best voxel from time-averaged data ──────────────────
    avg_spec = np.mean(recon_spec, axis=0)
    mag_map = np.mean(np.abs(avg_spec), axis=-1)
    vy, vx = np.unravel_index(mag_map.argmax(), mag_map.shape)
    print(f"[dyn] Best voxel (time-avg): [{vy},{vx}]")

    # ── 6. Figures ──────────────────────────────────────────────────
    peaks = [(0, 'PCr'), (4.9, 'Pi'), (-2.48, 'g-ATP'), (-7.52, 'a-ATP'), (-16.26, 'b-ATP')]

    # Time-averaged spectrum
    fig, ax = plt.subplots(figsize=(10, 4))
    sp = np.abs(avg_spec[vy, vx, :])
    ax.plot(PPM, sp, 'k-', lw=0.8)
    ax.fill_between(PPM, 0, sp, alpha=0.15, color='blue')
    ax.set_xlim(args.ppmlim[1], args.ppmlim[0])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ppm"); ax.set_ylabel("|spectrum|")
    ax.set_title(f"Time-averaged spectrum [{vy},{vx}] ({n_reps} reps)")
    for pos, name in peaks:
        ax.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.text(pos, ax.get_ylim()[1]*0.92, name, fontsize=8, ha='center', color='red')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "avg_spectrum.png"), dpi=150)
    plt.close(fig)

    # PCr dynamics (peak height at 0 ppm over time)
    pcr_idx = np.argmin(np.abs(PPM - 0.0))
    pcr_time = np.abs(recon_spec[:, vy, vx, pcr_idx])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(n_reps), pcr_time, 'b.-', lw=1, ms=4)
    ax.set_xlabel("Repetition"); ax.set_ylabel("|PCr| at 0 ppm")
    ax.set_title(f"PCr dynamics — Voxel [{vy},{vx}]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pcr_dynamics.png"), dpi=150)
    plt.close(fig)

    # Spectral waterfall (time vs ppm)
    spec_vox = np.abs(recon_spec[:, vy, vx, :])
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(spec_vox, aspect='auto', origin='lower',
                   extent=[PPM[0], PPM[-1], 0, n_reps],
                   cmap='hot', interpolation='nearest')
    ax.set_xlim(args.ppmlim[1], args.ppmlim[0])
    ax.set_xlabel("ppm"); ax.set_ylabel("Repetition")
    ax.set_title(f"Spectral waterfall — Voxel [{vy},{vx}]")
    plt.colorbar(im, ax=ax, label="|spectrum|")
    for pos, name in peaks:
        ax.axvline(pos, color='cyan', ls='--', lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spectral_waterfall.png"), dpi=150)
    plt.close(fig)

    # Magnitude map (time-averaged)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mag_map, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title("Time-averaged magnitude map")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "magnitude_map.png"), dpi=120)
    plt.close(fig)

    print(f"[dyn] Figures saved to {fig_dir}")
    print("[dyn] Done.")


if __name__ == "__main__":
    main()