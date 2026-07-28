#!/usr/bin/env python3
"""
data_proc_01_twix2npy.py  — Siemens twix (.dat) → pipeline .npy conversion

Each MRSI .dat file is treated as one subject and written to its own output
folder.  The subject ID is parsed from the trailing number in the filename
(e.g. MRSI_64_cr_2.dat → _02).  wref files are saved once to the first
subject folder; the remaining folders receive relative symlinks so all share
one copy on disk.

Folder layout created under <out-prefix>_NN/:
    mrsi_data.npy    (1, K_mrsi, N_shots, N_coils)  complex128
    mrsi_ksp.npy     (3, K_mrsi, N_shots)           float64   kx, ky, t ∈ [0,2π]
    wref_data.npy    (1, K_wref, N_wref, N_coils)   complex128
    wref_ksp.npy     (3, K_wref, N_wref)            float64
    wref_o.npy       (Ny, Nx, 1)                    complex64  adjoint recon
    sigma_noise.npy  scalar                          float32    noise σ

File layout
----------
    data/
      raw/
        invivo_260623/          ← twix .dat + .seq files go here
      processed/
        invivo_260623_01/       ← created by this script (per subject)
        invivo_260623_02/
        …
    output/
        invivo_260623_01/       ← created by pipeline scripts 01-12

Usage:
    python scripts/data_preproc/data_proc_01_twix2npy.py \
        --raw-dir data/raw/invivo_260623 \
        --out-prefix data/processed/invivo_260623 \
        --mrsi-dat meas_MID00066_FID92027_MRSI_64_cr_1.dat \
                    meas_MID00067_FID92028_MRSI_64_cr_2.dat \
                    meas_MID00068_FID92029_MRSI_64_cr_3.dat \
                    meas_MID00069_FID92030_MRSI_64_cr_4.dat \
                    meas_MID00070_FID92031_MRSI_64_cr_5.dat \
        --wref-dat meas_MID00071_FID92032_wref_64_cr_300.dat \
        --mrsi-seq 260622_SMF_MRSI_cr_64.seq \
        --wref-seq 260622_SMF_wref_cr_64.seq \
        --wref-start-index 238 \
        --mrsi-start-index 252 \
        --fid-trunc 500
        [--smap-npy ecalib_pp.npy]
        

Creates: data/processed/invivo_260623_01/ … data/processed/invivo_260623_05/
  wref files live in _01, symlinked from _02 … _05.
  Then run pipeline scripts with --data-dir data/processed/invivo_260623_01 etc.

Notes on trim indices
---------------------
--mrsi-start-index 252  : start index value for defined sequence;
                           removes pre-echo ringing → fixes first-order phase.
--wref-start-index 238    : wref does not need the phase fix;
                           default 0 = no trim. Confirm the correct pre-echo
                           length with your supervisor before running.
--chop-index 40000      : end of useful ADC window.
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from mrinufft.io.siemens import read_siemens_rawdat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.scan_params import save_scan_params
from utils.coil_sens import morse_pi
import mrinufft

from spec2nii.dcm2niiOrientation.orientationFuncs import dcm_to_nifti_orientation
from spec2nii.Siemens.twixfunctions import twix2DCMOrientation
from spec2nii.GSL import gslfunctions as GSL

from warnings import filterwarnings
filterwarnings("ignore")


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_twix(filepath, n_coils):
    """Read a twix .dat file.

    Returns
    -------
    data : (n_coils, n_transients, pts_per_transient)  complex64
    hdrs : dict with ADC_size, ADC_dwell, spectrometer_freq, fullHdr
    """
    data, _, twixobj = read_siemens_rawdat(filename=str(filepath), doAverage=False)

    data = data.squeeze().swapaxes(-1, -2)                   # (coils, N, blocks, adc)
    data = data.reshape(n_coils, data.shape[1], -1)          # (coils, N, pts_per_transient)

    adc_size        = int(twixobj.hdr.MeasYaps[('sWipMemBlock', 'alFree', '9')])
    adc_dwell       = twixobj.hdr.MeasYaps[('sRXSPEC', 'alDwellTime', '0')] * 1e-9
    spec_freq_mhz   = twixobj.hdr.Dicom['lFrequency'] / 1e6

    return data, {
        'adc_size':     adc_size,
        'adc_dwell':    adc_dwell,
        'spec_freq':    spec_freq_mhz,
        'fullHdr':      twixobj['hdr'],
    }


def _read_traj(seq_path, n_transients, pts_per_transient, strip_pts=0):
    """Read k-space trajectory from a PyPulseq .seq file.

    Returns
    -------
    traj : (n_transients, pts_per_transient, 3)  float64   in natural seq units (cycles/m)
    fov  : (3,)  float64  FOV in metres
    """
    seq = pp.Sequence()
    seq.read(str(seq_path))
    ktraj = seq.calculate_kspace()[0]       # (3, total_pts)
    ktraj = ktraj[:, strip_pts:]            # strip noise prefix

    expected = n_transients * pts_per_transient
    if ktraj.shape[1] != expected:
        raise ValueError(
            f"Trajectory has {ktraj.shape[1]} pts after stripping, "
            f"expected {expected} ({n_transients} × {pts_per_transient}). "
            f"Check --noise-reps / --start-index / --chop-index.")

    ktraj = ktraj.reshape(3, n_transients, pts_per_transient)  # (3, N, K)
    ktraj = np.permute_dims(ktraj, (1, 2, 0))                  # (N, K, 3)
    fov = seq.get_definition("FOV")

    return ktraj, fov


def _scale_traj(traj, fov, res, start, chop):
    """Scale kx/ky to ±π, trim to [start:chop], set t ∈ [0, 2π].

    Parameters
    ----------
    traj : (N, K_full, 3)
    Returns (N, K, 3)  where K = chop - start
    """
    traj = traj[:, start:chop, :].copy()   # trim spectral/time axis
    K = traj.shape[1]

    traj[..., :2] *= fov[0] / res * 2 * np.pi   # kx, ky → ±π
    traj[..., 2]   = np.linspace(0, 2 * np.pi, K)  # t ∈ [0, 2π] per shot

    return traj                             # (N, K, 3)


def _calc_pos_info(twix_hdr, recon_res, recon_res_z):
    """Extract affine, slice_normal, inplane_rot, slice_position from twix header."""
    twix_hdr['Meas']['lFinalMatrixSizePhase'] = recon_res
    twix_hdr['Meas']['lFinalMatrixSizeRead']  = recon_res
    twix_hdr['Meas']['lFinalMatrixSizeSlice'] = recon_res_z

    orient = twix2DCMOrientation(twix_hdr, verbose=False)
    iop, ipp, pix, sliceth, _ = orient

    affine_obj = dcm_to_nifti_orientation(
        iop, ipp, np.append(pix, sliceth),
        (recon_res, recon_res, recon_res_z),
        half_shift=True, verbose=False)

    slice_normal = np.cross(iop[0], iop[1])

    def _try(key):
        try:
            return twix_hdr['MeasYaps'][key]
        except KeyError:
            return 0.0

    inplane_rot    = _try(('sSliceArray', 'asSlice', '0', 'dInPlaneRot'))
    slice_position = np.asarray([
        _try(('sSliceArray', 'asSlice', '0', 'sPosition', 'dSag')),
        _try(('sSliceArray', 'asSlice', '0', 'sPosition', 'dCor')),
        _try(('sSliceArray', 'asSlice', '0', 'sPosition', 'dTra')),
    ])

    return {
        'affine':          affine_obj.Q44,
        'slice_normal':    slice_normal,
        'inplane_rot':     inplane_rot,
        'slice_position':  slice_position,
    }


def _calc_shifts(pos_info, fov_mm, res):
    """Compute voxel-shift vector for phase correction (matches slow_recon convention)."""
    sn = pos_info['slice_normal']
    ir = pos_info['inplane_rot']
    sp = pos_info['slice_position']

    dcol, drow = GSL.calc_prs(sn, ir, True)
    rot = np.stack((dcol, drow, sn))
    base = rot @ sp

    shifts = np.asarray([
        base[0] / (fov_mm[0] / res),
        base[1] / (fov_mm[1] / res),
        base[2] / (fov_mm[2] / 1),     # z resolution = 1
    ])
    # Swap x/y (same HACK as orientation_position.py)
    shifts = np.asarray([shifts[1], shifts[0], shifts[2]])
    shifts[2] = 0
    shifts /= 2 * np.pi

    print(f"  slice position: {sp}")
    print(f"  shifts (normalised): {shifts}")
    return shifts


def _phase_correction(traj, shifts):
    """Complex phase to apply to k-space data for slice-position correction.
    traj : (N, K, 3) or (N*K, 3)
    Returns same shape complex64.
    """
    cycles = np.sum(traj * shifts, axis=-1)
    return np.exp(1j * 2 * np.pi * cycles).astype(np.complex64)


_KSP_SCALE = 15.713940692571413 / 16.0   # matches 01_coil_correction.py default ksp-scale


def _save_wref_o_png(wref_o, out_dir):
    """Save magnitude / phase / real of wref_o as a quick-check PNG."""
    img = wref_o.squeeze()   # (Ny, Nx)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("wref_o — quick check", fontsize=9)
    for ax, data, title, cmap in [
        (axes[0], np.abs(img),    "|wref_o| magnitude", "gray"),
        (axes[1], np.angle(img),  "phase(wref_o)",      "hsv"),
        (axes[2], img.real,       "Re(wref_o)",         "RdBu_r"),
    ]:
        im = ax.imshow(data, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(out_dir, "wref_o_check.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved wref_o_check.png")


def _reconstruct_wref_image(data_raw, data_phcorr, traj_2d, res,
                             smap_npy=None, n_ref=6, smoothing_sd=3,
                             max_iter=50, calib_width=16, backend="finufft",
                             traj_2d_img=None, data_img=None):
    """Sensitivity-weighted adjoint NUFFT → (Ny, Nx, 1) complex64.

    Smaps come from MORSE-PI (run inline, using all data_raw/traj_2d) or a
    precomputed file (--smap-npy). The final adjoint image uses traj_2d_img /
    data_img if given (e.g. truncated FID for PD-weighting), otherwise falls
    back to the full data_phcorr / traj_2d.

    data_raw    : (n_coils, N_total) raw k-space (no phase correction) — for MORSE-PI
    data_phcorr : (n_coils, N_total) phase-corrected k-space — fallback for image
    traj_2d     : (N_total, 2) full trajectory kx/ky ∈ [-π, π]
    smap_npy    : optional path to precomputed smaps (n_coils, Ny, Nx); skips MORSE-PI
    traj_2d_img : (N_img, 2)  trajectory for final adjoint (e.g. truncated FID)
    data_img    : (n_coils, N_img)  data for final adjoint (e.g. truncated FID)
    """
    NufftOp = mrinufft.get_operator("finufft")
    n_coils = data_raw.shape[0]

    if smap_npy is not None:
        print(f"[twix2npy] Loading precomputed smaps from {smap_npy} …")
        smap = np.load(smap_npy).astype(np.complex64)   # (n_coils, Ny, Nx)
        if smap.shape[0] != n_coils:
            raise ValueError(f"smap_npy has {smap.shape[0]} coils, expected {n_coils}")
    else:
        print("[twix2npy] Running MORSE-PI for wref_o smaps …")
        sens_out = morse_pi(
            data         = data_raw,
            trajectory   = traj_2d * _KSP_SCALE,
            resolution   = (res, res),
            backend      = backend,
            N_ref        = n_ref,
            smoothing_sd = smoothing_sd,
            max_iter     = max_iter,
            calib_width  = calib_width,
        )
        # morse_pi returns (Ny, Nx, NCoils, NRef) → take first ref → (NCoils, Ny, Nx)
        smap = np.moveaxis(sens_out[:, :, :, 0], -1, 0).astype(np.complex64)
        rss  = np.sqrt(np.sum(np.abs(smap) ** 2, axis=0, keepdims=True))
        rss  = np.where(rss < 1e-10, 1.0, rss)
        smap = smap / rss

    # Final adjoint — use truncated FID traj/data if provided
    traj_img = traj_2d_img if traj_2d_img is not None else traj_2d
    d_img    = data_img    if data_img    is not None else data_phcorr
    n_pts_label = traj_img.shape[0]
    print(f"[twix2npy] Sensitivity-weighted adjoint NUFFT for wref_o  ({n_pts_label} pts) …")
    nufft_img = NufftOp(traj_img, shape=[res, res], n_coils=1, squeeze_dims=False)
    wref_o = np.zeros((res, res), dtype=np.complex64)
    for c in range(n_coils):
        img_c = nufft_img.adj_op(np.ascontiguousarray(d_img[c:c+1]))[0, 0]
        wref_o += np.conj(smap[c]) * img_c

    return wref_o[..., np.newaxis]   # (Ny, Nx, 1)


# ── argparse ──────────────────────────────────────────────────────────────────

def _parse_subject_num(filename):
    """Extract trailing integer from filename, e.g. 'MRSI_64_cr_2.dat' → 2."""
    m = re.search(r'_(\d+)\.dat$', os.path.basename(filename))
    if m is None:
        raise ValueError(
            f"Cannot parse subject number from filename: {filename!r}. "
            "Expected a trailing _N before .dat")
    return int(m.group(1))


def _symlink_wref(src_dir, dst_dir, wref_files):
    """Create relative symlinks in dst_dir pointing to src_dir for each wref file."""
    for fname in wref_files:
        src  = os.path.relpath(os.path.join(src_dir, fname), dst_dir)
        link = os.path.join(dst_dir, fname)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(src, link)
        print(f"  symlink  {os.path.basename(link)}  →  {src}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Siemens twix → pipeline .npy conversion (per-subject)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir",    required=True,
                   help="Directory containing the .dat and .seq files")
    p.add_argument("--out-prefix", required=True,
                   help="Output folder prefix; subject number is appended: "
                        "<out-prefix>_01/, <out-prefix>_02/, …")

    p.add_argument("--mrsi-dat", nargs="+", required=True, metavar="FILE",
                   help="MRSI .dat files; subject ID parsed from trailing _N")
    p.add_argument("--mrsi-seq", required=True, help="MRSI PyPulseq .seq file")
    p.add_argument("--wref-dat", required=True, help="Water-reference .dat file")
    p.add_argument("--wref-seq", required=True, help="Water-reference PyPulseq .seq file")

    p.add_argument("--n-coils",         type=int, default=32)
    p.add_argument("--res",             type=int, default=64,
                   help="Reconstruction resolution (square)")
    p.add_argument("--noise-reps",      type=int, default=10,
                   help="Noise-only transients at start of wref")
    p.add_argument("--wref-start-index",type=int, default=0,
                   help="First useful ADC sample for wref (0=no trim; "
                        "confirm pre-echo length with supervisor)")
    p.add_argument("--mrsi-start-index",type=int, default=252,
                   help="First useful ADC sample for MRSI "
                        "(252 = supervisor value for SMF, fixes 1st-order phase)")
    p.add_argument("--chop-index",      type=int, default=40000,
                   help="Last+1 useful ADC sample (from reference slow_recon.py; "
                        "confirm for SMF sequence)")
    p.add_argument("--smap-npy",     default=None,
                   help="Path to precomputed smaps (n_coils, Ny, Nx) for wref_o; "
                        "skips inline MORSE-PI. E.g. output/subj/coilmap/ecalib_pp.npy")
    p.add_argument("--n-ref",        type=int, default=6,
                   help="MORSE-PI: number of reference coils")
    p.add_argument("--max-iter",     type=int, default=50,
                   help="MORSE-PI: CG iterations")
    p.add_argument("--calib-width",  type=int, default=16,
                   help="MORSE-PI: calibration region half-width")
    p.add_argument("--fid-trunc",    type=int, default=500,
                   help="ADC samples per transient used for the final wref_o image "
                        "(early FID → PD-weighted); 0 = use all K points")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    raw  = args.raw_dir.rstrip("/")

    N_COILS  = args.n_coils
    RES      = args.res
    SI_WREF  = args.wref_start_index
    SI_MRSI  = args.mrsi_start_index
    CI       = args.chop_index
    K_WREF   = CI - SI_WREF
    K_MRSI   = CI - SI_MRSI

    print(f"[twix2npy] wref  K = {K_WREF}  ({SI_WREF}:{CI})")
    print(f"[twix2npy] MRSI  K = {K_MRSI}  ({SI_MRSI}:{CI})  [first-order phase fix]")

    # ── 1. Water reference (processed once, shared across all subjects) ────────
    print("\n[twix2npy] Reading wref dat …")
    wref_raw, wref_hdrs = _read_twix(os.path.join(raw, args.wref_dat), N_COILS)

    N_total_wref = wref_raw.shape[1]
    pts_per_tr   = wref_raw.shape[2]
    N_sig_wref   = N_total_wref - args.noise_reps
    print(f"  transients: {N_total_wref} total  |  {args.noise_reps} noise  |  {N_sig_wref} signal")
    print(f"  pts/transient: {pts_per_tr}   ADC dwell: {wref_hdrs['adc_dwell']*1e6:.2f} µs")
    print(f"  spectrometer freq: {wref_hdrs['spec_freq']:.6f} MHz")

    noise_data = wref_raw[:, :args.noise_reps, :]
    sigma_noise = float(
        np.sqrt(np.mean(np.var(noise_data.reshape(N_COILS, -1), axis=1) / 2))
    )
    print(f"  sigma_noise = {sigma_noise:.4e}")

    wref_sig = wref_raw[:, args.noise_reps:, SI_WREF:CI]   # (C, N_sig, K_WREF)

    print("[twix2npy] Computing slice orientation …")
    pos_info = _calc_pos_info(wref_hdrs['fullHdr'], RES, 1)

    print("[twix2npy] Reading wref seq trajectory …")
    noise_pts_strip = args.noise_reps * pts_per_tr
    wref_traj_full, seq_fov = _read_traj(
        os.path.join(raw, args.wref_seq),
        N_sig_wref, pts_per_tr,
        strip_pts=noise_pts_strip,
    )

    wref_traj = _scale_traj(wref_traj_full, seq_fov, RES, SI_WREF, CI)  # (N_sig, K_WREF, 3)

    fov_mm = seq_fov * 1000
    shifts = _calc_shifts(pos_info, fov_mm, RES)
    phase  = _phase_correction(wref_traj, shifts)

    traj_2d = wref_traj[:, :, :2].reshape(-1, 2).astype(np.float32)

    data_2d_raw    = wref_sig.reshape(N_COILS, -1)
    wref_sig_ph    = wref_sig * phase[np.newaxis, ...]       # phase correction, applied once
    data_2d_phcorr = wref_sig_ph.reshape(N_COILS, -1)

    # Truncated FID for wref_o image (PD-weighted): first fid_trunc pts per transient
    N_trunc = args.fid_trunc if args.fid_trunc > 0 else K_WREF
    N_trunc = min(N_trunc, K_WREF)
    if N_trunc < K_WREF:
        print(f"[twix2npy] Truncated FID: using first {N_trunc}/{K_WREF} pts per transient for wref_o image")
        traj_2d_img  = wref_traj[:, :N_trunc, :2].reshape(-1, 2).astype(np.float32)
        data_img     = wref_sig_ph[:, :, :N_trunc].reshape(N_COILS, -1)
    else:
        traj_2d_img  = None
        data_img     = None

    print("[twix2npy] Reconstructing wref_o (MORSE-PI smaps + sensitivity-weighted adjoint) …")
    wref_o = _reconstruct_wref_image(
        data_2d_raw, data_2d_phcorr, traj_2d, RES,
        smap_npy    = args.smap_npy,
        n_ref       = args.n_ref,
        max_iter    = args.max_iter,
        calib_width = args.calib_width,
        traj_2d_img = traj_2d_img,
        data_img    = data_img,
    )

    wref_sig = wref_sig_ph   # phase-corrected, saved as wref_data.npy

    wref_data_npy = wref_sig.transpose(2, 1, 0)[np.newaxis, ...]   # (1, K, N_sig, C)
    wref_ksp_npy  = wref_traj.transpose(2, 1, 0)                   # (3, K, N_sig)
    print(f"  wref_data shape: {wref_data_npy.shape}")
    print(f"  wref_ksp  shape: {wref_ksp_npy.shape}")
    print(f"  wref_o    shape: {wref_o.shape}")

    WREF_FILES = ["wref_data.npy", "wref_ksp.npy", "wref_o.npy", "sigma_noise.npy", "wref_o_check.png",
                  "affine.npy"]

    # ── 2. MRSI trajectory (read once; same seq for all files) ────────────────
    print("\n[twix2npy] Reading MRSI seq trajectory …")
    # peek at first MRSI file to know N_per_file and get MRSI header for affine
    first_mrsi_raw, first_mrsi_hdrs = _read_twix(os.path.join(raw, args.mrsi_dat[0]), N_COILS)
    N_per_file = first_mrsi_raw.shape[1]
    del first_mrsi_raw
    # affine must use the MRSI header so spec2nii takes the CSI orientation path
    # (pixel spacing = FOV/N_voxels), not the SVS path used by the wref header
    _q = _calc_pos_info(first_mrsi_hdrs['fullHdr'], RES, 1)['affine']
    mrsi_affine = _q.copy()

    mrsi_traj_one, _ = _read_traj(
        os.path.join(raw, args.mrsi_seq),
        N_per_file, pts_per_tr,
        strip_pts=0,
    )
    mrsi_traj_one = _scale_traj(mrsi_traj_one, seq_fov, RES, SI_MRSI, CI)  # (N, K_MRSI, 3)
    phase_mrsi_one = _phase_correction(mrsi_traj_one, shifts)               # (N, K_MRSI)

    mrsi_ksp_npy_one = mrsi_traj_one.transpose(2, 1, 0)   # (3, K_MRSI, N)

    # ── 3. Per-subject loop ────────────────────────────────────────────────────
    first_subj_dir = None

    for fname in args.mrsi_dat:
        subj_num = _parse_subject_num(fname)
        out_dir  = f"{args.out_prefix}_{subj_num:02d}"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[twix2npy] Subject {subj_num:02d}  →  {out_dir}")
        print(f"  reading {fname} …")

        data_i, _ = _read_twix(os.path.join(raw, fname), N_COILS)  # (C, N, pts)
        N_tr = data_i.shape[1]
        data_i = data_i[:, :, SI_MRSI:CI]                           # (C, N, K_MRSI)
        print(f"  transients: {N_tr}  →  data shape {data_i.shape}")

        data_i = data_i * phase_mrsi_one[np.newaxis, ...]           # phase correction
        mrsi_data_npy = data_i.transpose(2, 1, 0)[np.newaxis, ...]  # (1, K, N, C)
        print(f"  mrsi_data shape: {mrsi_data_npy.shape}")
        print(f"  mrsi_ksp  shape: {mrsi_ksp_npy_one.shape}")

        np.save(os.path.join(out_dir, "mrsi_data.npy"), mrsi_data_npy)
        np.save(os.path.join(out_dir, "mrsi_ksp.npy"),  mrsi_ksp_npy_one)
        print(f"  saved mrsi_data.npy  mrsi_ksp.npy")

        if first_subj_dir is None:
            # first subject: save wref files directly
            first_subj_dir = out_dir
            np.save(os.path.join(out_dir, "wref_data.npy"),   wref_data_npy)
            np.save(os.path.join(out_dir, "wref_ksp.npy"),    wref_ksp_npy)
            np.save(os.path.join(out_dir, "wref_o.npy"),      wref_o)
            np.save(os.path.join(out_dir, "sigma_noise.npy"),  np.float32(sigma_noise))
            np.save(os.path.join(out_dir, "affine.npy"),       mrsi_affine.astype(np.float64))
            np.save(os.path.join(out_dir, "wref_affine.npy"),  pos_info['affine'].astype(np.float64))
            _save_wref_o_png(wref_o, out_dir)
            print(f"  saved wref files + affine.npy + wref_affine.npy")
        else:
            # subsequent subjects: symlink to first subject's wref files
            _symlink_wref(first_subj_dir, out_dir, WREF_FILES)

        # save scan parameters so pipeline scripts can auto-load them
        p = save_scan_params(
            out_dir,
            dwelltime   = wref_hdrs['adc_dwell'],
            center_freq = wref_hdrs['spec_freq'],
            n_coils     = N_COILS,
            res         = RES,
            k_mrsi      = K_MRSI,
            k_wref      = K_WREF,
        )
        print(f"  saved {os.path.basename(p)}")

    print("\n[twix2npy] Done.")
    print(f"  ADC dwell:         {wref_hdrs['adc_dwell']*1e6:.2f} µs")
    print(f"  Spectrometer freq: {wref_hdrs['spec_freq']:.6f} MHz")
    print(f"  K wref:            {K_WREF}  (trim {SI_WREF}:{CI})")
    print(f"  K MRSI:            {K_MRSI}  (trim {SI_MRSI}:{CI}, first-order phase fix)")
    print(f"  wref shots:        {N_sig_wref}")
    print(f"  sigma_noise:       {sigma_noise:.4e}")
    print(f"  subjects:          {len(args.mrsi_dat)}")
    print()
    print("Next steps — run pipeline scripts with e.g.:")
    print(f"  python scripts/data_preproc/data_proc_02_coil_correction.py --data-dir {args.out_prefix}_01")


if __name__ == "__main__":
    main()
