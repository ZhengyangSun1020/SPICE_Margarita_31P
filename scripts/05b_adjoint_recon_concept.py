#!/usr/bin/env python3
"""
31P CONCEPT CRT adjoint NUFFT reconstruction from Siemens Twix .dat files.

Usage:
    python 05b_adjoint_recon_concept.py --dat "path/to/meas_*.dat" --out-dir ./output
    python 05b_adjoint_recon_concept.py --dat "path/to/meas_*.dat" --out-dir ./output --no-dcf
    python 05b_adjoint_recon_concept.py --dat "path/to/meas_*.dat" --out-dir ./output --no-hamming
    python 05b_adjoint_recon_concept.py --dat "path/to/meas_*.dat" --out-dir ./output --no-dcf --no-hamming

Options:
    --dat             Siemens Twix .dat file (required)
    --out-dir         Output directory (default: ./output)
    --no-dcf          Disable radial density compensation
    --no-hamming      Disable Hamming k-space weighting (keeps DCF)
    --no-phase-corr   Disable temporal interleave phase correction
    --no-fov-shift    Disable FOV shift correction
    --ppmlim L H      PPM display range (default: -20 20)
    --plot-voxel Y X  Plot a specific voxel instead of best-SNR

Pipeline:
    1. Load & rearrange raw Twix data (pymapVBVD)
    2. Phase correction for temporal interleaving
    3. Build concentric-ring trajectory from ICE parameters
    4. FOV shift via Siemens fGSLCalcPRS rotation
    5. Density compensation (radial) + Hamming k-space weighting
    6. 2D NUFFT adjoint per spectral point (finufft)
    7. FFT to spectrum domain

Output structure:
    <out-dir>/
        data/
            recon_fid.npy         (Ny, Nx, n_spec) complex64 — FID domain
            recon_spec.npy        (Ny, Nx, n_spec) complex64 — spectrum domain
            kdata_phase_corr.npy  (TrajPts, n_spec, 1, n_rings) — phase-corrected k-space
            adj_recon.nii.gz      NIfTI-MRS (if nifti_mrs installed)
        trajectory/
            traj_raw.npy          (n_total_pts, 2) — raw kx,ky in mT·µs/m
            traj_cpp.npy          (n_total_pts, 2) — kx,ky in cycles/pixel
            dcf.npy               (n_total_pts,)   — density compensation weights
        figures/
            magnitude_map.png     spatial mean-|spectrum| map
            spectrum_best_snr.png spectrum at best-SNR voxel
            all_spectra.png       all-voxel spectra grid
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from warnings import filterwarnings
filterwarnings("ignore")

import mrinufft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recon_concept_twix_31p import (
    load_concept_twix,
    concept_phase_correction,
    FIDToSpec,
)

try:
    from nifti_mrs.create_nmrs import gen_nifti_mrs
except ImportError:
    gen_nifti_mrs = None


# ── Siemens fGSLCalcPRS ────────────────────────────────────────────────


def _fGSLClassOri(dGs):
    """Siemens fGSLClassOri: classify slice orientation (0=sag, 1=cor, 2=tra)."""
    s, c, t = abs(dGs[0]), abs(dGs[1]), abs(dGs[2])
    eq_sc = abs(s - c) <= 1e-6
    eq_st = abs(s - t) <= 1e-6
    eq_ct = abs(c - t) <= 1e-6

    if ((eq_sc and eq_st) or (eq_sc and s < t) or (eq_st and s > c) or
            (eq_ct and c > s) or (s > c and s < t) or (s < c and c < t) or
            (s < t and t > c) or (c < t and t > s)):
        return 2  # transversal
    elif ((eq_sc and s > t) or (eq_st and s < c) or (s < c and c > t) or
            (s > t and s < c) or (s < t and t < c)):
        return 1  # coronal
    elif ((eq_ct and c < s) or (s > c and s > t) or (c > t and c < s) or
            (c < t and t < s)):
        return 0  # sagittal
    else:
        raise ValueError("Invalid slice orientation")


def _fGSLCalcPRS(dGs, dPhi):
    """Siemens fGSLCalcPRS: compute PE (phase) and RO (read) direction vectors."""
    ori = _fGSLClassOri(dGs)

    if ori == 2:  # transversal
        d = np.sqrt(dGs[1]**2 + dGs[2]**2)
        dGp = np.array([0.0, dGs[2] / d, -dGs[1] / d])
    elif ori == 1:  # coronal
        d = np.sqrt(dGs[0]**2 + dGs[1]**2)
        dGp = np.array([dGs[1] / d, -dGs[0] / d, 0.0])
    else:  # sagittal
        d = np.sqrt(dGs[0]**2 + dGs[1]**2)
        dGp = np.array([-dGs[1] / d, dGs[0] / d, 0.0])

    dGr = np.cross(dGs, dGp)

    if abs(dPhi) > 1e-10:
        co, si = np.cos(dPhi), np.sin(dPhi)
        dGp = co * dGp - si * dGr
        dGr = np.cross(dGs, dGp)

    return dGp, dGr


# ── Reconstruction steps ───────────────────────────────────────────────


def build_trajectory(first_circ, n_rings, traj_pts, gamma_mhz_per_mt, fov_m, N):
    """Build CRT trajectory from ICE parameters."""
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


def compute_fov_shift(info):
    """Compute FOV shift in voxels using Siemens fGSLCalcPRS convention."""
    pos = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]])
    snv = np.array([info["snv_x"], info["snv_y"], info["snv_z"]])
    ipr = info["in_plane_rotation"]

    phase_vec, read_vec = _fGSLCalcPRS(snv, ipr)
    rot_mat = np.column_stack([phase_vec, read_vec, snv])
    base_pos = rot_mat.T @ pos
    vox_size = info["fov_mm"] / info["matrix_size"]

    dim1_shift = base_pos[0] / vox_size  # PE
    dim2_shift = base_pos[1] / vox_size  # RO
    return dim1_shift, dim2_shift


def compute_dcf(kx_cpp, ky_cpp, kx_raw, ky_raw, use_hamming=True):
    """Compute density compensation weights with optional Hamming k-space weighting."""
    r_per_pt = np.sqrt(kx_cpp**2 + ky_cpp**2)
    weights = np.maximum(r_per_pt / (r_per_pt.max() + 1e-12), 0.01)

    if use_hamming:
        kx_norm = np.pi * kx_raw / (np.abs(kx_raw).max() + 1e-30)
        ky_norm = np.pi * ky_raw / (np.abs(ky_raw).max() + 1e-30)
        r_norm = np.sqrt(kx_norm**2 + ky_norm**2)
        ham = 0.54 + 0.46 * np.cos(r_norm)
        weights = weights * ham

    return weights.astype(np.float32)


def apply_fov_shift(kdata_flat, kx_cpp, ky_cpp, dim1_shift, dim2_shift):
    """Apply FOV shift as phase ramp on k-space data."""
    shift_phase = np.exp(-2j * np.pi * (kx_cpp * dim1_shift + ky_cpp * dim2_shift))
    return kdata_flat * shift_phase[:, np.newaxis]


def adjoint_nufft_recon(kdata_flat, traj, shape, n_spec, dcf):
    """Run 2D adjoint NUFFT per spectral point."""
    Ny, Nx = shape
    nufft_op = mrinufft.get_operator(
        "finufft", samples=traj, shape=(Ny, Nx),
        n_coils=1, density=dcf if dcf is not None else False,
    )

    recon_fid = np.zeros((n_spec, Ny, Nx), dtype=np.complex64)
    t0 = time.time()
    for s in range(n_spec):
        if s % 100 == 0:
            print(f"  spectral {s}/{n_spec}")
        recon_fid[s] = nufft_op.adj_op(kdata_flat[:, s]).astype(np.complex64)
    elapsed = time.time() - t0
    print(f"[recon] Adjoint done in {elapsed:.1f}s")

    image_fid = recon_fid.transpose(1, 2, 0)       # (Ny, Nx, n_spec)
    image_spec = FIDToSpec(image_fid, axis=-1)
    return image_fid, image_spec


# ── Plotting ───────────────────────────────────────────────────────────


def plot_magnitude_map(image_spec, fig_dir):
    """Save spatial mean-|spectrum| map."""
    mag_map = np.mean(np.abs(image_spec), axis=-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mag_map, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title("Adjoint recon — mean |spectrum|")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "magnitude_map.png"), dpi=120)
    plt.close(fig)
    np.save(os.path.join(fig_dir, "magnitude_map.npy"), mag_map)


def plot_best_snr_spectrum(image_spec, PPM, ppmlim, fig_dir):
    """Find best-SNR voxel and save its spectrum."""
    Ny, Nx, n_spec = image_spec.shape
    snr_map = np.zeros((Ny, Nx))
    for y in range(Ny):
        for x in range(Nx):
            spec = np.abs(image_spec[y, x, :])
            snr_map[y, x] = spec.max() / (np.std(spec[int(0.8 * n_spec):]) + 1e-20)
    vy, vx = np.unravel_index(snr_map.argmax(), snr_map.shape)
    print(f"[recon] Best SNR voxel: [{vy},{vx}] SNR={snr_map[vy, vx]:.1f}")

    spec_vox = np.abs(image_spec[vy, vx, :])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(PPM, spec_vox, 'k-', lw=0.8)
    ax.fill_between(PPM, 0, spec_vox, alpha=0.15, color='blue')
    ax.set_xlim(ppmlim[1], ppmlim[0])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ppm")
    ax.set_ylabel("|spectrum|")
    ax.set_title(f"Voxel [{vy},{vx}] — magnitude (best SNR)")
    for pos, name in [(0, 'PCr'), (5, 'Pi'), (-2.5, 'γ-ATP'),
                      (-7.5, 'α-ATP'), (-16, 'β-ATP')]:
        ax.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.text(pos, ax.get_ylim()[1] * 0.92, name, fontsize=8,
                ha='center', color='red')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spectrum_best_snr.png"), dpi=150)
    plt.close(fig)
    np.save(os.path.join(fig_dir, "snr_map.npy"), snr_map)


def plot_all_spectra(image_spec, PPM, ppmlim, fig_dir):
    """Save all-voxel spectra grid."""
    Ny, Nx = image_spec.shape[:2]
    fig, axes = plt.subplots(Ny, Nx, figsize=(Nx * 2.5, Ny * 2),
                             sharex=True, sharey=True)
    if Ny == 1:
        axes = axes[np.newaxis, :]
    if Nx == 1:
        axes = axes[:, np.newaxis]
    for y in range(Ny):
        for x in range(Nx):
            ax = axes[Ny - 1 - y, x]
            sp = np.abs(image_spec[y, x, :])
            ax.plot(PPM, sp, 'k-', lw=0.4)
            ax.fill_between(PPM, 0, sp, alpha=0.08, color='blue')
            ax.set_xlim(ppmlim[1], ppmlim[0])
            ax.set_title(f"[{y},{x}]", fontsize=7, pad=1)
            ax.tick_params(labelsize=4)
            ax.axvline(0, color='gray', ls='--', lw=0.3, alpha=0.4)
    fig.suptitle("All voxel spectra — magnitude", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(fig_dir, "all_spectra.png"), dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="31P CONCEPT CRT adjoint NUFFT reconstruction")
    p.add_argument("--dat", required=True, help="Siemens Twix .dat file")
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift", action="store_true")
    p.add_argument("--no-dcf", action="store_true")
    p.add_argument("--no-hamming", action="store_true")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--plot-voxel", type=int, nargs=2, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Output directories
    data_dir = os.path.join(args.out_dir, "data")
    traj_dir = os.path.join(args.out_dir, "trajectory")
    fig_dir = os.path.join(args.out_dir, "figures")
    for d in [data_dir, traj_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load & rearrange ────────────────────────────────────────────
    info = load_concept_twix(args.dat)

    kdata = info["kdata"]
    N = info["matrix_size"]
    Ny, Nx = N, N
    n_spec = info["n_spec"]
    n_rings = info["n_rings"]
    traj_pts = int(info["traj_pts"][0])
    dwell_s = info["dwell_s"] * 2
    larmor_mhz = info["larmor_freq_mhz"]
    gamma_mhz_per_mt = info["gamma_2pi"] / 1e9

    print(f"\n[recon] Matrix={Ny}x{Nx}, vecSize={n_spec}, "
          f"rings={n_rings}, TrajPts={traj_pts}")

    # ── 2. Phase correction ────────────────────────────────────────────
    if not args.no_phase_corr:
        ti_per_ring = info["_ti_per_ring"]
        print(f"[recon] Phase correction TI={ti_per_ring}")
        kdata = concept_phase_correction(kdata, ti_per_ring)

    # ── 3. Trajectory ──────────────────────────────────────────────────
    kx_raw, ky_raw, kx_cpp, ky_cpp = build_trajectory(
        info["first_circle"], n_rings, traj_pts,
        gamma_mhz_per_mt, info["fov_m"], N)
    traj = np.stack([kx_cpp, ky_cpp], axis=-1).astype(np.float32)
    total_kpts = len(kx_raw)

    print(f"[recon] Trajectory: {total_kpts} pts, "
          f"k_range=[{kx_cpp.min():.4f}, {kx_cpp.max():.4f}] cyc/pix")

    np.save(os.path.join(traj_dir, "traj_raw.npy"),
            np.stack([kx_raw, ky_raw], axis=-1))
    np.save(os.path.join(traj_dir, "traj_cpp.npy"), traj)

    # ── 4. FOV shift ───────────────────────────────────────────────────
    dim1_shift = dim2_shift = 0.0
    if not args.no_fov_shift:
        dim1_shift, dim2_shift = compute_fov_shift(info)
        print(f"[recon] FOV shift: dim1(PE)={dim1_shift:.4f}, "
              f"dim2(RO)={dim2_shift:.4f} vox")

    # ── 5. Flatten & apply FOV shift ───────────────────────────────────
    kdata_flat = kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
    if not args.no_fov_shift:
        kdata_flat = apply_fov_shift(kdata_flat, kx_cpp, ky_cpp,
                                     dim1_shift, dim2_shift)

    # ── 6. DCF + Hamming ───────────────────────────────────────────────
    dcf = None
    if not args.no_dcf:
        dcf = compute_dcf(kx_cpp, ky_cpp, kx_raw, ky_raw,
                          use_hamming=not args.no_hamming)
        if not args.no_hamming:
            print(f"[recon] DCF + Hamming weighting applied")
        else:
            print(f"[recon] DCF applied (no Hamming)")
        np.save(os.path.join(traj_dir, "dcf.npy"), dcf)

    # ── 7. NUFFT adjoint ───────────────────────────────────────────────
    print(f"[recon] Building finufft operator: {total_kpts} pts -> {Ny}x{Nx}")
    image_fid, image_spec = adjoint_nufft_recon(
        kdata_flat, traj, (Ny, Nx), n_spec, dcf)
    print(f"[recon] image shape={image_spec.shape}  "
          f"|max|={np.abs(image_spec).max():.4e}")

    # ── 8. Save data ───────────────────────────────────────────────────
    np.save(os.path.join(data_dir, "recon_fid.npy"), image_fid)
    np.save(os.path.join(data_dir, "recon_spec.npy"), image_spec)
    np.save(os.path.join(data_dir, "kdata_phase_corr.npy"), kdata)
    print(f"[recon] Saved recon data to {data_dir}")

    if gen_nifti_mrs is not None:
        nii_data = image_fid[:, :, np.newaxis, :]
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = info["fov_mm"] / N
        nifti_adj = gen_nifti_mrs(nii_data, dwelltime=dwell_s,
                                  spec_freq=larmor_mhz, affine=affine)
        nifti_adj.save(os.path.join(data_dir, "adj_recon.nii.gz"))
        print(f"[recon] Saved adj_recon.nii.gz")

    # ── 9. Figures ─────────────────────────────────────────────────────
    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    PPM = FREQ / larmor_mhz + args.ppm_center

    plot_magnitude_map(image_spec, fig_dir)
    plot_best_snr_spectrum(image_spec, PPM, args.ppmlim, fig_dir)
    plot_all_spectra(image_spec, PPM, args.ppmlim, fig_dir)

    print(f"[recon] Saved figures to {fig_dir}")
    print("[recon] Done.")


if __name__ == "__main__":
    main()
