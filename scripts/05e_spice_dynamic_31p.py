#!/usr/bin/env python3
"""
Dynamic SPICE 31P reconstruction — all reps, shared subspace.

Loads all repetitions at once, computes a shared subspace V from synthetic
training data (or from the time-averaged adjoint), then solves the SPICE
normal equation per rep independently.  Outputs PCr dynamics per voxel.

Usage:
    python scripts/05e_spice_dynamic_31p.py \\
        --dat "/path/to/dynamic.dat" \\
        --out-dir ./output_spice_dynamic \\
        --rank 6 --lambda1 1e-3 --n-iter 50 \\
        --apodization 30 \\
        --localizer "/path/to/localizer.IMA" \\
        --roi-mask roi_mask.npy
"""

import argparse
import importlib.util as _ilu
import os
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mrinufft
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg as sp_cg, LinearOperator
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

# ── Import loaders and helpers from sibling scripts ──────────────────────

_here = pathlib.Path(__file__).parent

def _import(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_05d = _import("_05d", _here / "05d_dynamic_adjoint_31p.py")
_05c = _import("_05c", _here / "05c_spice_recon_31p.py")

load_dynamic_concept_twix = _05d.load_dynamic_concept_twix
concept_phase_correction   = _05d.concept_phase_correction
build_trajectory_2d        = _05c.build_trajectory_2d
compute_crt_dcf_hamming    = _05c.compute_crt_dcf_hamming
compute_fov_shift          = _05c.compute_fov_shift
calc_Bmatrix_simple        = _05c.calc_Bmatrix_simple
generate_training_data        = _05c.generate_training_data
generate_training_data_pk2pi  = _05c.generate_training_data_pk2pi
load_localizer_registered  = _05c.load_localizer_registered
spice_reconstruct          = _05c.spice_reconstruct


def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)


def extract_pcr_amplitude(spec, PPM, ppm_lo=-1.5, ppm_hi=1.5):
    """Max |spectrum| in the PCr window per voxel. spec: (Nx, Ny, n_spec)."""
    mask = (PPM >= ppm_lo) & (PPM <= ppm_hi)
    return np.abs(spec[:, :, mask]).max(axis=-1)   # (Nx, Ny)


def extract_pi_amplitude(spec, PPM, ppm_lo=4.2, ppm_hi=5.2):
    """Max |spectrum| in the Pi window (Pi + Pi2) per voxel."""
    mask = (PPM >= ppm_lo) & (PPM <= ppm_hi)
    return np.abs(spec[:, :, mask]).max(axis=-1)


def plot_voxel_dynamics(spec_all, PPM, voxels, fig_dir, tr_s=None):
    """PCr and Pi amplitude vs rep for a list of specific (xi, yi) voxels."""
    n_reps = spec_all.shape[0]
    t_axis = np.arange(n_reps) * tr_s if tr_s is not None else np.arange(n_reps)
    xlabel = "Time (s)" if tr_s is not None else "Rep index"

    n_vox = len(voxels)
    fig, axes = plt.subplots(2, n_vox, figsize=(5 * n_vox, 8), squeeze=False)

    for col, (xi, yi) in enumerate(voxels):
        pcr = np.array([extract_pcr_amplitude(spec_all[r], PPM)[xi, yi] for r in range(n_reps)])
        pi  = np.array([extract_pi_amplitude(spec_all[r], PPM)[xi, yi]  for r in range(n_reps)])

        ax_pcr = axes[0, col]
        ax_pcr.plot(t_axis, pcr, '-o', ms=4, lw=1.5, color='steelblue')
        ax_pcr.set_xlabel(xlabel); ax_pcr.set_ylabel("|PCr| (a.u.)")
        ax_pcr.set_title(f"PCr  voxel ({xi},{yi})")

        ax_pi = axes[1, col]
        ax_pi.plot(t_axis, pi, '-o', ms=4, lw=1.5, color='firebrick')
        ax_pi.set_xlabel(xlabel); ax_pi.set_ylabel("|Pi| (a.u.)")
        ax_pi.set_title(f"Pi  voxel ({xi},{yi})")

    fig.suptitle("Per-voxel PCr and Pi dynamics", fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(fig_dir, "voxel_pcr_pi_dynamics.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[dyn-spice] Saved {out}")


def plot_pcr_dynamics(pcr_all, roi, fig_dir, tr_s=None):
    """Plot PCr amplitude vs rep for each in-ROI voxel + mean."""
    n_reps, Nx, Ny = pcr_all.shape
    reps = np.arange(n_reps)
    t_axis = reps * tr_s if tr_s is not None else reps
    xlabel = "Time (s)" if tr_s is not None else "Rep index"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-voxel traces
    ax = axes[0]
    for xi in range(Nx):
        for yi in range(Ny):
            if roi is None or roi[xi, yi]:
                ax.plot(t_axis, pcr_all[:, xi, yi], alpha=0.4, lw=0.8)
    mean_pcr = pcr_all[:, roi].mean(axis=-1) if roi is not None else pcr_all.reshape(n_reps, -1).mean(axis=-1)
    ax.plot(t_axis, mean_pcr, 'k-', lw=2, label='ROI mean')
    ax.set_xlabel(xlabel); ax.set_ylabel("|PCr| (a.u.)")
    ax.set_title("PCr dynamics — per voxel"); ax.legend()

    # Mean ± std
    ax2 = axes[1]
    vox_data = pcr_all[:, roi] if roi is not None else pcr_all.reshape(n_reps, -1)
    mean_v = vox_data.mean(axis=-1)
    std_v  = vox_data.std(axis=-1)
    ax2.fill_between(t_axis, mean_v - std_v, mean_v + std_v, alpha=0.25, color='blue')
    ax2.plot(t_axis, mean_v, 'b-', lw=1.5)
    ax2.set_xlabel(xlabel); ax2.set_ylabel("|PCr| (a.u.)")
    ax2.set_title("PCr dynamics — ROI mean ± std")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pcr_dynamics.png"), dpi=150)
    plt.close(fig)
    print(f"[dyn-spice] Saved pcr_dynamics.png")


def plot_pcr_map(pcr_mean, anatomy, fig_dir):
    """Mean PCr amplitude map overlaid on anatomy."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(anatomy, cmap='gray', origin='lower')
    axes[0].set_title("Localizer"); axes[0].axis('off')
    im = axes[1].imshow(pcr_mean.T, origin='lower', cmap='hot')
    plt.colorbar(im, ax=axes[1], label="|PCr| mean")
    axes[1].set_title("Mean PCr map"); axes[1].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pcr_mean_map.png"), dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Dynamic SPICE 31P reconstruction")
    p.add_argument("--dat",          required=True)
    p.add_argument("--out-dir",      default="./output_spice_dynamic")
    p.add_argument("--rank",         type=int, default=6)
    p.add_argument("--lambda1",      type=float, default=1e-3)
    p.add_argument("--n-iter",       type=int, default=50,
                   help="CG iterations per rep (0 = direct solve)")
    p.add_argument("--apodization",  type=float, default=0.0,
                   help="Gaussian apodization FWHM in Hz")
    p.add_argument("--subspace-src", default="pk2pi",
                   choices=["synthetic", "pk2pi", "average-adjoint"],
                   help="Subspace source: pk2pi (default, PK_3T_Skeletal_2Pi multiplets), synthetic, or average-adjoint")
    p.add_argument("--training-size", type=int, default=5000)
    p.add_argument("--localizer",    default=None)
    p.add_argument("--roi-mask",     default=None,
                   help=".npy boolean mask from roi_selector_31p.py")
    p.add_argument("--wmax",         type=float, default=5e3)
    p.add_argument("--prior-gamma",  type=float, default=1.0,
                   help="Optional gamma on normalized spatial prior before graph build")
    p.add_argument("--prior-scale",  type=float, default=1.0,
                   help="Scale for displayed prior map only")
    p.add_argument("--prior-scale-graph", type=float, default=1.0,
                   help="Scale for graph-construction prior only")
    p.add_argument("--prior-flip-native-y", action="store_true",
                   help="Flip registered localizer prior in native y before graph build")
    p.add_argument("--prior-only", action="store_true",
                   help="Build and save prior/graph diagnostics only; skip SPICE solve")
    p.add_argument("--lambda-out",   type=float, default=0.0,
                   help="Tikhonov penalty on out-of-ROI voxels (e.g. 1.0). Requires --roi-mask.")
    p.add_argument("--no-phase-corr", action="store_true")
    p.add_argument("--no-fov-shift",  action="store_true")
    p.add_argument("--tr",           type=float, default=None,
                   help="Repetition time in seconds (for time axis)")
    p.add_argument("--ppm-center",   type=float, default=0.0)
    p.add_argument("--ppmlim",       type=float, nargs=2, default=[-20, 20])
    p.add_argument("--plot-voxels",  type=int, nargs="*", default=None,
                   help="Voxel pairs xi yi to plot individually, e.g. --plot-voxels 6 5 6 6 6 7")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    fig_dir  = os.path.join(args.out_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir,  exist_ok=True)

    # ── 1. Load all reps ────────────────────────────────────────────────
    all_kdata, info = load_dynamic_concept_twix(args.dat)
    n_reps   = info["n_reps"]
    n_rings  = info["n_rings"]
    N        = info["matrix_size"]
    n_spec   = info["n_spec"]
    traj_pts = int(info["traj_pts"].max())
    dwell_s  = info["dwell_s"] * 2
    larmor_mhz = info["larmor_freq_mhz"]

    print(f"\n[dyn-spice] {N}×{N}, {n_spec} pts, {n_rings} rings, {n_reps} reps")

    # ── 2. Trajectory + DCF (shared) ────────────────────────────────────
    traj = build_trajectory_2d(info["first_circle"], n_rings, traj_pts)
    dcf  = compute_crt_dcf_hamming(traj)   # DCF from max-norm radius (unchanged)
    # Reconstruct on a FIXED PHYSICAL FOV (matching MATLAB CONCEPTPerform2DDirectFT):
    # rescale the max-normalized trajectory (outer ring -> ±0.5) back to physical
    # cycles/pixel (outer -> k_norm_max_phys < 0.5) AFTER computing the DCF, so the
    # recon FOV / voxel registration is identical for any ring count (max-norm ties
    # the FOV to the outermost ring -> ~1-voxel offset between 5-ring and 80-ring).
    traj = traj.copy()
    traj[:, 0] *= float(info["k_norm_max_phys"]) / 0.5
    traj[:, 1] *= float(info["k_norm_max_phys"]) / 0.5
    nufft_op = mrinufft.get_operator(
        "finufft", samples=traj, shape=(N, N), n_coils=1, density=dcf)
    n_kpts = traj.shape[0]

    # ── 3. FOV shift (shared) ───────────────────────────────────────────
    if not args.no_fov_shift:
        d_pe, d_ro = compute_fov_shift(info)
        shift_phase = np.exp(-2j * np.pi * (traj[:, 0] * d_pe + traj[:, 1] * d_ro))
    else:
        shift_phase = None

    # ── 4. Apodization window ───────────────────────────────────────────
    if args.apodization > 0:
        t_apo = np.arange(n_spec) * dwell_s
        apo = np.exp(-np.pi**2 * args.apodization**2 * t_apo**2 / (4 * np.log(2)))
    else:
        apo = None

    # ── 5. Spatial prior + Laplacian ────────────────────────────────────
    mag_norm = None
    if args.localizer is None:
        # Use first-rep adjoint magnitude as fallback spatial prior.
        kdata_r0 = all_kdata[0]
        if not args.no_phase_corr:
            kdata_r0 = concept_phase_correction(kdata_r0, info["_ti_per_ring"])
        kflat_r0 = kdata_r0[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
        if shift_phase is not None:
            kflat_r0 = kflat_r0 * shift_phase[:, np.newaxis]
        adj_r0 = np.stack([nufft_op.adj_op(kflat_r0[:, t]).reshape(N, N)
                           for t in range(n_spec)], axis=-1)
        mag_map = np.abs(adj_r0).mean(axis=-1)
        mag_norm = mag_map / (mag_map.max() + 1e-30)

    # Spatial prior for Laplacian: use registered localizer when provided,
    # otherwise fall back to adjoint magnitude.
    if args.localizer:
        anatomy_bg = load_localizer_registered(
            args.localizer, N,
            fov_mm  = info["fov_mm"],
            pos_sag = info["pos_sag"],
            pos_cor = info["pos_cor"],
            pos_tra = info["pos_tra"])
        if args.prior_flip_native_y:
            anatomy_bg = np.flipud(anatomy_bg)
            print("[dyn-spice] prior native-y flip: enabled")
        spatial_prior = anatomy_bg
        _prior_label  = "registered localizer (1H)"
    else:
        anatomy_bg    = mag_norm
        spatial_prior = mag_norm
        _prior_label  = "31P adjoint mag"

    roi = None
    if args.roi_mask:
        roi = np.load(args.roi_mask).astype(bool)
        if roi.shape != (N, N):
            sys.exit(f"ROI mask shape {roi.shape} != ({N},{N})")
        print(f"[dyn-spice] ROI: {int(roi.sum())}/{N*N} voxels — decoupling edges at boundary")

    if args.prior_gamma != 1.0:
        spatial_prior = np.power(np.clip(spatial_prior, 0.0, 1.0), args.prior_gamma).astype(np.float32)
        _prior_label = f"{_prior_label}, gamma={args.prior_gamma:g}"

    if roi is not None:
        spatial_prior = spatial_prior * roi.T.astype(np.float32)
        _prior_label = f"{_prior_label} + ROI masked"

    spatial_prior_graph = spatial_prior.astype(np.float32)
    spatial_prior_plot = spatial_prior.astype(np.float32)
    if args.prior_scale != 1.0:
        spatial_prior_plot = (spatial_prior_plot * float(args.prior_scale)).astype(np.float32)
        _prior_label = f"{_prior_label}, plot_scale={args.prior_scale:g}"
    if args.prior_scale_graph != 1.0:
        spatial_prior_graph = (spatial_prior_graph * float(args.prior_scale_graph)).astype(np.float32)

    # ── Save spatial prior map ──────────────────────────────────────────────
    _fig_sp, _ax_sp = plt.subplots(figsize=(6, 5))
    _im_sp = _ax_sp.imshow(spatial_prior_plot, origin='lower', cmap='Greens', aspect='equal')
    plt.colorbar(_im_sp, ax=_ax_sp)
    _ax_sp.set_title(f"Voxel-level sum prior map ({_prior_label})", fontsize=11)
    _ax_sp.set_xlabel("x (voxel)"); _ax_sp.set_ylabel("y (voxel)")
    _fig_sp.tight_layout()
    _fig_sp.savefig(os.path.join(fig_dir, "spatial_prior_map.png"), dpi=150, bbox_inches='tight')
    plt.close(_fig_sp)
    print(f"[dyn-spice] Saved spatial prior map -> {fig_dir}/spatial_prior_map.png")

    if args.localizer:
        _fig_or, (_ax_or0, _ax_or1) = plt.subplots(1, 2, figsize=(10, 4.5))
        _ax_or0.imshow(anatomy_bg, origin='lower', cmap='gray', aspect='equal')
        _ax_or0.set_title("Registered localizer")
        _ax_or0.set_xlabel("x (voxel)")
        _ax_or0.set_ylabel("y (voxel)")
        _im_or = _ax_or1.imshow(spatial_prior_plot, origin='lower', cmap='Greens', aspect='equal')
        plt.colorbar(_im_or, ax=_ax_or1)
        _ax_or1.set_title("Spatial prior (same orientation)")
        _ax_or1.set_xlabel("x (voxel)")
        _ax_or1.set_ylabel("y (voxel)")
        _fig_or.tight_layout()
        _fig_or.savefig(os.path.join(fig_dir, "localizer_vs_prior_orientation_check.png"), dpi=150, bbox_inches='tight')
        plt.close(_fig_or)
        print(f"[dyn-spice] Saved orientation check -> {fig_dir}/localizer_vs_prior_orientation_check.png")
    WW, _, A_graph = calc_Bmatrix_simple(spatial_prior_graph, wmax=args.wmax, adj=8, mask=roi, return_graph=True)

    voxel_sum_map = np.sum(A_graph, axis=0).reshape(N, N)
    n_neighbors_max = 8.0
    expected_max = float(args.wmax) * n_neighbors_max
    vmax_actual = float(voxel_sum_map.max())
    print(f"[dyn-spice] Graph weight-sum max={vmax_actual:.3g} (expected scale <= {expected_max:.3g} for adj=8)")

    _fig_vs, _ax_vs = plt.subplots(figsize=(6, 5))
    _im_vs = _ax_vs.imshow(voxel_sum_map, origin='lower', cmap='Greens', aspect='equal')
    plt.colorbar(_im_vs, ax=_ax_vs)
    _ax_vs.set_title("Voxel-level sum prior map (graph weights)", fontsize=11)
    _ax_vs.set_xlabel("x (voxel)")
    _ax_vs.set_ylabel("y (voxel)")
    _fig_vs.tight_layout()
    _fig_vs.savefig(os.path.join(fig_dir, "spatial_prior_voxel_sum_map.png"), dpi=150, bbox_inches='tight')
    plt.close(_fig_vs)
    print(f"[dyn-spice] Saved graph weight-sum prior map -> {fig_dir}/spatial_prior_voxel_sum_map.png")

    voxel_sum_norm = np.clip(voxel_sum_map / (expected_max + 1e-12), 0.0, 1.0)
    _fig_vsn, _ax_vsn = plt.subplots(figsize=(6, 5))
    _im_vsn = _ax_vsn.imshow(voxel_sum_norm, origin='lower', cmap='Greens', aspect='equal', vmin=0.0, vmax=1.0)
    plt.colorbar(_im_vsn, ax=_ax_vsn, label="normalized weight sum")
    _ax_vsn.set_title("Voxel-level sum prior map (graph weights, normalized)", fontsize=11)
    _ax_vsn.set_xlabel("x (voxel)")
    _ax_vsn.set_ylabel("y (voxel)")
    _fig_vsn.tight_layout()
    _fig_vsn.savefig(os.path.join(fig_dir, "spatial_prior_voxel_sum_map_normalized.png"), dpi=150, bbox_inches='tight')
    plt.close(_fig_vsn)
    print(f"[dyn-spice] Saved normalized graph weight-sum map -> {fig_dir}/spatial_prior_voxel_sum_map_normalized.png")

    _fig_ov, _ax_ov = plt.subplots(figsize=(6, 5))
    _ax_ov.imshow(spatial_prior_plot, origin='lower', cmap='gray', aspect='equal')
    _ax_ov.contour(voxel_sum_norm, levels=[0.2, 0.4, 0.6, 0.8], colors='lime', linewidths=1.0, origin='lower')
    _ax_ov.set_title("Prior + graph-weight contours (orientation check)", fontsize=11)
    _ax_ov.set_xlabel("x (voxel)")
    _ax_ov.set_ylabel("y (voxel)")
    _fig_ov.tight_layout()
    _fig_ov.savefig(os.path.join(fig_dir, "spatial_prior_graph_overlay.png"), dpi=150, bbox_inches='tight')
    plt.close(_fig_ov)
    print(f"[dyn-spice] Saved prior/graph overlay -> {fig_dir}/spatial_prior_graph_overlay.png")

    if args.prior_only:
        print("[dyn-spice] Prior-only mode done.")
        return

    # Out-of-ROI mask for Tikhonov suppression (row-major flat)
    out_roi_flat = None
    if roi is not None and args.lambda_out > 0:
        out_roi_flat = ~roi.T.flatten()

    # ── 6. Subspace V ───────────────────────────────────────────────────
    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    PPM  = -FREQ / larmor_mhz + args.ppm_center

    if args.subspace_src == "average-adjoint":
        print("[dyn-spice] Building subspace from time-averaged adjoint ...")
        avg_adj = np.zeros((N * N, n_spec), dtype=np.complex128)
        for kdata_r in tqdm(all_kdata, desc="Avg adjoint"):
            if not args.no_phase_corr:
                kdata_r = concept_phase_correction(kdata_r, info["_ti_per_ring"])
            kf = kdata_r[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
            if shift_phase is not None:
                kf = kf * shift_phase[:, np.newaxis]
            for t in range(n_spec):
                avg_adj[:, t] += nufft_op.adj_op(kf[:, t]).reshape(N * N)
        avg_adj /= n_reps
        training = avg_adj
    else:  # "pk2pi" or "synthetic"
        cache_name = ("training_data_pk2pi.npy" if args.subspace_src == "pk2pi"
                      else "training_data.npy")
        training_cache = os.path.join(data_dir, cache_name)
        if os.path.exists(training_cache):
            print(f"[dyn-spice] Loading cached training data: {cache_name}")
            training = np.load(training_cache)
        else:
            if args.subspace_src == "pk2pi":
                print(f"[dyn-spice] Generating {args.training_size} PK2Pi multiplet training FIDs ...")
                training = generate_training_data_pk2pi(
                    n_spec, dwell_s, larmor_mhz, args.training_size)
            else:
                print(f"[dyn-spice] Generating {args.training_size} synthetic training FIDs ...")
                training = generate_training_data(n_spec, dwell_s, larmor_mhz, args.training_size)
            np.save(training_cache, training)

    _, s_vals, Vh = np.linalg.svd(training, full_matrices=False)
    V = Vh[:args.rank, :].conj().T   # (n_spec, rank)

    # Plot singular values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(s_vals[:min(30, len(s_vals))], 'x-'); ax1.set_title("Singular values")
    for k in range(min(6, args.rank)):
        ax2.plot(PPM, np.abs(FIDToSpec(Vh[k, :].conj())), label=f"V{k}")
    ax2.set_xlim(20, -20); ax2.set_xlabel("ppm")
    ax2.set_title("Top subspace vectors"); ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "subspace.png"), dpi=120)
    plt.close(fig)
    print(f"[dyn-spice] Subspace: rank={args.rank}, "
          f"energy={np.sum(s_vals[:args.rank]**2)/np.sum(s_vals**2)*100:.1f}%")

    # ── 7. Per-rep SPICE solve ───────────────────────────────────────────
    pcr_all    = np.zeros((n_reps, N, N), dtype=np.float32)
    spec_all   = np.zeros((n_reps, N, N, n_spec), dtype=np.complex64)

    t0_total = time.time()
    for rep_idx, kdata_r in enumerate(tqdm(all_kdata, desc="SPICE per rep")):
        if not args.no_phase_corr:
            kdata_r = concept_phase_correction(kdata_r, info["_ti_per_ring"])
        kflat = kdata_r[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
        if shift_phase is not None:
            kflat = kflat * shift_phase[:, np.newaxis]

        # Adjoint (FID domain)
        adj_flat = np.zeros((N * N, n_spec), dtype=np.complex128)
        for t in range(n_spec):
            adj_flat[:, t] = nufft_op.adj_op(kflat[:, t]).reshape(N * N)

        spice_fid, _ = spice_reconstruct(
            adj_flat, nufft_op, V, WW, N, n_spec,
            rank=args.rank, lamda=args.lambda1, n_iter=args.n_iter,
            out_roi_mask=out_roi_flat, lambda_out=args.lambda_out)

        if apo is not None:
            spice_fid = spice_fid * apo

        spec = FIDToSpec(spice_fid.conj(), axis=-1)
        spec_all[rep_idx]  = spec.astype(np.complex64)
        pcr_all[rep_idx]   = extract_pcr_amplitude(spec, PPM)

    print(f"[dyn-spice] All reps done in {time.time()-t0_total:.0f}s")

    # ── 8. Save ─────────────────────────────────────────────────────────
    np.save(os.path.join(data_dir, f"spice_spec_all_rank{args.rank}.npy"), spec_all)
    np.save(os.path.join(data_dir, f"pcr_dynamics_rank{args.rank}.npy"),   pcr_all)
    print(f"[dyn-spice] Saved spec_all and pcr_dynamics")

    # ── 9. Plots ────────────────────────────────────────────────────────
    plot_pcr_dynamics(pcr_all, roi, fig_dir, tr_s=args.tr)
    plot_pcr_map(pcr_all.mean(axis=0), anatomy_bg, fig_dir)

    if args.plot_voxels and len(args.plot_voxels) >= 2:
        vox_pairs = [(args.plot_voxels[i], args.plot_voxels[i+1])
                     for i in range(0, len(args.plot_voxels) - 1, 2)]
        plot_voxel_dynamics(spec_all, PPM, vox_pairs, fig_dir, tr_s=args.tr)

    # Best-voxel spectrum from rep 0
    spec_r0 = spec_all[0]
    snr_map = np.abs(spec_r0).max(axis=-1) / (np.abs(spec_r0).std(axis=-1) + 1e-30)
    vy, vx = np.unravel_index(snr_map.argmax(), snr_map.shape)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(PPM, np.abs(spec_r0[vy, vx, :]), 'k-', lw=0.9)
    ax.set_xlim(args.ppmlim[1], args.ppmlim[0])
    ax.set_xlabel("ppm"); ax.set_title(f"SPICE rep0 — best voxel [{vy},{vx}]")
    for pos, name in [(0,'PCr'),(4.85,'Pi'),(-2.53,'γATP'),(-7.58,'αATP'),(-15.86,'βATP')]:
        ax.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.text(pos, ax.get_ylim()[1]*0.9, name, fontsize=7, ha='center', color='red')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "spice_rep0_best_voxel.png"), dpi=150)
    plt.close(fig)

    print(f"\n[dyn-spice] Done.  Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
