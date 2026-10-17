#!/usr/bin/env python3
"""Sum adjoint (model-free) recon over rep groups and test for a Pi peak.

Reproduces 05c's adjoint pipeline per rep, spatially sums over the ROI to one
complex spectrum, phase-aligns each rep to its PCr peak (removes rep-to-rep
global phase drift), then coherently sums the group. Compares group A (rep0-5)
vs group B (rep20-24) in the Pi region and reports a Pi SNR / position.

Usage:  python _adj_multirep_sum.py
"""
import os, importlib.util as ilu
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrinufft

ROOT = "/home/zsun/SPICE_Margarita"
DAT  = ("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
        "meas_MID00277_FID221482_240FOV_10x10_5rings.dat")
ROI  = os.path.join(ROOT, "roi_mask_mid00276_manual.npy")
EXTRA_SHIFT_RO = -1.0
EXTRA_SHIFT_PE = -1.0
GROUPS = {"A_rep0-5": [0, 1, 2, 3, 4, 5], "B_rep20-24": [20, 21, 22, 23, 24]}

s = ilu.spec_from_file_location("m", os.path.join(ROOT, "scripts",
                                                  "05c_spice_recon_31p.py"))
m = ilu.module_from_spec(s); s.loader.exec_module(m)


def adj_spec_for_rep(rep, apod_hz=0.0):
    """Return (adj_spec [N,N,n_spec] complex, PPM) reproducing 05c exactly.
    apod_hz: Gaussian apodization FWHM in Hz applied to the FID (0 = none)."""
    info = m.load_concept_twix_3d(DAT, rep_sel=rep)
    kdata = info["kdata"]; info["kdata"] = None
    N = info["matrix_size"]; n_spec = info["n_spec"]; n_rings = info["n_rings"]
    traj_pts = int(info["traj_pts"].max())
    dwell_s = info["dwell_s"] * 2; larmor = info["larmor_freq_mhz"]
    kdata = m.concept_phase_correction(kdata, info["_ti_per_ring"])
    traj = m.build_trajectory_2d(info["first_circle"], n_rings, traj_pts)
    kf = kdata[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec)
    d_pe, d_ro = m.compute_fov_shift(info)
    d_ro += EXTRA_SHIFT_RO; d_pe += EXTRA_SHIFT_PE
    kf = kf * np.exp(-2j*np.pi*(traj[:, 0]*d_pe + traj[:, 1]*d_ro))[:, None]
    r_xy = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    ham = r_xy * (float(info["k_norm_max_phys"]) / 0.5)
    dcf = m.compute_crt_dcf_hamming(traj, hamming_radius=ham)
    op = mrinufft.get_operator("finufft", samples=traj, shape=(N, N),
                               n_coils=1, density=dcf)
    adj = np.zeros((N*N, n_spec), np.complex64)
    for t in range(n_spec):
        adj[:, t] = op.adj_op(kf[:, t]).reshape(N*N)
    adj_img = adj.reshape(N, N, n_spec)
    if apod_hz > 0:                                     # match 05c Gaussian apodization
        t_apo = np.arange(n_spec) * dwell_s
        apo = np.exp(-np.pi**2 * apod_hz**2 * t_apo**2 / (4*np.log(2)))
        adj_img = adj_img * apo
    spec = m.FIDToSpec(adj_img.conj(), axis=-1)
    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    PPM = -FREQ / larmor
    return spec, PPM


def roi_complex_sum(spec, roi):
    return np.sum([spec[x, y] for x, y in np.argwhere(roi)], axis=0)


def main():
    roi = np.load(ROI).astype(bool)
    PPM = None
    summed = {}       # phase-aligned coherent sum per group
    summed_mag = {}   # magnitude sum per group (phase-insensitive sanity)
    per_rep = {}
    for gname, reps in GROUPS.items():
        acc = None; accm = None
        for rep in reps:
            spec, ppm = adj_spec_for_rep(rep)
            if PPM is None: PPM = ppm
            z = roi_complex_sum(spec, roi)                 # 1D complex, ROI spatial sum
            pcr_i = np.argmin(np.abs(PPM - 0.0))
            # search PCr peak within +/-1 ppm, align its phase to 0
            band = (PPM >= -1) & (PPM <= 1)
            pk = np.where(band)[0][np.argmax(np.abs(z[band]))]
            z_al = z * np.exp(-1j*np.angle(z[pk]))
            acc = z_al if acc is None else acc + z_al
            accm = np.abs(z) if accm is None else accm + np.abs(z)
            per_rep.setdefault(gname, []).append(np.abs(z))
            print(f"  {gname} rep{rep}: PCr|z|={np.abs(z[pk]):.3e} @ {PPM[pk]:+.2f}ppm")
        summed[gname] = acc
        summed_mag[gname] = accm

    # ── Pi analysis on the coherent sum ──
    pcr_m = (PPM >= -1) & (PPM <= 1)
    pi_m  = (PPM >= 3.6) & (PPM <= 6.0)
    noise_m = (PPM >= 10) & (PPM <= 25)     # signal-free downfield band
    print("\n===== Pi test on phase-aligned coherent group sum =====")
    for g in GROUPS:
        z = np.abs(summed[g])
        nf = z[noise_m].std() + 1e-12
        pcr_snr = z[pcr_m].max() / nf
        pi_i = np.where(pi_m)[0][np.argmax(z[pi_m])]
        pi_snr = z[pi_i] / nf
        cm = (PPM >= PPM[pi_i]-0.4) & (PPM <= PPM[pi_i]+0.4)
        pi_cen = float(np.sum(PPM[cm]*z[cm]) / (np.sum(z[cm])+1e-12))
        print(f"  {g}: PCr_SNR={pcr_snr:6.1f} | Pi peak@{PPM[pi_i]:.2f} "
              f"cen@{pi_cen:.2f}ppm  Pi_SNR={pi_snr:.2f}  "
              f"({'PLAUSIBLE Pi' if pi_snr>3 else 'no clear Pi (noise)'})")

    # ── plots ──
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    full = (PPM >= -20) & (PPM <= 12); piz = (PPM >= 2.5) & (PPM <= 6.5)
    cols = {"A_rep0-5": "tab:blue", "B_rep20-24": "tab:red"}
    for j, (mask, ttl) in enumerate(((full, "full"), (piz, "Pi zoom"))):
        for g in GROUPS:
            z = np.abs(summed[g]); gN = z[full].max()+1e-12
            axs[0][j].plot(PPM[mask], z[mask]/gN, color=cols[g], lw=1.0, label=g)
            zm = summed_mag[g]; gM = zm[full].max()+1e-12
            axs[1][j].plot(PPM[mask], zm[mask]/gM, color=cols[g], lw=1.0, label=g)
        for row, tag in ((0, "coherent (PCr-aligned) sum"), (1, "magnitude sum")):
            axs[row][j].axvline(4.85, color='k', ls='--', lw=0.5, alpha=0.5)
            axs[row][j].set_xlim(PPM[mask].max(), PPM[mask].min())
            axs[row][j].set_title(f"{tag} — {ttl}", fontsize=9)
            axs[row][j].set_xlabel("ppm"); axs[row][j].legend(fontsize=7)
    fig.suptitle("Adjoint rep-group sums (ROI): A=rep0-5 vs B=rep20-24  "
                 "(dashed = 4.85 ppm Pi)", fontsize=11)
    fig.tight_layout()
    outp = os.path.join(ROOT, "_adj_multirep_pi.png")
    fig.savefig(outp, dpi=140); plt.close(fig)
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
