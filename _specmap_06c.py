#!/usr/bin/env python3
"""Per-voxel spectra maps (10x10 grid) from 06c iterative recon, rep0 & rep30.

Reads iter3d_recon_spec_rep{NN}.npy (shape Nx,Ny,Nz,n_spec) from the 06c out-dir.
Builds the ppm axis from the .dat exactly as 06c does. Shared scale, no ROI tint.
"""
import os, sys, importlib.util as ilu
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/zsun/SPICE_Margarita"
OUT  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "output_dynamic_3T_0724_5rings_06c")
DAT  = ("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
        "meas_MID00277_FID221482_240FOV_10x10_5rings.dat")
REPS = [0, 30]
PPM_LO, PPM_HI = -20.0, 10.0

def load_ppm(n_spec):
    # 06c: dwell_s = ADC dwell*2 after OS removal; PPM = -fftshift(fftfreq)/larmor
    s = ilu.spec_from_file_location("m", os.path.join(ROOT, "scripts",
                                                      "05c_spice_recon_31p.py"))
    m = ilu.module_from_spec(s); s.loader.exec_module(m)
    info = m.load_concept_twix_3d(DAT, rep_sel=0)
    dwell_s = info["dwell_s"] * 2; larmor = info["larmor_freq_mhz"]
    return -np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s)) / larmor

def main():
    specs = {}
    for rep in REPS:
        p = os.path.join(OUT, "data", f"iter3d_recon_spec_rep{rep:02d}.npy")
        if not os.path.exists(p):
            print(f"[missing] {p}"); return
        a = np.load(p)
        if a.ndim == 4:  a = a[:, :, a.shape[2] // 2, :]   # central slice (Nz=1 here)
        specs[rep] = np.abs(a)
    PPM = load_ppm(specs[REPS[0]].shape[-1])
    N = specs[REPS[0]].shape[0]
    win = (PPM >= PPM_LO) & (PPM <= PPM_HI); xw = PPM[win]
    allv = np.concatenate([specs[r][:, :, win].ravel() for r in REPS])
    g = np.percentile(allv, 99.5) + 1e-12
    peaks = [0.0, 4.85, -2.53, -7.58, -15.86]

    for rep in REPS:
        fig, axs = plt.subplots(N, N, figsize=(N*1.25, N*1.05),
                                gridspec_kw=dict(wspace=0.05, hspace=0.05))
        for xi in range(N):
            for yi in range(N):
                ax = axs[N-1-xi][yi]
                ax.plot(xw, specs[rep][xi, yi, win]/g, 'k-', lw=0.5)
                for pos in peaks: ax.axvline(pos, color='r', ls=':', lw=0.3, alpha=0.5)
                ax.set_xlim(PPM_HI, PPM_LO); ax.set_ylim(0, 1.05)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"06c iterative per-voxel spectra map — rep{rep}  "
                     f"(ppm {PPM_HI}..{PPM_LO}; shared scale; red=PCr/Pi/ATP)",
                     fontsize=10)
        outp = os.path.join(ROOT, f"_specmap_06c_rep{rep:02d}.png")
        fig.savefig(outp, dpi=140, bbox_inches='tight'); plt.close(fig)
        print("saved ->", outp)

    fig, axs = plt.subplots(N, N, figsize=(N*1.25, N*1.05),
                            gridspec_kw=dict(wspace=0.05, hspace=0.05))
    for xi in range(N):
        for yi in range(N):
            ax = axs[N-1-xi][yi]
            ax.plot(xw, specs[REPS[0]][xi, yi, win]/g, 'b-', lw=0.5)
            ax.plot(xw, specs[REPS[1]][xi, yi, win]/g, 'r-', lw=0.5)
            for pos in peaks: ax.axvline(pos, color='k', ls=':', lw=0.3, alpha=0.4)
            ax.set_xlim(PPM_HI, PPM_LO); ax.set_ylim(0, 1.05)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"06c spectra map overlay — rep{REPS[0]}(blue) vs rep{REPS[1]}(red)",
                 fontsize=10)
    outp = os.path.join(ROOT, "_specmap_06c_overlay.png")
    fig.savefig(outp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print("saved ->", outp)

if __name__ == "__main__":
    main()
