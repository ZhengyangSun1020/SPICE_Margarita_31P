#!/usr/bin/env python3
"""Per-voxel adjoint spectra maps (10x10 grid) for rep0 and rep30, shared scale.

Reuses adj_spec_for_rep() from _adj_multirep_sum.py (reproduces 05c's adjoint
pipeline). ROI voxels are tinted. Also writes an overlay figure with both reps
in each cell for direct comparison.
"""
import os, importlib.util as ilu
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/zsun/SPICE_Margarita"
REPS = [0, 30]
APOD_HZ = 20.0                 # Gaussian apodization (line broadening) in Hz
PPM_LO, PPM_HI = -20.0, 10.0   # display window (covers bATP..downfield of Pi)

s = ilu.spec_from_file_location("mr", os.path.join(ROOT, "_adj_multirep_sum.py"))
mr = ilu.module_from_spec(s); s.loader.exec_module(mr)

def main():
    specs = {}; PPM = None
    for rep in REPS:
        sp, ppm = mr.adj_spec_for_rep(rep, apod_hz=APOD_HZ)
        specs[rep] = np.abs(sp)
        if PPM is None: PPM = ppm
    N = specs[REPS[0]].shape[0]
    win = (PPM >= PPM_LO) & (PPM <= PPM_HI)
    xw = PPM[win]
    # shared y-scale = 99.5th pct over all voxels across both reps
    allv = np.concatenate([specs[r][:, :, win].ravel() for r in REPS])
    g = np.percentile(allv, 99.5) + 1e-12
    peaks = [(0, 'PCr'), (4.85, 'Pi'), (-2.53, 'γ'), (-7.58, 'α'), (-15.86, 'β')]

    # ── one map per rep ──
    for rep in REPS:
        fig, axs = plt.subplots(N, N, figsize=(N*1.25, N*1.05),
                                gridspec_kw=dict(wspace=0.05, hspace=0.05))
        for xi in range(N):
            for yi in range(N):
                ax = axs[N-1-xi][yi]          # xi upward so row0 at bottom
                ax.plot(xw, specs[rep][xi, yi, win]/g, 'k-', lw=0.5)
                for pos, _ in peaks:
                    ax.axvline(pos, color='r', ls=':', lw=0.3, alpha=0.5)
                ax.set_xlim(PPM_HI, PPM_LO); ax.set_ylim(0, 1.05)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Adjoint per-voxel spectra map — rep{rep}  "
                     f"({APOD_HZ:.0f}Hz apod; ppm {PPM_HI}..{PPM_LO}; shared "
                     f"scale; red=PCr/Pi/ATP)", fontsize=10)
        outp = os.path.join(ROOT, f"_specmap_adj_rep{rep:02d}.png")
        fig.savefig(outp, dpi=140, bbox_inches='tight'); plt.close(fig)
        print("saved ->", outp)

    # ── overlay (rep0 blue vs rep30 red) ──
    fig, axs = plt.subplots(N, N, figsize=(N*1.25, N*1.05),
                            gridspec_kw=dict(wspace=0.05, hspace=0.05))
    for xi in range(N):
        for yi in range(N):
            ax = axs[N-1-xi][yi]
            ax.plot(xw, specs[REPS[0]][xi, yi, win]/g, 'b-', lw=0.5)
            ax.plot(xw, specs[REPS[1]][xi, yi, win]/g, 'r-', lw=0.5)
            for pos, _ in peaks:
                ax.axvline(pos, color='k', ls=':', lw=0.3, alpha=0.4)
            ax.set_xlim(PPM_HI, PPM_LO); ax.set_ylim(0, 1.05)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Adjoint spectra map overlay — rep{REPS[0]}(blue) vs "
                 f"rep{REPS[1]}(red)  ({APOD_HZ:.0f}Hz apod; shared scale)",
                 fontsize=10)
    outp = os.path.join(ROOT, "_specmap_adj_overlay.png")
    fig.savefig(outp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print("saved ->", outp)

if __name__ == "__main__":
    main()
