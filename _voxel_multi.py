#!/usr/bin/env python3
"""Overlay a voxel's 06c spectrum across reps, with correct (+freq/larmor) axis."""
import os, importlib.util as ilu
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/zsun/SPICE_Margarita"
OUT  = os.path.join(ROOT, "output_dynamic_3T_0724_5rings_06c")
DAT  = ("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
        "meas_MID00277_FID221482_240FOV_10x10_5rings.dat")
VOXELS = [(4, 5), (4, 4)]
REPS   = [0, 25, 30, 40]

def load_ppm(n_spec):
    s = ilu.spec_from_file_location("m", os.path.join(ROOT, "scripts",
                                                      "05c_spice_recon_31p.py"))
    m = ilu.module_from_spec(s); s.loader.exec_module(m)
    info = m.load_concept_twix_3d(DAT, rep_sel=0)
    # CORRECT convention = what 06c actually PLOTS with (06c line 1230, and 05c):
    #   PPM = -FREQ/larmor  (NOT the stored ppm_axis at line 512, which is +freq and unused for plots)
    return -np.fft.fftshift(np.fft.fftfreq(n_spec, d=info["dwell_s"]*2)) / info["larmor_freq_mhz"]

def main():
    specs = {}
    for rep in REPS:
        p = os.path.join(OUT, "data", f"iter3d_recon_spec_rep{rep:02d}.npy")
        if not os.path.exists(p):
            print(f"[skip] rep{rep} missing"); continue
        a = np.load(p); a = a[:, :, a.shape[2]//2, :]
        specs[rep] = np.abs(a)
    reps = [r for r in REPS if r in specs]
    PPM = load_ppm(specs[reps[0]].shape[-1])
    peaks = [(0.0,'PCr'), (4.85,'Pi'), (-2.53,'γ-ATP'), (-7.58,'α-ATP'), (-15.86,'β-ATP')]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(reps)))

    fig, axs = plt.subplots(len(VOXELS), 1, figsize=(12, 4.2*len(VOXELS)), squeeze=False)
    for vi, (xi, yi) in enumerate(VOXELS):
        ax = axs[vi][0]
        for ri, rep in enumerate(reps):
            ax.plot(PPM, specs[rep][xi, yi], color=cmap[ri], lw=1.1, label=f"rep{rep}")
        top = ax.get_ylim()[1]
        for pos, name in peaks:
            ax.axvline(pos, color='k', ls='--', lw=0.5, alpha=0.4)
            ax.text(pos, top*0.97, name, fontsize=8, ha='center', va='top', rotation=90)
        ax.set_xlim(PPM.max(), PPM.min())   # full spectral width (~+28..-28 ppm)
        ax.set_xlabel("ppm"); ax.set_ylabel("|spectrum|")
        ax.set_title(f"06c voxel [{xi},{yi}] across reps  (20 Hz apod)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    outp = os.path.join(ROOT, "_voxel_multi_06c.png")
    fig.savefig(outp, dpi=150); plt.close(fig)
    print("saved ->", outp)

if __name__ == "__main__":
    main()
