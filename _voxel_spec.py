#!/usr/bin/env python3
"""Plot a single voxel's 06c spectrum with a ppm axis + metabolite markers."""
import os, importlib.util as ilu
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/zsun/SPICE_Margarita"
OUT  = os.path.join(ROOT, "output_dynamic_3T_0724_5rings_06c")
DAT  = ("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
        "meas_MID00277_FID221482_240FOV_10x10_5rings.dat")
VOX  = (6, 6)
REPS = [0, 30]     # rep30 requested; rep0 overlaid for reference

def load_ppm(n_spec):
    s = ilu.spec_from_file_location("m", os.path.join(ROOT, "scripts",
                                                      "05c_spice_recon_31p.py"))
    m = ilu.module_from_spec(s); s.loader.exec_module(m)
    info = m.load_concept_twix_3d(DAT, rep_sel=0)
    # match 06c EXACTLY (06c line 510-512): spectral_dwell=dwell_s*2, ppm=+freq/larmor
    return np.fft.fftshift(np.fft.fftfreq(n_spec, d=info["dwell_s"]*2)) / info["larmor_freq_mhz"]

def main():
    specs = {}
    for rep in REPS:
        p = os.path.join(OUT, "data", f"iter3d_recon_spec_rep{rep:02d}.npy")
        a = np.load(p)
        if a.ndim == 4: a = a[:, :, a.shape[2]//2, :]
        specs[rep] = a
    PPM = load_ppm(specs[REPS[0]].shape[-1])
    peaks = [(0.0,'PCr'), (4.85,'Pi'), (-2.53,'γ-ATP'), (-7.58,'α-ATP'), (-15.86,'β-ATP')]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    cols = {0:'tab:blue', 30:'tab:red'}
    for rep in REPS:
        z = np.abs(specs[rep][VOX[0], VOX[1]])
        ax.plot(PPM, z, color=cols[rep], lw=1.1, label=f"rep{rep}")
    for pos, name in peaks:
        ax.axvline(pos, color='k', ls='--', lw=0.5, alpha=0.4)
        ax.text(pos, ax.get_ylim()[1]*0.97, name, fontsize=8, ha='center',
                va='top', color='k', rotation=90)
    ax.set_xlim(12, -22); ax.set_xlabel("ppm"); ax.set_ylabel("|spectrum|")
    ax.set_title(f"06c iterative recon — voxel {VOX}  (20 Hz apod)")
    ax.legend()
    fig.tight_layout()
    outp = os.path.join(ROOT, f"_voxel_{VOX[0]}_{VOX[1]}_06c.png")
    fig.savefig(outp, dpi=150); plt.close(fig)
    print("saved ->", outp)

if __name__ == "__main__":
    main()
