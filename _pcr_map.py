#!/usr/bin/env python3
"""PCr-intensity map (0 ppm) from 06c rep30, with array indices labeled,
plus SNR ranking of voxels so we can pick a real signal voxel."""
import os, importlib.util as ilu
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/zsun/SPICE_Margarita"
OUT  = os.path.join(ROOT, "output_dynamic_3T_0724_5rings_06c")
DAT  = ("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
        "meas_MID00277_FID221482_240FOV_10x10_5rings.dat")
REP  = 30

def load_ppm(n_spec):
    s = ilu.spec_from_file_location("m", os.path.join(ROOT, "scripts",
                                                      "05c_spice_recon_31p.py"))
    m = ilu.module_from_spec(s); s.loader.exec_module(m)
    info = m.load_concept_twix_3d(DAT, rep_sel=0)
    return np.fft.fftshift(np.fft.fftfreq(n_spec, d=info["dwell_s"]*2)) / info["larmor_freq_mhz"]

def main():
    a = np.load(os.path.join(OUT, "data", f"iter3d_recon_spec_rep{REP:02d}.npy"))
    a = a[:, :, a.shape[2]//2, :]
    mag = np.abs(a); Nx, Ny, ns = mag.shape
    PPM = load_ppm(ns)
    pcr = (PPM >= -0.6) & (PPM <= 0.6)
    pi  = (PPM >= 4.3) & (PPM <= 5.4)
    noise = (PPM >= 10) & (PPM <= 20)
    pcr_map = mag[:, :, pcr].max(axis=-1)
    pi_map  = mag[:, :, pi].max(axis=-1)
    nf = mag[:, :, noise].std(axis=-1) + 1e-12
    snr_map = pcr_map / nf

    # rank voxels by PCr SNR
    order = np.argsort(snr_map.ravel())[::-1]
    print("Top voxels by PCr SNR (array index [xi,yi]):")
    for k in order[:8]:
        xi, yi = divmod(k, Ny)
        print(f"  [{xi},{yi}]  PCr_SNR={snr_map[xi,yi]:.1f}  "
              f"PCr={pcr_map[xi,yi]:.2e}  Pi={pi_map[xi,yi]:.2e}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, M, ttl in ((axs[0], pcr_map, "PCr (0 ppm) max"),
                       (axs[1], pi_map, "Pi (4.3-5.4 ppm) max"),
                       (axs[2], snr_map, "PCr SNR")):
        im = ax.imshow(M.T, origin="lower", cmap="viridis")  # .T so x=cols
        ax.set_title(f"{ttl} — rep{REP}"); ax.set_xlabel("xi"); ax.set_ylabel("yi")
        plt.colorbar(im, ax=ax, fraction=0.046)
        for xi in range(Nx):
            for yi in range(Ny):
                ax.text(xi, yi, f"{xi},{yi}", ha='center', va='center',
                        fontsize=5, color='w')
        ax.add_patch(plt.Rectangle((6-0.5, 6-0.5), 1, 1, fill=False,
                                   ec='red', lw=2))   # mark [6,6]
    fig.suptitle(f"06c rep{REP} maps (red box = array index [6,6])", fontsize=11)
    fig.tight_layout()
    outp = os.path.join(ROOT, "_pcr_map_06c_rep30.png")
    fig.savefig(outp, dpi=140); plt.close(fig)
    print("saved ->", outp)

if __name__ == "__main__":
    main()
