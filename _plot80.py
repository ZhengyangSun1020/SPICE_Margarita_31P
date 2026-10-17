#!/usr/bin/env python3
"""High-SNR 80-ring adjoint: best voxel + FOV-sum spectrum + PCr/Pi maps.
Correct spectral axis: PPM = -freq/larmor (05c / 06c-plot convention)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/zsun/SPICE_Margarita"
OUT  = os.path.join(ROOT, "output_dynamic_3T_0724_80rings_ADJ")
LARMOR = 49.895            # MHz (3T 31P, from loader)
SPEC_DWELL = 3.6e-4        # s  (ADC dwell 180us *2 OS removal) -> BW 2778 Hz

a = np.load(os.path.join(OUT, "data", "adj_spec.npy"))
mag = np.abs(a); Nx, Ny, ns = mag.shape
PPM = -np.fft.fftshift(np.fft.fftfreq(ns, d=SPEC_DWELL)) / LARMOR
peaks = [(0.0,'PCr'), (4.85,'Pi'), (-2.53,'γ-ATP'), (-7.58,'α-ATP'), (-15.86,'β-ATP')]

pcr_m = (PPM >= -1) & (PPM <= 1)
pi_m  = (PPM >= 3.8) & (PPM <= 6.0)
noise_m = (PPM >= 12) & (PPM <= 25)
nf = mag[:, :, noise_m].std(axis=-1) + 1e-12
snr = mag[:, :, pcr_m].max(axis=-1) / nf
bx, by = np.unravel_index(np.argmax(snr), snr.shape)
print(f"best PCr-SNR voxel [{bx},{by}] SNR={snr[bx,by]:.1f}")

# coherent FOV sum, phase-aligned to each voxel's PCr
acc = np.zeros(ns, complex)
for xi in range(Nx):
    for yi in range(Ny):
        z = a[xi, yi]
        pk = np.where(pcr_m)[0][np.argmax(np.abs(z[pcr_m]))]
        acc += z * np.exp(-1j*np.angle(z[pk]))
zsum = np.abs(acc)

fig = plt.figure(figsize=(15, 9))
# best voxel
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(PPM, mag[bx, by], 'k-', lw=1.0)
ax1.set_title(f"80-ring adjoint — best voxel [{bx},{by}] (SNR {snr[bx,by]:.1f})")
# FOV sum
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(PPM, zsum, 'b-', lw=1.0)
ax2.set_title("80-ring adjoint — phase-aligned FOV sum (max SNR)")
for ax in (ax1, ax2):
    top = ax.get_ylim()[1]
    for pos, name in peaks:
        ax.axvline(pos, color='r', ls='--', lw=0.5, alpha=0.4)
        ax.text(pos, top*0.97, name, fontsize=8, ha='center', va='top', rotation=90, color='r')
    ax.set_xlim(PPM.max(), PPM.min()); ax.set_xlabel("ppm"); ax.set_ylabel("|spectrum|")
# maps
pcr_map = mag[:, :, pcr_m].max(axis=-1); pi_map = mag[:, :, pi_m].max(axis=-1)
for k, (M, ttl) in enumerate([(pcr_map, "PCr (0 ppm) map"), (pi_map, "Pi (+5 ppm) map")]):
    ax = fig.add_subplot(2, 2, 3+k)
    im = ax.imshow(M.T, origin="lower", cmap="viridis")
    ax.set_title(ttl); ax.set_xlabel("xi"); ax.set_ylabel("yi")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.add_patch(plt.Rectangle((bx-0.5, by-0.5), 1, 1, fill=False, ec='r', lw=2))
fig.suptitle("80-ring adjoint (MID00276), correct axis  |  red box = best voxel", fontsize=12)
fig.tight_layout()
outp = os.path.join(ROOT, "_plot80_adj.png")
fig.savefig(outp, dpi=140); plt.close(fig)
print("saved ->", outp)
