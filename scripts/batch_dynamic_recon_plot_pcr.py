"""Batch-reconstruct all dynamic reps, then plot the PCr time course."""
import subprocess, sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAT  = "/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_crt_10_10/meas_MID00644_FID214692_LV_crt_dynamic_10x10_5rings.dat"
OUT  = "./output_dynamic_3T"
SCRIPT = os.path.join(os.path.dirname(__file__), "06c_iterative_recon_31p_3d.py")
N_REPS = 52
LAM    = 0.5
MAXITER = 50

# ── 1. Reconstruct each rep ───────────────────────────────────────────────────
for rep in range(N_REPS):
    out_npy = os.path.join(OUT, "data", f"iter3d_recon_spec_rep{rep:02d}.npy")
    if os.path.exists(out_npy):
        print(f"[batch] Rep {rep:02d} already done — skipping")
        continue
    print(f"\n[batch] === Reconstructing Rep {rep:02d}/{N_REPS-1} ===")
    cmd = [
        sys.executable, SCRIPT,
        "--dat", DAT,
        "--out-dir", OUT,
        "--rep", str(rep),
        "--maxiter", str(MAXITER),
        "--lam", str(LAM),
    ]
    ret = subprocess.run(cmd, check=True)

# ── 2. Load all recon specs and extract PCr peak ──────────────────────────────
# Derive ppm axis from first saved spec + known acquisition params
spec0 = np.load(os.path.join(OUT, "data", "iter3d_recon_spec_rep00.npy"))
n_spec = spec0.shape[-1]
larmor_mhz  = 49.894571      # 31P at ~3T
dwell_s     = 180e-9 * 2     # spectral dwell = 2x ADC dwell (2778 Hz BW)
FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
PPM  = -FREQ / larmor_mhz

# PCr window: 0 ppm ± 1 ppm
pcr_mask = (PPM >= -1.0) & (PPM <= 1.0)

print(f"\n[batch] ppm axis: [{PPM.min():.1f}, {PPM.max():.1f}] ppm  "
      f"res={abs(PPM[1]-PPM[0])*1000:.1f} mppm/bin")
print(f"[batch] PCr window: {PPM[pcr_mask].min():.2f}..{PPM[pcr_mask].max():.2f} ppm  "
      f"({pcr_mask.sum()} bins)")

# Collect PCr peak per rep per voxel
Nx, Ny, Nz = spec0.shape[:3]
pcr_amp   = np.zeros((N_REPS, Nx, Ny, Nz))   # peak |S| in PCr window
noise_std = np.zeros((N_REPS, Nx, Ny, Nz))   # noise from spectral tails

# Noise region: far from all metabolites (|ppm| > 20 ppm)
noise_mask = np.abs(PPM) > 20.0

for rep in range(N_REPS):
    path = os.path.join(OUT, "data", f"iter3d_recon_spec_rep{rep:02d}.npy")
    spec = np.load(path)                           # (Nx, Ny, Nz, n_spec)
    mag  = np.abs(spec)
    pcr_amp[rep]   = mag[..., pcr_mask].max(axis=-1)
    noise_std[rep] = mag[..., noise_mask].std(axis=-1)
    if rep % 10 == 0:
        print(f"  loaded rep {rep:02d}")

# ── 3. Find best voxel (highest mean PCr/noise across reps) ───────────────────
mean_snr = np.where(noise_std.mean(0) > 0,
                    pcr_amp.mean(0) / noise_std.mean(0), 0.0)
best_flat = int(np.argmax(mean_snr))
bx, by, bz = np.unravel_index(best_flat, (Nx, Ny, Nz))
print(f"\n[batch] Best voxel (mean PCr SNR): [{bx},{by},{bz}]  "
      f"mean SNR={mean_snr[bx,by,bz]:.1f}")

# ── 4. Plot ────────────────────────────────────────────────────────────────────
fig_dir = os.path.join(OUT, "figures_dynamic")
os.makedirs(fig_dir, exist_ok=True)

time_ax = np.arange(N_REPS)   # rep index; multiply by TR if known

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# ── 4a. Best voxel PCr time course ──
ax = axes[0]
pcr_best = pcr_amp[:, bx, by, bz]
noise_best = noise_std[:, bx, by, bz]
ax.plot(time_ax, pcr_best, 'b-o', ms=3, lw=1.2, label="PCr peak |S|")
ax.fill_between(time_ax,
                pcr_best - noise_best, pcr_best + noise_best,
                alpha=0.25, color='blue', label="±noise σ")
ax.set_xlabel("Rep (dynamic frame)")
ax.set_ylabel("|PCr| amplitude")
ax.set_title(f"PCr time course — best voxel [{bx},{by},{bz}]")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 4b. PCr/noise (SNR) time course ──
ax = axes[1]
snr_course = np.where(noise_std[:, bx, by, bz] > 0,
                      pcr_amp[:, bx, by, bz] / noise_std[:, bx, by, bz], 0.0)
ax.plot(time_ax, snr_course, 'g-o', ms=3, lw=1.2)
ax.axhline(snr_course.mean(), color='g', ls='--', lw=0.8,
           label=f"mean SNR={snr_course.mean():.1f}")
ax.set_xlabel("Rep (dynamic frame)")
ax.set_ylabel("PCr SNR")
ax.set_title(f"PCr SNR time course — voxel [{bx},{by},{bz}]")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 4c. PCr spatial map (mean over reps) ──
ax = axes[2]
mean_pcr_map = pcr_amp.mean(0)[:, :, bz]     # (Nx, Ny)
im = ax.imshow(mean_pcr_map.T, origin="upper", cmap="hot", aspect="equal")
ax.plot(bx, by, "c+", ms=10, mew=1.5)
plt.colorbar(im, ax=ax, label="|PCr| mean over reps")
ax.set_title(f"Mean PCr amplitude map — z={bz}")
ax.set_xlabel("x voxel")
ax.set_ylabel("y voxel")

fig.tight_layout()
out_png = os.path.join(fig_dir, "pcr_dynamic_timecourse.png")
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"[batch] Saved: {out_png}")

# ── 4d. PCr map per rep (all 52 in one figure) ──────────────────────────────
ncols = 13
nrows = int(np.ceil(N_REPS / ncols))
fig2, axes2 = plt.subplots(nrows, ncols,
                            figsize=(ncols * 1.5, nrows * 1.5))
axes2 = np.array(axes2).reshape(nrows, ncols)
vmin, vmax = 0, float(pcr_amp[:, :, :, bz].max())
for rep in range(N_REPS):
    r, c = divmod(rep, ncols)
    ax2 = axes2[r, c]
    ax2.imshow(pcr_amp[rep, :, :, bz].T, origin="upper",
               cmap="hot", vmin=vmin, vmax=vmax, aspect="equal")
    ax2.set_title(f"r{rep}", fontsize=6)
    ax2.axis("off")
for rep in range(N_REPS, nrows * ncols):
    r, c = divmod(rep, ncols)
    axes2[r, c].axis("off")
fig2.suptitle("PCr peak amplitude maps — all dynamic frames (z=%d)" % bz, fontsize=11)
fig2.tight_layout()
out_png2 = os.path.join(fig_dir, "pcr_all_reps_map.png")
fig2.savefig(out_png2, dpi=150)
plt.close(fig2)
print(f"[batch] Saved: {out_png2}")
print("\n[batch] All done.")
