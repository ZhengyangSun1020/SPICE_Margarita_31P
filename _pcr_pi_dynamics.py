#!/usr/bin/env python3
"""PCr / Pi dynamics across all reps for muscle voxels (MID00644 direct-FT adjoint).

Muscle voxels = PCr SNR above threshold in the rep-averaged spectrum.
Correct spectral axis: PPM = -freq/larmor.
Outputs: PCr map + muscle mask, PCr/Pi/ratio vs rep curves, and a summary table.
"""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/home/zsun/SPICE_Margarita"
OUT  = os.path.join(ROOT, "output_dynamic_3T_MID00644_10x10_5rings_dft_adj")
DATA = os.path.join(OUT, "data")
LARMOR = 49.895; SPEC_DWELL = 3.6e-4
PCR_WIN = 0.8          # +/- ppm around 0
PI_LO, PI_HI = 3.8, 5.6   # Pi search window (allow pH shift)
NOISE_LO, NOISE_HI = 10, 25
MUSCLE_FRAC = 0.40     # muscle = voxels with rep-avg PCr >= frac * max(PCr) (core high-PCr region)

def load_stack():
    files = glob.glob(os.path.join(DATA, "iter3d_recon_spec_rep*.npy"))
    reps = sorted(int(re.search(r"rep(\d+)\.npy", f).group(1)) for f in files)
    arrs = []
    for r in reps:
        a = np.load(os.path.join(DATA, f"iter3d_recon_spec_rep{r:02d}.npy"))
        arrs.append(a[:, :, a.shape[2]//2, :])
    return np.array(reps), np.abs(np.array(arrs))    # (n_rep, N, N, ns)

def main():
    reps, mag = load_stack()
    n_rep, N, _, ns = mag.shape
    PPM = -np.fft.fftshift(np.fft.fftfreq(ns, d=SPEC_DWELL)) / LARMOR
    pcr_m = np.abs(PPM) <= PCR_WIN
    pi_m  = (PPM >= PI_LO) & (PPM <= PI_HI)
    noise_m = (PPM >= NOISE_LO) & (PPM <= NOISE_HI)
    print(f"loaded {n_rep} reps, grid {N}x{N}, ns={ns}")

    # muscle mask from rep-averaged spectrum
    meanmag = mag.mean(axis=0)                        # (N,N,ns)
    nf = meanmag[:, :, noise_m].std(axis=-1) + 1e-12
    pcr_map = meanmag[:, :, pcr_m].max(axis=-1)
    pcr_snr = pcr_map / nf
    muscle = pcr_map >= MUSCLE_FRAC * pcr_map.max()      # core high-PCr voxels only
    mvox = np.argwhere(muscle)
    print(f"muscle voxels (PCr>={MUSCLE_FRAC}*max): {len(mvox)} -> {mvox.tolist()}")
    print("  PCr SNR of muscle voxels:", [round(float(pcr_snr[x,y]),1) for x,y in mvox])

    # per-rep, per-voxel PCr & Pi amplitude and Pi ppm
    def amp(spec, m): return spec[:, :, :, m].max(axis=-1)      # (n_rep,N,N)
    pcr_amp = amp(mag, pcr_m)
    pi_amp  = amp(mag, pi_m)
    # Pi ppm (position of Pi max) per rep/voxel
    pi_idx  = np.argmax(mag[:, :, :, pi_m], axis=-1)
    pi_ppm  = PPM[pi_m][pi_idx]                                  # (n_rep,N,N)

    # aggregate over muscle voxels
    mm = muscle[None].repeat(n_rep, 0)
    def mstat(x):
        v = np.where(mm, x, np.nan)
        return np.nanmean(v.reshape(n_rep, -1), axis=1), np.nanstd(v.reshape(n_rep, -1), axis=1)
    pcr_mean, pcr_std = mstat(pcr_amp)
    pi_mean,  pi_std  = mstat(pi_amp)
    ratio = pi_amp / (pcr_amp + 1e-12)
    ratio_mean, ratio_std = mstat(ratio)

    # ---- figure ----
    fig = plt.figure(figsize=(15, 10))
    # PCr map + muscle mask
    ax = fig.add_subplot(2, 3, 1)
    im = ax.imshow(pcr_map.T, origin="lower", cmap="viridis")
    for (xi, yi) in mvox:
        ax.add_patch(plt.Rectangle((xi-0.5, yi-0.5), 1, 1, fill=False, ec='r', lw=1.5))
    ax.set_title(f"PCr map (rep-avg) + muscle ({len(mvox)} vox)"); ax.set_xlabel("xi"); ax.set_ylabel("yi")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # PCr vs rep
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(reps, pcr_mean, 'b-o', ms=3); ax.fill_between(reps, pcr_mean-pcr_std, pcr_mean+pcr_std, alpha=0.2)
    ax.set_title("PCr amplitude vs rep (muscle mean±std)"); ax.set_xlabel("rep"); ax.set_ylabel("PCr")
    # Pi vs rep
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(reps, pi_mean, 'r-o', ms=3); ax.fill_between(reps, pi_mean-pi_std, pi_mean+pi_std, alpha=0.2, color='r')
    ax.set_title("Pi amplitude vs rep (muscle mean±std)"); ax.set_xlabel("rep"); ax.set_ylabel("Pi")
    # ratio
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(reps, ratio_mean, 'g-o', ms=3); ax.fill_between(reps, ratio_mean-ratio_std, ratio_mean+ratio_std, alpha=0.2, color='g')
    ax.set_title("Pi/PCr ratio vs rep (muscle mean)"); ax.set_xlabel("rep"); ax.set_ylabel("Pi/PCr")
    # per-voxel PCr (faint)
    ax = fig.add_subplot(2, 3, 5)
    for (xi, yi) in mvox:
        ax.plot(reps, pcr_amp[:, xi, yi], '-', lw=0.6, alpha=0.5)
    ax.plot(reps, pcr_mean, 'k-', lw=2, label="mean")
    ax.set_title("PCr per muscle voxel"); ax.set_xlabel("rep"); ax.set_ylabel("PCr"); ax.legend()
    # muscle-summed spectrum, a few reps
    ax = fig.add_subplot(2, 3, 6)
    show = [reps[0], reps[len(reps)//2], reps[-1]]
    for r in show:
        ri = list(reps).index(r)
        z = mag[ri][muscle].sum(axis=0)
        ax.plot(PPM, z/z.max(), lw=0.9, label=f"rep{r}")
    ax.axvline(0, color='k', ls=':', lw=0.5); ax.axvline(4.85, color='k', ls=':', lw=0.5)
    ax.set_xlim(10, -20); ax.set_title("muscle-sum spectrum (norm)"); ax.set_xlabel("ppm"); ax.legend(fontsize=7)
    fig.suptitle(f"MID00644 dynamic — PCr/Pi dynamics over {n_rep} reps (direct-FT adjoint)", fontsize=13)
    fig.tight_layout()
    outp = os.path.join(ROOT, "_pcr_pi_dynamics.png")
    fig.savefig(outp, dpi=140); plt.close(fig)
    print("saved ->", outp)

    # summary table (downsampled)
    print(f"\n{'rep':>4} {'PCr':>10} {'Pi':>10} {'Pi/PCr':>8}")
    for i in range(0, n_rep, max(1, n_rep//12)):
        print(f"{reps[i]:>4} {pcr_mean[i]:10.2e} {pi_mean[i]:10.2e} {ratio_mean[i]:8.3f}")
    # save arrays for reuse
    np.savez(os.path.join(ROOT, "_pcr_pi_dynamics.npz"),
             reps=reps, pcr_mean=pcr_mean, pi_mean=pi_mean, ratio_mean=ratio_mean,
             muscle=muscle, pcr_amp=pcr_amp, pi_amp=pi_amp, pi_ppm=pi_ppm)

if __name__ == "__main__":
    main()
