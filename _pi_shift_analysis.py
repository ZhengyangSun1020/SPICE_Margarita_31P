#!/usr/bin/env python3
"""Compare Pi chemical-shift (pH marker) between two reps, for SPICE and adjoint.

For each ROI voxel it locates PCr (near 0 ppm) and the Pi peak (search window),
reports the Pi-PCr shift in ppm and the implied pH, and overlays rep0 vs rep25
spectra zoomed on the Pi region. Runs on whatever it finds: the pk2pi PHRANGE
SPICE runs and/or the ADJ adjoint-only runs.

Usage:
    python _pi_shift_analysis.py
"""
import os, importlib.util as ilu
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/zsun/SPICE_Margarita"
DAT  = ("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
        "meas_MID00277_FID221482_240FOV_10x10_5rings.dat")
ROI  = os.path.join(ROOT, "roi_mask_mid00276_manual.npy")

# (label, rep0_dir, rep25_dir, spec_filename)
SETS = [
    ("SPICE_rank10",
     "output_dynamic_3T_0724_5rings_05c_pk2pi_PHRANGE_rep00",
     "output_dynamic_3T_0724_5rings_05c_pk2pi_PHRANGE_rep25",
     "spice_spec_rank10.npy"),
    ("Adjoint",
     "output_dynamic_3T_0724_5rings_05c_ADJ_rep00",
     "output_dynamic_3T_0724_5rings_05c_ADJ_rep25",
     "adj_spec.npy"),
]

# Pi-PCr titration curve (Petroff/Moon-Richards): pH = 6.75 + log10((d-3.27)/(5.69-d))
def shift_to_pH(d):
    d = np.clip(d, 3.29, 5.67)
    return 6.75 + np.log10((d - 3.27) / (5.69 - d))

# ── spectral axis: replicate 05c exactly from the .dat ──────────────────
def load_ppm_axis(n_spec):
    s = ilu.spec_from_file_location("m", os.path.join(ROOT, "scripts",
                                                      "05c_spice_recon_31p.py"))
    m = ilu.module_from_spec(s); s.loader.exec_module(m)
    info = m.load_concept_twix_3d(DAT, rep_sel=0)
    dwell_s = info["dwell_s"] * 2
    larmor  = info["larmor_freq_mhz"]
    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    PPM  = -FREQ / larmor
    print(f"[axis] n_spec={n_spec} dwell_s={dwell_s:.3e} BW={1/dwell_s:.1f}Hz "
          f"larmor={larmor:.3f}MHz  ppm/pt={(PPM[1]-PPM[0]):.4f}")
    return PPM

def pi_position(spec, PPM, roi, pcr_win=(-1.0, 1.0), pi_win=(3.6, 6.0)):
    """Per-ROI-voxel Pi peak ppm (intensity-weighted centroid in the Pi window),
    plus PCr-referenced shift. Returns list of dicts."""
    mag = np.abs(spec)
    pcr_m = (PPM >= pcr_win[0]) & (PPM <= pcr_win[1])
    pi_m  = (PPM >= pi_win[0]) & (PPM <= pi_win[1])
    out = []
    for (xi, yi) in np.argwhere(roi):
        v = mag[xi, yi]
        pcr_ppm = PPM[pcr_m][np.argmax(v[pcr_m])]      # PCr peak (should be ~0)
        pi_sub  = v[pi_m]
        if pi_sub.max() < 1e-9:
            continue
        pk = PPM[pi_m][np.argmax(pi_sub)]              # Pi peak (argmax)
        # intensity-weighted centroid in a +/-0.4 ppm window around the peak
        cen_m = (PPM >= pk - 0.4) & (PPM <= pk + 0.4)
        cen = float(np.sum(PPM[cen_m] * v[cen_m]) / (np.sum(v[cen_m]) + 1e-12))
        out.append(dict(vox=(xi, yi), pcr=pcr_ppm, pi_peak=pk, pi_cen=cen,
                        pi_amp=float(pi_sub.max()), pcr_amp=float(v[pcr_m].max())))
    return out

def roi_sum_spectrum(spec, roi):
    """Coherent (complex) sum over ROI voxels -> one high-SNR spectrum."""
    idx = np.argwhere(roi)
    return np.sum([spec[x, y] for x, y in idx], axis=0)

def analyze_roi_sum(label, s0, s25, PPM, roi):
    """Robust readout: Pi peak in the ROI-summed spectrum, rep0 vs rep25,
    plus an SNR estimate (Pi peak / noise std in a signal-free band)."""
    z0 = np.abs(roi_sum_spectrum(s0, roi))
    z25 = np.abs(roi_sum_spectrum(s25, roi))
    pcr_m = (PPM >= -1) & (PPM <= 1)
    pi_m = (PPM >= 3.6) & (PPM <= 6.0)
    noise_m = (PPM >= 15) & (PPM <= 25)          # signal-free downfield band
    def pk(z):
        pcr = PPM[pcr_m][np.argmax(z[pcr_m])]
        pi = PPM[pi_m][np.argmax(z[pi_m])]
        cm = (PPM >= pi - 0.4) & (PPM <= pi + 0.4)
        cen = float(np.sum(PPM[cm]*z[cm])/(np.sum(z[cm])+1e-12))
        nf = z[noise_m].std() + 1e-12
        return pcr, pi, cen, float(z[pi_m].max()/nf), float(z[pcr_m].max()/nf)
    p0, p25 = pk(z0), pk(z25)
    print(f"\n----- {label}: ROI-SUMMED spectrum (robust) -----")
    print(f"  rep0 : PCr@{p0[0]:+.2f}  Pi_cen@{p0[2]:.3f}  Pi-PCr={p0[2]-p0[0]:.3f}ppm"
          f"  pH={shift_to_pH(p0[2]-p0[0]):.2f}  |  Pi SNR={p0[3]:.1f} PCr SNR={p0[4]:.1f}")
    print(f"  rep25: PCr@{p25[0]:+.2f}  Pi_cen@{p25[2]:.3f}  Pi-PCr={p25[2]-p25[0]:.3f}ppm"
          f"  pH={shift_to_pH(p25[2]-p25[0]):.2f}  |  Pi SNR={p25[3]:.1f} PCr SNR={p25[4]:.1f}")
    print(f"  --> Pi shift rep25-rep0 = {p25[2]-p0[2]:+.3f} ppm "
          f"(dpH={shift_to_pH(p25[2]-p0[0])-shift_to_pH(p0[2]-p0[0]):+.2f}); "
          f"trustworthy only if Pi SNR >~3 in BOTH reps")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    full = (PPM >= -20) & (PPM <= 12)
    for a, m, ttl in ((ax[0], full, "full"), (ax[1], (PPM >= 2.5)&(PPM <= 6.5), "Pi zoom")):
        g = max(z0[m].max(), z25[m].max())+1e-12
        a.plot(PPM[m], z0[m]/g, 'b-', lw=1.0, label="rep0")
        a.plot(PPM[m], z25[m]/g, 'r-', lw=1.0, label="rep25")
        a.axvline(4.85, color='k', ls='--', lw=0.5, alpha=0.5)
        a.axvline(p0[2], color='b', ls=':', lw=0.8); a.axvline(p25[2], color='r', ls=':', lw=0.8)
        a.set_xlim(m.nonzero()[0].size and PPM[m].max(), PPM[m].min()); a.invert_xaxis()
        a.set_title(f"{label} ROI-sum ({ttl})", fontsize=9); a.set_xlabel("ppm"); a.legend(fontsize=7)
    fig.tight_layout()
    outp = os.path.join(ROOT, f"_pi_roisum_{label}.png")
    fig.savefig(outp, dpi=140); plt.close(fig)
    print(f"  saved -> {outp}")

def main():
    roi = np.load(ROI).astype(bool)
    print(f"[roi] {roi.shape} nnz={int(roi.sum())} voxels={np.argwhere(roi).tolist()}\n")
    PPM = None
    for label, d0, d25, fn in SETS:
        p0 = os.path.join(ROOT, d0, "data", fn)
        p25 = os.path.join(ROOT, d25, "data", fn)
        if not (os.path.exists(p0) and os.path.exists(p25)):
            print(f"[skip] {label}: missing ({os.path.exists(p0)},{os.path.exists(p25)})")
            continue
        s0 = np.load(p0); s25 = np.load(p25)
        if PPM is None:
            PPM = load_ppm_axis(s0.shape[-1])
        analyze_roi_sum(label, s0, s25, PPM, roi)
        r0 = pi_position(s0, PPM, roi)
        r25 = pi_position(s25, PPM, roi)
        m0 = {d["vox"]: d for d in r0}; m25 = {d["vox"]: d for d in r25}
        print(f"\n===== {label} : Pi-PCr shift & pH, rep0 vs rep25 =====")
        print(f"{'vox':>7} | {'Pi0':>6} {'Pi25':>6} {'dPi':>7} | "
              f"{'pH0':>5} {'pH25':>5} {'dpH':>6} | {'shift0':>6} {'shift25':>6}")
        dpis = []
        for vox in sorted(set(m0) & set(m25)):
            a, b = m0[vox], m25[vox]
            sh0 = a["pi_cen"] - a["pcr"]; sh25 = b["pi_cen"] - b["pcr"]
            pH0, pH25 = shift_to_pH(sh0), shift_to_pH(sh25)
            dpis.append(b["pi_cen"] - a["pi_cen"])
            print(f"{str(vox):>7} | {a['pi_cen']:6.3f} {b['pi_cen']:6.3f} "
                  f"{b['pi_cen']-a['pi_cen']:+7.3f} | {pH0:5.2f} {pH25:5.2f} "
                  f"{pH25-pH0:+6.2f} | {sh0:6.3f} {sh25:6.3f}")
        if dpis:
            dpis = np.array(dpis)
            print(f"  mean dPi (rep25-rep0) = {dpis.mean():+.3f} ppm "
                  f"(std {dpis.std():.3f}, n={len(dpis)})")

        # overlay plot: rep0 vs rep25, zoomed on Pi region, per ROI voxel
        vox = sorted(set(m0) & set(m25))
        nc = min(4, len(vox)); nr = int(np.ceil(len(vox) / nc)) or 1
        fig, axs = plt.subplots(nr, nc, figsize=(3.2*nc, 2.4*nr), squeeze=False)
        pim = (PPM >= 2.5) & (PPM <= 6.5)
        for k, v in enumerate(vox):
            ax = axs[k // nc][k % nc]
            g = max(np.abs(s0[v[0], v[1], pim]).max(),
                    np.abs(s25[v[0], v[1], pim]).max()) + 1e-12
            ax.plot(PPM[pim], np.abs(s0[v[0], v[1], pim])/g, 'b-', lw=0.9, label="rep0")
            ax.plot(PPM[pim], np.abs(s25[v[0], v[1], pim])/g, 'r-', lw=0.9, label="rep25")
            ax.axvline(m0[v]["pi_cen"], color='b', ls=':', lw=0.7)
            ax.axvline(m25[v]["pi_cen"], color='r', ls=':', lw=0.7)
            ax.axvline(4.85, color='k', ls='--', lw=0.4, alpha=0.5)
            ax.set_xlim(6.5, 2.5); ax.set_title(f"vox {v}", fontsize=8)
            ax.tick_params(labelsize=6)
            if k == 0: ax.legend(fontsize=6)
        for k in range(len(vox), nr*nc):
            axs[k // nc][k % nc].axis('off')
        fig.suptitle(f"{label}: Pi region, rep0(blue) vs rep25(red) — dashed k=4.85ppm",
                     fontsize=10)
        fig.tight_layout()
        outp = os.path.join(ROOT, f"_pi_shift_{label}.png")
        fig.savefig(outp, dpi=140); plt.close(fig)
        print(f"  saved -> {outp}")

if __name__ == "__main__":
    main()
