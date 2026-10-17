#!/usr/bin/env python3
"""Why is CA-SPICE resting soleus Pi high? MID00<ds> (argv[1]). Reconstruct pure noise (measured k-t noise level, no signal) with the exact NUFFT and
CA-SPICE (spice_final) pipelines; ROI spectrum of the core GM / soleus voxels with the real data's per-voxel phases (as Fig 4). Compare per ROI:
  real rest Pi band integral (3.8-5.4 ppm) / real rest PCr integral   vs   noise-only Pi band integral / real rest PCr integral (mean over replicas).
Also: noise-only magnitude spectrum shape (where the subspace puts the noise) and the real rest ROI spectra. -> scratch figure + summary."""
import os, sys, json
DS = sys.argv[1] if len(sys.argv) > 1 else "283"; NREP = int(os.environ.get("NREP", 40))
os.environ.update(VFILE="VD12.npy", PIXMASK="1", LABELS="mask", MUS="30", HOUT="pr_tmp")
S_ = "/mnt/c/Users/zsun/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-zsun-SPICE-Margarita/6aafd315-8b8b-4771-8d0e-7a49ffd29de0/scratchpad/hybrid_slam_spice.py"
src = open(S_).read(); sys.argv = ["x", DS]; exec(src[:src.index("RES = {}")])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
s0 = np.load(f"{O}/system.npz"); s11 = np.load(f"{O}/system11.npz"); G_h = s0["G"].astype(np.complex128)
comp = np.zeros(N * N); comp[legv] = 3; comp[vGM] = 1; comp[vSOL] = 2; comp = comp.reshape(N, N); legm = legv.reshape(N, N); ph = phant.reshape(N, N)
_, (ii, jj) = ndi.distance_transform_edt(~legm, return_indices=True); cmp_ = np.where(legm, comp, comp[ii, jj]); p_raw = s0["prior"]
lo, hi = p_raw[legm].min(), p_raw[legm].max(); pp = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1) + 2.0 * np.select([cmp_ == 1, cmp_ == 2], [0.0, 1.0], 2.0); pp[ph] = 0.0
WW, _ = m.calc_Bmatrix_simple(pp.astype(np.float32), wmax=5e4, adj=8, label_mask=np.where(ph, 0, cmp_).astype(int).T)
dd = ndi.distance_transform_cdt(~legm, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2); lout[ph] = 1.0; lout = lout.ravel()
Gd = (Fc.conj().T * dcf) @ Fc; rel = np.real(np.diag(Gd)).mean() / np.real(np.diag(G_h)).mean()
Ai = np.linalg.inv(Gd + float(s11["lam"]) * rel * WW + np.diag(lout * rel)); PS = Ai @ (Fc.conj().T * dcf); PN = Fc.conj().T * dcf_h
kr_ = np.hypot(tj[:, 0], tj[:, 1]); Kt = K[:, kr_ >= np.median(kr_), -200:]; sig2 = float(np.mean(np.abs(Kt - Kt.mean(0))**2) * NR / (NR - 1))
XS_real = np.load(f"{O}/spice_final/fid_spice.npy"); XN_real = fid
# core voxels (same QC rule as Fig 4: CA-SPICE PCr rep-to-rep noise in rest + late recovery < 25 %)
h = np.load(f"{O}/spice_final/hybrid_MID00{DS}.npz", allow_pickle=True); names = list(h["names"]); li = list(h["leg_idx"]); iS = names.index("SPICE (compartment, CSI grid)")
cc = h[f"{iS}_vox_pcr"]; quiet = np.r_[R0:R1, NR - int(round(144 / REP)):NR]
core = {mus: [q for q in np.flatnonzero(msk) if np.std(np.diff(cc[quiet, li.index(q)])) / np.sqrt(2) < 0.25] for mus, msk in (("GM", vGM), ("soleus", vSOL))}
APOg = np.exp(-np.pi**2 * 10.0**2 * t**2 / (4 * np.log(2))); sp = lambda X_: np.fft.fftshift(np.fft.fft(np.conj(X_ * APOg), axis=-1), axes=-1)
PPMr = -np.fft.fftshift(np.fft.fftfreq(n, d=dw)) / f0
OUT = {}; rng = np.random.default_rng(9)
for meth, Xr, Pop, sub in (("NUFFT", XN_real, PN, False), ("CA-SPICE", XS_real, PS, True)):
    for mus, qs in core.items():
        Sr = sp(Xr[R0:R1].mean(0)[qs]); k0 = np.flatnonzero(np.abs(PPMr) < 1.5)[np.argmax(np.abs(Sr).sum(0)[np.abs(PPMr) < 1.5])]; Q_ = PPMr - PPMr[k0]
        bP, bPi = np.abs(Q_) <= 1, (Q_ >= 3.8) & (Q_ <= 5.4)
        c = Sr[:, k0]; ph_ = c.conj() / np.abs(c).sum()                                             # real-data phase alignment
        Rr = np.abs((Sr * ph_[:, None]).sum(0)); pci = Rr[bP].sum()
        Nm = []
        for it in range(NREP):
            Kn = np.sqrt(sig2 / 2) * (rng.standard_normal(K[0].shape) + 1j * rng.standard_normal(K[0].shape))
            Xn = ((Pop @ (Kn @ V)) @ V.conj().T) if sub else (Pop @ Kn)
            Nm.append(np.abs((sp(Xn[qs]) * ph_[:, None]).sum(0)))
        Nm = np.array(Nm); Nmean = Nm.mean(0) / np.sqrt(R1 - R0)                                   # noise of the rest-rep average (R1-R0 reps)
        OUT[(meth, mus)] = dict(Q=Q_, real=Rr / Rr[bP].max(), noise=Nmean / Rr[bP].max(), pi_real=Rr[bPi].sum() / pci, pi_noise=Nmean[bPi].sum() / pci,
                                pcr_noise=Nmean[bP].sum() / pci)
log(f"\n=== MID00{DS}: rest Pi band integral / rest PCr integral (core ROIs: GM {len(core['GM'])}, soleus {len(core['soleus'])} voxels) ===")
for k, d in OUT.items():
    log(f"  {k[0]:8s} {k[1]:6s}: real {d['pi_real']:.2f} | noise alone in the Pi band {d['pi_noise']:.2f} ({100 * d['pi_noise'] / d['pi_real']:.0f}% of real) | noise in PCr band {d['pcr_noise']:.2f}")
fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
for i, mus in enumerate(("GM", "soleus")):
    for j, meth in enumerate(("NUFFT", "CA-SPICE")):
        a = axs[i, j]; d = OUT[(meth, mus)]; a.plot(d["Q"], d["real"], "k", lw=1.5, label="real rest ROI spectrum"); a.plot(d["Q"], d["noise"], "r", lw=1.3, label="noise alone (same recon)")
        a.axvspan(3.8, 5.4, color="#fdd49e", alpha=0.4); a.set_xlim(10, -20); a.set_ylim(0, 1.1); a.set_title(f"{meth}, {mus} core ROI: Pi/PCr real {d['pi_real']:.2f}, noise {d['pi_noise']:.2f}", fontsize=11)
        if i == 1: a.set_xlabel("ppm (PCr = 0)")
axs[0, 0].legend(fontsize=9); fig.tight_layout(); fig.savefig(f"/mnt/e/claude_tmp/study/paper_abstract/check_soleus_pi_MID00{DS}.png", dpi=110); log("saved")
