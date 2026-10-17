#!/usr/bin/env python3
"""Pseudo-multiple-replica SNR (Robson et al., MRM 2008) of NUFFT vs CA-SPICE, MID00<ds> (argv[1]); same encoding / masks / labels / guidance as spice_final.
Noise sigma^2 per k-t sample measured from the data (outer half of k-space, last 200 time points, rep-to-rep). For one rest rep (the last one):
NREP replicas = measured k-space + complex white noise of that variance; each replica reconstructed with NUFFT (Hamming DCF, as the study) and
CA-SPICE (hard compartment boundary, spectral subspace rank 12). Per voxel: PCr peak height of the magnitude spectrum (8 Hz LB, |ppm| < 0.4);
SNR = mean / SD over replicas. Note: noise is added on top of the real data (which already has noise once); the SD measures the reconstruction's
noise propagation, the mean the signal. -> output_study_mid<ds>/spice_final/pseudo_replica.npz + summary"""
import os, sys
DS = sys.argv[1] if len(sys.argv) > 1 else "283"; NREP = int(os.environ.get("NREP", 60))
os.environ.update(VFILE="VD12.npy", PIXMASK="1", LABELS="mask", MUS="30", HOUT="pr_tmp")
S_ = "/mnt/c/Users/zsun/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-zsun-SPICE-Margarita/6aafd315-8b8b-4771-8d0e-7a49ffd29de0/scratchpad/hybrid_slam_spice.py"
src = open(S_).read(); sys.argv = ["x", DS]; exec(src[:src.index("RES = {}")])
# CA-SPICE operator (as spice_final: hard compartment boundary, offset 2, published lambda)
s0 = np.load(f"{O}/system.npz"); s11 = np.load(f"{O}/system11.npz"); G_h = s0["G"].astype(np.complex128)
comp = np.zeros(N * N); comp[legv] = 3; comp[vGM] = 1; comp[vSOL] = 2; comp = comp.reshape(N, N); legm = legv.reshape(N, N); ph = phant.reshape(N, N)
_, (ii, jj) = ndi.distance_transform_edt(~legm, return_indices=True); cmp_ = np.where(legm, comp, comp[ii, jj]); p_raw = s0["prior"]
lo, hi = p_raw[legm].min(), p_raw[legm].max(); pp = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1) + 2.0 * np.select([cmp_ == 1, cmp_ == 2], [0.0, 1.0], 2.0); pp[ph] = 0.0
WW, _ = m.calc_Bmatrix_simple(pp.astype(np.float32), wmax=5e4, adj=8, label_mask=np.where(ph, 0, cmp_).astype(int).T)
dd = ndi.distance_transform_cdt(~legm, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2); lout[ph] = 1.0; lout = lout.ravel()
Gd = (Fc.conj().T * dcf) @ Fc; rel = np.real(np.diag(Gd)).mean() / np.real(np.diag(G_h)).mean()
Ai = np.linalg.inv(Gd + float(s11["lam"]) * rel * WW + np.diag(lout * rel)); PS = Ai @ (Fc.conj().T * dcf); PN = Fc.conj().T * dcf_h; PN0 = Fc.conj().T * dcf                                    # NUFFT with / without the Hamming filter
# measured noise variance per k-t sample (outer k, signal-free tail of the FID, across reps)
kr_ = np.hypot(tj[:, 0], tj[:, 1]); Kt = K[:, kr_ >= np.median(kr_), -200:]; sig2 = float(np.mean(np.abs(Kt - Kt.mean(0))**2) * NR / (NR - 1))
r0 = R1 - 1; K0 = K[r0]; rng = np.random.default_rng(5)
APOs = np.exp(-np.pi * 8 * t); spec = lambda F_: np.abs(np.fft.fftshift(np.fft.fft(np.conj(F_ * APOs), axis=-1), axes=-1))
PPMr = -np.fft.fftshift(np.fft.fftfreq(n, d=dw)) / f0; S0 = spec(fid[R0:R1].mean(0)); k0 = int(np.argmax(S0[vGM].sum(0) * (np.abs(PPMr) < 1.5))); QQ = PPMr - PPMr[k0]; bPk = np.abs(QQ) < 0.4
HN, HS, H0 = [], [], []
for it in range(NREP):
    Kn = K0 + np.sqrt(sig2 / 2) * (rng.standard_normal(K0.shape) + 1j * rng.standard_normal(K0.shape))
    XN = PN @ Kn; XS = (PS @ (Kn @ V)) @ V.conj().T                                              # NUFFT keeps the full spectral dimension; CA-SPICE in the rank-12 subspace
    HN.append(spec(XN)[:, bPk].max(1)); HS.append(spec(XS)[:, bPk].max(1)); H0.append(spec(PN0 @ Kn)[:, bPk].max(1))
HN, HS, H0 = np.array(HN), np.array(HS), np.array(H0); snrN, snrS, snr0 = HN.mean(0) / HN.std(0), HS.mean(0) / HS.std(0), H0.mean(0) / H0.std(0)
disp = lambda q: (N - 1 - q // N) * N + q % N
log(f"\n=== MID00{DS} pseudo-replica SNR ({NREP} replicas, rest rep {r0 + 1}, noise var {sig2:.3g}) ===")
for nm, msk in (("GM", vGM), ("soleus", vSOL)):
    log(f"{nm:7s} voxels: median SNR NUFFT no-Hamming {np.median(snr0[msk]):5.1f} | NUFFT {np.median(snrN[msk]):5.1f}  CA-SPICE {np.median(snrS[msk]):5.1f}  (ratio x{np.median(snrS[msk] / snrN[msk]):.2f}); "
        + ", ".join(f"{disp(q)}: {snrN[q]:.0f}->{snrS[q]:.0f}" for q in sorted(np.flatnonzero(msk), key=disp)))
np.savez(f"{O}/spice_final/pseudo_replica.npz", snr_nufft_noham=snr0, snr_nufft=snrN, snr_caspice=snrS, nrep=NREP, rep=r0, sig2=sig2)
