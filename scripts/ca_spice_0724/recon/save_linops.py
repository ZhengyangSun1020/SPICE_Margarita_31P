#!/usr/bin/env python3
"""Save the linear reconstruction operators of NUFFT (Hamming DCF, as adj_fid) and CA-SPICE (spice_final: hard boundary, rank 12) for MID00<ds>,
plus the measured k-t noise variance, so that noise variance per voxel / ROI / frequency can be computed exactly:
  voxel FIDs  NUFFT: X = PN @ K (all time points)      CA-SPICE: X = (PS @ (K @ V)) @ V^H
-> output_study_mid<ds>/spice_final/lin_ops.npz"""
import os, sys
DS = sys.argv[1] if len(sys.argv) > 1 else "283"
os.environ.update(VFILE="VD12.npy", PIXMASK="1", LABELS="mask", MUS="30", HOUT="pr_tmp")
S_ = "/mnt/c/Users/zsun/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-zsun-SPICE-Margarita/6aafd315-8b8b-4771-8d0e-7a49ffd29de0/scratchpad/hybrid_slam_spice.py"
src = open(S_).read(); sys.argv = ["x", DS]; exec(src[:src.index("RES = {}")])
s0 = np.load(f"{O}/system.npz"); s11 = np.load(f"{O}/system11.npz"); G_h = s0["G"].astype(np.complex128)
comp = np.zeros(N * N); comp[legv] = 3; comp[vGM] = 1; comp[vSOL] = 2; comp = comp.reshape(N, N); legm = legv.reshape(N, N); ph = phant.reshape(N, N)
_, (ii, jj) = ndi.distance_transform_edt(~legm, return_indices=True); cmp_ = np.where(legm, comp, comp[ii, jj]); p_raw = s0["prior"]
lo, hi = p_raw[legm].min(), p_raw[legm].max(); pp = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1) + 2.0 * np.select([cmp_ == 1, cmp_ == 2], [0.0, 1.0], 2.0); pp[ph] = 0.0
WW, _ = m.calc_Bmatrix_simple(pp.astype(np.float32), wmax=5e4, adj=8, label_mask=np.where(ph, 0, cmp_).astype(int).T)
dd = ndi.distance_transform_cdt(~legm, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2); lout[ph] = 1.0; lout = lout.ravel()
Gd = (Fc.conj().T * dcf) @ Fc; rel = np.real(np.diag(Gd)).mean() / np.real(np.diag(G_h)).mean()
Ai = np.linalg.inv(Gd + float(s11["lam"]) * rel * WW + np.diag(lout * rel)); PS = Ai @ (Fc.conj().T * dcf); PN = Fc.conj().T * dcf_h
kr_ = np.hypot(tj[:, 0], tj[:, 1]); Kt = K[:, kr_ >= np.median(kr_), -200:]; sig2 = float(np.mean(np.abs(Kt - Kt.mean(0))**2) * NR / (NR - 1))
# consistency: operators reproduce the stored voxel FIDs (rep 0)
XS = np.load(f"{O}/spice_final/fid_spice.npy")[0]; XS2 = (PS @ (K[0] @ V)) @ V.conj().T; XN2 = PN @ K[0]
log(f"MID00{DS}: CA-SPICE operator mismatch {100 * np.linalg.norm(XS2 - XS) / np.linalg.norm(XS):.2e}%, NUFFT {100 * np.linalg.norm(XN2 - fid[0]) / np.linalg.norm(fid[0]):.2e}%; sig2 {sig2:.3g}")
np.savez(f"{O}/spice_final/lin_ops.npz", PN=PN.astype(np.complex64), PS=PS.astype(np.complex64), V=V.astype(np.complex64), sig2=sig2, dw=dw, f0=f0, N=N,
         legv=legv, vGM=vGM, vSOL=vSOL, rest=np.array([R0, R1]), ex=np.array([E0, E1]), REP=REP, NR=NR); log("saved lin_ops")
