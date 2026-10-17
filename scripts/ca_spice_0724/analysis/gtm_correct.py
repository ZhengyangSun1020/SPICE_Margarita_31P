#!/usr/bin/env python3
"""Geometric-transfer-matrix (GTM) correction of SPICE ROI curves, MID00<ds> (argv[1]); same encoding / sensitivity / masks / labels as hybrid_slam_spice.py.
ROI estimate i (GM, soleus, other voxels; fixed per-voxel phase weights, so it is linear in k-space) responds to uniform tissue of compartment j with
T[i, j] = w_i . (A M_j). Measured ROI FIDs m = T s  ->  corrected compartment FIDs s = T^-1 m, rep by rep, before any peak picking / fitting.
Compared: NUFFT, compartment SPICE (soft, offset 2, per-bout lambda = as published), each raw and GTM-corrected; CC-SPICE soleus/GM voxel ROI (mu 30) for reference.
Metrics as hybrid_slam_spice.evaluate (matched-filter PCr / Pi heights, rest vs last 96 s of exercise, rep-to-rep noise from rest + late recovery).
-> output_study_mid<ds>/gtm/gtm_MID00<ds>.npz + summary line"""
import os, sys
DS = sys.argv[1] if len(sys.argv) > 1 else "283"
os.environ.update(VFILE="VD12.npy", PIXMASK="1", LABELS="mask", MUS="30", HOUT="gtm")
S_ = "/mnt/c/Users/zsun/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-zsun-SPICE-Margarita/6aafd315-8b8b-4771-8d0e-7a49ffd29de0/scratchpad/hybrid_slam_spice.py"
src = open(S_).read(); sys.argv = ["x", DS]; exec(src[:src.index("RES = {}")])
A = Fh[:, pix] * Ss[None]; AMj = A @ Mc                                                       # k x 3: uniform GM / soleus / other tissue -> k-space
oth = legv & ~vGM & ~vSOL; ROIS = (vGM, vSOL, oth)
# ---- compartment SPICE (soft boundary, offset 2, as published) ----
s0 = np.load(f"{O}/system.npz"); s11 = np.load(f"{O}/system11.npz"); G_h = s0["G"].astype(np.complex128)
comp = np.zeros(N * N); comp[legv] = 3; comp[vGM] = 1; comp[vSOL] = 2; comp = comp.reshape(N, N); legm = legv.reshape(N, N); ph = phant.reshape(N, N)
_, (ii, jj) = ndi.distance_transform_edt(~legm, return_indices=True); cmp_ = np.where(legm, comp, comp[ii, jj]); p_raw = s0["prior"]
lo, hi = p_raw[legm].min(), p_raw[legm].max(); pp = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1) + 2.0 * np.select([cmp_ == 1, cmp_ == 2], [0.0, 1.0], 2.0); pp[ph] = 0.0
WW, _ = m.calc_Bmatrix_simple(pp.astype(np.float32), wmax=5e4, adj=8, label_mask=np.where(ph, 0, 1).T)
dd = ndi.distance_transform_cdt(~legm, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2); lout[ph] = 1.0; lout = lout.ravel()
Gd = (Fc.conj().T * dcf) @ Fc; rel = np.real(np.diag(Gd)).mean() / np.real(np.diag(G_h)).mean()
Ai = np.linalg.inv(Gd + float(s11["lam"]) * rel * WW + np.diag(lout * rel))
OPS = {"NUFFT": Fc.conj().T * dcf_h, "SPICE": Ai @ (Fc.conj().T * dcf)}                       # voxel <- k-space
KV = K @ V
def roi_weights(P):
    resp = P @ AMj.sum(1)                                                                       # response of each voxel to uniform muscle tissue
    return [np.where(r_, np.exp(-1j * np.angle(resp)), 0) / max(1, r_.sum()) for r_ in ROIS]      # fixed phase-aligning voxel weights per ROI
OUTD = {}
def metrics(F):                                                                                # F: (NR, n) ROI or compartment FID
    Sx = cspec(F); pc = mf(Sx[:, None, :], bP)[:, 0]; pa = pi_mf(Sx[:, None, :])[:, 0]; r0 = pc[R0:R1].mean()
    return 100 * (1 - pc[EX].mean() / r0), 100 * (pa[EX].mean() - pa[R0:R1].mean()) / r0, 100 * nz(pc / r0), pc / r0, pa / r0
lines = []
for nm, P in OPS.items():
    Wr = roi_weights(P); Wk = np.stack([w @ P for w in Wr])                                     # 3 x k: ROI estimators in k-space
    T = Wk @ AMj                                                                                # 3 x 3 transfer matrix
    Mroi = np.einsum("ik,rkc->ric", Wk, KV) @ V.conj().T                                        # (NR, 3, n) measured ROI FIDs
    Scor = np.einsum("ij,rjc->ric", np.linalg.inv(T), Mroi)                                     # GTM-corrected compartment FIDs
    S2 = np.einsum("ij,rjc->ric", np.linalg.inv(T[:2, :2]), Mroi[:, :2])                       # 2x2 GTM: GM and soleus only ('other' left out)
    Tn = np.abs(T) / np.abs(np.diag(T))[:, None]
    for tag, X in (("raw", Mroi), ("GTM", Scor), ("GTM2", S2)):
        mg, ms = metrics(X[:, 0]), metrics(X[:, 1]); OUTD[f"{nm}_{tag}_gm_pcr"], OUTD[f"{nm}_{tag}_sol_pcr"] = mg[3], ms[3]; OUTD[f"{nm}_{tag}_gm_pi"], OUTD[f"{nm}_{tag}_sol_pi"] = mg[4], ms[4]
        lines.append(f"{nm:5s} {tag:4s} | GM dep {mg[0]:5.1f}  Pi {mg[1]:+5.1f}  noise {mg[2]:4.1f} | SOL dep {ms[0]:5.1f}  Pi {ms[1]:+5.1f}  noise {ms[2]:4.1f}")
    OUTD[f"{nm}_T"] = T
    lines.append(f"      T (|T_ij| / |T_ii|; rows = ROI GM, SOL, other; cols = tissue GM, SOL, other): {np.round(Tn, 2).tolist()}   cond {np.linalg.cond(T):.1f}, 2x2 cond {np.linalg.cond(T[:2, :2]):.1f}")
log(f"\n=== MID00{DS} GTM test ===\n" + "\n".join(lines))
os.makedirs(f"{O}/gtm", exist_ok=True); np.savez(f"{O}/gtm/gtm_MID00{DS}.npz", REP=REP, rest=np.array([R0, R1]), ex=np.array([E0, E1]), **OUTD); log("saved")
