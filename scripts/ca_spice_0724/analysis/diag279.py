#!/usr/bin/env python3
"""MID00279 diagnostics (argv[1], default 279; compare with 281): labels vs NUFFT signal, regularisation strength, Hamming in the data term.
Prints display-grid maps (row 0 = top): labels, NUFFT rest PCr (% of max), muscle area fraction per CSI voxel; then GM / soleus ROI depletion
and median voxel tSNR for CA-SPICE variants (lambda x 0 / 1 / 1000; with Hamming-weighted DCF)."""
import os, sys
DS = sys.argv[1] if len(sys.argv) > 1 else "279"
os.environ.update(VFILE="VD12.npy", PIXMASK="1", LABELS="mask", MUS="30", HOUT="diag")
S_ = "/mnt/c/Users/zsun/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-zsun-SPICE-Margarita/6aafd315-8b8b-4771-8d0e-7a49ffd29de0/scratchpad/hybrid_slam_spice.py"
src = open(S_).read(); sys.argv = ["x", DS]; exec(src[:src.index("RES = {}")])
d2r = lambda M: M.reshape(N, N)[::-1]                                                             # recon frame -> display (row 0 = top)
lab = np.where(vGM, "G", np.where(vSOL, "S", np.where(legv, "o", np.where(phant, "P", "."))))
Sn = np.abs(cspec(fid[R0:R1].mean(0))); pcr = Sn[:, bP].max(1); pcr = 100 * pcr / pcr[legv].max()
fm = np.zeros(N * N); np.add.at(fm, blk[pix], fMUS[pix]); fm = 100 * fm / (F_UP**2)
log(f"\n=== MID00{DS} ({N}x{N}, {FOV / N:.0f} mm) — display grid, row 0 = top ===")
for name, M, fmt in (("labels (G GM, S soleus, o other leg, P phantom)", lab, "{:>4s}"), ("NUFFT rest PCr (% of leg max)", np.round(pcr).astype(int), "{:4d}"), ("muscle area in voxel (%)", np.round(fm).astype(int), "{:4d}")):
    log(name); A2 = d2r(M)
    for r in range(N): log("   " + "".join(fmt.format(x) for x in A2[r]))
log(f"PCr (% max) median: GM voxels {np.median(pcr[vGM]):.0f}, soleus {np.median(pcr[vSOL]):.0f}, other leg {np.median(pcr[legv & ~vGM & ~vSOL]):.0f}")
# CA-SPICE variants
s0 = np.load(f"{O}/system.npz"); s11 = np.load(f"{O}/system11.npz"); G_h = s0["G"].astype(np.complex128)
comp = np.zeros(N * N); comp[legv] = 3; comp[vGM] = 1; comp[vSOL] = 2; comp = comp.reshape(N, N); legm = legv.reshape(N, N); ph = phant.reshape(N, N)
_, (ii, jj) = ndi.distance_transform_edt(~legm, return_indices=True); cmp_ = np.where(legm, comp, comp[ii, jj]); p_raw = s0["prior"]
lo, hi = p_raw[legm].min(), p_raw[legm].max(); pn = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1); pp = pn + 2.0 * np.select([cmp_ == 1, cmp_ == 2], [0.0, 1.0], 2.0); pp[ph] = 0.0
log("80-ring prior (0-100 within leg), display grid:"); A2 = np.round(100 * pn)[::-1].astype(int)
for r in range(N): log("   " + "".join(f"{x:4d}" for x in A2[r]))
c80 = np.corrcoef(pn.ravel()[legv], pcr[legv])[0, 1]; log(f"corr(80-ring prior, NUFFT rest PCr) over leg voxels: {c80:.2f}")
WW, _ = m.calc_Bmatrix_simple(pp.astype(np.float32), wmax=5e4, adj=8, label_mask=np.where(ph, 0, cmp_).astype(int).T)
dd = ndi.distance_transform_cdt(~legm, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2); lout[ph] = 1.0; lout = lout.ravel()
KV = K @ V
for dname, dcf_use in (("plain DCF", dcf), ("Hamming DCF", dcf_h)):
    Gd = (Fc.conj().T * dcf_use) @ Fc; rel = np.real(np.diag(Gd)).mean() / np.real(np.diag(G_h)).mean()
    lam = float(s11["lam"]); wdiag = np.real(np.diag(WW))[vGM]
    log(f"[{dname}] lambda {lam:.3g}, rel {rel:.3g}: GM-voxel graph degree (median) {np.median(wdiag):.3g}; lambda*rel*degree / Gd diag = {np.median(lam * rel * wdiag) / np.real(np.diag(Gd)).mean():.3g}")
    for kl in (0.0, 1.0, 1000.0):
        Ai = np.linalg.inv(Gd + kl * lam * rel * WW + np.diag(lout * rel))
        Fx = np.einsum("pq,rqc->rpc", Ai, np.einsum("vk,rkc->rvc", Fc.conj().T * dcf_use, KV)) @ V.conj().T; r = evaluate(Fx)
        log(f"   lambda x{kl:g}: GM dep {r['g']['pcr']:5.1f}  SOL dep {r['s']['pcr']:5.1f} | median voxel tSNR GM {np.nanmedian(r['tsnr'][vGM]):5.1f}  SOL {np.nanmedian(r['tsnr'][vSOL]):5.1f}")
rN = evaluate(fid); log(f"NUFFT (Hamming): GM dep {rN['g']['pcr']:5.1f}  SOL dep {rN['s']['pcr']:5.1f} | median voxel tSNR GM {np.nanmedian(rN['tsnr'][vGM]):5.1f}  SOL {np.nanmedian(rN['tsnr'][vSOL]):5.1f}")
