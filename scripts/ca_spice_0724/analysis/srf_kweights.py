#!/usr/bin/env python3
"""Where does the soleus 'leakage' come from? MID00283 (or argv[1]), same encoding / sensitivity / masks / labels as hybrid_slam_spice.py.
Every estimator here is linear in k-space: estimate = sum_k w_k y_k (one weight per trajectory sample). Its spatial response function
SRF(p) = sum_k w_k A[k, p] (A = fine-grid encoding x coil sensitivity) says how much tissue at pixel p contributes. For a uniform tissue signal
the soleus estimate splits into: soleus tissue | GM tissue INSIDE the soleus CSI voxels (partial volume) | GM tissue from OUTSIDE (PSF) | other.
Estimators: NUFFT soleus ROI (mean of soleus voxels) | compartment SPICE soleus ROI (hard boundary) | CC-SPICE soleus ROI (voxel readout, mu 30)
| CC-SPICE soleus COMPARTMENT spectrum s_sol (no voxel readout). -> /mnt/e/claude_tmp/study/paper_abstract/F_leakage_srf_MID00<ds>.png"""
import os, sys
DS = sys.argv[1] if len(sys.argv) > 1 else "283"
os.environ.update(VFILE="VD12.npy", PIXMASK="1", LABELS="mask", MUS="30", HOUT="srf_tmp")
S_ = "/mnt/c/Users/zsun/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-zsun-SPICE-Margarita/6aafd315-8b8b-4771-8d0e-7a49ffd29de0/scratchpad/hybrid_slam_spice.py"
src = open(S_).read(); sys.argv = ["x", DS]; exec(src[:src.index("RES = {}")])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors
A = Fh[:, pix] * Ss[None]                                                                    # k x nu: tissue pixel -> k-space sample
insol = vSOL[blk[pix]]; inGMv = vGM[blk[pix]]
CLS = {"soleus": Mc[:, 1], "GM inside soleus voxels\n(partial volume)": Mc[:, 0] * insol, "GM from outside\n(PSF spill)": Mc[:, 0] * ~insol, "other muscle": Mc[:, 2]}
# --- compartment SPICE (CSI grid), hard boundary, as the runs ---
s0 = np.load(f"{O}/system.npz"); s11 = np.load(f"{O}/system11.npz"); G_h = s0["G"].astype(np.complex128)
comp = np.zeros(N * N); comp[legv] = 3; comp[vGM] = 1; comp[vSOL] = 2; comp = comp.reshape(N, N); legm = legv.reshape(N, N); ph = phant.reshape(N, N)
_, (ii, jj) = ndi.distance_transform_edt(~legm, return_indices=True); cmp_ = np.where(legm, comp, comp[ii, jj]); p_raw = s0["prior"]
lo, hi = p_raw[legm].min(), p_raw[legm].max(); pp = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1) + 2.0 * np.select([cmp_ == 1, cmp_ == 2], [0.0, 1.0], 2.0); pp[ph] = 0.0
WW, _ = m.calc_Bmatrix_simple(pp.astype(np.float32), wmax=5e4, adj=8, label_mask=np.where(ph, 0, cmp_).astype(int).T)
dd = ndi.distance_transform_cdt(~legm, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2); lout[ph] = 1.0; lout = lout.ravel()
Gd = (Fc.conj().T * dcf) @ Fc; rel = np.real(np.diag(Gd)).mean() / np.real(np.diag(G_h)).mean()
Ai = np.linalg.inv(Gd + float(s11["lam"]) * rel * WW + np.diag(lout * rel))
# --- CC-SPICE, mu 30 ---
AM = A @ Mc; Af = np.hstack([AM, A]); Gf = (Af.conj().T * dcf) @ Af; sc = np.real(np.trace((A.conj().T * dcf) @ A)) / nu
posm = -np.ones(Nh * Nh, int); posm[pix] = np.arange(nu); L = np.zeros((nu, nu)); pi_, pj_ = np.divmod(pix, Nh)
for dx_, dy_ in ((0, 1), (1, 0), (1, 1), (1, -1)):
    ok = (pi_ + dx_ < Nh) & (pj_ + dy_ >= 0) & (pj_ + dy_ < Nh); q = np.where(ok, (pi_ + dx_) * Nh + pj_ + dy_, 0); k2 = np.where(ok, posm[q], -1); a_ = k2 >= 0
    i1, i2 = np.flatnonzero(a_), k2[a_]; w = np.where(plab[i1] == plab[i2], 1.0, EPS_X) / np.hypot(dx_, dy_)
    np.add.at(L, (i1, i1), w); np.add.at(L, (i2, i2), w); np.add.at(L, (i1, i2), -w); np.add.at(L, (i2, i1), -w)
Reg = np.zeros_like(Gf); Reg[3:, 3:] = sc * (LAM_D * L + 30.0 * np.eye(nu)); Ahi = np.linalg.inv(Gf + Reg + 1e-9 * sc * np.eye(nu + 3))
Rd = np.zeros((N * N, nu), complex); Rd[blk[pix], np.arange(nu)] = Ss
Wz = Ahi @ (Af.conj().T * dcf)                                                               # (3 + nu) x k
EST = {"NUFFT\nsoleus voxels": (Fc[:, vSOL].conj().T * dcf_h).mean(0),
       "compartment SPICE\nsoleus voxels": (Ai @ (Fc.conj().T * dcf))[vSOL].mean(0),
       "CC-SPICE\nsoleus voxels": (Rd @ np.vstack([Mc.T, np.eye(nu)]).T @ Wz)[vSOL].mean(0) if False else (Rd @ np.hstack([Mc, np.eye(nu)]) @ Wz)[vSOL].mean(0),
       "CC-SPICE\nsoleus compartment s_sol": Wz[1]}
res = {}
for nm, w in EST.items():
    r = w @ A; tot = (r * Mc.sum(1)).sum(); ph0 = np.exp(-1j * np.angle(tot)); rr = np.real(r * ph0) / np.abs(tot)
    shares = {k: float((rr * v).sum()) for k, v in CLS.items()}; noise = np.linalg.norm(w * np.sqrt(1 / np.maximum(dcf, 1e-12)) * np.sqrt(dcf)) / np.abs(tot)
    res[nm] = dict(rr=rr, shares=shares, w=w, noise=np.linalg.norm(w) / np.abs(tot))
n0 = res["NUFFT\nsoleus voxels"]["noise"]
log("\nshare of the soleus estimate for a uniform tissue signal (sum = 1), noise relative to NUFFT:")
for nm, d in res.items(): log(f"  {nm.replace(chr(10), ' '):38s} " + "  ".join(f"{k.split(chr(10))[0]} {100 * v:5.1f}%" for k, v in d["shares"].items()) + f"   noise x{d['noise'] / n0:.2f}")
# GM compartment estimator too (for the k-space weights)
wG = Wz[0]
# ---------------- figure ----------------
def to_img(v):
    Z = np.full(Nh * Nh, np.nan); Z[pix] = v; return Z.reshape(Nh, Nh)
colc = lambda X_: (X_ - ipp18[0]) / ps18; rowc = lambda Y_: (Y_ - ipp18[1]) / ps18
ext = [colc(CX + (-0.5 - Nh / 2) * dh + 0.5 * D_), colc(CX + (Nh - 0.5 - Nh / 2) * dh + 0.5 * D_), rowc(CY + (Nh / 2 - (Nh - 0.5)) * dh - 0.5 * D_), rowc(CY + (Nh / 2 + 0.5) * dh - 0.5 * D_)]
supp_img = to_img(np.ones(nu)) == 1; rows_, cols_ = np.where(supp_img)
fig = plt.figure(figsize=(22, 12.5)); g = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.28, wspace=0.12, left=0.04, right=0.99, top=0.9, bottom=0.07)
CC = {"soleus": "#1f77b4", "GM inside soleus voxels\n(partial volume)": "#98df8a", "GM from outside\n(PSF spill)": "#2ca02c", "other muscle": "#d9a441"}
cm_ = matplotlib.colormaps["RdBu_r"]
for j, (nm, d) in enumerate(res.items()):
    a = fig.add_subplot(g[0, j]); a.imshow(im18, cmap="gray", interpolation="lanczos")
    v = d["rr"] / np.max(np.abs(d["rr"])); a.imshow(to_img(v), cmap=cm_, norm=colors.Normalize(-1, 1), extent=ext, alpha=0.85, interpolation="nearest")
    a.contour(GMp.astype(float), [0.5], colors="#2ca02c", linewidths=1.6); a.contour(SOLp.astype(float), [0.5], colors="#1f77b4", linewidths=1.6)
    for vv in np.flatnonzero(vSOL):                                                          # soleus CSI voxels (dashed)
        bx, by = vv // N, vv % N; xs_ = [colc(CX + ((by * F_UP - 0.5 - 1) - Nh / 2) * dh + 0.5 * D_ + dh)]
    xl = [ext[0] + (cols_.min() - 1) * (ext[1] - ext[0]) / Nh, ext[0] + (cols_.max() + 2) * (ext[1] - ext[0]) / Nh]
    yl = [ext[3] + (rows_.max() + 2) * (ext[2] - ext[3]) / Nh, ext[3] + (rows_.min() - 1) * (ext[2] - ext[3]) / Nh]
    a.set_xlim(*sorted(xl)); a.set_ylim(*sorted(yl, reverse=True)); a.set_xticks([]); a.set_yticks([])
    a.set_title(nm + f"\nnoise ×{d['noise'] / n0:.2f} vs NUFFT", fontsize=12)
    ia = a.inset_axes([0.0, -0.2, 1.0, 0.12]); left = 0
    for k, val in d["shares"].items():
        ia.barh(0, val, left=left, color=CC[k]);
        if abs(val) > 0.06: ia.text(left + val / 2, 0, f"{100 * val:.0f}%", ha="center", va="center", fontsize=10, color="w", fontweight="bold")
        left += val
    ia.set_xlim(min(0, min(d["shares"].values())), max(1, left)); ia.axis("off")
fig.legend(handles=[matplotlib.patches.Patch(color=c, label=k.replace("\n", " ")) for k, c in CC.items()], loc="upper center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.text(0.5, 0.94, f"MID00{DS}: spatial response of each soleus estimate (red = adds signal, blue = subtracts); bar = where a uniform-tissue soleus estimate comes from",
         ha="center", fontsize=13, fontweight="bold")
kr = np.hypot(tj[:, 0], tj[:, 1])
for j, (nm, w) in enumerate((("NUFFT soleus voxels", EST["NUFFT\nsoleus voxels"]), ("CC-SPICE soleus compartment", Wz[1]), ("CC-SPICE GM compartment", wG))):
    a = fig.add_subplot(g[1, j]); wa = np.abs(w) / np.abs(w).max()
    s = a.scatter(tj[:, 0], tj[:, 1], c=wa, s=10, cmap="viridis", vmin=0, vmax=1); a.set_aspect("equal"); a.set_title(f"k-space weights |w|: {nm}", fontsize=12)
    a.set_xlabel("kx"); a.set_ylabel("ky")
a = fig.add_subplot(g[1, 3])
for nm, w, c in (("NUFFT soleus voxels", EST["NUFFT\nsoleus voxels"], "0.4"), ("CC-SPICE soleus compartment", Wz[1], "#1f77b4"), ("CC-SPICE GM compartment", wG, "#2ca02c")):
    rb = np.unique(np.round(kr, 3)); prof = [np.abs(w[np.round(kr, 3) == r_]).mean() for r_ in rb]; a.plot(rb, np.array(prof) / max(prof), "o-", color=c, label=nm)
a.set_xlabel("|k| (ring radius)"); a.set_ylabel("mean |w| per ring (norm.)"); a.legend(fontsize=10); a.grid(alpha=0.3); a.set_title("which rings each estimate relies on", fontsize=12)
fig.savefig(f"/mnt/e/claude_tmp/study/paper_abstract/F_leakage_srf_MID00{DS}.png", dpi=110); log("saved figure")
