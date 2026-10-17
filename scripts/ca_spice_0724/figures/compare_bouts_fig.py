#!/usr/bin/env python3
"""NUFFT vs CC-SPICE (rank 12, mu 30), unfitted, all four 24-07 bouts (rows).
Columns: PCr depletion NUFFT | CC-SPICE, Pi increase NUFFT | CC-SPICE (GM + soleus voxels, values printed),
GM voxel and soleus voxel PCr (solid) + GM Pi (dashed) dynamics, 3-rep running mean within each phase, shared y.
Plotted voxels: yellow dashed (MID00283: 63 / 53 as Fig 3; others: the standard GM / soleus pair).
-> paper_abstract/F_nufft_vs_ccspice_4bouts.png/.pdf"""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
os.environ.setdefault("HDIR", "hybrid_v2_r12"); os.environ.setdefault("MU_SHOW", "30")
src = open("/mnt/e/claude_tmp/study/hybrid_fig.py").read(); exec(src[:src.index("for key, cmap, lo, hi, qname in")])
src2 = open("/mnt/e/claude_tmp/study/combined_fig.py").read(); exec(src2[src2.index("L_GM, L_SOL"):src2.index("\n", src2.index("L_GM, L_SOL"))])
OUTF = "/mnt/e/claude_tmp/study/paper_abstract"
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
STK = [patheffects.withStroke(linewidth=2, foreground="k")]; BO = tuple(os.environ.get("BOUTS_SHOW", "283,279,281,277").split(",")); PAIRS = dict(PAIRS); PAIRS["283"] = (int(os.environ.get("GM283", 63)), int(os.environ.get("SOL283", 53)))
SECOND = os.environ.get("SECOND", "CC")                                                     # CC = CC-SPICE (fine grid), SP = compartment SPICE on the CSI grid
if SECOND.startswith("SPx"): M2 = (f"SPICE comp λx{SECOND[3:]}", f"compartment SPICE, λ ×{SECOND[3:]}", "#1b9e77")    # e.g. SECOND=SPx100
else: M2 = {"CC": (HYB, "CC-SPICE", "#7b3294"), "SP": (SP, "compartment SPICE", "#1b9e77"), "CA": (SP, "CA-SPICE", "#7b3294")}[SECOND]
METH = (("NUFFT", "NUFFT", os.environ.get("C_NUFFT", "0.45")), M2)
MAPS = (("dep", "RdBu_r", -60, 60), ("inc", "PuOr_r", -40, 40))
fig = plt.figure(figsize=(25, 5.25 * len(BO))); g = fig.add_gridspec(len(BO), 6, width_ratios=[1, 1, 1, 1, 0.16, 1.9], wspace=0.06, hspace=0.2, left=0.05, right=0.99, top=0.945, bottom=0.07)
_AVG = {}
def AVG(B, nm):
    """Maps from averaged spectra (as Fig 2): rest = mean of rest reps, end of exercise = mean of the last 4 reps fully inside exercise;
    magnitude, 8 Hz LB, peak heights: dep = 100 (1 - PCr_ex / PCr_rest), inc = 100 (Pi_ex - Pi_rest) / PCr_rest (Pi peak in 3.8-5.4 ppm)."""
    if (B, nm) in _AVG: return _AVG[(B, nm)]
    h_ = H[B]; R0_, R1_ = map(int, h_["rest"]); E1_ = int(h_["ex"][1]); O_ = f"{ROOT}/output_study_mid{B}"
    X_ = np.load(f"{O_}/adj_fid.npy") if nm == "NUFFT" else np.load(f"{O_}/{os.environ.get('HDIR', 'spice_final')}/fid_spice.npy")
    n_ = X_.shape[-1]; t_ = np.arange(n_) * 0.00036; apo = np.exp(-np.pi * 8 * t_); mag = lambda x: np.abs(np.fft.fftshift(np.fft.fft(np.conj(x * apo), axis=-1), axes=-1))
    Sr, Se = mag(X_[R0_:R1_].mean(0)), mag(X_[E1_ - 5:E1_ - 1].mean(0)); ppm_raw = -np.fft.fftshift(np.fft.fftfreq(n_, 0.00036)) / 49.8946
    vg_ = h_["vGM"].astype(bool) & h_["legv"].astype(bool); k0 = np.flatnonzero(np.abs(ppm_raw) < 1.5)[np.argmax(Sr[vg_].sum(0)[np.abs(ppm_raw) < 1.5])]; Q = ppm_raw - ppm_raw[k0]
    bP, bPi = np.abs(Q) < 1, (Q >= 3.8) & (Q <= 5.4); r0 = Sr[:, bP].max(1)
    _AVG[(B, nm)] = dict(dep=100 * (1 - Se[:, bP].max(1) / r0), inc=100 * (Se[:, bPi].max(1) - Sr[:, bPi].max(1)) / r0); return _AVG[(B, nm)]
rows = []
for i, B in enumerate(BO):
    h = H[B]; N = int(h["N"]); IMG, gmL, solL = loc_show(h); _, x0, y0, cw, ch, box, ext = geom(h); leg = h["legv"].astype(bool); vg, vs = h["vGM"] & leg, h["vSOL"] & leg
    flip = lambda d: (N - 1 - d // N) * N + d % N; GV, SV = PAIRS[B]; qg, qs = flip(GV), flip(SV)
    E0, E1 = map(int, h["ex"]); rep = float(h["REP"]); li = list(h["leg_idx"]); NR = len(arr(h, "NUFFT", "g_curve")); tm = np.arange(NR) * rep / 60
    def rm3(y):
        out = np.array(y, float).copy()
        for s0, s1 in ((0, E0), (E0, E1), (E1, len(y))):
            seg = np.asarray(y[s0:s1], float); out[s0:s1] = np.convolve(seg, np.ones(3), "same") / np.convolve(np.ones(len(seg)), np.ones(3), "same")
        return out
    qq = np.flatnonzero(vg | vs); rr_ = [N - 1 - q // N for q in qq]; cc_ = [q % N for q in qq]
    for jm, (key, cmap, lo, hi) in enumerate(MAPS):
        for k, (m, nm, _) in enumerate(METH):
            a = fig.add_subplot(g[i, 2 * jm + k]); a.imshow(IMG, cmap="gray", interpolation="lanczos"); nrm = colors.Normalize(lo, hi); cm_ = matplotlib.colormaps[cmap]; val = AVG(B, nm)[key]
            for q in qq:
                l, tp = box(q); a.add_patch(plt.Rectangle((l, tp), cw, ch, fc=cm_(nrm(val[q])), ec="none", alpha=0.9))
                a.text(l + cw / 2, tp + ch / 2, f"{val[q]:.0f}", ha="center", va="center", fontsize=10 if N <= 8 else 9, color="w", fontweight="bold", path_effects=STK)
            outline(a, vg, N, x0, y0, cw, ch, L_GM, lw=2.2); outline(a, vs, N, x0, y0, cw, ch, L_SOL, lw=2.2)
            for q in (qg, qs): pm = np.zeros(N * N, bool); pm[q] = True; outline(a, pm, N, x0, y0, cw, ch, "yellow", lw=2.2, ls="--")
            a.set_xlim(x0 + (min(cc_) - 0.6) * cw, x0 + (max(cc_) + 1.6) * cw); a.set_ylim(y0 + (max(rr_) + 1.6) * ch, y0 + (min(rr_) - 0.6) * ch)
            a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
            if i == 0: a.set_title(f"{'PCr depletion (%)' if key == 'dep' else 'Pi increase (% rest PCr)'}\n{nm}", fontsize=12.5, color="0.15")
            if jm == 0 and k == 0: a.set_ylabel(f"MID00{B}\n{float(h['FOV']) / N:.0f} mm, {rep:.0f} s/rep", fontsize=12.5, fontweight="bold")
    a = fig.add_subplot(g[i, 5]); a.axvspan(E0 * rep / 60, E1 * rep / 60, color="0.93", zorder=0)
    for m, nm, c_ in METH:
        a.plot(tm, rm3(arr(h, m, "vox_pcr")[:, li.index(qg)]), "-", color=c_, lw=2.6)
        a.plot(tm, rm3(arr(h, m, "vox_pcr")[:, li.index(qs)]), "-", color=c_, lw=1.6, alpha=0.75)
        a.plot(tm, rm3(arr(h, m, "vox_pi")[:, li.index(qg)]), "--", color=c_, lw=2.0)
    a.set_ylim(0, 1.4); a.set_xlim(0, tm[-1]); a.grid(alpha=0.3); a.set_ylabel("relative to rest PCr")
    a.set_title(f"GM voxel {GV} (thick) and soleus voxel {SV} (thin) PCr; GM Pi dashed", loc="left", fontsize=11.5)
    if i == len(BO) - 1: a.set_xlabel("time (min)")
    d = {nm: (AVG(B, nm)["dep"][qg], AVG(B, nm)["dep"][qs], AVG(B, nm)["inc"][qg], AVG(B, nm)["inc"][qs]) for m, nm, _ in METH}
    md = {nm: (np.median(AVG(B, nm)["dep"][vg]), np.median(AVG(B, nm)["dep"][vs]), np.median(AVG(B, nm)["inc"][vg]), np.median(AVG(B, nm)["inc"][vs])) for m, nm, _ in METH}
    rows.append((B, d, md))
fig.legend(handles=[Line2D([], [], color=METH[0][2], lw=2.6, label="NUFFT"), Line2D([], [], color=M2[2], lw=2.6, label=M2[1]),
                    Line2D([], [], color="0.2", lw=2.4, label="PCr"), Line2D([], [], color="0.2", lw=2.0, ls="--", label="Pi"),
                    Patch(fc="none", ec=L_GM, lw=2.2, label="gastrocnemius"), Patch(fc="none", ec=L_SOL, lw=2.2, label="soleus"),
                    Patch(fc="none", ec="yellow", lw=2.2, ls="--", label="plotted voxels")], loc="upper center", ncol=7, fontsize=12.5, frameon=True, edgecolor="0.7", bbox_to_anchor=(0.5, 0.995))
for j, (key, cmap, lo, hi) in enumerate(MAPS):
    cax = fig.add_axes([0.07 + 0.31 * j, 0.035, 0.24, 0.009])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=colors.Normalize(lo, hi), cmap=cmap), cax=cax, orientation="horizontal").set_label("PCr depletion (%)" if key == "dep" else "Pi increase (% of rest PCr)", fontsize=11)
tag = ("compSPICE_lam" + SECOND[3:] if SECOND.startswith("SPx") else {"CC": "ccspice", "SP": "compSPICE", "CA": "caspice"}[SECOND]) + f"_{len(BO)}bouts" + os.environ.get("TAG", "")
fig.savefig(f"{OUTF}/F_nufft_vs_{tag}.png", dpi=110); fig.savefig(f"{OUTF}/F_nufft_vs_{tag}.pdf"); print("saved", tag)
print("median over voxels [GM dep, SOL dep, GM Pi inc, SOL Pi inc] and voxel tSNR [GM, SOL]:")
for B in BO:
    h = H[B]; leg = h["legv"].astype(bool); vg, vs = h["vGM"] & leg, h["vSOL"] & leg; out = []
    MM = [("NUFFT", "NUFFT"), (SP, "compSPICE")] + ([(M2[0], M2[1])] if SECOND.startswith("SPx") else []) + [(HYB, "CC-SPICE")]
    for m, nm in MM:
        v = [np.median(arr(h, m, k)[msk]) for k, msk in (("dep", vg), ("dep", vs), ("inc", vg), ("inc", vs))] + [np.nanmedian(arr(h, m, "tsnr")[vg]), np.nanmedian(arr(h, m, "tsnr")[vs])]
        out.append(f"{nm} {np.round(v).astype(int).tolist()}")
    s = {nm: arr(h, m, "summary") for m, nm in MM}
    print(f"  MID00{B}: " + "   ".join(out) + "   | GM->soleus spill " + ", ".join(f"{k} {100 * v[10]:.0f}%" for k, v in s.items()))
