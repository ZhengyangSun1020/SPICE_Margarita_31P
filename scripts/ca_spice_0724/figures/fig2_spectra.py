#!/usr/bin/env python3
"""Abstract Fig 2: spectra across protocols (two bouts per figure), NUFFT vs CA-SPICE.
Layout per bout (row): [gastrocnemius area: NUFFT / CA-SPICE spectra of the GM voxel] [NUFFT map] [CA-SPICE map] [soleus area: NUFFT / CA-SPICE spectra
of the soleus voxel directly above]; lines connect the marked voxels to their spectra. Maps: whole localizer with the CSI grid, spectra in GM + soleus
voxels (rest black, end of exercise red, one y-scale per map). Spectra: magnitude, 8 Hz LB; rest = mean of rest reps, end of exercise = last 4 reps
fully inside exercise. Labels from the displayed spectra: PCr change = peak height change (%), Pi/PCr = Pi peak height (3.8-5.4 ppm) / rest PCr peak.
SET=a: MID00283 + MID00279 (smallest / largest voxels), SET=b: MID00281 + MID00277. -> paper_abstract/F2_spectra_<bouts>.png/.pdf"""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, ConnectionPatch
os.environ.setdefault("HDIR", "spice_final"); os.environ.setdefault("MU_SHOW", "30")
src = open("/mnt/e/claude_tmp/study/hybrid_fig.py").read(); exec(src[:src.index("for key, cmap, lo, hi, qname in")])
OUTF = "/mnt/e/claude_tmp/study/paper_abstract"; SPD = os.environ["HDIR"]
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
DW, F0 = 0.00036, 49.8946; STK = [patheffects.withStroke(linewidth=2.4, foreground="k")]; C_N, C_C = "#b35806", "#7b3294"
SET = os.environ.get("SET", "a"); BOUTSET = {"a": (("283", 64, 54), ("279", 43, 35)), "b": (("281", 43, 35), ("277", 64, 54))}[SET]
fig = plt.figure(figsize=(21.5, 12.5)); AREA = {0: [], 3: []}
g = fig.add_gridspec(2, 4, width_ratios=[0.82, 1, 1, 0.82], wspace=0.09, hspace=0.16, left=0.03, right=0.985, top=0.86, bottom=0.055)
for i, (B, GV, SV) in enumerate(BOUTSET):
    h = H[B]; N = int(h["N"]); IMG, gmL, solL = loc_show(h); _, x0, y0, cw, ch, box, ext = geom(h); leg = h["legv"].astype(bool); vg, vs = h["vGM"] & leg, h["vSOL"] & leg
    flip = lambda d: (N - 1 - d // N) * N + d % N; O = f"{ROOT}/output_study_mid{B}"; R0, R1 = map(int, h["rest"]); E0, E1 = map(int, h["ex"]); SHOW = vg | vs
    SP_ = {}
    for meth, X in (("NUFFT", np.load(f"{O}/adj_fid.npy")), ("CA-SPICE", np.load(f"{O}/{SPD}/fid_spice.npy"))):
        n = X.shape[-1]; t = np.arange(n) * DW; APO = np.exp(-np.pi * 8 * t); mag = lambda x: np.abs(np.fft.fftshift(np.fft.fft(np.conj(x * APO), axis=-1), axes=-1))
        Sr, Se = mag(X[R0:R1].mean(0)), mag(X[E1 - 5:E1 - 1].mean(0)); ppm_raw = -np.fft.fftshift(np.fft.fftfreq(n, DW)) / F0
        k0 = int(np.argmax(Sr[vg].sum(0) * (np.abs(ppm_raw) < 1.5))); SP_[meth] = (ppm_raw - ppm_raw[k0], Sr, Se)
    maps = {}
    for j, meth in enumerate(("NUFFT", "CA-SPICE")):
        ppm, Sr, Se = SP_[meth]; bP = np.abs(ppm) < 1; SC = Sr[SHOW][:, bP].max(); w = (ppm < 8) & (ppm > -18); px = (8 - ppm[w]) / 26
        a = fig.add_subplot(g[i, 1 + j]); a.imshow(IMG, cmap="gray", interpolation="lanczos"); maps[meth] = a
        for k_ in range(N + 1):
            a.plot([x0, x0 + N * cw], [y0 + k_ * ch] * 2, color="#00e5ff", lw=0.8, alpha=0.7); a.plot([x0 + k_ * cw] * 2, [y0, y0 + N * ch], color="#00e5ff", lw=0.8, alpha=0.7)
        for q in np.flatnonzero(SHOW):
            l, tp = box(q)
            for S, col in ((Sr, "w"), (Se, "#ff3b3b")): a.plot(l + 0.05 * cw + 0.9 * cw * px, tp + 0.93 * ch - 0.82 * ch * np.clip(S[q, w] / SC, 0, 1.1), color=col, lw=1.3, path_effects=STK)
        outline(a, vg, N, x0, y0, cw, ch, L_GM, lw=2.0); outline(a, vs, N, x0, y0, cw, ch, L_SOL, lw=2.0)
        for d_, cc_ in ((GV, "#7CFC00"), (SV, "#4fc3ff")): pm = np.zeros(N * N, bool); pm[flip(d_)] = True; outline(a, pm, N, x0, y0, cw, ch, cc_, lw=3.2, ls="--")
        a.set_xlim(x0, x0 + N * cw); a.set_ylim(y0 + N * ch, y0); a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
        a.set_title(f"{meth}   (MID00{B}: {float(h['FOV']) / N:.0f} mm voxels, {float(h['REP']):.0f} s/rep)", fontsize=12.5, color=C_N if meth == "NUFFT" else C_C)
    for side, (d_, col, gcol) in enumerate(((GV, L_GM, 0), (SV, L_SOL, 3))):
        q = flip(d_); sub = g[i, gcol].subgridspec(2, 1, hspace=0.18); axs_ = AREA[gcol]; axs_ = []
        for k_, meth in enumerate(("NUFFT", "CA-SPICE")):
            ppm, Sr, Se = SP_[meth]; bP, bPi = np.abs(ppm) < 1, (ppm >= 3.8) & (ppm <= 5.4); sc_ = Sr[q, bP].max(); a = fig.add_subplot(sub[k_]); axs_.append(a); AREA[gcol].append(a)
            a.plot(ppm, Sr[q] / sc_, color="k", lw=1.3); a.plot(ppm, Se[q] / sc_, color="#e41a1c", lw=1.3)
            a.set_xlim(10, -20); a.set_ylim(0, 1.3); a.set_yticks([0, 1]); a.grid(alpha=0.25)
            dep = 100 * (1 - Se[q, bP].max() / sc_); pr, pe = Sr[q, bPi].max() / sc_, Se[q, bPi].max() / sc_
            a.text(-19.6, 1.24, f"{meth}\nPCr −{dep:.0f}%\nPi  {100 * (pe - pr):+.0f}%", ha="right", va="top", fontsize=10.5, color=C_N if meth == "NUFFT" else C_C,
                   fontweight="bold", linespacing=1.25)
            if k_ == 0:
                a.set_title(f"voxel {d_}  (MID00{B})", color=col, fontsize=12.5, pad=16); a.set_xticklabels([])
                for p_, lb in ((0, "PCr"), (4.9, "Pi"), (-7.6, "α"), (-16.2, "β")): a.text(p_, 1.32, lb, ha="center", fontsize=9, color="0.4")
            elif i == 1: a.set_xlabel("ppm (PCr = 0)")
        # connector: marked voxel (nearest map) -> its spectra area
        m_ax = maps["NUFFT"] if side == 0 else maps["CA-SPICE"]; l, tp = box(q); xy_vox = (l + (0 if side == 0 else cw), tp + ch / 2)
        con = ConnectionPatch(xyA=xy_vox, coordsA=m_ax.transData, xyB=(1.0 if side == 0 else 0.0, -0.09), coordsB=axs_[0].transAxes, color=col, lw=2.2, ls="-",
                              arrowstyle="-|>", mutation_scale=18, zorder=10)
        fig.add_artist(con)
for gcol, col, ttl in ((0, L_GM, "Gastrocnemius voxel"), (3, L_SOL, "Soleus voxel")):                   # coloured areas around the spectra panels
    bb = [a_.get_position() for a_ in AREA[gcol]]; xa, xb = min(b.x0 for b in bb) - 0.022, max(b.x1 for b in bb) + 0.008; ya, yb = min(b.y0 for b in bb) - 0.04, max(b.y1 for b in bb) + 0.035
    fig.add_artist(FancyBboxPatch((xa, ya), xb - xa, yb - ya, boxstyle="round,pad=0.003,rounding_size=0.01", transform=fig.transFigure, fc=col, alpha=0.07, ec=col, lw=1.5, zorder=-5))
    fig.text((xa + xb) / 2, yb + 0.008, ttl, ha="center", fontsize=15, fontweight="bold", color=col)
fig.legend(handles=[Line2D([], [], color="k", lw=2, label="rest"), Line2D([], [], color="#e41a1c", lw=2, label="end of exercise"),
                    Patch(fc="none", ec=L_GM, lw=2, label="gastrocnemius"), Patch(fc="none", ec=L_SOL, lw=2, label="soleus"),
                    Patch(fc="none", ec="#7CFC00", lw=2.4, ls="--", label="plotted GM voxel"), Patch(fc="none", ec="#4fc3ff", lw=2.4, ls="--", label="plotted soleus voxel")],
           loc="upper center", ncol=6, fontsize=12.5, frameon=True, edgecolor="0.7", bbox_to_anchor=(0.5, 0.999))
TG = "_".join(b for b, _, _ in BOUTSET); fig.savefig(f"{OUTF}/F2_spectra_{TG}.png", dpi=120); fig.savefig(f"{OUTF}/F2_spectra_{TG}.pdf"); print("saved", TG)
