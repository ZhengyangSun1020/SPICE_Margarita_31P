#!/usr/bin/env python3
"""One figure, all four 24-07 bouts (rows): PCr depletion and Pi increase maps (NUFFT, SPICE) on the whole localizer, gastrocnemius / soleus
outlined, plus one GM voxel and the soleus voxel above it (marked yellow on the maps) with PCr and Pi over time, NUFFT and SPICE together.
Reads output_study_mid<ds>/$FINAL; writes $PAPER_OUT/combined_maps_voxels.png"""
import sys, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.lines import Line2D
sys.argv = [sys.argv[0], "283"]; exec(open("/mnt/e/claude_tmp/study/paper_figs_0724.py").read().split("# ================= Figure 1 =================")[0])
L_GM, L_SOL = "#2ca02c", "#1f77b4"; PAIRS = {"279": (43, 35), "281": (43, 35), "277": (64, 54), "283": (64, 54)}
TITLE = os.environ.get("TITLE", "")
def outline(a, msk, N, x0, y0, cw, ch, col, lw=2.2, ls="-"):
    m2 = msk.reshape(N, N)[::-1]
    for r in range(N):
        for c in range(N):
            if not m2[r, c]: continue
            l, t, rg, bt = x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch
            if r == 0 or not m2[r - 1, c]: a.plot([l, rg], [t, t], color=col, lw=lw, ls=ls)
            if r == N - 1 or not m2[r + 1, c]: a.plot([l, rg], [bt, bt], color=col, lw=lw, ls=ls)
            if c == 0 or not m2[r, c - 1]: a.plot([l, l], [t, bt], color=col, lw=lw, ls=ls)
            if c == N - 1 or not m2[r, c + 1]: a.plot([rg, rg], [t, bt], color=col, lw=lw, ls=ls)
def compmap(a, z, IMG, val, cmap, vmin, vmax, pair):
    N, x0, y0, cw, ch, box, _ = geom(z); a.imshow(IMG, cmap="gray", interpolation="lanczos")
    leg = z["leg"]; vg, vs = z["vGM"] & leg, z["vSOL"] & leg; nrm = colors.Normalize(vmin, vmax); cm_ = matplotlib.colormaps[cmap]
    for q in np.flatnonzero(vg | vs):
        if not np.isfinite(val[q]): continue
        l, t = box(q); a.add_patch(plt.Rectangle((l, t), cw, ch, fc=cm_(nrm(val[q])), ec="none", alpha=0.75))
        a.text(l + cw / 2, t + ch / 2, f"{val[q]:.0f}", ha="center", va="center", fontsize=7.5, color="w", fontweight="bold",
               path_effects=[patheffects.withStroke(linewidth=1.6, foreground="0.15")])
    outline(a, vg, N, x0, y0, cw, ch, L_GM); outline(a, vs, N, x0, y0, cw, ch, L_SOL)
    for q in pair: pm = np.zeros(N * N, bool); pm[q] = True; outline(a, pm, N, x0, y0, cw, ch, "yellow", lw=1.6, ls="--")
    a.set_xlim(x0, x0 + N * cw); a.set_ylim(y0 + N * ch, y0); a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
fig = plt.figure(figsize=(34, 21)); gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 1.45, 1.45], wspace=0.2, hspace=0.22)
MAPS = (("NUFFT", "dep", "RdBu_r", -60, 60), ("SPICE", "dep", "RdBu_r", -60, 60), ("NUFFT", "inc", "PuOr_r", -40, 40), ("SPICE", "inc", "PuOr_r", -40, 40))
for i, b in enumerate(BOUTS):
    z = Z[b]; IMG = loc_img(z); N = int(z["N"]); rec = lambda d: (N - 1 - d // N) * N + d % N; gid, sid = PAIRS[b]; qg, qs = rec(gid), rec(sid)
    for j, (k, key, cm_, lo, hi) in enumerate(MAPS):
        a = fig.add_subplot(gs[i, j]); compmap(a, z, IMG, z[f"{k}_{key}"], cm_, lo, hi, (qg, qs))
        if i == 0: a.set_title(f"{k}\n{'PCr depletion (%)' if key == 'dep' else 'Pi increase (% of rest PCr)'}", fontsize=15, fontweight="bold")
        if j == 0: a.set_ylabel(f"MID00{b}\n{float(z['FOV']) / N:.0f} mm voxels", fontsize=15, fontweight="bold")
    leg_idx = list(z["leg_idx"]); t_ = tmin(z)
    for j, (q, nm, idd, col0) in enumerate(((qg, "GM", gid, L_GM), (qs, "SOL", sid, L_SOL))):
        a = fig.add_subplot(gs[i, 4 + j]); i_ = leg_idx.index(q)
        for k, col in (("NUFFT", C_NU), ("SPICE", C_SP)):
            a.plot(t_, z[f"{k}_vox_pcr"][:, i_], "-", color=col, lw=2.0, label=f"{k} PCr"); a.plot(t_, z[f"{k}_vox_pi"][:, i_], "--", color=col, lw=1.6, label=f"{k} Pi")
        exspan(a, z); a.set_ylim(-0.1, 1.6); a.grid(alpha=0.3); a.set_ylabel("normalised to rest PCr", fontsize=11)
        if i == 3: a.set_xlabel("time (min)", fontsize=12)
        a.set_title(f"{nm} voxel {idd}: PCr depletion NUFFT {z['NUFFT_dep'][q]:.0f}%, SPICE {z['SPICE_dep'][q]:.0f}%", loc="left", fontweight="bold", color=col0, fontsize=12)
        if i == 0: a.legend(frameon=False, ncol=2, fontsize=10, loc="upper right")
cax1 = fig.add_axes([0.13, 0.065, 0.24, 0.008]); cax2 = fig.add_axes([0.40, 0.065, 0.24, 0.008])
fig.colorbar(matplotlib.cm.ScalarMappable(norm=colors.Normalize(-60, 60), cmap="RdBu_r"), cax=cax1, orientation="horizontal").set_label("PCr depletion (%)", fontsize=13)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=colors.Normalize(-40, 40), cmap="PuOr_r"), cax=cax2, orientation="horizontal").set_label("Pi increase (% of rest PCr)", fontsize=13)
fig.legend(handles=[Line2D([], [], color=L_GM, lw=3, label="gastrocnemius"), Line2D([], [], color=L_SOL, lw=3, label="soleus"),
                    Line2D([], [], color="yellow", lw=2, ls="--", label="voxels plotted on the right")],
           loc="upper left", ncol=3, fontsize=14, frameon=False, bbox_to_anchor=(0.13, 0.955))
if TITLE: fig.suptitle(TITLE, fontsize=17, fontweight="bold", y=0.985)
fig.savefig(f"{OUT}/combined_maps_voxels.png", dpi=100, bbox_inches="tight"); fig.savefig(f"{OUT}/combined_maps_voxels.pdf", bbox_inches="tight"); print("saved", f"{OUT}/combined_maps_voxels.png")
