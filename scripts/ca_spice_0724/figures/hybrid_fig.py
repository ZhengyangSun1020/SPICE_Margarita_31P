#!/usr/bin/env python3
"""CC-SPICE (pixel-level masks) vs NUFFT vs compartment SPICE, all four 24-07 bouts.
Fig A (PCr) / Fig A_Pi: per bout - localizer with the drawn masks, maps (NUFFT, compartment SPICE, hybrid), GM and soleus voxel time courses.
Fig B: validation - leakage test vs true partial volume (2 mm truth, + 3 mm GM boundary error) and the mu trade-off. -> paper_hybrid/"""
import os, re, sys, numpy as np, pydicom, scipy.io as sio, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.argv = [sys.argv[0], "283"]; exec(open("/mnt/e/claude_tmp/study/paper_figs_0724.py").read().split("# ================= Figure 1 =================")[0])
src = open("/mnt/e/claude_tmp/study/combined_fig.py").read(); exec(src[src.index("L_GM, L_SOL"):src.index("fig = plt.figure(")])
sys.path.insert(0, "/mnt/e/claude_tmp/study/masks_hr"); from phantom_mask import phantom_mask
HDIR = os.environ.get("HDIR", "hybrid_pix"); OUTH = os.environ.get("OUTH", "/mnt/e/claude_tmp/study/paper_hybrid"); os.makedirs(OUTH, exist_ok=True)
MU_SHOW = os.environ.get("MU_SHOW", "100"); HYB = f"hybrid μ {MU_SHOW}"; SP = "SPICE (compartment, CSI grid)"
COLM = {"NUFFT": "0.35", SP: "#d62728", HYB: "#7b3294"}; LABM = {"NUFFT": "NUFFT", SP: "compartment SPICE", HYB: f"CC-SPICE (μ {MU_SHOW})"}
SG = sio.loadmat("/mnt/e/claude_tmp/study/masks_hr/seg_series18.mat"); GMp = SG["GM"].astype(bool); SOLp = SG["SOL"].astype(bool) & ~GMp
L18 = pydicom.dcmread([f"/mnt/e/claude_tmp/study/loc/{f}" for f in os.listdir("/mnt/e/claude_tmp/study/loc") if f.startswith("Series0018")][0])
ipp18 = np.array(L18.ImagePositionPatient, float); ps18 = float(L18.PixelSpacing[0])
H = {}
for b in BOUTS:
    h = dict(np.load(f"{ROOT}/output_study_mid{b}/{HDIR}/hybrid_MID00{b}.npz", allow_pickle=True)); h["leg"] = h["legv"]; h["names"] = list(h["names"])
    h["log"] = open(f"{ROOT}/output_study_mid{b}/{HDIR}/log.txt").read(); H[b] = h
def arr(h, meth, key): return h[f"{h['names'].index(meth)}_{key}"]
def loc_show(h):                                                                                  # localizer with the phantom (display only) + drawn masks resampled
    ds = pydicom.dcmread("/mnt/e/claude_tmp/study/loc_all/" + str(h["loc"])); im = ds.pixel_array.astype(float); ph = phantom_mask(im)
    v = im[~ph]; IMG = np.clip((im - np.percentile(v, 1)) / (1.15 * (np.percentile(v, 99.5) - np.percentile(v, 1))), 0, 1)
    vp = im[ph]; IMG[ph] = 0.85 * np.clip((im[ph] - np.percentile(vp, 1)) / (np.percentile(vp, 99.5) - np.percentile(vp, 1)), 0, 1)
    ipp = np.array(ds.ImagePositionPatient, float); ps = float(ds.PixelSpacing[0]); rr, cc = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    r18 = np.clip(np.round((ipp[1] + rr * ps - ipp18[1]) / ps18).astype(int), 0, 511); c18 = np.clip(np.round((ipp[0] + cc * ps - ipp18[0]) / ps18).astype(int), 0, 511)
    return IMG, GMp[r18, c18], SOLp[r18, c18]
tmn = lambda h: np.arange(len(arr(h, "NUFFT", "g_curve"))) * float(h["REP"]) / 60
def exsp(a, h): E0, E1 = h["ex"]; a.axvspan(E0 * float(h["REP"]) / 60, E1 * float(h["REP"]) / 60, color="0.92", zorder=0)
for key, cmap, lo, hi, qname in (("dep", "RdBu_r", -60, 60, "PCr depletion (%)"), ("inc", "PuOr_r", -40, 40, "Pi increase (% of rest PCr)")):
    fig = plt.figure(figsize=(34, 22)); gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 1.45, 1.45], wspace=0.2, hspace=0.28)
    for i, b in enumerate(BOUTS):
        h = H[b]; N = int(h["N"]); IMG, gmL, solL = loc_show(h); rec = lambda d: (N - 1 - d // N) * N + d % N; gid, sid = PAIRS[b]; qg, qs = rec(gid), rec(sid)
        a = fig.add_subplot(gs[i, 0]); a.imshow(IMG, cmap="gray", interpolation="lanczos")
        a.contour(gmL.astype(float), [0.5], colors=L_GM, linewidths=2); a.contour(solL.astype(float), [0.5], colors=L_SOL, linewidths=2)
        Nn, x0, y0, cw, ch, box, _ = geom(h)
        for i_ in range(N + 1):
            a.plot([x0, x0 + N * cw], [y0 + i_ * ch] * 2, color="w", lw=0.3, alpha=0.35); a.plot([x0 + i_ * cw] * 2, [y0, y0 + N * ch], color="w", lw=0.3, alpha=0.35)
        for q in (qg, qs): pm = np.zeros(N * N, bool); pm[q] = True; outline(a, pm, N, x0, y0, cw, ch, "yellow", lw=1.6, ls="--")
        a.set_xlim(x0, x0 + N * cw); a.set_ylim(y0 + N * ch, y0); a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
        a.set_ylabel(f"MID00{b}\n{float(h['FOV']) / N:.0f} mm voxels", fontsize=15, fontweight="bold")
        if i == 0: a.set_title("localizer, drawn masks\n(phantom shown for display only)", fontsize=14, fontweight="bold")
        for j, meth in enumerate(("NUFFT", SP, HYB)):
            a = fig.add_subplot(gs[i, 1 + j]); compmap(a, h, IMG, arr(h, meth, key), cmap, lo, hi, (qg, qs))
            if i == 0: a.set_title(f"{LABM[meth]}\n{qname}", fontsize=14, fontweight="bold")
        li = list(h["leg_idx"]); t_ = tmn(h)
        for j, (q, nm, idd, col0) in enumerate(((qg, "GM", gid, L_GM), (qs, "SOL", sid, L_SOL))):
            a = fig.add_subplot(gs[i, 4 + j]); k_ = li.index(q)
            for meth in ("NUFFT", SP, HYB):
                a.plot(t_, arr(h, meth, "vox_pcr")[:, k_], "-", color=COLM[meth], lw=1.9, label=LABM[meth]); a.plot(t_, arr(h, meth, "vox_pi")[:, k_], "--", color=COLM[meth], lw=1.4)
            exsp(a, h); a.set_ylim(-0.1, 1.6); a.grid(alpha=0.3); a.set_ylabel("normalised to rest PCr", fontsize=11)
            if i == 3: a.set_xlabel("time (min)", fontsize=12)
            a.set_title(f"{nm} voxel {idd} — PCr depletion\nNUFFT {arr(h, 'NUFFT', 'dep')[q]:.0f}%, comp. SPICE {arr(h, SP, 'dep')[q]:.0f}%, hybrid {arr(h, HYB, 'dep')[q]:.0f}%",
                        loc="left", fontweight="bold", color=col0, fontsize=12)
            if i == 0: a.legend(frameon=False, fontsize=9.5, loc="upper right", title="solid PCr, dashed Pi", title_fontsize=9)
    cax = fig.add_axes([0.2, 0.065, 0.3, 0.008]); fig.colorbar(matplotlib.cm.ScalarMappable(norm=colors.Normalize(lo, hi), cmap=cmap), cax=cax, orientation="horizontal").set_label(qname, fontsize=13)
    fig.legend(handles=[Line2D([], [], color=L_GM, lw=3, label="gastrocnemius"), Line2D([], [], color=L_SOL, lw=3, label="soleus"),
                        Line2D([], [], color="yellow", lw=2, ls="--", label="voxels plotted on the right")], loc="upper left", ncol=3, fontsize=14, frameon=False, bbox_to_anchor=(0.13, 0.955))
    fig.suptitle(f"NUFFT vs compartment SPICE vs CC-SPICE (pixel-level masks, coil sensitivity, real CRT k-space; μ {MU_SHOW}) — {qname}", fontsize=17, fontweight="bold", y=0.985)
    nm_ = "hybrid_maps_PCr" if key == "dep" else "hybrid_maps_Pi"; fig.savefig(f"{OUTH}/{nm_}.png", dpi=95, bbox_inches="tight"); fig.savefig(f"{OUTH}/{nm_}.pdf", bbox_inches="tight"); plt.close(fig)
# ---------------- Fig B: validation and mu trade-off ----------------
def parse(h):
    true = float(re.search(r"true GM share .*?: (\d+)%", h["log"]).group(1)); out = {}
    for line in h["log"].splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 8 and parts[0] in h["names"]:
            nums = re.findall(r"-?\d+\.?\d*", parts[6]); out[parts[0]] = (float(nums[0]), float(nums[1])) if len(nums) >= 2 else (np.nan, np.nan)
    return true, out
fig, axs = plt.subplots(2, 3, figsize=(25, 13)); vox = [float(H[b]["FOV"]) / int(H[b]["N"]) for b in BOUTS]; x = np.arange(4)
a = axs[0, 0]; w = 0.2
for i, b in enumerate(BOUTS):
    true, P = parse(H[b])
    a.bar(i - 1.5 * w, true, w, color="k", label="true partial volume" if i == 0 else None)
    for j, meth in enumerate(("NUFFT", SP, HYB)):
        a.bar(i + (j - 0.5) * w, P[meth][0], w, color=COLM[meth], label=LABM[meth] if i == 0 else None)
        a.bar(i + (j - 0.5) * w, P[meth][1] - P[meth][0], w, bottom=P[meth][0], color="none", edgecolor=COLM[meth], hatch="///", lw=0, label="+ 3 mm GM boundary error" if (i == 0 and j == 0) else None)
a.set_xticks(x); a.set_xticklabels([f"{v:.0f} mm\nMID00{b}" for v, b in zip(vox, BOUTS)]); a.set_ylabel("GM share of the soleus-voxel signal (%)"); a.grid(alpha=0.3, axis="y")
a.legend(fontsize=9, frameon=False); a.set_title("a  Leakage test (independent 2 mm truth from the drawn masks)", loc="left", fontweight="bold")
MU = sorted(float(nm.split("μ ")[1]) for nm in H[BOUTS[0]]["names"] if nm.startswith("hybrid μ")); BCOL = {"279": "#8c564b", "281": "#1f77b4", "277": "#2ca02c", "283": "#d62728"}
SUMIDX = {"GM PCr depletion (%)": 0, "soleus PCr depletion (%)": 4, "soleus PCr noise per rep (%)": 5, "GM β-ATP change (%), should be 0": 3, "GM voxel PCr tSNR": 8}
for a, (ttl, ix) in zip(axs.ravel()[1:], SUMIDX.items()):
    for b in BOUTS:
        h = H[b]; y = [float(arr(h, f"hybrid μ {mu:g}", "summary")[ix]) for mu in MU]
        a.plot(MU, y, "o-", color=BCOL[b], lw=2, label=f"MID00{b} ({float(h['FOV']) / int(h['N']):.0f} mm)")
        a.axhline(float(arr(h, "NUFFT", "summary")[ix]), color=BCOL[b], ls="--", lw=1); a.axhline(float(arr(h, SP, "summary")[ix]), color=BCOL[b], ls=":", lw=1.3)
    if "β" in ttl: a.axhline(0, color="k", lw=0.8)
    a.set_xscale("log"); a.set_xlabel("μ (weight holding each pixel to its compartment; large = SLAM)"); a.grid(alpha=0.3); a.set_title(ttl, loc="left", fontweight="bold")
axs[0, 1].legend(fontsize=9, frameon=False); axs[0, 1].text(0.02, 0.02, "dashed = NUFFT, dotted = compartment SPICE (same bout)", transform=axs[0, 1].transAxes, fontsize=9)
fig.suptitle("CC-SPICE: leakage validation and the μ trade-off", fontsize=16, fontweight="bold"); fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{OUTH}/hybrid_validation.png", dpi=90); fig.savefig(f"{OUTH}/hybrid_validation.pdf"); plt.close(fig); print("saved", OUTH)
