#!/usr/bin/env python3
"""Abstract Fig 3: MID00283 (20 mm voxels) PCr depletion and Pi increase maps, NUFFT vs CA-SPICE (2 x 2), GM + soleus voxels on the localizer.
Values from averaged spectra (as Fig 2): rest = mean of rest reps, end of exercise = mean of the last 4 reps fully inside exercise; magnitude, 8 Hz LB,
peak heights; depletion = 100 (1 - PCr_ex / PCr_rest), Pi increase = 100 (Pi_ex - Pi_rest) / PCr_rest (Pi peak 3.8-5.4 ppm).
Not shown: voxels excluded by the rule of Fig 4 (noisy, < 50 % muscle, or next to the phantom); same set for both methods.
Dashed: the Fig 2 voxels (GM 64 / soleus 54). -> paper_abstract/F3_maps_MID00283.png/.pdf"""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.patches import Patch
os.environ.setdefault("HDIR", "spice_final"); os.environ.setdefault("MU_SHOW", "30")
src = open("/mnt/e/claude_tmp/study/hybrid_fig.py").read(); exec(src[:src.index("for key, cmap, lo, hi, qname in")])
OUTF = "/mnt/e/claude_tmp/study/paper_abstract"; SPD = os.environ["HDIR"]
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "font.family": "DejaVu Sans"})
import json as _json
_LAB = _json.load(open("/mnt/e/claude_tmp/study/masks_hr/mask_labels.json"))
def excluded(B, frac_min=0.5):
    """Voxel exclusion (same rule in Figs 3 and 4): CA-SPICE PCr rep-to-rep noise in rest + late recovery >= 25 %, OR muscle area fraction < 50 %,
    OR 8-neighbour of a phantom voxel (signal blurred in from the reference phantom)."""
    h = H[B]; N = int(h["N"]); leg = h["legv"].astype(bool); vg, vs = h["vGM"] & leg, h["vSOL"] & leg; li = list(h["leg_idx"])
    R0, R1 = map(int, h["rest"]); rep = float(h["REP"]); cc = arr(h, SP, "vox_pcr"); NR = cc.shape[0]; quiet = np.r_[R0:R1, NR - int(round(144 / rep)):NR]
    ph = (np.load(f"{ROOT}/output_study_mid{B}/reg_TRUE.npy") == 2).reshape(N, N); fS, fG = np.array(_LAB[B]["fS"]), np.array(_LAB[B]["fG"])
    disp = lambda q: (N - 1 - q // N) * N + q % N; out = set()
    for q in np.flatnonzero(vg | vs):
        d = disp(q); r, c = divmod(q, N)
        if (np.std(np.diff(cc[quiet, li.index(q)])) / np.sqrt(2) >= 0.25 or (fG if vg[q] else fS)[d // N][d % N] < frac_min
                or any(0 <= r + dr < N and 0 <= c + dc < N and ph[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1))): out.add(q)
    return out
STK = [patheffects.withStroke(linewidth=2, foreground="k")]; C_N, C_C = "#b35806", "#7b3294"
B = os.environ.get("BOUT", "283"); GV, SV = 64, 54
h = H[B]; N = int(h["N"]); IMG, gmL, solL = loc_show(h); _, x0, y0, cw, ch, box, ext = geom(h); leg = h["legv"].astype(bool); vg, vs = h["vGM"] & leg, h["vSOL"] & leg
flip = lambda d: (N - 1 - d // N) * N + d % N; R0, R1 = map(int, h["rest"]); E0, E1 = map(int, h["ex"]); rep = float(h["REP"]); li = list(h["leg_idx"]); O = f"{ROOT}/output_study_mid{B}"
cc = arr(h, SP, "vox_pcr"); NR = cc.shape[0]; quiet = np.r_[R0:R1, NR - int(round(144 / rep)):NR]
BAD = excluded(B); SHOW = np.array([q for q in np.flatnonzero(vg | vs) if q not in BAD]); vgk = vg.copy(); vsk = vs.copy(); vgk[list(BAD)] = False; vsk[list(BAD)] = False
def avg_maps(X):
    n = X.shape[-1]; t = np.arange(n) * 0.00036; apo = np.exp(-np.pi * 8 * t); mag = lambda x: np.abs(np.fft.fftshift(np.fft.fft(np.conj(x * apo), axis=-1), axes=-1))
    Sr, Se = mag(X[R0:R1].mean(0)), mag(X[E1 - 5:E1 - 1].mean(0)); ppm_raw = -np.fft.fftshift(np.fft.fftfreq(n, 0.00036)) / 49.8946
    k0 = np.flatnonzero(np.abs(ppm_raw) < 1.5)[np.argmax(Sr[vg].sum(0)[np.abs(ppm_raw) < 1.5])]; Q = ppm_raw - ppm_raw[k0]; bP, bPi = np.abs(Q) < 1, (Q >= 3.8) & (Q <= 5.4)
    r0 = Sr[:, bP].max(1); return dict(dep=100 * (1 - Se[:, bP].max(1) / r0), inc=100 * (Se[:, bPi].max(1) - Sr[:, bPi].max(1)) / r0)
VAL = {"NUFFT": avg_maps(np.load(f"{O}/adj_fid.npy")), "CA-SPICE": avg_maps(np.load(f"{O}/{SPD}/fid_spice.npy"))}
MAPS = (("dep", "PCr depletion (%)", "RdBu_r", -60, 60), ("inc", "Pi increase (% of rest PCr)", "PuOr_r", -40, 40))
rr = [N - 1 - q // N for q in SHOW]; cl = [q % N for q in SHOW]
fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.6), gridspec_kw=dict(wspace=0.04, hspace=0.2, left=0.1, right=0.9, top=0.86, bottom=0.12))
for i, (key, ttl, cmap, lo, hi) in enumerate(MAPS):
    for j, meth in enumerate(("NUFFT", "CA-SPICE")):
        a = axs[i, j]; a.imshow(IMG, cmap="gray", interpolation="lanczos"); nrm = colors.Normalize(lo, hi); cm_ = matplotlib.colormaps[cmap]; v = VAL[meth][key]
        for q in SHOW:
            l, tp = box(q)
            a.add_patch(plt.Rectangle((l, tp), cw, ch, fc=cm_(nrm(v[q])), ec="none", alpha=0.9))
            a.text(l + cw / 2, tp + ch / 2, f"{v[q]:.0f}", ha="center", va="center", fontsize=11, color="w", fontweight="bold", path_effects=STK)
        outline(a, vgk, N, x0, y0, cw, ch, L_GM, lw=2.4); outline(a, vsk, N, x0, y0, cw, ch, L_SOL, lw=2.4)
        for d_, c_ in ((GV, "#7CFC00"), (SV, "#4fc3ff")): pm = np.zeros(N * N, bool); pm[flip(d_)] = True; outline(a, pm, N, x0, y0, cw, ch, c_, lw=2.6, ls="--")
        a.set_xlim(x0 + (min(cl) - 0.5) * cw, x0 + (max(cl) + 1.5) * cw); a.set_ylim(y0 + (max(rr) + 1.5) * ch, y0 + (min(rr) - 0.5) * ch)
        a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
        if i == 0: a.set_title(meth, fontsize=14, color=C_N if meth == "NUFFT" else C_C)
        if j == 0: a.set_ylabel(ttl, fontsize=12.5, fontweight="bold")
    pos = axs[i, 1].get_position(); cax = fig.add_axes([pos.x1 + 0.006, pos.y0 + 0.01, 0.012, pos.height - 0.02])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=colors.Normalize(lo, hi), cmap=cmap), cax=cax).ax.tick_params(labelsize=9)
fig.legend(handles=[Patch(fc="none", ec=L_GM, lw=2.4, label="gastrocnemius"), Patch(fc="none", ec=L_SOL, lw=2.4, label="soleus"),
                    Patch(fc="none", ec="#7CFC00", lw=2.4, ls="--", label="GM voxel 64 (Fig 2)"),
                    Patch(fc="none", ec="#4fc3ff", lw=2.4, ls="--", label="soleus voxel 54 (Fig 2)")], loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.54, 0.0))
fig.text(0.54, 0.93, f"MID00{B}: {float(h['FOV']) / N:.0f} mm voxels, {rep:.0f} s/rep — end of exercise vs rest", ha="center", fontsize=13.5, fontweight="bold")
fig.savefig(f"{OUTF}/F3_maps_MID00{B}.png", dpi=130); fig.savefig(f"{OUTF}/F3_maps_MID00{B}.pdf")
disp = lambda q: (N - 1 - q // N) * N + q % N
for meth in VAL:
    for mus, msk in (("GM", vg), ("soleus", vs)):
        qs = [q for q in np.flatnonzero(msk) if q not in BAD]
        print(f"{meth:8s} {mus:6s} (n={len(qs)}): median depletion {np.median(VAL[meth]['dep'][qs]):.0f}%, median Pi increase {np.median(VAL[meth]['inc'][qs]):+.0f}")
print("excluded voxels:", sorted(disp(q) for q in BAD))
