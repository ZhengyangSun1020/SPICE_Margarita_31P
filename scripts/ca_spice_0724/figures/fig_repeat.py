#!/usr/bin/env python3
"""Abstract Fig 4: repeatability across the four 24-07 bouts (same leg, 20-30 mm voxels, 10-14 s/rep), NUFFT vs CA-SPICE.
Core ROIs: GM / soleus voxels not excluded by the rule in excluded() (noisy, < 50 % muscle, or next to the phantom); same voxels for both methods and Fig 3.
ROI spectrum = rest-PCr-phase-aligned mean of the core voxels; magnitude, 8 Hz LB (as Figs 2 and 3).
Time courses: per rep, noise-power-corrected magnitude (exact ROI noise variance, lin_quant.py); PCr = PCr peak height / mean rest PCr peak;
  Pi = change of the Pi peak height (3.8-5.4 ppm) from its rest mean, in units of rest PCr; 5-rep moving mean within rest / exercise / recovery.
Summary (right): from the AVERAGED ROI spectra, exactly as Figs 2 / 3: rest = mean of rest reps, end of exercise = mean of the last 4 reps fully inside
  exercise; PCr depletion = 100 (1 - PCr_ex / PCr_rest), Pi increase = 100 (Pi_ex - Pi_rest) / PCr_rest (peak heights). -> paper_abstract/F4_repeatability.png/.pdf"""
import os, sys, numpy as np, matplotlib; sys.path.insert(0, "/mnt/e/claude_tmp/study"); import lin_quant as lq; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
os.environ.setdefault("HDIR", "spice_final"); os.environ.setdefault("MU_SHOW", "30")
src = open("/mnt/e/claude_tmp/study/hybrid_fig.py").read(); exec(src[:src.index("for key, cmap, lo, hi, qname in")])
OUTF = "/mnt/e/claude_tmp/study/paper_abstract"
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
BO = ("283", "277", "281", "279"); BC = {"283": "#222222", "277": "#d62728", "281": "#e7298a", "279": "#a6761d"}; C_N, C_C = "#f1a340", "#7b3294"
METHS = ("NUFFT", "CA-SPICE"); DW, F0 = 0.00036, 49.8946; NSM = int(os.environ.get("SMOOTH", 5)); QC = float(os.environ.get("QC_NOISE", 25)) / 100
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
def smooth(y, E0, E1):
    out = np.array(y, float).copy(); k = np.ones(NSM)
    for s0, s1 in ((0, E0), (E0, E1), (E1, len(y))):
        seg = np.asarray(y[s0:s1], float); out[s0:s1] = (np.convolve(seg, k, "same") / np.convolve(np.ones(len(seg)), k, "same"))[:len(seg)]
    return out
def place_labels(a, items, gap, x=31.0):
    items = sorted(items); ys = [it[0] for it in items]
    for k in range(1, len(ys)): ys[k] = max(ys[k], ys[k - 1] + gap)
    for (_, txt, col), y in zip(items, ys): a.text(x, y, txt, fontsize=9.5, color=col, va="center")
def roi(B, meth, qs, LO):
    h = H[B]; R0, R1 = map(int, h["rest"]); E1 = int(h["ex"][1])
    X = np.load(f"{ROOT}/output_study_mid{B}/" + ("adj_fid.npy" if meth == "NUFFT" else "spice_final/fid_spice.npy"))[:, qs]
    n = X.shape[-1]; t = np.arange(n) * DW; apo = np.exp(-np.pi * 8 * t); Sx = lq.spec(X, apo)                                  # (NR, nq, n)
    ppm_raw = -np.fft.fftshift(np.fft.fftfreq(n, DW)) / F0; k0 = np.flatnonzero(np.abs(ppm_raw) < 1.5)[np.argmax(np.abs(Sx[R0:R1].mean(0)).sum(0)[np.abs(ppm_raw) < 1.5])]
    Q = ppm_raw - ppm_raw[k0]; bP, bPi = np.abs(Q) < 1, (Q >= 3.8) & (Q <= 5.4)
    c = Sx[R0:R1].mean(0)[:, k0]; w = c.conj() / np.abs(c).sum(); S = (Sx * w[None, :, None]).sum(1)                           # phase-aligned ROI spectrum per rep
    var = lq.noise_var_spec(LO, meth, np.zeros(LO["PN"].shape[0], complex) if False else _wfull(w, qs, LO), apo=apo)            # exact ROI noise variance (1 rep)
    M = lq.denoised_mag(S, var[None, :]); r0 = M[R0:R1][:, bP].max(1).mean()
    pcr = M[:, bP].max(1) / r0; pi = (M[:, bPi].max(1) - M[R0:R1][:, bPi].max(1).mean()) / r0
    Sr, Se = np.abs(S[R0:R1].mean(0)), np.abs(S[E1 - 5:E1 - 1].mean(0)); a0 = Sr[bP].max()                                      # averaged spectra (as Figs 2/3)
    return pcr, pi, 100 * (1 - Se[bP].max() / a0), 100 * (Se[bPi].max() - Sr[bPi].max()) / a0
def _wfull(w, qs, LO):
    wf = np.zeros(LO["PN"].shape[0], complex); wf[qs] = w; return wf
R = {}; vox = {B: float(H[B]["FOV"]) / int(H[B]["N"]) for B in BO}
for B in BO:
    h = H[B]; leg = h["legv"].astype(bool); li = list(h["leg_idx"]); R0, R1 = map(int, h["rest"]); rep = float(h["REP"]); cc = arr(h, SP, "vox_pcr"); NR = cc.shape[0]
    quiet = np.r_[R0:R1, NR - int(round(144 / rep)):NR]; LO = lq.load_ops(B)
    for mus, msk in (("GM", h["vGM"] & leg), ("soleus", h["vSOL"] & leg)):
        EXC = excluded(B); qs = [q for q in np.flatnonzero(msk) if q not in EXC]
        for meth in METHS: R[(B, meth, mus)] = roi(B, meth, qs, LO)
fig = plt.figure(figsize=(21, 10.8)); g = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.72], wspace=0.2, hspace=0.3, left=0.05, right=0.985, top=0.88, bottom=0.07)
for j, meth in enumerate(METHS):
    for i, (mus, key) in enumerate((("gastrocnemius", "GM"), ("soleus", "soleus"))):
        a = fig.add_subplot(g[i, j]); a.axvspan(0, 5, color="0.93", zorder=0); a.axhline(0, color="0.6", lw=0.8)
        for B in BO:
            h = H[B]; rep = float(h["REP"]); E0, E1 = map(int, h["ex"]); pc, pi = R[(B, meth, key)][:2]; tt = (np.arange(len(pc)) - E0) * rep / 60
            a.plot(tt, smooth(pc, E0, E1), color=BC[B], lw=2.2); a.plot(tt, smooth(pi, E0, E1), color=BC[B], lw=1.5, ls="--")
        a.set_xlim(-1.05, 11.2); a.set_ylim(-0.22, 1.3); a.grid(alpha=0.3)
        if i == 1: a.set_xlabel("time from exercise onset (min)")
        a.set_title(f"{meth}: {mus} ROI", fontsize=13.5, color=C_N if meth == "NUFFT" else C_C); a.text(2.5, 1.22, "exercise", ha="center", fontsize=10, color="0.45")
        if j == 0: a.set_ylabel("PCr, and Pi change from rest\n(relative to rest PCr)")
xs = np.array([vox[B] for B in BO]); o = np.argsort(xs)
for k, (idx, ttl, ylab, yl, gap, sgn) in enumerate(((2, "End-exercise PCr depletion", "PCr depletion (%)", (0, 70), 6.5, False), (3, "End-exercise Pi increase", "Pi increase (% of rest PCr)", (-10, 45), 4.5, True))):
    a = fig.add_subplot(g[k, 2]); LB = []; a.axhline(0, color="0.6", lw=0.8)
    for meth, c_, mk in (("NUFFT", C_N, "o"), ("CA-SPICE", C_C, "s")):
        for mus, ls, fill in (("GM", "-", True), ("soleus", "--", False)):
            ys = np.array([R[(B, meth, mus)][idx] for B in BO])
            a.plot(xs[o], ys[o], ls=ls, color=c_, lw=1.8, marker=mk, ms=7.5, mfc=c_ if fill else "white", mec=c_, mew=1.8)
            LB.append((ys[o][-1], f"{mus} {ys.mean():{'+' if sgn else ''}.0f}±{ys.std(ddof=1):.0f}", c_))
    place_labels(a, LB, gap); a.set_xlim(19, 35.5); a.set_ylim(*yl); a.set_xticks(sorted(set(xs))); a.set_ylabel(ylab); a.grid(alpha=0.3)
    a.set_title(f"{ttl}: GM vs soleus", fontsize=12.5, loc="left")
    if k == 1: a.set_xlabel("voxel size (mm)")
    else: a.set_xticklabels([])
fig.legend(handles=[Line2D([], [], color=BC[B], lw=2.4, label=f"MID00{B}: {vox[B]:.0f} mm, {float(H[B]['REP']):.0f} s/rep") for B in BO] +
           [Line2D([], [], color="0.3", lw=2.2, label="PCr"), Line2D([], [], color="0.3", lw=1.5, ls="--", label="Pi change")],
           loc="upper center", ncol=3, fontsize=11, frameon=True, edgecolor="0.7", bbox_to_anchor=(0.36, 0.995))
fig.legend(handles=[Line2D([], [], color=C_N, marker="o", ls="-", label="NUFFT"), Line2D([], [], color=C_C, marker="s", ls="-", label="CA-SPICE"),
                    Line2D([], [], color="0.3", marker="o", mfc="0.3", ls="-", label="GM"), Line2D([], [], color="0.3", marker="o", mfc="white", ls="--", label="soleus")],
           loc="upper center", ncol=2, fontsize=11, frameon=True, edgecolor="0.7", bbox_to_anchor=(0.86, 0.995))
fig.savefig(f"{OUTF}/F4_repeatability.png", dpi=120); fig.savefig(f"{OUTF}/F4_repeatability.pdf"); print("saved")
for meth in METHS:
    for mus in ("GM", "soleus"):
        d = np.array([R[(B, meth, mus)][2:] for B in BO]); print(f"  {meth:8s} {mus:6s} depletion {np.round(d[:, 0]).astype(int).tolist()} -> {d[:, 0].mean():.0f}±{d[:, 0].std(ddof=1):.0f} | Pi increase {np.round(d[:, 1]).astype(int).tolist()} -> {d[:, 1].mean():+.0f}±{d[:, 1].std(ddof=1):.0f}")
