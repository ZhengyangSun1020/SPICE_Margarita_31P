#!/usr/bin/env python3
"""Simulation (MID00283 geometry): do alternating outer rings + a low-rank-in-time (partial-separability) SPICE recon improve soleus separation
at the same number of rings per rep?  Uses the real CRT ring geometry, 80-ring coil sensitivity, drawn GM / soleus / other pixel masks (fine grid
6.7 mm), CSI-grid compartment SPICE graph (soft, offset 2, published lambda) and the real protocol (rest 1-4, exercise 5-26, 14 s / rep).
Truth: PCr amplitude per tissue pixel (spectral dimension collapsed to the PCr peak):
   GM pixels: 1 - d_p E(t), d_p = 0.30 ... 0.60 across the GM (heterogeneous), soleus 1 - 0.15 E(t), other 1; E(t) = on-kinetics 45 s, recovery 30 s.
Noise: complex white per k sample, scaled so the CURRENT-scheme NUFFT GM ROI has the measured per-rep noise (4.3 % of rest PCr).
Schemes (same 5 rings per rep):  CUR = rings 1-5 every rep;  ALT = rings 1-3 every rep + rings 4-5 (odd reps) / rings 6-7 (even reps, larger radii).
Recons: NUFFT; SPICE per rep (CSI grid); SPICE-LR = joint over all reps, x(t) = U phi(t), phi = first L right singular vectors of the inner-ring data,
U on the CSI grid with the same SPICE graph. ROI readout = phase-aligned mean of the GM / soleus CSI voxels (as the real analysis).
Reported over NMC noise realisations: GM and soleus depletion (truth: GM ROI-weighted mean, soleus 15 %), soleus per-rep error, leakage = share of the
GM-soleus depletion difference that the soleus estimate takes on. -> scratch log + /mnt/e/claude_tmp/study/paper_abstract/F_sim_ktshare.png"""
import os, sys
DS = sys.argv[1] if len(sys.argv) > 1 else "283"
os.environ.update(VFILE="VD12.npy", PIXMASK="1", LABELS="mask", MUS="30", HOUT="sim_kt")
S_ = "/mnt/c/Users/zsun/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-zsun-SPICE-Margarita/6aafd315-8b8b-4771-8d0e-7a49ffd29de0/scratchpad/hybrid_slam_spice.py"
src = open(S_).read(); sys.argv = ["x", DS]; exec(src[:src.index("RES = {}")])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(7); NMC = int(os.environ.get("NMC", 30)); L_RANK = int(os.environ.get("LRANK", 3))
# ---------------- ring geometry ----------------
kr = np.round(np.hypot(tj[:, 0], tj[:, 1]), 4); RAD = np.unique(kr); ring = np.searchsorted(RAD, kr)            # ring index 0..4
log(f"current ring radii (N-grid units, Nyquist 0.5): {np.round(RAD, 3).tolist()}; points per ring {[int((ring == i).sum()) for i in range(len(RAD))]}")
dR = RAD[-1] - RAD[-2]; EXTRA_R = [RAD[-1] + dR, RAD[-1] + 2 * dR]
def ring_pts(r_, n_, rot):
    a = 2 * np.pi * (np.arange(n_) + rot) / n_; return np.stack([r_ * np.cos(a), r_ * np.sin(a)], 1)
n5 = int((ring == len(RAD) - 1).sum())
TJ_CUR = tj[:, :2].astype(float)
TJ_B = np.vstack([tj[ring <= 2, :2]] + [ring_pts(r_, n5, 0.5) for r_ in EXTRA_R]).astype(float)                    # rings 1-3 + rings 6-7
log(f"ALT even reps: rings 1-3 + radii {np.round(EXTRA_R, 3).tolist()} (k_max {EXTRA_R[-1]:.3f} vs {RAD[-1]:.3f}: nominal resolution x{EXTRA_R[-1] / RAD[-1]:.2f})")
def ops(tj2):
    tj2 = tj2.astype(np.float32); Fh_ = hr_matrix(tj2); opc = mrinufft.get_operator("finufft", samples=tj2, shape=(N, N), n_coils=1, density=False)
    Fc_ = np.stack([opc.op(np.eye(N * N, dtype=np.complex64)[v].reshape(N, N)).ravel() for v in range(N * N)], 1).astype(np.complex128)
    r_ = np.hypot(tj2[:, 0], tj2[:, 1])
    d0 = np.asarray(m.compute_crt_dcf_hamming(tj2, hamming_radius=np.zeros(len(tj2))), float).ravel(); dh_ = np.asarray(m.compute_crt_dcf_hamming(tj2, hamming_radius=r_), float).ravel()
    return dict(A=Fh_[:, pix] * Ss[None], Fc=Fc_, dcf=d0, dcfh=dh_, inner=np.round(np.hypot(tj2[:, 0], tj2[:, 1]), 4) <= RAD[2] + 1e-4)
OPA, OPB = ops(TJ_CUR), ops(TJ_B)
# ---------------- truth ----------------
NRs = NR; tsec = (np.arange(NRs) + 0.5) * REP; t0, t1 = E0 * REP, E1 * REP
E = np.where(tsec < t0, 0, np.where(tsec < t1, 1 - np.exp(-(tsec - t0) / 45), 0)); Et1 = 1 - np.exp(-(t1 - t0) / 45)
E = np.where(tsec >= t1, Et1 * np.exp(-(tsec - t1) / 30), E)
xg = X.ravel()[pix]; gsel = Mc[:, 0] > 0; u = (xg - xg[gsel].min()) / (np.ptp(xg[gsel]) + 1e-9); dp = 0.30 + 0.30 * u
RHO = Mc[:, 0][None] * (1 - dp[None] * E[:, None]) + Mc[:, 1][None] * (1 - 0.15 * E[:, None]) + Mc[:, 2][None]       # (NR, nu)
# true ROI depletion of the GM ROI (signal-weighted over the GM tissue in GM voxels) and soleus 15 %
wgt = np.abs(Ss) * Mc[:, 0] * vGM[blk[pix]]; GM_TRUE = 100 * (wgt * dp).sum() / wgt.sum() * E[EX].mean() / 1.0
SOL_TRUE = 100 * 0.15 * E[EX].mean()
log(f"truth: GM ROI depletion {GM_TRUE:.1f}% (pixel range {100 * dp.min() * E[EX].mean():.0f}-{100 * dp.max() * E[EX].mean():.0f}%), soleus {SOL_TRUE:.1f}% (end-exercise window)")
# ---------------- SPICE pieces (CSI grid) ----------------
s0 = np.load(f"{O}/system.npz"); s11 = np.load(f"{O}/system11.npz"); G_h = s0["G"].astype(np.complex128)
comp = np.zeros(N * N); comp[legv] = 3; comp[vGM] = 1; comp[vSOL] = 2; comp = comp.reshape(N, N); legm = legv.reshape(N, N); ph = phant.reshape(N, N)
_, (ii, jj) = ndi.distance_transform_edt(~legm, return_indices=True); cmp_ = np.where(legm, comp, comp[ii, jj]); p_raw = s0["prior"]
lo, hi = p_raw[legm].min(), p_raw[legm].max(); pp = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1) + 2.0 * np.select([cmp_ == 1, cmp_ == 2], [0.0, 1.0], 2.0); pp[ph] = 0.0
WW, _ = m.calc_Bmatrix_simple(pp.astype(np.float32), wmax=5e4, adj=8, label_mask=np.where(ph, 0, 1).T)
dd = ndi.distance_transform_cdt(~legm, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2); lout[ph] = 1.0; lout = lout.ravel()
LAM = float(s11["lam"])
def gram(o): return (o["Fc"].conj().T * o["dcf"]) @ o["Fc"]
relA = np.real(np.diag(gram(OPA))).mean() / np.real(np.diag(G_h)).mean()
REG = LAM * relA * WW + np.diag(lout * relA)
for o in (OPA, OPB): o["Ai"] = np.linalg.inv(gram(o) + REG); o["P"] = o["Ai"] @ (o["Fc"].conj().T * o["dcf"]); o["Pn"] = o["Fc"].conj().T * o["dcfh"]
SCHEME = {"CUR": [OPA] * NRs, "ALT": [OPA if r % 2 == 0 else OPB for r in range(NRs)]}
# ROI weights: phase of the noise-free rest image (per method/scheme fixed)
def roi_read(Xv, ref):                                                                        # Xv: (NR, N^2) images; ref: (N^2,) rest image
    cg = np.exp(-1j * np.angle(ref)); g_ = np.real((Xv[:, vGM] * cg[vGM]).mean(1)); s_ = np.real((Xv[:, vSOL] * cg[vSOL]).mean(1)); return g_, s_
# noise calibration: CUR NUFFT GM ROI per-rep noise = 4.3 % of rest
y_rest = OPA["A"] @ RHO[0]; ref_n = OPA["Pn"] @ y_rest; cg = np.exp(-1j * np.angle(ref_n)); wG = (cg[vGM][:, None] * OPA["Pn"][vGM]).mean(0)
SIG = 0.043 * np.abs(np.real(wG @ y_rest)) / (np.linalg.norm(wG) / np.sqrt(2))
log(f"noise sigma per k sample {SIG:.3g} (CUR NUFFT GM ROI per-rep noise 4.3 %)")
def recon(scheme, method, Y):
    opsr = SCHEME[scheme]
    if method in ("NUFFT", "SPICE"):
        key = "Pn" if method == "NUFFT" else "P"; return np.stack([opsr[r][key] @ Y[r] for r in range(NRs)])
    # SPICE-LR: temporal basis from inner-ring data (present in every rep, same samples in both schemes)
    Yin = np.stack([Y[r][opsr[r]["inner"]] for r in range(NRs)], 1)                            # n_inner x NR
    _, _, Vh = np.linalg.svd(Yin, full_matrices=False); PHI = Vh[:L_RANK]                       # L x NR (complex)
    if os.environ.get("PHI_TRUE") == "1":                                                       # oracle: true temporal basis {1, E(t)}
        PHI = np.linalg.qr(np.stack([np.ones(NRs), E], 1))[0].T.astype(complex)
    elif os.environ.get("PHI_SMOOTH") == "1":                                                   # basis from inner data after temporal mean removal + smoothing
        Yc = Yin - Yin[:, R0:R1].mean(1, keepdims=True); Ys = np.zeros_like(Yc)
        for a_, b_ in ((0, E0), (E0, E1), (E1, NRs)):
            nk = min(5, b_ - a_); k_ = np.ones(nk) / nk; den = np.convolve(np.ones(b_ - a_), k_, "same")[:b_ - a_]
            Ys[:, a_:b_] = np.array([np.convolve(row, k_, "same")[:b_ - a_] / den for row in Yc[:, a_:b_]])
        _, _, Vh2 = np.linalg.svd(Ys, full_matrices=False); PHI = np.linalg.qr(np.vstack([np.ones(NRs), Vh2[:L_RANK - 1]]).T)[0].T.astype(complex)
    nv = N * N; H = np.kron(np.eye(L_RANK), REG).astype(complex); b = np.zeros(nv * L_RANK, complex)
    for r in range(NRs):
        o = opsr[r]; Gr = (o["Fc"].conj().T * o["dcf"]) @ o["Fc"]; ph_ = PHI[:, r]
        H += np.kron(np.outer(ph_.conj(), ph_), Gr); b += np.kron(ph_.conj(), (o["Fc"].conj().T * o["dcf"]) @ Y[r])
    U = np.linalg.solve(H, b).reshape(L_RANK, nv).T                                            # nv x L
    return (U @ PHI).T                                                                          # NR x nv
def depl(c): r0 = c[R0:R1].mean(); return 100 * (1 - c[EX].mean() / r0), c / r0
truth_s = 1 - 0.15 * E; RUNS = [("CUR", "NUFFT"), ("CUR", "SPICE"), ("CUR", "SPICE-LR"), ("ALT", "SPICE"), ("ALT", "SPICE-LR")]
OUT = {k: dict(gd=[], sd=[], serr=[], gerr=[], scurve=[], gcurve=[]) for k in RUNS}
Y0 = {sc: [SCHEME[sc][r]["A"] @ RHO[r] for r in range(NRs)] for sc in SCHEME}
if os.environ.get("DEBUG") == "1":
    for sc in SCHEME:
        Yin = np.stack([Y0[sc][r][SCHEME[sc][r]["inner"]] for r in range(NRs)], 1); sv = np.linalg.svd(Yin, compute_uv=False)
        _, _, Vh = np.linalg.svd(Yin, full_matrices=False); PHI = Vh[:L_RANK]; Pp = PHI.conj().T @ PHI
        resid = np.linalg.norm(RHO.T - RHO.T @ Pp.T) / np.linalg.norm(RHO.T)
        log(f"[debug {sc}] inner-data singular values {np.round(sv[:5] / sv[0], 4).tolist()}; truth outside span(phi): {100 * resid:.2f}%")
        for mth in ("SPICE", "SPICE-LR"):
            Xv = recon(sc, mth, Y0[sc]); ref = Xv[R0:R1].mean(0); g_, s_ = roi_read(Xv, ref); log(f"[debug {sc}] noise-free {mth}: GM {depl(g_)[0]:.1f}%  soleus {depl(s_)[0]:.1f}%")
    sys.exit(0)
for it in range(NMC):
    for sc in SCHEME:
        Y = [y + SIG / np.sqrt(2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y))) for y in Y0[sc]]
        for (s2, mth) in RUNS:
            if s2 != sc: continue
            Xv = recon(sc, mth, Y); ref = recon(sc, mth, Y0[sc])[R0:R1].mean(0) if it == 0 and "ref_" + sc + mth not in OUT else OUT.get("ref_" + sc + mth)
            if it == 0: OUT["ref_" + sc + mth] = ref
            g_, s_ = roi_read(Xv, OUT["ref_" + sc + mth]); gd, gc = depl(g_); sd, scv = depl(s_)
            d = OUT[(sc, mth)]; d["gd"].append(gd); d["sd"].append(sd); d["serr"].append(np.std(scv - truth_s)); d["scurve"].append(scv); d["gcurve"].append(gc)
log(f"\n{'scheme':6s} {'recon':9s} | GM depletion (truth {GM_TRUE:.0f}%) | soleus depletion (truth {SOL_TRUE:.0f}%) | soleus per-rep error | leakage GM->soleus")
for k in RUNS:
    d = OUT[k]; gd, sd = np.mean(d["gd"]), np.mean(d["sd"]); leak = 100 * (sd - SOL_TRUE) / (GM_TRUE - SOL_TRUE)
    log(f"{k[0]:6s} {k[1]:9s} | {gd:5.1f} ± {np.std(d['gd']):4.1f}            | {sd:5.1f} ± {np.std(d['sd']):4.1f}                | {100 * np.mean(d['serr']):5.1f} %             | {leak:5.0f} %")
# ---------------- figure ----------------
tm = tsec / 60; fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3], wspace=0.25, left=0.04, right=0.99, top=0.85, bottom=0.12))
a = axs[0]
for rr, c, lab in ((RAD, "0.3", "current rings 1-5 (every rep)"), (EXTRA_R, "#d62728", "added rings 6-7 (even reps, replace 4-5)")):
    for r_ in rr: a.add_patch(plt.Circle((0, 0), r_, fill=False, color=c, lw=2, ls="-" if c == "0.3" else "--"))
a.plot([], [], color="0.3", lw=2, label="rings 1-5"); a.plot([], [], color="#d62728", lw=2, ls="--", label="rings 6-7 (alternate reps)")
a.set_xlim(-0.5, 0.5); a.set_ylim(-0.5, 0.5); a.set_aspect("equal"); a.legend(fontsize=10, loc="upper right"); a.set_title("k-space: same number of rings per rep", fontsize=12)
COL = {("CUR", "NUFFT"): "0.55", ("CUR", "SPICE"): "#1b9e77", ("CUR", "SPICE-LR"): "#66a61e", ("ALT", "SPICE"): "#e6ab02", ("ALT", "SPICE-LR"): "#d62728"}
for j, (key, ttl, tr) in enumerate((("scurve", "soleus ROI PCr (mean of %d noise realisations)" % NMC, truth_s), ("gcurve", "GM ROI PCr", None))):
    a = axs[1 + j]; a.axvspan(t0 / 60, t1 / 60, color="0.93")
    for k in RUNS:
        cm_ = np.mean(OUT[k][key], 0); sd_ = np.std(OUT[k][key], 0); a.plot(tm, cm_, color=COL[k], lw=2, label=f"{k[0]} {k[1]}"); a.fill_between(tm, cm_ - sd_, cm_ + sd_, color=COL[k], alpha=0.12)
    if tr is not None: a.plot(tm, tr, "k--", lw=1.5, label="truth")
    a.set_ylim(0.3, 1.15); a.grid(alpha=0.3); a.set_xlabel("time (min)"); a.set_ylabel("relative to rest"); a.set_title(ttl, fontsize=12)
axs[1].legend(fontsize=9, loc="lower left", ncol=2)
fig.suptitle(f"Simulation, MID00{DS} geometry: alternating outer rings + low-rank-in-time SPICE (band = ±1 SD over noise realisations)", fontsize=13, fontweight="bold")
fig.savefig("/mnt/e/claude_tmp/study/paper_abstract/F_sim_ktshare.png", dpi=110); log("saved figure")
