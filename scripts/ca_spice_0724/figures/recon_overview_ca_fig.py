#!/usr/bin/env python3
"""Abstract Fig 1 (MID00283 example): CA-SPICE (compartment-aware SPICE) pipeline, framed modules + block arrows, light theme.
(a) Acquisition | (b) Prior information: spatial (muscle masks -> CSI-voxel compartment labels -> compartment-aware smoothing graph on the 80-ring
PCr guidance image, edges across muscle boundaries cut) + spectral (simulated 31P spectra -> SVD -> basis) | (c) CA-SPICE reconstruction |
(d) Dynamic MRSI output (rest PCr map, rest vs end-exercise spectrum of one GM voxel, tilted waterfall).
Data: output_study_mid283/spice_final (CA-SPICE: CSI grid, hard compartment boundary, edited voxel labels, rank 12).
-> paper_abstract/F1_recon_overview_CA.png/.pdf"""
import os, sys, numpy as np, matplotlib; sys.path.insert(0, "/mnt/e/claude_tmp/study"); import lin_quant as lq; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch, FancyArrow, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
os.environ.setdefault("HDIR", "spice_final"); os.environ.setdefault("MU_SHOW", "30")
src = open("/mnt/e/claude_tmp/study/hybrid_fig.py").read(); exec(src[:src.index("for key, cmap, lo, hi, qname in")])
OUTF = "/mnt/e/claude_tmp/study/paper_abstract"; SPD = os.environ["HDIR"]
plt.rcParams.update({"font.size": 10.5, "axes.titleweight": "bold", "font.family": "DejaVu Sans", "axes.titlesize": 10.5, "axes.titlepad": 3})
B = "283"; h = H[B]; N = int(h["N"]); IMG, gmL, solL = loc_show(h); _, x0, y0, cw, ch, box, ext = geom(h); leg = h["legv"].astype(bool); vg, vs = h["vGM"] & leg, h["vSOL"] & leg
flip = lambda d: (N - 1 - d // N) * N + d % N; GV = int(os.environ.get("GMVOX", 65)); qg = flip(GV)
U = dict(np.load(f"{ROOT}/output_study_mid{B}/hybrid_v2_unc/unc_MID00{B}.npz"))                 # raw k-t magnitude, trajectory, spectral basis (first columns = rank-12 basis)
C = np.load(f"{ROOT}/output_study_mid{B}/{SPD}/spice_constraints.npz")
X = np.load(f"{ROOT}/output_study_mid{B}/{SPD}/fid_spice.npy"); NR, _, n = X.shape
DW, F0 = 0.00036, 49.8946; t = np.arange(n) * DW; R0, R1 = map(int, h["rest"]); E0, E1 = map(int, h["ex"])
APO = np.exp(-np.pi * 6 * t); mag = lambda x: np.abs(np.fft.fftshift(np.fft.fft(np.conj(x * APO), axis=-1), axes=-1))
cs = lambda x: np.fft.fftshift(np.fft.fft(np.conj(x * APO), axis=-1), axes=-1); LO = lq.load_ops(B)
VAR = float(LO["sig2"]) * np.sum(np.abs(lq.op(LO, "CA-SPICE"))**2, 1)[:, None] * np.sum(np.abs(lq.basis_spectra(LO, "CA-SPICE", APO))**2, 0)[None, :]   # noise var per voxel, freq (1 rep)
dmag = lambda x, nav, v: lq.denoised_mag(cs(x), v / nav)                                     # noise-power subtracted magnitude
ppm_raw = -np.fft.fftshift(np.fft.fftfreq(n, DW)) / F0; k0 = int(np.argmax(mag(X[R0:R1, qg].mean(0)) * (np.abs(ppm_raw) < 1.5))); ppm = ppm_raw - ppm_raw[k0]
REPS = [float(H[b]["REP"]) for b in BOUTS]; VOXMM = [float(H[b]["FOV"]) / int(H[b]["N"]) for b in BOUTS]
C_G, C_S, C_O = L_GM, L_SOL, "#bdbdbd"
fig = plt.figure(figsize=(21, 6.0)); fig.patch.set_facecolor("white")
def module(x, y, w, hgt, title, col):
    fig.add_artist(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.004,rounding_size=0.012", fc=col, ec="0.6", lw=1.0, transform=fig.transFigure, zorder=-5))
    fig.text(x + 0.008, y + hgt - 0.012, title, fontsize=12.5, fontweight="bold", va="top")
def arrow(x, y, dx):
    fig.add_artist(FancyArrow(x, y, dx, 0, width=0.03, head_width=0.06, head_length=0.012, length_includes_head=True, transform=fig.transFigure, fc="#6baed6", ec="none", zorder=5))
def noax(a): a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
def locframe(a): a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_xticks([]); a.set_yticks([])
TOP, BOT = 0.965, 0.03; HG = TOP - BOT
module(0.008, BOT, 0.142, HG, "(a) Acquisition", "#f7f7f7"); module(0.163, BOT, 0.268, HG, "(b) Prior information", "#f7f7f7")
module(0.457, BOT, 0.088, HG, "", "#deebf7"); module(0.571, BOT, 0.421, HG, "(d) Dynamic MRSI output", "#f7f7f7")
arrow(0.150, 0.5, 0.0125); arrow(0.431, 0.5, 0.0255); arrow(0.545, 0.5, 0.0255)
# ---------- (a) acquisition ----------
a = fig.add_axes([0.03, 0.62, 0.11, 0.26]); kt = np.log10(U["kt_abs"][:, :300] + 1e-30)
a.imshow(kt, aspect="auto", cmap="magma", vmin=np.percentile(kt, 20), vmax=np.percentile(kt, 99.9)); a.set_xticks([]); a.set_yticks([])
a.set_title("raw k–t data"); a.set_xlabel("time", fontsize=9, labelpad=1); a.set_ylabel("k-space", fontsize=9, labelpad=1)
a = fig.add_axes([0.035, 0.32, 0.10, 0.23]); tj = U["traj"]; a.scatter(tj[:, 0], tj[:, 1], s=3, c=np.hypot(tj[:, 0], tj[:, 1]), cmap="viridis")
a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title("2D CRT trajectory")
a = fig.add_axes([0.02, 0.175, 0.12, 0.04])
for s0, s1, col, lab in ((0, 1, "0.82", ""), (1, 6, "#fcbba1", "exercise"), (6, 12, "#c6dbef", "recovery")):
    a.barh(0, s1 - s0, left=s0, color=col); a.text((s0 + s1) / 2, 0, lab, ha="center", va="center", fontsize=9)
a.text(0.5, 0.45, "rest", ha="center", va="bottom", fontsize=8.5, color="0.3")
a.set_xlim(0, 12); a.set_ylim(-0.4, 0.4); a.set_yticks([]); a.set_xticks([0, 6, 12]); a.tick_params(labelsize=8, length=2, pad=1)
[a.spines[s].set_visible(False) for s in ("top", "right", "left")]; a.set_xlabel("min", fontsize=8.5, labelpad=0); a.set_title("dynamic exercise protocol", fontsize=10, pad=14)
fig.text(0.079, 0.105, f"time resolution: {min(REPS):.0f}–{max(REPS):.0f} s\nvoxel size: {min(VOXMM):.0f}–{max(VOXMM):.0f} mm", ha="center", va="top", fontsize=9.5, color="0.2", linespacing=1.35)
# ---------- (b) prior information ----------
fig.text(0.169, 0.72, "spatial", fontsize=10.5, color="0.35", style="italic", rotation=90, va="center", ha="center"); fig.text(0.169, 0.27, "spectral", fontsize=10.5, color="0.35", style="italic", rotation=90, va="center", ha="center")
a = fig.add_axes([0.175, 0.615, 0.078, 0.225]); a.imshow(IMG, cmap="gray", interpolation="lanczos")
a.contour(gmL.astype(float), [0.5], colors=C_G, linewidths=1.5); a.contour(solL.astype(float), [0.5], colors=C_S, linewidths=1.5); locframe(a); a.set_title("muscle\nmasks")
cmp_, legm, ph, pp = C["cmp_"].ravel(), C["legm"].ravel(), C["ph"].ravel(), C["pp"].reshape(N, N)
a = fig.add_axes([0.262, 0.615, 0.078, 0.225]); a.imshow(IMG, cmap="gray", interpolation="lanczos")
for q in np.flatnonzero(legm):                                                                  # recon-frame voxel q -> display rectangle
    l, tp = box(q); a.add_patch(plt.Rectangle((l, tp), cw, ch, fc={1: C_G, 2: C_S, 3: C_O}[int(cmp_[q])], ec="w", lw=0.5, alpha=0.8))
locframe(a); a.set_title("compartment\nlabels")
a = fig.add_axes([0.349, 0.615, 0.078, 0.225]); a.imshow(IMG, cmap="gray", interpolation="lanczos")
pn = np.where(legm.reshape(N, N), pp - 2.0 * np.select([C["cmp_"] == 1, C["cmp_"] == 2], [0.0, 1.0], 2.0), np.nan)       # 80-ring PCr guidance (0-1) inside the leg
cmg = matplotlib.colormaps["magma"]
for q in np.flatnonzero(legm): l, tp = box(q); a.add_patch(plt.Rectangle((l, tp), cw, ch, fc=cmg(0.15 + 0.8 * np.clip(pn.ravel()[q], 0, 1)), ec="none", alpha=0.85))
ctr = lambda q: (box(q)[0] + cw / 2, box(q)[1] + ch / 2); keep, cut = [], []
for r in range(N):
    for c in range(N):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < N and 0 <= c2 < N): continue
            q1, q2 = r * N + c, r2 * N + c2
            if not (legm[q1] and legm[q2]): continue
            (keep if cmp_[q1] == cmp_[q2] else cut).append([ctr(q1), ctr(q2)])
a.add_collection(LineCollection(keep, colors="w", linewidths=1.1, alpha=0.9)); a.add_collection(LineCollection(cut, colors="#ff3030", linewidths=1.2, linestyles=":"))
locframe(a); a.set_title("compartment-aware\nsmoothing")
fig.legend(handles=[Patch(color=C_G, label="GM"), Patch(color=C_S, label="soleus"), Patch(color=C_O, label="other"),
                    Line2D([], [], color="0.25", lw=1.6, label="smoothing edge"), Line2D([], [], color="#ff3030", lw=1.6, ls=":", label="cut at boundary")],
           loc="center", bbox_to_anchor=(0.30, 0.58), ncol=5, fontsize=8.5, frameon=False, handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
J_B, J_A, J_G = 15 / 51, 16 / 51, 15 / 51
PKS = [("ATP", -15.86, (10, 25), (0.5, 1.5), [(-J_B, 0.5), (0, 1.0), (J_B, 0.5)], None), ("ATP", -7.58, (10, 25), (0.5, 1.5), [(-J_A / 2, 1.0), (J_A / 2, 1.0)], None),
       ("ATP", -2.53, (10, 25), (0.5, 1.5), [(-J_G / 2, 1.0), (J_G / 2, 1.0)], None), ("PCr", 0.0, (2.5, 15), (0.3, 3.0), [(0, 1.0)], None),
       ("Pi", 4.85, (8, 20), (0.0, 1.2), [(0, 1.0)], (4.3, 5.1)), ("Pi2", 4.25, (8, 25), (0.0, 0.8), [(0, 1.0)], (3.9, 4.3)), ("PDE", 3.07, (15, 50), (0.0, 0.2), [(0, 1.0)], (2.75, 3.50))]
rng = np.random.default_rng(11); a = fig.add_axes([0.177, 0.06, 0.11, 0.42]); fax = np.fft.fftshift(np.fft.fftfreq(n, DW))[::-1] / 49.894558
for k_ in range(6):
    sig = np.zeros(n, complex)
    for nm_, cs_, lwr, ar, comps, bnd in PKS:
        c_ = rng.uniform(*bnd) if bnd else (cs_ if nm_ == "ATP" else cs_ + rng.uniform(-0.1, 0.1)); lw_ = rng.uniform(*lwr); am = rng.uniform(*ar)
        for dd, ww in comps: sig += am * ww * np.exp(-np.pi * lw_ * t) * np.exp(2j * np.pi * (c_ + dd) * 49.894558 * t)
    sig *= np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.exp(2j * np.pi * rng.uniform(-0.1, 0.1) * 49.894558 * t)
    s_ = np.real(np.fft.fftshift(np.fft.fft(sig * np.exp(-np.pi * 4 * t))))[::-1]; a.plot(fax, s_ / np.abs(s_).max() + 1.2 * k_, color="0.25", lw=0.8)
a.set_xlim(10, -20); noax(a); a.set_title("simulated ³¹P spectra")
fig.text(0.297, 0.27, "SVD\n→", ha="center", va="center", fontsize=11, fontweight="bold", color="0.3")
a = fig.add_axes([0.31, 0.06, 0.11, 0.42]); APOb = np.exp(-np.pi**2 * 100 * t**2 / (4 * np.log(2))); Vb = np.fft.fftshift(np.fft.fft(U["V"][:, :12] * APOb[:, None], axis=0), axes=0)
for c_ in range(5): sb = np.abs(Vb[:, c_]); a.plot(ppm, sb / sb.max() + 1.1 * c_, color="k", lw=0.9)
a.set_xlim(10, -20); noax(a); a.set_title("spectral basis")
# ---------- (c) reconstruction ----------
fig.text(0.501, 0.5, "(c)\nCA-SPICE\nReconstruction", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#7b3294", linespacing=1.35)
# ---------- (d) output ----------
pmap = dmag(X[R0:R1].mean(0), R1 - R0, VAR)[:, np.abs(ppm) < 1].max(1)
a = fig.add_axes([0.585, 0.50, 0.105, 0.38]); a.imshow(IMG, cmap="gray", interpolation="lanczos"); nrm = colors.Normalize(0, np.percentile(pmap[leg], 99)); cmv = matplotlib.colormaps["viridis"]
for q in np.flatnonzero(leg): l, tp = box(q); a.add_patch(plt.Rectangle((l, tp), cw, ch, fc=cmv(nrm(pmap[q])), ec="none", alpha=0.88))
outline(a, vg, N, x0, y0, cw, ch, C_G, lw=1.4); outline(a, vs, N, x0, y0, cw, ch, C_S, lw=1.4)
pm = np.zeros(N * N, bool); pm[qg] = True; outline(a, pm, N, x0, y0, cw, ch, "#e41a1c", lw=2.8)
locframe(a); a.set_title("PCr map (rest)")
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=nrm, cmap=cmv), cax=fig.add_axes([0.692, 0.52, 0.005, 0.34])); cb.set_ticks([]); cb.set_label("signal", fontsize=9)
a = fig.add_axes([0.588, 0.08, 0.20, 0.33]); Sr, Se = dmag(X[R0:R1, qg].mean(0), R1 - R0, VAR[qg]), dmag(X[E1 - 5:E1 - 1, qg].mean(0), 4, VAR[qg]); sc = Sr.max()
a.plot(ppm, Sr / sc, color="k", lw=1.5, label="rest"); a.plot(ppm, Se / sc, color="#e41a1c", lw=1.5, label="end of exercise")
for p_, lab in ((0, "PCr"), (4.9, "Pi"), (-2.5, "γ"), (-7.6, "α"), (-16.2, "β-ATP")): a.text(p_, 1.04, lab, ha="center", fontsize=8.5, color="0.3")
a.set_xlim(10, -20); a.set_ylim(0, 1.13); a.set_yticks([]); a.set_xlabel("ppm", fontsize=9, labelpad=1); a.tick_params(labelsize=8.5)
[a.spines[s].set_visible(False) for s in ("top", "right", "left")]; a.legend(fontsize=9, frameon=False, loc="center right"); a.set_title("spectrum, marked voxel", color="#e41a1c")
a = fig.add_axes([0.80, 0.05, 0.185, 0.83]); Sd = dmag(X[:, qg], 1, VAR[qg][None, :]); scd = Sd[R0:R1].max(); w = (ppm < 8) & (ppm > -18); DX, DY = -0.12, 0.05
for r_ in range(NR - 1, -1, -1):
    col = "0.35" if r_ < E0 else "#de2d26" if r_ < E1 else "#3182bd"; xx, yy = ppm[w] + DX * r_, Sd[r_, w] / scd + DY * r_
    a.fill_between(xx, DY * r_, yy, color="white", zorder=NR - r_); a.plot(xx, yy, color=col, lw=0.8, zorder=NR - r_ + 0.5)
a.set_xlim(9, -18 + DX * NR - 1); a.set_ylim(-0.12, DY * NR + 1.05); noax(a); a.set_title("dynamic spectra, marked voxel")
for p_, lab in ((0, "PCr"), (4.9, "Pi"), (-16.2, "β-ATP")): a.text(p_, -0.1, lab, ha="center", va="top", fontsize=9, color="0.25")
a.annotate("", xy=(8.5 + DX * NR, DY * NR + 0.1), xytext=(8.5, 0.1), arrowprops=dict(arrowstyle="->", color="0.4", lw=1.1)); a.text(9.0 + DX * NR / 2, DY * NR / 2 + 0.25, "time", rotation=72, fontsize=10, color="0.4")
a.legend(handles=[Line2D([], [], color="0.35", lw=2, label="rest"), Line2D([], [], color="#de2d26", lw=2, label="exercise"), Line2D([], [], color="#3182bd", lw=2, label="recovery")],
         fontsize=9, frameon=False, loc="upper right")
fig.savefig(f"{OUTF}/F1_recon_overview_CA.png", dpi=140); fig.savefig(f"{OUTF}/F1_recon_overview_CA.pdf"); print("saved", SPD, "edges kept", len(keep), "cut", len(cut))
