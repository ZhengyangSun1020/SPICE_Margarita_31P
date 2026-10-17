#!/usr/bin/env python3
"""Hybrid SLAM + SPICE on the fine grid (8 mm for 24 mm voxels): tissue signal per fine pixel
     rho(r, t) = sum_c m_c(r) s_c(t) + delta(r, t)
   m_c  = sub-voxel area fraction of compartment c (GM / soleus / other muscle; user voxel labels on THIS bout's grid, upsampled, intersected with
          the localizer muscle mask), s_c = compartment spectrum (SLAM part), delta = local deviation (SPICE part, label graph, small).
   min || D^1/2 (y V - F S (M s + delta)) ||^2 + lam ||W delta||^2 + mu ||delta||^2       (real CRT k-space, DCF weighting, 80-ring sensitivity S, rank 16)
   mu -> inf = pure SLAM (one spectrum per compartment), mu -> 0 = SPICE-S (voxel model on the fine grid). Read out per CSI voxel: sum_p S_p rho_p.
Compared, with identical metrics and an identical sub-voxel leakage test (uniform GM tissue signal through the real encoding), with NUFFT (Hamming)
and the current compartment SPICE (soft delta 2, CSI grid). Usage: hybrid_slam_spice.py <ds>  -> output_study_mid<ds>/hybrid/"""
import os, sys, io, json, contextlib, importlib.util as ilu, numpy as np, pydicom, mrinufft
from scipy import ndimage as ndi
ROOT = "/home/zsun/SPICE_Margarita"; RAW = "/mnt/e/claude_tmp/study/raw"; sys.path.insert(0, ROOT + "/scripts")
sys.path.insert(0, "/mnt/e/claude_tmp/study/masks_hr"); from phantom_mask import phantom_mask
m = ilu.module_from_spec(ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py")); ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py").loader.exec_module(m)
from recon_concept_twix_31p import concept_phase_correction
DATS = {"277": "meas_MID00277_FID221482_240FOV_10x10_5rings.dat", "279": "meas_MID00279_FID221484_240FOV_8x8_5rings.dat",
        "281": "meas_MID00281_FID221486_200FOV_8x8_5rings.dat", "283": "meas_MID00283_FID221488_200FOV_10x10_5rings.dat",
        "1034": "/mnt/e/claude_tmp/study0207/CRT/meas_MID01034_FID216041_LV_crt_dynamic.dat"}
datpath = lambda ds: DATS[ds] if DATS[ds].startswith("/") else f"{RAW}/{DATS[ds]}"
# per-session localizer and drawn pixel masks (24-07: Series 18 for all bouts; 02-07: MR000006 at the CSI slice)
LOCF = {"1034": "/mnt/e/claude_tmp/study0207/dicoms/MR localizer_tra/MR000006.dcm"}
SEGF = {"1034": "/mnt/e/claude_tmp/study0207/masks/seg_MID01034.mat"}
USER = {"277": ([47, 56, 57, 63, 64, 65, 66], [53, 54, 55]), "283": ([47, 56, 57, 63, 64, 65, 66], [53, 54, 55]),
        "279": ([37, 42, 43, 44, 45], [34, 35, 36]), "281": ([37, 42, 43, 44, 45], [28, 34, 35, 36]), "1034": ([26, 34, 42, 43, 44, 45], [27, 35, 36])}
DAT80 = f"{RAW}/meas_MID00276_FID221481_240FOV_10x10_80rings.dat"
PIXMASK = os.environ.get("PIXMASK", "") == "1"                                                 # use the drawn pixel-level masks (seg_series18.mat)
MLAB = os.environ.get("LABELS", "") == "mask"                                                  # CSI voxel labels from the drawn masks (>= 50 % of voxel area)
if MLAB: USER.update({k: (v["gm"], v["so"]) for k, v in json.load(open("/mnt/e/claude_tmp/study/masks_hr/mask_labels.json")).items()})
DS = sys.argv[1]; O = f"{ROOT}/output_study_mid{DS}"; OUT = f"{O}/" + (os.environ.get("HOUT") or ("hybrid" + ("_pix" if PIXMASK else "") + ("_mlab" if MLAB else ""))); os.makedirs(OUT, exist_ok=True); LOG = open(f"{OUT}/log.txt", "w")
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
MUS_ = [float(x) for x in os.environ.get("MUS", "0.001,0.01,0.1,1,10,100").split(",")]; LAM_D = float(os.environ.get("LAM_D", 1.0)); F_UP = 3; EPS_X = 1e-3
meta = json.load(open(f"{O}/meta.json")); N, NR, FOV, APOD = meta["n"], meta["nreps"], meta["fov"], meta["apod_hz"]; R0, R1 = meta["rest"]; E0, E1 = meta["ex"]
EXTRA = meta.get("fov_extra_shift", -0.5); REP = {"277": 718.2, "279": 718.4, "281": 718.4, "283": 726.2, "1034": 718.3}[DS] / NR
EX = slice(E1 - int(round(96 / REP)), E1); TAIL = slice(NR - int(round(144 / REP)), NR); TPL = slice(max(E1 + 4, NR - 20), NR); BASE = np.r_[R0:R1, min(E1 + 3, NR - 1):NR]
fid = np.load(f"{O}/adj_fid.npy").astype(np.complex128); n = fid.shape[-1]; V = np.load(f"{O}/{os.environ.get('VFILE', 'VD.npy')}")   # VFILE=VS1.npy: single-Pi subspace
with contextlib.redirect_stdout(io.StringIO()): info0 = m.load_concept_twix_3d(datpath(DS), rep_sel=0)
CX, CY = float(info0["pos_sag"]), float(info0["pos_cor"]); f0 = info0["larmor_freq_mhz"]; dw = info0["dwell_s"] * 2; t = np.arange(n) * dw
disp2rec = lambda ids: (N - 1 - np.asarray(ids) // N) * N + np.asarray(ids) % N
def make_traj(info):
    tj = m.build_trajectory_2d(info["first_circle"], info["n_rings"], int(info["traj_pts"].max())).copy()
    tj[:, :2] *= float(info["k_norm_max_phys"]) / 0.5 * (FOV / N) / (float(info["fov_mm"]) / float(info["matrix_size"]))
    r_ = np.sqrt(tj[:, 0]**2 + tj[:, 1]**2)
    return tj, np.asarray(m.compute_crt_dcf_hamming(tj, hamming_radius=np.zeros(len(tj))), float).ravel(), np.asarray(m.compute_crt_dcf_hamming(tj, hamming_radius=r_), float).ravel()
tj, dcf, dcf_h = make_traj(info0)
if os.environ.get("DCF_HAM") == "1": dcf = dcf_h.copy()                                         # Hamming-weighted DCF in the data term (as NUFFT)
if os.environ.get("OUTERW"):                                                                      # down-weight the outer rings (2 temporal interleaves)
    _r = np.round(np.hypot(tj[:, 0], tj[:, 1]), 3); _out = _r >= np.sort(np.unique(_r))[-2]; dcf = dcf.copy(); dcf[_out] *= float(os.environ["OUTERW"])
if not os.path.exists(f"{O}/kspace_fs.npy"):                                                   # build phase-corrected, FOV-shifted k-space (as rebuild_study.py)
    dpe, dro = m.compute_fov_shift(info0); shift = np.exp(-2j * np.pi * (tj[:, 0] * (dpe + EXTRA) + tj[:, 1] * (dro + EXTRA))); Kl = []
    for r in range(NR):
        with contextlib.redirect_stdout(io.StringIO()): info = info0 if r == 0 else m.load_concept_twix_3d(datpath(DS), rep_sel=r)
        Kl.append((concept_phase_correction(info["kdata"], info["_ti_per_ring"])[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n) * shift[:, None]).astype(np.complex64))
    np.save(f"{O}/kspace_fs.npy", np.stack(Kl)); log("built kspace_fs.npy")
K = np.load(f"{O}/kspace_fs.npy").astype(np.complex128)
if os.environ.get("RECENTER") == "1":                                                          # put PCr at 0 Hz in the training (V) convention
    _S = np.abs(np.fft.fft(fid[R0:R1].mean(0), 8 * n, axis=-1)).sum(0); _fr = np.fft.fftfreq(8 * n, dw); _w = np.flatnonzero(np.abs(_fr) < 60)
    _i = _w[np.argmax(_S[_w])]; _a, _b, _c = _S[_i - 1], _S[_i], _S[(_i + 1) % (8 * n)]; F_D = _fr[_i] + 0.5 * (_a - _c) / (_a - 2 * _b + _c) * (_fr[1] - _fr[0])
    _ph = np.exp(-2j * np.pi * F_D * t); fid = fid * _ph[None, None, :]; K = K * _ph[None, None, :]
    log(f"re-centred: PCr was at {F_D:+.2f} Hz in the subspace convention -> 0 Hz")
# ---------------- CSI-grid encoding (NUFFT and current SPICE) ----------------
opF = mrinufft.get_operator("finufft", samples=tj, shape=(N, N), n_coils=1, density=False)
Fc = np.stack([opF.op(np.eye(N * N, dtype=np.complex64)[v].reshape(N, N)).ravel() for v in range(N * N)], 1).astype(np.complex128)
log(f"MID{int(DS):05d}: CSI encoding reproduces the stored NUFFT adjoint: mismatch {100 * np.linalg.norm(Fc.conj().T @ (dcf_h * K[0][:, 0]) - fid[0][:, 0]) / np.linalg.norm(fid[0][:, 0]):.1e}%")
# ---------------- fine grid (centred on the CSI grid), labels, muscle mask ----------------
Nh = N * F_UP; dh = FOV / Nh; D_ = FOV / N; xi, yi = np.mgrid[0:Nh, 0:Nh]
X = CX + (yi - Nh / 2) * dh + 0.5 * D_; Y = CY + (Nh / 2 - xi) * dh - 0.5 * D_
blk = ((((xi + F_UP // 2) // F_UP) % N) * N + ((yi + F_UP // 2) // F_UP) % N).ravel()
def hr_matrix(tj_):
    tjh = tj_.copy(); tjh[:, :2] *= N / Nh; oph = mrinufft.get_operator("finufft", samples=tjh, shape=(Nh, Nh), n_coils=1, density=False)
    return np.stack([oph.op(np.eye(Nh * Nh, dtype=np.complex64)[p].reshape(Nh, Nh)).ravel() for p in range(Nh * Nh)], 1).astype(np.complex128)
Fh = hr_matrix(tj)
LOC18 = LOCF.get(DS) or "/mnt/e/claude_tmp/study/loc/" + [f for f in os.listdir("/mnt/e/claude_tmp/study/loc") if f.startswith("Series0018")][0]; L18 = pydicom.dcmread(LOC18)
ipp18 = np.array(L18.ImagePositionPatient, float); ps18 = float(L18.PixelSpacing[0]); im18 = L18.pixel_array.astype(float); PH18 = phantom_mask(im18)
bw = ndi.binary_fill_holes(ndi.binary_opening((im18 > np.percentile(im18[~PH18], 60)) & ~PH18, iterations=3)); lab, nl = ndi.label(bw)
LEG18 = lab == (1 + np.argmax(ndi.sum(bw, lab, range(1, nl + 1)))); sm18 = ndi.gaussian_filter(im18, 2.0); lv = sm18[LEG18]
MUS18 = ndi.binary_opening(LEG18 & (sm18 < np.percentile(lv, 88)) & (sm18 > np.percentile(lv, 5)), iterations=1)
def sample_grid(mask, X_, Y_, d_):                                                              # area fraction of a localizer mask in grid pixels
    out = np.zeros(X_.shape)
    for a in np.linspace(-0.375, 0.375, 4):
        for b in np.linspace(-0.375, 0.375, 4):
            c_ = np.round((X_ + a * d_ - ipp18[0]) / ps18).astype(int); r_ = np.round((Y_ + b * d_ - ipp18[1]) / ps18).astype(int)
            ok = (r_ >= 0) & (r_ < mask.shape[0]) & (c_ >= 0) & (c_ < mask.shape[1]); v = np.zeros(X_.shape); v[ok] = mask[r_[ok], c_[ok]]; out += v / 16
    return out.ravel()
sample18 = lambda mask: sample_grid(mask, X, Y, dh)
fMUS = sample18(MUS18)
gm_ids, so_ids = USER[DS]; vGM = np.zeros(N * N, bool); vGM[disp2rec(gm_ids)] = True; vSOL = np.zeros(N * N, bool); vSOL[disp2rec(so_ids)] = True
regv = np.load(f"{O}/reg_TRUE.npy").ravel(); legv = (regv == 1) | vGM | vSOL; phant = regv == 2
vlab = np.where(vGM, 1, np.where(vSOL, 2, np.where(legv, 3, 0)))                                 # CSI-voxel label
if PIXMASK:
    import scipy.io as sio
    SG = sio.loadmat(SEGF.get(DS, "/mnt/e/claude_tmp/study/masks_hr/seg_series18.mat")); GMp = SG["GM"].astype(bool); SOLp = SG["SOL"].astype(bool) & ~GMp
    MUSall = MUS18 | GMp | SOLp; fMUS = sample18(MUSall); fG, fS = sample18(GMp), sample18(SOLp); fO = np.clip(fMUS - fG - fS, 0, None)
    supp = fMUS > 0.25; pix = np.flatnonzero(supp); nu = len(pix)
    Mc = np.stack([fG[pix], fS[pix], fO[pix]], 1); plab = 1 + np.argmax(Mc, 1)                   # pixel-level compartments from the drawn masks
    CM = os.environ.get("COMPMODE", "")                                                          # compartment-split test
    if CM == "merge_other_sol": Mc[:, 1] += Mc[:, 2]; Mc[:, 2] = 0; plab = np.where(plab == 3, 2, plab)   # other muscle shares the soleus spectrum
    elif CM == "drop_other": Mc[:, 2] = 0                                                        # other-muscle pixels: local deviation only
    if CM: log(f"compartment mode: {CM}")
    log(f"pixel-level masks: GM {GMp.sum()} px, soleus {SOLp.sum()} px on the localizer")
else:
    supp = (fMUS > 0.25) & (vlab[blk] > 0); pix = np.flatnonzero(supp); nu = len(pix); plab = vlab[blk][pix]
    Mc = np.stack([fMUS[pix] * (plab == c) for c in (1, 2, 3)], 1)                               # sub-voxel compartment area fractions (nu x 3)
log(f"fine grid {Nh}x{Nh} @ {dh:.1f} mm: {nu} muscle pixels (GM {int((plab == 1).sum())}, soleus {int((plab == 2).sum())}, other {int((plab == 3).sum())}); "
    f"GM {gm_ids}, soleus {so_ids} (display ids)")
# ---------------- sensitivity: 80-ring scan if the session has one, else the 5-ring rest reps (smooth complex polynomial, k-space fit) ----------------
if DS in ("277", "279", "281", "283") and os.environ.get("SENS_SRC") != "rest":                 # SENS_SRC=rest: sensitivity from the bout's own rest reps
    with contextlib.redirect_stdout(io.StringIO()): i80 = m.load_concept_twix_3d(DAT80, rep_sel=0)
    t80, dcf80, _ = make_traj(i80); tgt = dict(i80); tgt.update(pos_sag=CX, pos_cor=CY, pos_tra=info0.get("pos_tra", 0.0), fov_mm=FOV, matrix_size=N)
    p_pe, p_ro = m.compute_fov_shift(tgt); ns80, dw80 = i80["n_spec"], i80["dwell_s"] * 2; tt80 = np.arange(ns80) * dw80
    kf80 = concept_phase_correction(i80["kdata"], i80["_ti_per_ring"])[:, :, 0, :].transpose(2, 0, 1).reshape(-1, ns80) * \
        np.exp(-2j * np.pi * (t80[:, 0] * (p_pe + EXTRA) + t80[:, 1] * (p_ro + EXTRA)))[:, None]
    S80sp = np.fft.fftshift(np.fft.fft(kf80 * np.exp(-np.pi * 5 * tt80), axis=-1), axes=-1); kc80 = np.argsort(np.hypot(t80[:, 0], t80[:, 1]))[:20]
    yP80 = S80sp[:, int(np.argmax(np.abs(S80sp[kc80]).sum(0)))]; Fh80 = hr_matrix(t80); S_SRC = "80-ring scan"
else:
    Sr = np.fft.fftshift(np.fft.fft(K[R0:R1].mean(0) * np.exp(-np.pi * 5 * t), axis=-1), axes=-1); kc5 = np.argsort(np.hypot(tj[:, 0], tj[:, 1]))[:20]
    yP80 = Sr[:, int(np.argmax(np.abs(Sr[kc5]).sum(0)))]; Fh80 = Fh; dcf80 = dcf; S_SRC = "5-ring rest reps (no 80-ring scan)"
xn, yn = (yi / Nh * 2 - 1).ravel(), ((Nh - xi) / Nh * 2 - 1).ravel(); Bp = np.stack([np.ones_like(xn), xn, yn, xn**2, xn * yn, yn**2]).astype(float)
sw = np.sqrt(dcf80)[:, None]; Mx = (Fh80 * fMUS) @ Bp.T; cS = np.linalg.lstsq(sw * Mx, sw[:, 0] * yP80, rcond=None)[0]; S = (cS @ Bp).astype(np.complex128)
Ss = S[pix]; log(f"sensitivity from {S_SRC}: mean |S| GM/soleus {np.abs(Ss[plab == 1]).mean() / np.abs(Ss[plab == 2]).mean():.2f}")
# ---------------- metrics (identical for all methods) ----------------
PPM = -np.fft.fftshift(np.fft.fftfreq(n, d=dw)) / f0; APO = np.exp(-np.pi**2 * APOD**2 * t**2 / (4 * np.log(2)))
cspec = lambda F_: m.FIDToSpec((F_ * APO).conj(), axis=-1)
_s = np.abs(cspec(fid[R0:R1].mean(0)[legv])).sum(0); _k = np.flatnonzero(np.abs(PPM) < 1.5); p0 = PPM[_k[np.argmax(_s[_k])]]; Q = PPM - p0
bP, bB = np.abs(Q) <= 1, (Q >= -17.2) & (Q <= -15.0); kc = int(np.argmin(np.abs(Q))); kP = np.flatnonzero(bP); offs = kP - kc; cand = np.flatnonzero((Q >= 3.8) & (Q <= 5.3))
def mf(Sx, band): Sb = Sx[..., band]; T = Sb[TPL].mean(0); return np.abs((Sb * T.conj()).sum(-1)) / (np.abs(T)**2).sum(-1)
def pi_mf(Sx):
    T = Sx[TPL][..., kP].mean(0); Sc = Sx[..., cand[:, None] + offs[None, :]]
    am = np.abs((Sc * T.conj()[..., None, :]).sum(-1)) / (np.abs(T)**2).sum(-1)[..., None]; return am.max(-1)
def roi_spec(F_):
    Sx = cspec(F_); Tm = Sx[TPL].mean(0); c = Tm[np.arange(Tm.shape[0]), kP[np.argmax(np.abs(Tm[:, bP]), axis=1)]]
    return (Sx * c.conj()[None, :, None]).sum(1) / np.abs(c).sum()
nz = lambda y: np.std(np.diff(y[TAIL])) / np.sqrt(2)
def evaluate(Fv):                                                                                # Fv: (NR, N*N, n) voxel FIDs
    r = {}
    for nm, msk in (("g", vGM), ("s", vSOL)):
        Sx = roi_spec(Fv[:, msk]); pc = mf(Sx, bP); r0 = pc[R0:R1].mean(); pa = pi_mf(Sx); b = mf(Sx, bB)
        r[nm] = dict(pcr=100 * (1 - pc[EX].mean() / r0), noise=100 * nz(pc / r0), pi=100 * (pa[EX].mean() - pa[R0:R1].mean()) / r0, beta=100 * (b[E0:E1].mean() / b[BASE].mean() - 1),
                     curve=pc / r0, pic=pa / r0)
    Sx = cspec(Fv[:, legv]); pcv = mf(Sx, bP); r0v = pcv[R0:R1].mean(0); pav = pi_mf(Sx)
    dep = np.full(N * N, np.nan); inc = np.full(N * N, np.nan); ts = np.full(N * N, np.nan)
    dep[legv] = 100 * (1 - pcv[EX].mean(0) / r0v); inc[legv] = 100 * (pav[EX].mean(0) - pav[R0:R1].mean(0)) / r0v
    ts[legv] = 1 / (np.std(np.diff(pcv[TAIL] / r0v, axis=0), axis=0) / np.sqrt(2))
    r.update(dep=dep, inc=inc, tsnr=ts, vox_pcr=pcv / r0v, vox_pi=pav / r0v, vox_r0=r0v)
    if os.environ.get("SAVE_SPEC") == "1": r.update(srest=np.abs(cspec(Fv[R0:R1].mean(0))).astype(np.float32), sex=np.abs(cspec(Fv[EX].mean(0))).astype(np.float32))
    return r
# sub-voxel leakage test: uniform GM (or soleus) TISSUE signal on the fine grid through the real encoding; response per CSI voxel
y_g = Fh[:, pix] @ (Ss * Mc[:, 0]); y_s = Fh[:, pix] @ (Ss * Mc[:, 1]); y_all = Fh[:, pix] @ (Ss * Mc.sum(1))
def share(part, total, msk):                                                                     # phase-independent: projection of the part onto the total signal
    return float(np.real(np.vdot(total[msk], part[msk])) / np.real(np.vdot(total[msk], total[msk])))
def spill(rec):                                                                                  # rec: k-space vector -> CSI voxel image
    ra = rec(y_all); rg = rec(y_g); rs = rec(y_s)
    out = dict(g2s=share(rg, ra, vSOL), s2g=share(rs, ra, vGM))                                  # share of the soleus (GM) voxel signal from GM (soleus)
    for tag, (yg_, ya_) in TESTS.items(): out[f"g2s_{tag}"] = share(rec(yg_), rec(ya_), vSOL)
    return out
TESTS = {}
if PIXMASK:                                                                                      # independent 2 mm truth (no inverse crime) + 3 mm GM boundary error
    Nh2 = int(round(FOV / 2.0)); dh2 = FOV / Nh2; xi2, yi2 = np.mgrid[0:Nh2, 0:Nh2]
    X2 = CX + (yi2 - Nh2 / 2) * dh2 + 0.5 * D_; Y2 = CY + (Nh2 / 2 - xi2) * dh2 - 0.5 * D_
    xn2, yn2 = (2 * (X2 - CX - 0.5 * D_) / FOV).ravel(), (2 * (Y2 - CY + 0.5 * D_) / FOV).ravel()
    S2 = cS @ np.stack([np.ones_like(xn2), xn2, yn2, xn2**2, xn2 * yn2, yn2**2]).astype(float)
    tj2 = tj.copy(); tj2[:, :2] *= N / Nh2; op2 = mrinufft.get_operator("finufft", samples=tj2, shape=(Nh2, Nh2), n_coils=1, density=False)
    fw = lambda img: op2.op((S2 * img).reshape(Nh2, Nh2).astype(np.complex64)).ravel().astype(np.complex128)
    fM2 = sample_grid(MUSall, X2, Y2, dh2); fG2 = sample_grid(GMp, X2, Y2, dh2)
    GMd = ndi.binary_dilation(GMp, iterations=int(round(3.0 / ps18))) & (GMp | SOLp); fGd = sample_grid(GMd, X2, Y2, dh2)
    TESTS = {"2mm": (fw(fG2), fw(fM2)), "2mm_shift3": (fw(fGd), fw(fM2))}
    pv = Rd_true = None
Rd0 = np.zeros((N * N, nu), complex); Rd0[blk[pix], np.arange(nu)] = Ss                          # partial-volume truth of the soleus voxels (fine grid)
tg = Rd0 @ Mc[:, 0]; ta = Rd0 @ Mc.sum(1); log(f"true GM share of the soleus-voxel signal (partial volume): {100 * share(tg, ta, vSOL):.0f}%")
RES = {}
# ---- UNC=1: noise propagation (exact for these linear estimators), repeated-scan uncertainty, L-curve -> OUT/unc_MID00<ds>.npz ----
UNC = os.environ.get("UNC") == "1"; SPEC_MUS = [float(x) for x in os.environ.get("SPEC_MUS", "").split(",") if x]; UO = {}
if UNC:
    kr = np.hypot(tj[:, 0], tj[:, 1]); Kt = K[:, kr >= np.median(kr), -200:]                     # outer k, last 200 points: noise only
    sig2 = float(np.mean(np.abs(Kt - Kt.mean(0))**2) * NR / (NR - 1)); log(f"noise variance per k-t sample (outer k, last 200 pts): {sig2:.3e}")
    UO.update(sig2=sig2, shape_white=(np.abs(cspec(np.eye(n)))**2).sum(0), shape_sub=(np.abs(cspec(V @ V.conj().T))**2).sum(0))
    KV = K @ V
    def rep_unc(F_, tag):                                                                        # repeated-scan SD: rep-to-rep differences / sqrt 2 in the rest block and the recovery plateau
        Sx = cspec(F_); d = np.concatenate([np.diff(Sx[R0:R1], axis=0), np.diff(Sx[TAIL], axis=0)])
        UO[f"{tag}_emp"] = np.sqrt(np.mean(np.abs(d)**2, axis=0) / 2).astype(np.float32); UO[f"{tag}_ndiff"] = len(d)
        UO[f"{tag}_rest"] = Sx[R0:R1].mean(0).astype(np.complex64); UO[f"{tag}_single"] = Sx[R1 - 1].astype(np.complex64)
    Pn = Fc.conj().T * dcf_h; UO["nufft_var"] = sig2 * (np.abs(Pn)**2).sum(1); rep_unc(fid, "nufft")
    UO.update(kt_abs=np.abs(K[R0]).astype(np.float32), traj=tj[:, :2], V=V, Nh=Nh, pix=pix, Mc=Mc, S=S, legv=legv, vGM=vGM, vSOL=vSOL)
# ---- NUFFT ----
RES["NUFFT"] = evaluate(fid); RES["NUFFT"]["spill"] = spill(lambda y: Fc.conj().T @ (dcf_h * y))
# ---- current compartment SPICE (soft delta 2, CSI grid, DCF weighting) ----
s0 = np.load(f"{O}/system.npz"); s11 = np.load(f"{O}/system11.npz" if os.path.exists(f"{O}/system11.npz") else f"{O}/system11_B.npz"); G_h = s0["G"].astype(np.complex128)
comp = np.zeros(N * N); comp[legv] = 3; comp[vGM] = 1; comp[vSOL] = 2; comp = comp.reshape(N, N); legm = legv.reshape(N, N); ph = phant.reshape(N, N)
_, (ii, jj) = ndi.distance_transform_edt(~legm, return_indices=True); cmp_ = np.where(legm, comp, comp[ii, jj]); p_raw = s0["prior"]
CDEL = float(os.environ.get("COMP_DELTA", 2.0))                                                  # compartment offset in the guidance image (cross-compartment W ~ 1 / CDEL^2)
lo, hi = p_raw[legm].min(), p_raw[legm].max(); pp = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1) + CDEL * np.select([cmp_ == 1, cmp_ == 2], [0.0, 1.0], 2.0); pp[ph] = 0.0
log(f"compartment SPICE: compartment offset {CDEL:g}")
CB = os.environ.get("COMP_BOUND", "soft")                                                        # soft: cross-compartment edges W ~ 1/2^2 (offset 2); hard: cut (W = 0)
WW, _ = m.calc_Bmatrix_simple(pp.astype(np.float32), wmax=5e4, adj=8, label_mask=(np.where(ph, 0, cmp_) if CB == "hard" else np.where(ph, 0, 1)).astype(int).T)
log(f"compartment SPICE: {CB} compartment boundary")
dd = ndi.distance_transform_cdt(~legm, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2); lout[ph] = 1.0; lout = lout.ravel()
Gd = (Fc.conj().T * dcf) @ Fc; rel = np.real(np.diag(Gd)).mean() / np.real(np.diag(G_h)).mean()
Ai = np.linalg.inv(Gd + float(s11["lam"]) * rel * WW + np.diag(lout * rel))
Fsp = np.einsum("pq,rqc->rpc", Ai, np.einsum("vk,rkc->rvc", Fc.conj().T * dcf, K @ V)) @ V.conj().T
RES["SPICE (compartment, CSI grid)"] = evaluate(Fsp); RES["SPICE (compartment, CSI grid)"]["spill"] = spill(lambda y: Ai @ (Fc.conj().T @ (dcf * y)))
if os.environ.get("SAVE_SP") == "1":                                                              # compartment-SPICE voxel FIDs (NR, N*N, n) + its constraint inputs, for figures / MATLAB export
    np.save(f"{OUT}/fid_spice.npy", Fsp.astype(np.complex64)); np.savez(f"{OUT}/spice_constraints.npz", comp=comp, cmp_=cmp_, pp=pp, lout=lout.reshape(N, N), ph=ph, legm=legm, lam=float(s11["lam"]), bound=CB)
    log(f"saved compartment-SPICE voxel FIDs {OUT}/fid_spice.npy")
def temporal_basis(Ls):
    """Low-rank-in-time (partial separability) basis from the data: inner rings (1-3) of the spectrally projected k-space K @ V,
    resting mean removed, 5-rep moving mean within rest / exercise / recovery, first L-1 right singular vectors + a constant."""
    kr_ = np.round(np.hypot(tj[:, 0], tj[:, 1]), 4); inner = kr_ <= np.unique(kr_)[2] + 1e-4
    Y_ = (K[:, inner] @ V).reshape(NR, -1).T; Yc = Y_ - Y_[:, R0:R1].mean(1, keepdims=True); Ys = np.zeros_like(Yc)
    for a_, b_ in ((0, E0), (E0, E1), (E1, NR)):
        nk = min(5, b_ - a_); k_ = np.ones(nk) / nk; den = np.convolve(np.ones(b_ - a_), k_, "same")[:b_ - a_]
        Ys[:, a_:b_] = np.array([np.convolve(row, k_, "same")[:b_ - a_] / den for row in Yc[:, a_:b_]])
    sv = np.linalg.svd(Ys, compute_uv=False); _, _, Vh_ = np.linalg.svd(Ys, full_matrices=False)
    log(f"temporal basis: smoothed inner-ring singular values (rel.) {np.round(sv[:8] / sv[0], 3).tolist()}")
    return {L_: np.linalg.qr(np.vstack([np.ones(NR), Vh_[:L_ - 1]]).T)[0].T for L_ in Ls}         # L x NR, orthonormal rows
def temporal_basis_img(Ls, X_):
    """Basis from the per-rep recon itself: all leg voxels x spectral coefficients, no smoothing, no mean removal."""
    Z_ = (X_[:, legv] @ V).reshape(NR, -1).T; sv = np.linalg.svd(Z_, compute_uv=False); _, _, Vh_ = np.linalg.svd(Z_, full_matrices=False)
    log(f"temporal basis (per-rep SPICE images, leg voxels): singular values (rel.) {np.round(sv[:10] / sv[0], 3).tolist()}")
    return {L_: Vh_[:L_] for L_ in Ls}
TLR = [int(x) for x in os.environ.get("TLR", "").split(",") if x]
if TLR:                                                                                           # same rings every rep + linear SPICE -> joint LR recon = SPICE of time-projected data
    BAS = temporal_basis_img(TLR, Fsp) if os.environ.get("TLR_SRC") == "img" else temporal_basis(TLR)
    # honest noise test: white k-space noise at the measured level through SPICE, ROI PCr amplitude, with / without the temporal projection
    kr_ = np.hypot(tj[:, 0], tj[:, 1]); Kt = K[:, kr_ >= np.median(kr_), -200:]; sig2_ = float(np.mean(np.abs(Kt - Kt.mean(0))**2) * NR / (NR - 1))
    rngn = np.random.default_rng(3); PS = Ai @ (Fc.conj().T * dcf)
    tplG = (Fsp[R0:R1][:, vGM] @ V).mean((0, 1)); tplS = (Fsp[R0:R1][:, vSOL] @ V).mean((0, 1))
    def roi_amp(Xc, msk, tpl): return np.real((Xc[:, msk].mean(1) @ tpl.conj())) / np.real(tpl @ tpl.conj())     # ROI amplitude along the rest template, rel. to rest
    NZ = {}
    for itn in range(8):
        Kn = np.sqrt(sig2_ / 2) * (rngn.standard_normal(K.shape[:2] + (V.shape[1],)) + 1j * rngn.standard_normal(K.shape[:2] + (V.shape[1],)))   # noise already in subspace coords
        Xn = np.einsum("pq,rqc->rpc", PS, Kn); NZ.setdefault("raw", []).append([roi_amp(Xn, vGM, tplG), roi_amp(Xn, vSOL, tplS)])
        for L_, PHI in BAS.items():
            Pt = PHI.conj().T @ PHI; Xp = np.einsum("st,tvc->svc", Pt, Xn); NZ.setdefault(L_, []).append([roi_amp(Xp, vGM, tplG), roi_amp(Xp, vSOL, tplS)])
    nz0 = np.std(np.array(NZ["raw"]), axis=(0, 2))
    log("noise test (synthetic k-space noise at the measured level, ROI PCr amplitude, % of rest): plain SPICE GM {:.1f} / SOL {:.1f}".format(*(100 * nz0)))
    for L_ in BAS:
        nzl = np.std(np.array(NZ[L_]), axis=(0, 2)); log(f"   LR{L_}: GM {100 * nzl[0]:.1f} / SOL {100 * nzl[1]:.1f}  (x{nz0[0] / nzl[0]:.1f} / x{nz0[1] / nzl[1]:.1f} lower)")
    for L_, PHI in BAS.items():
        Pt = PHI.conj().T @ PHI                                                                   # NR x NR temporal projector
        nmk = f"SPICE comp LR{L_}"; RES[nmk] = evaluate(np.einsum("st,tvc->svc", Pt, Fsp)); RES[nmk]["spill"] = RES["SPICE (compartment, CSI grid)"]["spill"]
        nmn = f"NUFFT LR{L_}"; RES[nmn] = evaluate(np.einsum("st,tvc->svc", Pt, fid)); RES[nmn]["spill"] = RES["NUFFT"]["spill"]
        if os.environ.get("SAVE_LR") == "1": np.save(f"{OUT}/fid_spice_lr{L_}.npy", np.einsum("st,tvc->svc", Pt, Fsp).astype(np.complex64))
for kl in [float(x) for x in os.environ.get("SP_LAMS", "").split(",") if x]:                     # lambda sweep: lambda x kl (same graph, boundary, lambda_out)
    Aik = np.linalg.inv(Gd + kl * float(s11["lam"]) * rel * WW + np.diag(lout * rel))
    Fk = np.einsum("pq,rqc->rpc", Aik, np.einsum("vk,rkc->rvc", Fc.conj().T * dcf, K @ V)) @ V.conj().T
    nmk = f"SPICE comp λx{kl:g}"; RES[nmk] = evaluate(Fk); RES[nmk]["spill"] = spill(lambda y, A_=Aik: A_ @ (Fc.conj().T @ (dcf * y)))
    log(f"compartment SPICE lambda x{kl:g} = {kl * float(s11['lam']):.3g}")
# ---- hybrid SLAM + SPICE on the fine grid ----
A = Fh[:, pix] * Ss[None]; AM = A @ Mc; Af = np.hstack([AM, A])                                  # unknowns z = [s (3); delta (nu)]
Gf = (Af.conj().T * dcf) @ Af; sc = np.real(np.trace((A.conj().T * dcf) @ A)) / nu
posm = -np.ones(Nh * Nh, int); posm[pix] = np.arange(nu); L = np.zeros((nu, nu)); pi_, pj_ = np.divmod(pix, Nh)
for dx_, dy_ in ((0, 1), (1, 0), (1, 1), (1, -1)):
    ok = (pi_ + dx_ < Nh) & (pj_ + dy_ >= 0) & (pj_ + dy_ < Nh); q = np.where(ok, (pi_ + dx_) * Nh + pj_ + dy_, 0); k2 = np.where(ok, posm[q], -1); a_ = k2 >= 0
    i1, i2 = np.flatnonzero(a_), k2[a_]; w = np.where(plab[i1] == plab[i2], 1.0, EPS_X) / np.hypot(dx_, dy_)
    np.add.at(L, (i1, i1), w); np.add.at(L, (i2, i2), w); np.add.at(L, (i1, i2), -w); np.add.at(L, (i2, i1), -w)
RHS = np.einsum("pk,rkc->rpc", Af.conj().T * dcf, K @ V)
Rd = np.zeros((N * N, nu), complex); Rd[blk[pix], np.arange(nu)] = Ss                            # CSI-voxel readout of the signal S * rho
for mu in MUS_:
    Reg = np.zeros_like(Gf); Reg[3:, 3:] = sc * (LAM_D * L + mu * np.eye(nu)); Ah = Gf + Reg + 1e-9 * sc * np.eye(nu + 3)
    Z = np.einsum("pq,rqc->rpc", np.linalg.inv(Ah), RHS); rho = np.einsum("pc,rcq->rpq", Mc, Z[:, :3]) + Z[:, 3:]
    Fv = np.einsum("vp,rpq->rvq", Rd, rho) @ V.conj().T
    Ahi = np.linalg.inv(Ah); recf = lambda y, Ahi=Ahi: Rd @ (Mc @ (Ahi @ (Af.conj().T @ (dcf * y)))[:3] + (Ahi @ (Af.conj().T @ (dcf * y)))[3:])
    nm = f"hybrid μ {mu:g}"; RES[nm] = evaluate(Fv); RES[nm]["spill"] = spill(recf)
    if os.environ.get("SAVE_FID") and mu in [float(x) for x in os.environ["SAVE_FID"].split(",")]:                     # voxel FIDs (NR, N*N, n), recon frame, same convention as adj_fid
        np.save(f"{OUT}/fid_hybrid_mu{mu:g}.npy", Fv.astype(np.complex64)); log(f"saved voxel FIDs {OUT}/fid_hybrid_mu{mu:g}.npy")
    if UNC:
        Rfull = Rd @ np.hstack([Mc, np.eye(nu)]); P = Rfull @ Ahi @ (Af.conj().T * dcf)            # CSI voxel <- k-space, per subspace coefficient
        UO[f"mu{mu:g}_var"] = sig2 * (np.abs(P)**2).sum(1)
        res_ = float(np.sum(dcf[None, :, None] * np.abs(KV - np.einsum("kp,rpc->rkc", Af, Z))**2)); dl = Z[:, 3:]
        UO[f"mu{mu:g}_lcurve"] = np.array([res_, float(np.sum(np.abs(dl)**2)), float(np.real(np.einsum("rpc,pq,rqc->", dl.conj(), L, dl)))])
        Zr = Z[R0:R1].mean(0); cpart = Rd @ (Mc @ Zr[:3]); dpart = Rd @ Zr[3:]
        UO[f"mu{mu:g}_dshare"] = np.linalg.norm(dpart, axis=1) / (np.linalg.norm(cpart + dpart, axis=1) + 1e-30)
        if mu in SPEC_MUS: rep_unc(Fv, f"mu{mu:g}")
    Sc_ = (Z[:, :3] @ V.conj().T); fr = np.linalg.norm(np.einsum("pc,rcq->rpq", Mc, Z[:, :3])) / np.linalg.norm(rho)
    if os.environ.get("SAVE_COMP") and mu in [float(x) for x in os.environ["SAVE_COMP"].split(",")]:                   # unmixed compartment spectra s_c (NR, 3, n): GM, soleus, other
        frac = np.zeros((N * N, 3)); np.add.at(frac, blk[pix], Mc); frac /= (Nh / N)**2                                 # compartment area fraction per CSI voxel
        np.savez(f"{OUT}/comp_mu{mu:g}.npz", s=Sc_.astype(np.complex64), frac=frac, delta_share=1 - fr); log(f"saved compartment spectra {OUT}/comp_mu{mu:g}.npz")
    RES[nm]["delta_share"] = 1 - fr
log(f"\n{'method':32s} | GM PCr  noise  Pi    β-ATP | SOL PCr  noise  Pi    β-ATP | GM−SOL | voxel tSNR GM / SOL | spill GM→SOL  SOL→GM | 2 mm test  +3 mm shift | δ share")
for nm, r in RES.items():
    g, s_ = r["g"], r["s"]
    log(f"{nm:32s} | {g['pcr']:5.1f} {g['noise']:5.1f} {g['pi']:+5.1f} {g['beta']:+6.1f} | {s_['pcr']:5.1f} {s_['noise']:6.1f} {s_['pi']:+5.1f} {s_['beta']:+6.1f} | {g['pcr'] - s_['pcr']:+5.1f}  | "
        f"{np.nanmedian(r['tsnr'][vGM]):5.1f} / {np.nanmedian(r['tsnr'][vSOL]):5.1f}      | {100 * r['spill']['g2s']:6.0f}%  {100 * r['spill']['s2g']:6.0f}% | "
        f"{100 * r['spill'].get('g2s_2mm', float('nan')):6.0f}%    {100 * r['spill'].get('g2s_2mm_shift3', float('nan')):6.0f}%    | {r.get('delta_share', float('nan')):.2f}")
np.savez(f"{OUT}/hybrid_MID00{DS}.npz", Q=Q, exwin=np.array([EX.start, EX.stop]), **{f"{i}_{k}": r[k] for i, r in enumerate(RES.values()) for k in ("srest", "sex") if k in r}, names=np.array(list(RES)), vGM=vGM, vSOL=vSOL, legv=legv, N=N, FOV=FOV, REP=REP, rest=np.array([R0, R1]), ex=np.array([E0, E1]),
         gm_ids=np.array(gm_ids), so_ids=np.array(so_ids), loc=meta["loc"].split("/")[-1], x0=meta["x0"], y0=meta["y0"], bw=meta["bw"], bh=meta["bh"], leg_idx=np.flatnonzero(legv),
         **{f"{i}_{k}": np.asarray(v) for i, (nm, r) in enumerate(RES.items()) for k, v in (("dep", r["dep"]), ("inc", r["inc"]), ("tsnr", r["tsnr"]), ("vox_pcr", r["vox_pcr"]),
            ("vox_pi", r["vox_pi"]), ("vox_r0", r["vox_r0"]), ("g_curve", r["g"]["curve"]), ("s_curve", r["s"]["curve"]), ("g_pic", r["g"]["pic"]), ("s_pic", r["s"]["pic"]),
            ("summary", np.array([r["g"]["pcr"], r["g"]["noise"], r["g"]["pi"], r["g"]["beta"], r["s"]["pcr"], r["s"]["noise"], r["s"]["pi"], r["s"]["beta"],
                                   np.nanmedian(r["tsnr"][vGM]), np.nanmedian(r["tsnr"][vSOL]), r["spill"]["g2s"], r["spill"]["s2g"]])))})
if UNC: np.savez(f"{OUT}/unc_MID00{DS}.npz", Q=Q, MUS=np.array(MUS_), N=N, **UO); log(f"saved {OUT}/unc_MID00{DS}.npz")
log(f"saved {OUT}")
