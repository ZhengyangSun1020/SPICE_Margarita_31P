#!/usr/bin/env python3
"""24-06 session, MID00644 (10x10, 200 mm, 52 reps): build the same per-scan inputs as the 24-07 study (study.py + make_fids.py
+ system11.py) so spiceD.py runs unchanged:
  reg_TRUE (leg from the localizer: Otsu + largest component, voxel = leg if >50% covered; no phantom), comp_TRUE (placeholder:
  whole leg = 'other calf' until GM/soleus masks are drawn), system.npz (G, 80-ring MID00643 PCr prior), system11.npz (soft
  boundary, lambda = 0.0251 as MID00283 = same 10x10/200 mm grid), adj_fid.npy (un-apodised), adj_spec.npy, ppm.npy, meta.json.
Usage: prep644.py [rest0 rest1 ex0 ex1]"""
import os, sys, json, time, glob, importlib.util as ilu
import numpy as np, pydicom, mrinufft
from scipy import ndimage as ndi

def threshold_otsu(x, nb=256):
    h, e = np.histogram(x.ravel(), nb); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
    w1 = w[-1] - w; v = (mu[-1] * w / w[-1] - mu) ** 2 / np.maximum(w * w1, 1)   # between-class variance (up to a constant)
    return c[np.argmax(v)]

ROOT = "/home/zsun/SPICE_Margarita"; D = "/mnt/e/claude_tmp/study644"
DAT = f"{D}/CRT/meas_MID00644_FID214692_LV_crt_dynamic_10x10_5rings.dat"; DAT80 = f"{D}/CRT/meas_MID00643_FID214691_LV_crt_static_80rings_TR2s.dat"
O = os.path.join(ROOT, "output_study_mid644"); os.makedirs(O, exist_ok=True)
REST, EX = ((0, 4), (4, 26)) if len(sys.argv) < 5 else ((int(sys.argv[1]), int(sys.argv[2])), (int(sys.argv[3]), int(sys.argv[4])))
RANK, LAM, APOD, DELTA = 16, 0.0251, 10.0, 2.0
m = ilu.module_from_spec(ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py")); ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py").loader.exec_module(m)
from recon_concept_twix_31p import concept_phase_correction
info0 = m.load_concept_twix_3d(DAT, rep_sel=0)
N = int(info0["matrix_size"]); FOV = float(info0["fov_mm"]); CX, CY, CZ = float(info0["pos_sag"]), float(info0["pos_cor"]), float(info0["pos_tra"])
n_spec = info0["n_spec"]; dwell = info0["dwell_s"] * 2; larmor = info0["larmor_freq_mhz"]
NR = len(np.unique(np.asarray(info0.get("reps", np.arange(52))))) if "reps" in info0 else 52

# ---------------- localizer slice closest to the MRSI slab centre ----------------
cands = [(abs(float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]) - CZ), f) for f in glob.glob(f"{D}/dicoms/MR localizer_tra*/*.dcm")]
LOC = min(cands)[1]; ds = pydicom.dcmread(LOC); ipp = np.array(ds.ImagePositionPatient, float); ps_r, ps_c = map(float, ds.PixelSpacing)
assert np.allclose(np.array(ds.ImageOrientationPatient, float), [1, 0, 0, 0, 1, 0], atol=1e-3)
img = ds.pixel_array.astype(float); bw = img > threshold_otsu(img); bw = ndi.binary_fill_holes(ndi.binary_opening(bw, iterations=2))
lab, nl = ndi.label(bw); body = lab == (1 + np.argmax(ndi.sum(bw, lab, range(1, nl + 1))))
d = FOV / N; sub = (np.arange(8) + 0.5) / 8 - 0.5; REG = np.zeros((N, N), int); FRAC = np.zeros((N, N))
for xi in range(N):                                 # recon frame: xi -> y reversed (display row = N-1-xi), yi -> x
    for yi in range(N):
        px, py = np.meshgrid(CX - FOV / 2 + (yi + 0.5 + sub) * d, CY - FOV / 2 + (N - 1 - xi + 0.5 + sub) * d)
        c = np.round((px - ipp[0]) / ps_c).astype(int); r = np.round((py - ipp[1]) / ps_r).astype(int)
        ok = (r >= 0) & (r < img.shape[0]) & (c >= 0) & (c < img.shape[1]); FRAC[xi, yi] = body[r[ok], c[ok]].mean() if ok.any() else 0
REG[FRAC > 0.5] = 1; LEG = REG == 1; COMP = np.where(LEG, 3, 0)
np.save(os.path.join(O, "reg_TRUE.npy"), REG); np.save(os.path.join(O, "comp_TRUE.npy"), COMP); np.save(os.path.join(O, "leg_frac.npy"), FRAC)
print(f"localizer {os.path.relpath(LOC, D)} (z={ipp[2]:.1f}, slab {CZ:.1f}); leg voxels {int(LEG.sum())}; display (row 1 = top):\n{REG[::-1]}")
meta = dict(loc=LOC, fov=FOV, n=N, voxel_mm=FOV / N, x0=(CX - FOV / 2 - ipp[0]) / ps_c - 0.5, y0=(CY - FOV / 2 - ipp[1]) / ps_r - 0.5,
            bw=FOV / ps_c, bh=FOV / ps_r, nreps=NR, rest=list(REST), ex=list(EX), rec=[EX[1], NR], rank=RANK, lam=LAM, apod_hz=APOD,
            note="comp_TRUE is a placeholder (whole leg = 3) until GM/soleus masks are drawn")
json.dump(meta, open(os.path.join(O, "meta.json"), "w"), indent=1)

# ---------------- operator + G (as make_fids.py / study.py) ----------------
def make_op(info):
    tj = m.build_trajectory_2d(info["first_circle"], info["n_rings"], int(info["traj_pts"].max())).copy()
    tj[:, :2] *= float(info["k_norm_max_phys"]) / 0.5 * (FOV / N) / (float(info["fov_mm"]) / float(info["matrix_size"]))
    r = np.sqrt(tj[:, 0]**2 + tj[:, 1]**2)
    return tj, mrinufft.get_operator("finufft", samples=tj, shape=(N, N), n_coils=1, density=m.compute_crt_dcf_hamming(tj, hamming_radius=r))
traj, op = make_op(info0)
# extra FOV shift: -0.5 voxel puts the mrinufft grid (k=0 on an integer pixel) on the symmetric voxel centres of
# CONCEPTPerform2DDirectFT (idx - N/2 - 0.5) and of the scanner geometry; was -1 (empirical) before 2026-10-03
EXTRA = float(os.environ.get("FOV_EXTRA_SHIFT", "-0.5"))
dpe, dro = m.compute_fov_shift(info0); dpe += EXTRA; dro += EXTRA
shift = np.exp(-2j * np.pi * (traj[:, 0] * dpe + traj[:, 1] * dro))
G = np.zeros((N * N, N * N), np.complex128)
for v in range(N * N):
    e = np.zeros((N, N), np.complex64); e.flat[v] = 1; G[:, v] = op.adj_op(op.op(e)).reshape(N * N)

# ---------------- 80-ring prior (MID00643: same FOV / centre) ----------------
i80 = m.load_concept_twix_3d(DAT80, rep_sel=0); t80, op80 = make_op(i80)
tgt = dict(i80); tgt.update(pos_sag=CX, pos_cor=CY, pos_tra=CZ, fov_mm=FOV, matrix_size=N)
p_pe, p_ro = m.compute_fov_shift(tgt); p_pe += EXTRA; p_ro += EXTRA
ns80, dw80 = i80["n_spec"], i80["dwell_s"] * 2
kf80 = concept_phase_correction(i80["kdata"], i80["_ti_per_ring"])[:, :, 0, :].transpose(2, 0, 1).reshape(-1, ns80)
kf80 = kf80 * np.exp(-2j * np.pi * (t80[:, 0] * p_pe + t80[:, 1] * p_ro))[:, None]
a80 = np.stack([op80.adj_op(kf80[:, k]).reshape(N, N) for k in range(ns80)], -1)
ap80 = np.exp(-np.pi**2 * APOD**2 * (np.arange(ns80) * dw80)**2 / (4 * np.log(2)))
pp80 = -np.fft.fftshift(np.fft.fftfreq(ns80, d=dw80)) / i80["larmor_freq_mhz"]
S80 = np.abs(m.FIDToSpec((a80 * ap80).conj(), axis=-1)); k0 = np.abs(pp80) < 1.5
pc80 = S80[..., k0].max(-1); prior = np.clip(pc80 / pc80.max(), 0, 1).astype(np.float32)
print("80-ring PCr prior (% of max), display:\n", np.round(100 * prior[::-1]).astype(int))
np.savez(os.path.join(O, "system.npz"), G=G, prior=prior, lam=LAM)

# ---------------- system11 (identical rule to system11.py; no phantom here) ----------------
_, (ii, jj) = ndi.distance_transform_edt(~LEG, return_indices=True); cmp = np.where(LEG, COMP, COMP[ii, jj])
lo, hi = prior[LEG].min(), prior[LEG].max()
p = np.clip((prior - lo) / (hi - lo + 1e-12), 0, 1) + DELTA * np.select([cmp == 1, cmp == 2], [0.0, 1.0], 2.0)
WW, _ = m.calc_Bmatrix_simple(p.astype(np.float32), wmax=5e4, adj=8, label_mask=np.ones((N, N), int).T)
dd = ndi.distance_transform_cdt(~LEG, metric="chessboard").astype(float); lout = np.minimum(1.0, 0.1 * dd**2)
np.savez(os.path.join(O, "system11.npz"), G=G, WW=WW, out=lout > 0, lout_map=lout, prior=p, lam=LAM, lout=1.0, delta=DELTA)

# ---------------- adjoint FIDs / spectra, all reps ----------------
apo = np.exp(-np.pi**2 * APOD**2 * (np.arange(n_spec) * dwell)**2 / (4 * np.log(2)))
fid = np.zeros((NR, N * N, n_spec), np.complex64); spec = np.zeros((NR, N, N, n_spec), np.complex64); t0 = time.time()
for r in range(NR):
    info = info0 if r == 0 else m.load_concept_twix_3d(DAT, rep_sel=r)
    kf = concept_phase_correction(info["kdata"], info["_ti_per_ring"])[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec) * shift[:, None]
    a = np.stack([op.adj_op(kf[:, k]).reshape(N * N) for k in range(n_spec)], -1); fid[r] = a
    spec[r] = m.FIDToSpec((a.reshape(N, N, n_spec) * apo).conj(), axis=-1)
    if r % 10 == 0: print(f"  rep {r+1}/{NR} ({time.time()-t0:.0f}s)", flush=True)
PPM = -np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell)) / larmor
np.save(os.path.join(O, "adj_fid.npy"), fid); np.save(os.path.join(O, "adj_spec.npy"), spec); np.save(os.path.join(O, "ppm.npy"), PPM)
M = np.abs(spec); kk = np.abs(PPM) < 1.5; pcr = M[..., kk].max(-1); np.save(os.path.join(O, "adj_pcr.npy"), pcr)
curve = pcr[:, LEG].mean(1); print("NUFFT leg-mean PCr per rep (% of rep 0):", np.round(100 * curve / curve[0]).astype(int).tolist())
print("done", O)
