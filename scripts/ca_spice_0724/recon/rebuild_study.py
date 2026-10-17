#!/usr/bin/env python3
"""24-07 study bout: redo everything that depends on the FOV shift with the corrected extra shift (-0.5/-0.5 instead of -1/-1):
80-ring PCr prior (-> system.npz 'prior', G kept), un-apodised adjoint FIDs (adj_fid.npy), apodised adjoint spectra and PCr/Pi
peak maps (adj_spec/adj_pcr/adj_pi.npy). Masks (geometric, from the localizer), lambda and G are unchanged.
Then run system11.py and spiceD.py. Usage: rebuild_study.py <277|279|281|283>"""
import os, sys, json, time, importlib.util as ilu
import numpy as np, mrinufft
ROOT = "/home/zsun/SPICE_Margarita"; RAW = "/mnt/e/claude_tmp/study/raw"
DATS = {"277": "meas_MID00277_FID221482_240FOV_10x10_5rings.dat", "279": "meas_MID00279_FID221484_240FOV_8x8_5rings.dat",
        "281": "meas_MID00281_FID221486_200FOV_8x8_5rings.dat", "283": "meas_MID00283_FID221488_200FOV_10x10_5rings.dat"}
DAT80 = f"{RAW}/meas_MID00276_FID221481_240FOV_10x10_80rings.dat"
EXTRA = float(os.environ.get("FOV_EXTRA_SHIFT", "-0.5"))
DS = sys.argv[1]; O = os.path.join(ROOT, f"output_study_mid{DS}"); meta = json.load(open(os.path.join(O, "meta.json")))
N, NR, APOD, FOV = meta["n"], meta["nreps"], meta["apod_hz"], meta["fov"]
reg = np.load(os.path.join(O, "reg_TRUE.npy")); PH = reg == 2
m = ilu.module_from_spec(ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py")); ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py").loader.exec_module(m)
from recon_concept_twix_31p import concept_phase_correction
DAT = os.path.join(RAW, DATS[DS]); info0 = m.load_concept_twix_3d(DAT, rep_sel=0)
CX, CY = float(info0["pos_sag"]), float(info0["pos_cor"]); n_spec = info0["n_spec"]; dwell = info0["dwell_s"] * 2; larmor = info0["larmor_freq_mhz"]
def make_op(info):
    tj = m.build_trajectory_2d(info["first_circle"], info["n_rings"], int(info["traj_pts"].max())).copy()
    tj[:, :2] *= float(info["k_norm_max_phys"]) / 0.5 * (FOV / N) / (float(info["fov_mm"]) / float(info["matrix_size"]))
    r = np.sqrt(tj[:, 0]**2 + tj[:, 1]**2)
    return tj, mrinufft.get_operator("finufft", samples=tj, shape=(N, N), n_coils=1, density=m.compute_crt_dcf_hamming(tj, hamming_radius=r))
# ---- 80-ring prior (as study.py, corrected shift) ----
i80 = m.load_concept_twix_3d(DAT80, rep_sel=0); t80, op80 = make_op(i80)
tgt = dict(i80); tgt.update(pos_sag=CX, pos_cor=CY, pos_tra=info0.get("pos_tra", 0.0), fov_mm=FOV, matrix_size=N)
p_pe, p_ro = m.compute_fov_shift(tgt); p_pe += EXTRA; p_ro += EXTRA
ns80, dw80 = i80["n_spec"], i80["dwell_s"] * 2
kf80 = concept_phase_correction(i80["kdata"], i80["_ti_per_ring"])[:, :, 0, :].transpose(2, 0, 1).reshape(-1, ns80)
kf80 = kf80 * np.exp(-2j * np.pi * (t80[:, 0] * p_pe + t80[:, 1] * p_ro))[:, None]
a80 = np.stack([op80.adj_op(kf80[:, k]).reshape(N, N) for k in range(ns80)], -1)
ap80 = np.exp(-np.pi**2 * APOD**2 * (np.arange(ns80) * dw80)**2 / (4 * np.log(2)))
pp80 = -np.fft.fftshift(np.fft.fftfreq(ns80, d=dw80)) / i80["larmor_freq_mhz"]
pc80 = np.abs(m.FIDToSpec((a80 * ap80).conj(), axis=-1))[..., (pp80 >= -1) & (pp80 <= 1)].max(-1)
prior = np.clip(pc80 / pc80[~PH].max(), 0, 1).astype(np.float32)
s0 = dict(np.load(os.path.join(O, "system.npz"))); s0["prior"] = prior; np.savez(os.path.join(O, "system.npz"), **s0)
# ---- adjoint FIDs / spectra, all reps ----
traj, op = make_op(info0)
dpe, dro = m.compute_fov_shift(info0); dpe += EXTRA; dro += EXTRA
shift = np.exp(-2j * np.pi * (traj[:, 0] * dpe + traj[:, 1] * dro))
apo = np.exp(-np.pi**2 * APOD**2 * (np.arange(n_spec) * dwell)**2 / (4 * np.log(2)))
PPM = -np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell)) / larmor; B_PCR = (PPM >= -1.0) & (PPM <= 1.0); B_PI = (PPM >= 3.8) & (PPM <= 5.3)
fid = np.zeros((NR, N * N, n_spec), np.complex64); spec = np.zeros((NR, N, N, n_spec), np.complex64); t0 = time.time()
for r in range(NR):
    info = info0 if r == 0 else m.load_concept_twix_3d(DAT, rep_sel=r)
    kf = concept_phase_correction(info["kdata"], info["_ti_per_ring"])[:, :, 0, :].transpose(2, 0, 1).reshape(-1, n_spec) * shift[:, None]
    a = np.stack([op.adj_op(kf[:, k]).reshape(N * N) for k in range(n_spec)], -1); fid[r] = a
    spec[r] = m.FIDToSpec((a.reshape(N, N, n_spec) * apo).conj(), axis=-1)
np.save(os.path.join(O, "adj_fid.npy"), fid); np.save(os.path.join(O, "adj_spec.npy"), spec); M = np.abs(spec)
np.save(os.path.join(O, "adj_pcr.npy"), M[..., B_PCR].max(-1)); np.save(os.path.join(O, "adj_pi.npy"), M[..., B_PI].max(-1)); np.save(os.path.join(O, "ppm.npy"), PPM)
meta["fov_extra_shift"] = EXTRA; json.dump(meta, open(os.path.join(O, "meta.json"), "w"), indent=1)
print(f"MID00{DS}: rebuilt prior + adjoint ({NR} reps, {time.time() - t0:.0f}s) with extra FOV shift {EXTRA}")
