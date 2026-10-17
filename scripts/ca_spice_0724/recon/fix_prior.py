#!/usr/bin/env python3
"""Rebuild the 80-ring PCr guidance image ('prior' in system.npz) for one bout, keeping only 80-ring k samples inside the bout grid's Nyquist
(|k| <= 0.5 after rescaling to the bout's voxel size). For a coarser bout grid (MID00279: 30 mm vs the 80-ring 24 mm) the rescaled 80-ring
k-space extends past 0.5 and the NUFFT adjoint aliases (periodic wrap) -> a ramp instead of a PCr map. Same steps as rebuild_study.py otherwise.
Old prior kept as system.npz['prior_old']. Usage: fix_prior.py <ds> [--dry]"""
import os, sys, json, importlib.util as ilu
import numpy as np, mrinufft
ROOT = "/home/zsun/SPICE_Margarita"; RAW = "/mnt/e/claude_tmp/study/raw"
DATS = {"277": "meas_MID00277_FID221482_240FOV_10x10_5rings.dat", "279": "meas_MID00279_FID221484_240FOV_8x8_5rings.dat",
        "281": "meas_MID00281_FID221486_200FOV_8x8_5rings.dat", "283": "meas_MID00283_FID221488_200FOV_10x10_5rings.dat"}
DAT80 = f"{RAW}/meas_MID00276_FID221481_240FOV_10x10_80rings.dat"
DS = sys.argv[1]; DRY = "--dry" in sys.argv; O = os.path.join(ROOT, f"output_study_mid{DS}"); meta = json.load(open(os.path.join(O, "meta.json")))
N, APOD, FOV = meta["n"], meta["apod_hz"], meta["fov"]; EXTRA = meta.get("fov_extra_shift", -0.5)
reg = np.load(os.path.join(O, "reg_TRUE.npy")); PH = reg == 2
m = ilu.module_from_spec(ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py")); ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py").loader.exec_module(m)
from recon_concept_twix_31p import concept_phase_correction
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): info0 = m.load_concept_twix_3d(os.path.join(RAW, DATS[DS]), rep_sel=0); i80 = m.load_concept_twix_3d(DAT80, rep_sel=0)
CX, CY = float(info0["pos_sag"]), float(info0["pos_cor"])
tj = m.build_trajectory_2d(i80["first_circle"], i80["n_rings"], int(i80["traj_pts"].max())).copy()
tj[:, :2] *= float(i80["k_norm_max_phys"]) / 0.5 * (FOV / N) / (float(i80["fov_mm"]) / float(i80["matrix_size"]))
kr = np.hypot(tj[:, 0], tj[:, 1]); keep = kr <= 0.5
print(f"MID00{DS}: {N}x{N}, {FOV / N:.0f} mm; 80-ring k max on this grid {kr.max():.3f} (Nyquist 0.5); samples beyond Nyquist {100 * (~keep).mean():.0f}%")
if DRY: sys.exit(0)
t80 = tj[keep]; op80 = mrinufft.get_operator("finufft", samples=t80, shape=(N, N), n_coils=1, density=m.compute_crt_dcf_hamming(t80, hamming_radius=np.hypot(t80[:, 0], t80[:, 1])))
tgt = dict(i80); tgt.update(pos_sag=CX, pos_cor=CY, pos_tra=info0.get("pos_tra", 0.0), fov_mm=FOV, matrix_size=N)
p_pe, p_ro = m.compute_fov_shift(tgt); p_pe += EXTRA; p_ro += EXTRA
ns80, dw80 = i80["n_spec"], i80["dwell_s"] * 2
kf80 = concept_phase_correction(i80["kdata"], i80["_ti_per_ring"])[:, :, 0, :].transpose(2, 0, 1).reshape(-1, ns80)[keep]
kf80 = kf80 * np.exp(-2j * np.pi * (t80[:, 0] * p_pe + t80[:, 1] * p_ro))[:, None]
a80 = np.stack([op80.adj_op(kf80[:, k]).reshape(N, N) for k in range(ns80)], -1)
ap80 = np.exp(-np.pi**2 * APOD**2 * (np.arange(ns80) * dw80)**2 / (4 * np.log(2)))
pp80 = -np.fft.fftshift(np.fft.fftfreq(ns80, d=dw80)) / i80["larmor_freq_mhz"]
pc80 = np.abs(m.FIDToSpec((a80 * ap80).conj(), axis=-1))[..., (pp80 >= -1) & (pp80 <= 1)].max(-1)
prior = np.clip(pc80 / pc80[~PH].max(), 0, 1).astype(np.float32)
s0 = dict(np.load(os.path.join(O, "system.npz")))
if "prior_old" not in s0: s0["prior_old"] = s0["prior"]
s0["prior"] = prior; np.savez(os.path.join(O, "system.npz"), **s0)
pn = np.load(os.path.join(O, "adj_pcr.npy")); pn = pn[: min(5, pn.shape[0])].mean(0) if pn.ndim == 3 else pn
print("new prior (display grid, row 0 = top, % of max):"); [print("   " + "".join(f"{int(round(100 * x)):4d}" for x in row)) for row in prior[::-1]]
