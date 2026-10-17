#!/usr/bin/env python3
"""Soft leg boundary (system11): one smoothing graph over all non-phantom voxels; prior = 80-ring PCr map scaled to the
leg range everywhere (not cut at the leg edge) + delta * compartment offset (outside the leg: offset of the nearest leg
voxel), so edges across the leg edge are only weakened by the prior difference, not removed. lambda_out grows with the
distance d (voxels, 8-connected chessboard) from the leg ROI: 0.1*d^2, capped at 1; phantom (no 31P) lambda_out = 1, no
edges. Same lambda / delta as system9/10. Saves output_study_mid{ds}/system11.npz. Usage: system11.py <ds>"""
import os, sys, json, importlib.util as ilu
import numpy as np
from scipy.ndimage import distance_transform_edt, distance_transform_cdt

DS = sys.argv[1]; DELTA = 2.0
ROOT = "/home/zsun/SPICE_Margarita"; O = os.path.join(ROOT, f"output_study_mid{DS}")
s10 = np.load(os.path.join(O, "system10.npz")); LAM = float(s10["lam"])
meta = json.load(open(os.path.join(O, "meta.json"))); N = meta["n"]
reg = np.load(os.path.join(O, "reg_TRUE.npy")); comp = np.load(os.path.join(O, "comp_TRUE.npy")); leg, ph = reg == 1, reg == 2
s0 = np.load(os.path.join(O, "system.npz")); G = s0["G"]; p_raw = s0["prior"]
m = ilu.module_from_spec(ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py")); ilu.spec_from_file_location("m", ROOT + "/scripts/05c_spice_recon_31p.py").loader.exec_module(m)
_, (ii, jj) = distance_transform_edt(~leg, return_indices=True)
cmp = np.where(leg, comp, comp[ii, jj])                                          # compartment of nearest leg voxel
lo, hi = p_raw[leg].min(), p_raw[leg].max()
p = np.clip((p_raw - lo) / (hi - lo + 1e-12), 0, 1) + DELTA * np.select([cmp == 1, cmp == 2], [0.0, 1.0], 2.0)
p[ph] = 0.0
lab = np.where(ph, 0, 1)
WW, _ = m.calc_Bmatrix_simple(p.astype(np.float32), wmax=5e4, adj=8, label_mask=lab.T)
d = distance_transform_cdt(~leg, metric="chessboard").astype(float)
lout = np.minimum(1.0, 0.1 * d**2); lout[ph] = 1.0
np.savez(os.path.join(O, "system11.npz"), G=G, WW=WW, out=lout > 0, lout_map=lout, prior=p, lam=LAM, lout=1.0, delta=DELTA)
print(f"MID00{DS}: system11 saved — soft boundary, lambda_out per distance: " + ", ".join(f"d={k}: {int((d == k).sum())} vox" for k in range(1, 5)) + f", phantom {int(ph.sum())}")
