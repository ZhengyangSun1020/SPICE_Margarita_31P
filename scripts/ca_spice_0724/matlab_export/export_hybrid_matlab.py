#!/usr/bin/env python3
"""Export the hybrid SLAM+SPICE voxel FIDs (mu 30, drawn pixel masks) for MATLAB fitting, same .mat format as export_matlab.py:
  /mnt/e/claude_tmp/study/recon_for_matlab/hybrid/MID00{ds}_hybrid.mat
fid [nPts x N x N x nRep] complex single, display orientation (row 1 = top of the localizer, col 1 = image left), no apodisation.
One global scale factor puts the leg-summed rest PCr at the level of the adjoint NUFFT (ratios are unchanged).
muscle_mask uses the voxel labels from the drawn masks (masks_hr/mask_labels.json). Usage: export_hybrid_matlab.py 283"""
import os, sys, json, numpy as np, scipy.io as sio
ROOT = "/home/zsun/SPICE_Margarita"; OUT = os.environ.get("EXPORT_DIR", "/mnt/e/claude_tmp/study/recon_for_matlab/hybrid"); os.makedirs(OUT, exist_ok=True)
DWELL, LARMOR = 0.00036, 49.894558; MU = os.environ.get("MU", "30"); ds = sys.argv[1] if len(sys.argv) > 1 else "283"
O = f"{ROOT}/output_study_mid{ds}"; meta = json.load(open(f"{O}/meta.json")); N = meta["n"]
H = np.load(f"{O}/{os.environ.get('HYBFID', 'hybrid_v2_fid')}/fid_hybrid_mu{MU}.npy").astype(np.complex128); NR, _, n = H.shape
adj = np.load(f"{O}/adj_fid.npy").astype(np.complex128); reg = np.load(f"{O}/reg_TRUE.npy")
lab = json.load(open("/mnt/e/claude_tmp/study/masks_hr/mask_labels.json"))[ds]; disp2rec = lambda ids: (N - 1 - np.asarray(ids) // N) * N + np.asarray(ids) % N
mus = np.where(reg.ravel() == 1, 3, 0); mus[disp2rec(lab["so"])] = 2; mus[disp2rec(lab["gm"])] = 1; leg = mus > 0
# global scale: leg-summed rest-mean magnitude spectrum (10 Hz LB), PCr peak, hybrid -> NUFFT units
t = np.arange(n) * DWELL; lb = np.exp(-np.pi * 10 * t); R0, R1 = meta["rest"]
pk = lambda X: np.abs(np.fft.fft(X[R0:R1].mean(0)[leg] * lb, axis=-1)).sum(0).max()
scale = pk(adj) / pk(H); H *= scale
nz = np.abs(H).max(axis=(0, 2)) > 0
print(f"MID00{ds}: hybrid mu {MU}, {NR} reps x {n} pts, scale to NUFFT units x{scale:.4g}; voxels with signal {nz.sum()}/{N * N} "
      f"(leg {leg.sum()}, leg voxels without muscle pixels {int((leg & ~nz).sum())})")
disp = lambda x: x.reshape(x.shape[0], N, N, n)[:, ::-1, :, :]
rc = lambda ids: np.array([[q // N + 1, q % N + 1] for q in ids], dtype=np.int16)              # display id -> 1-based [row col]
F = np.transpose(disp(H), (3, 1, 2, 0)).astype(np.complex64)
sio.savemat(f"{OUT}/MID00{ds}_hybrid.mat", dict(
    fid=F, dwell=DWELL, imagingFrequency=LARMOR, ppmAxis=np.fft.fftshift(np.fft.fftfreq(n, d=DWELL)) / LARMOR, voxel_mm=meta["voxel_mm"], fov_mm=meta["fov"], matrix=N,
    leg_phantom_mask=reg[::-1, :].astype(np.uint8), muscle_mask=mus.reshape(N, N)[::-1, :].astype(np.uint8),
    mask_labels="leg_phantom_mask: 1 leg, 2 phantom; muscle_mask: 1 GM, 2 soleus, 3 other calf (voxel labels = >= 50% of the voxel area in the drawn masks)",
    gm_voxels_rowcol=rc(lab["gm"]), soleus_voxels_rowcol=rc(lab["so"]),
    rest_reps=np.arange(*meta["rest"]) + 1, exercise_reps=np.arange(*meta["ex"]) + 1, recovery_reps=np.arange(meta["ex"][1], NR) + 1, localizer=meta["loc"],
    method=(f"hybrid SLAM+SPICE (mu {MU}): fine grid 3x the CSI grid; tissue per pixel = sum over GM/soleus/other of drawn-mask area fraction x compartment spectrum "
            "+ local deviation (label-graph smoothing lam 1, ridge mu); real CRT k-space, DCF weighting, rank-16 subspace (variant D), smooth complex coil sensitivity "
            "fitted to the 80-ring PCr map; read out per CSI voxel. Voxels with no muscle pixel (phantom, background) are exactly zero."),
    spice_lambda=1.0, spice_rank=int(os.environ.get("RANK", 16)), hybrid_mu=float(MU), scale_to_nufft=scale,
    note="fid(:, row, col, rep); row 1 = top of the localizer image (anterior), col 1 = image left. No apodisation applied."), do_compression=True)
print("saved", f"{OUT}/MID00{ds}_hybrid.mat", F.shape)
