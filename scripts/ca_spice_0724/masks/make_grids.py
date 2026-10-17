#!/usr/bin/env python3
"""24-07: the four localizers (same slice, z = -16.6 mm) with each bout's CSI grid (symmetric voxel centres = corrected recon geometry)
and voxel numbers (row*N+col, row 0 = top). -> E:/claude_tmp/study/masks_hr/grid_MID00xxx.png"""
import glob, json, numpy as np, pydicom, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import sys; sys.path.insert(0, "/mnt/e/claude_tmp/study/masks_hr"); from phantom_mask import phantom_mask
ROOT = "/home/zsun/SPICE_Margarita"; OUT = "/mnt/e/claude_tmp/study/masks_hr"
fig_all, axs = plt.subplots(1, 4, figsize=(36, 9.5))
for a2, ds_ in zip(axs, ("279", "281", "277", "283")):
    meta = json.load(open(f"{ROOT}/output_study_mid{ds_}/meta.json")); N = meta["n"]; p = meta["loc"]
    p = p if p.startswith("/mnt/e") else "/mnt/e/claude_tmp/study/loc/" + p.split("/")[-1]
    ds = pydicom.dcmread(p); im = ds.pixel_array.astype(float)
    ph = phantom_mask(im); im[ph] = 0                                   # phantom has no 31P
    v = im[~ph]; IMG = np.clip((im - np.percentile(v, 1)) / (np.percentile(v, 99.5) - np.percentile(v, 1)), 0, 1)
    x0, y0, cw, ch = meta["x0"], meta["y0"], meta["bw"] / N, meta["bh"] / N
    fig, ax = plt.subplots(figsize=(12, 12))
    for a in (ax, a2):
        a.imshow(IMG, cmap="gray")
        for i in range(N + 1): a.plot([x0, x0 + N * cw], [y0 + i * ch] * 2, color="y", lw=0.8); a.plot([x0 + i * cw] * 2, [y0, y0 + N * ch], color="y", lw=0.8)
        for r in range(N):
            for c in range(N): a.text(x0 + (c + 0.5) * cw, y0 + (r + 0.5) * ch, f"{r * N + c}", color="yellow", fontsize=12 if a is ax else 7, ha="center", va="center", fontweight="bold")
        a.set_xticks([]); a.set_yticks([]); a.set_title(f"MID00{ds_}: {N}x{N}, FOV {meta['fov']:.0f} mm ({meta['voxel_mm']:.0f} mm voxels) — {p.split('/')[-1][:44]}", fontsize=13 if a is ax else 11)
    fig.tight_layout(); fig.savefig(f"{OUT}/grid_MID00{ds_}.png", dpi=90); plt.close(fig)
fig_all.tight_layout(); fig_all.savefig(f"{OUT}/grids_0724_all.png", dpi=70); print("saved")
