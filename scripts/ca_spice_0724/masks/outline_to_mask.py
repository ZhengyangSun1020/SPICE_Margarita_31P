#!/usr/bin/env python3
"""Turn the user's coloured outlines on a screenshot of the Series-18 display (green = soleus, blue + red = gastrocnemius heads) into masks on the
Series-18 grid: (1) register the screenshot to the displayed image (loc_enhanced_s18.mat) - scale + shift from the leg masks, refined by NCC;
(2) fill each closed outline; (3) clean up: one shared boundary between neighbouring regions (gaps filled by the nearest region, overlaps given
to the region with the larger smoothed indicator), outer boundary and inner boundaries smoothed (Gaussian sigma ~1.5 mm).
-> seg_series18.mat (GM, SOL, plus GM_A / GM_B heads), seg_series18.png (check)"""
import numpy as np, pydicom, scipy.io as sio, glob, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import sys; sys.path.insert(0, "/mnt/e/claude_tmp/study/masks_hr"); from phantom_mask import phantom_mask
D = "/mnt/e/claude_tmp/study/masks_hr"
rgb = plt.imread(f"{D}/user_outline_s18.png")[..., :3].astype(float); H, W = rgb.shape[:2]
R_, G_, B_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
red = (R_ > 0.75) & (G_ < 0.35) & (B_ < 0.35); green = (G_ > 0.5) & (R_ < 0.4) & (B_ < 0.4) & (G_ - np.maximum(R_, B_) > 0.25)
blue = (B_ > 0.6) & (R_ < 0.35) & (B_ - R_ > 0.35) & ~green
colored = ndi.binary_dilation(red | green | blue, iterations=2); gray = rgb.mean(-1)
gray_f = gray.copy(); gray_f[colored] = ndi.median_filter(np.where(colored, np.nan, gray), size=7)[colored] if False else ndi.grey_closing(gray, size=5)[colored] * 0 + np.median(gray)
print(f"screenshot {W}x{H}; outline pixels: green {green.sum()}, blue {blue.sum()}, red {red.sum()}")
# ---- registration: screenshot <-> displayed Series-18 image ----
L = pydicom.dcmread(glob.glob("/mnt/e/claude_tmp/study/loc/Series0018*")[0]); im = L.pixel_array.astype(float); ph = phantom_mask(im)
IMG = sio.loadmat(f"{D}/loc_enhanced_s18.mat")["IMG"].astype(float)
def leg_of(a, frac):
    bw = ndi.binary_fill_holes(ndi.binary_opening(ndi.gaussian_filter(a, 3) > frac, iterations=3)); lab, nl = ndi.label(bw)
    return lab == 1 + np.argmax(ndi.sum(bw, lab, range(1, nl + 1)))
imz = np.where(ph, 0, im); legR = leg_of(imz / np.percentile(imz[~ph], 99), 0.35)
legS = leg_of(np.where(colored, np.median(gray), gray), np.percentile(gray, 35))
cR, cS = np.array(ndi.center_of_mass(legR)), np.array(ndi.center_of_mass(legS)); s0 = np.sqrt(legS.sum() / legR.sum())
print(f"leg area ratio -> scale {s0:.3f} screenshot px per Series-18 px; centroids S18 {np.round(cR, 1)}, screenshot {np.round(cS, 1)}")
rr, cc = np.mgrid[0:512, 0:512]
def warp(src, s, ty, tx, order=1):                                                              # sample screenshot-space array at Series-18 pixels
    return ndi.map_coordinates(src.astype(float), [ty + s * rr, tx + s * cc], order=order, mode="constant", cval=0)
grayS = np.where(colored, np.nan, gray); fillv = np.nanmedian(grayS); grayS = np.where(colored, fillv, gray)
msk = ndi.binary_erosion(legR, iterations=10)
best = None
for s in s0 * np.linspace(0.97, 1.03, 13):
    ty0, tx0 = cS - s * cR
    for dy in np.arange(-6, 6.1, 1.0):
        for dx in np.arange(-6, 6.1, 1.0):
            w = warp(grayS, s, ty0 + dy, tx0 + dx)[msk]; a = IMG[msk]
            ncc = np.corrcoef(w, a)[0, 1]
            if best is None or ncc > best[0]: best = (ncc, s, ty0 + dy, tx0 + dx)
ncc, s, ty, tx = best; print(f"registration: NCC {ncc:.3f}, scale {s:.4f}, offset ({ty:.1f}, {tx:.1f}) screenshot px")
# ---- fill outlines (closed curves), map to the Series-18 grid ----
# all outlines together act as walls; every enclosed area is assigned to the colour that dominates its border
Ewall = ndi.binary_closing(ndi.binary_dilation(red | green | blue, iterations=2), iterations=2)
inside = ndi.binary_fill_holes(Ewall) & ~Ewall; labI, nI = ndi.label(inside)
regS = {"SOL": np.zeros((H, W), bool), "GM_A": np.zeros((H, W), bool), "GM_B": np.zeros((H, W), bool)}
for i in range(1, nI + 1):
    comp = labI == i
    if comp.sum() < 2000: continue                                                              # slivers between parallel lines: left to the gap filling
    ring = ndi.binary_dilation(comp, iterations=6) & ~comp; votes = {"SOL": (ring & green).sum(), "GM_A": (ring & blue).sum(), "GM_B": (ring & red).sum()}
    k = max(votes, key=votes.get); regS[k] |= ndi.binary_dilation(comp, iterations=3) & ~(inside & ~comp)   # grow half-way into the outline itself
    print(f"  enclosed area {i}: {comp.sum()} px, border votes {votes} -> {k}")
reg18 = {k: warp(v, s, ty, tx, order=1) > 0.5 for k, v in regS.items()}
# ---- clean-up: shared boundaries, no gaps / overlaps, smooth ----
ps = float(L.PixelSpacing[0]); sig = 1.5 / ps; gapfill = int(round(3.0 / ps))
U = np.zeros((512, 512), bool)
for v in reg18.values(): U |= v
U = ndi.binary_fill_holes(ndi.binary_closing(U, structure=np.ones((3, 3)), iterations=gapfill))   # bridges gaps between neighbouring outlines
U = ndi.gaussian_filter(U.astype(float), sig) > 0.5                                             # smooth outer boundary
names = list(reg18); Sm = np.stack([ndi.gaussian_filter(reg18[k].astype(float), sig) for k in names])
lab = np.argmax(Sm, 0); none = Sm.max(0) < 1e-3
if none.any():                                                                                  # pixels far from every outline: nearest region
    _, (iy, ix) = ndi.distance_transform_edt(~np.any(np.stack(list(reg18.values())), 0), return_indices=True); near = np.zeros((512, 512), int)
    for i, k in enumerate(names): near[reg18[k]] = i
    lab = np.where(none, near[iy, ix], lab)
OUTM = {k: U & (lab == i) for i, k in enumerate(names)}
for k in OUTM:                                                                                  # remove specks, keep the main piece
    l_, n_ = ndi.label(OUTM[k]); OUTM[k] = l_ == 1 + np.argmax(ndi.sum(OUTM[k], l_, range(1, n_ + 1))) if n_ else OUTM[k]
GM = OUTM["GM_A"] | OUTM["GM_B"]; SOL = OUTM["SOL"] & ~GM
print(f"final: GM {GM.sum()} px ({GM.sum() * ps * ps / 100:.1f} cm2; heads {OUTM['GM_A'].sum()} + {OUTM['GM_B'].sum()}), soleus {SOL.sum()} px ({SOL.sum() * ps * ps / 100:.1f} cm2); "
      f"gap between GM and soleus: {int((ndi.binary_dilation(GM, iterations=2) & ndi.binary_dilation(SOL, iterations=2) & ~GM & ~SOL).sum())} px")
sio.savemat(f"{D}/seg_series18.mat", dict(GM=GM, SOL=SOL, GM_A=OUTM["GM_A"], GM_B=OUTM["GM_B"], PHANTOM=ph, IPP=np.array(L.ImagePositionPatient, float)[:, None],
                                          IOP=np.array(L.ImageOrientationPatient, float)[:, None], PS=np.array(L.PixelSpacing, float)[:, None],
                                          locFile=glob.glob("/mnt/e/claude_tmp/study/loc/Series0018*")[0], series=18, source="user outline screenshot 2026-10-07, smoothed"))
yy, xx = np.where(legR); sl = (slice(yy.min() - 20, yy.max() + 20), slice(xx.min() - 20, xx.max() + 20))
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(rgb); axs[0].set_title("your outlines (screenshot)", fontsize=14)
a = axs[1]; a.imshow(IMG[sl], cmap="gray")
for m_, col in ((OUTM["GM_A"], "#2ca02c"), (OUTM["GM_B"], "#7fdc7f"), (SOL, "#1f77b4")): a.contour(m_[sl].astype(float), [0.5], colors=col, linewidths=2)
a.set_title(f"masks on the Series-18 grid (registration NCC {ncc:.2f}): gastrocnemius (two heads, greens) {GM.sum()} px, soleus (blue) {SOL.sum()} px", fontsize=12)
for x_ in axs: x_.set_xticks([]); x_.set_yticks([])
fig.tight_layout(); fig.savefig(f"{D}/seg_series18.png", dpi=80); print("saved")
