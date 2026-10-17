"""Phantom mask for the 24-07 localizers. Leg tissue ~100-250, phantom ~600-800 (a.u.): threshold halfway between the 80th (tissue) and
99th (phantom) percentile, keep large blobs touching the image border (the leg does not), fill holes, dilate 4 px."""
import numpy as np
from scipy import ndimage as ndi
def phantom_mask(im):
    thr = 0.5 * (np.percentile(im, 80) + np.percentile(im, 99))
    lab, nl = ndi.label(ndi.binary_opening(im > thr, iterations=2)); keep = np.zeros(im.shape, bool)
    for k in range(1, nl + 1):
        r = lab == k
        if r.sum() > 3000 and (r[0].any() or r[-1].any() or r[:, 0].any() or r[:, -1].any()): keep |= r
    return ndi.binary_dilation(ndi.binary_fill_holes(keep), iterations=4)
