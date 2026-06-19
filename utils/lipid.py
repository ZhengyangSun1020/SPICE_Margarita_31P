"""
Lipid-signal-statistic (LSS) helpers shared between the lipid-removal step
(03) and the joint lipid+metabolite refit step (04b).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def compute_lss(data, ppm_axis, low_ppm=0.7, high_ppm=1.8):
    lipid_idx = np.where((ppm_axis >= low_ppm) & (ppm_axis <= high_ppm))[0]
    return np.sum(np.abs(data[..., lipid_idx]), axis=-1), lipid_idx


def select_lipid_mask_gmm_simple(lss_map, out_dir, nsigma=0.2,
                                  max_voxels=500, topN_fallback=100,
                                  save_plots=False, out_fname="fig_03b_lss_gmm.png",
                                  title_prefix="LSS"):
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm

    lss2d = np.squeeze(np.asarray(lss_map))
    flat  = lss2d.ravel()
    vals  = flat[np.isfinite(flat) & (flat > 0)]

    thr, method, gmm_model = None, None, None
    try:
        logvals = np.log(vals).reshape(-1, 1)
        gmm     = GaussianMixture(n_components=2, random_state=0).fit(logvals)
        means   = gmm.means_.ravel()
        covs    = gmm.covariances_.ravel()
        li      = int(np.argmax(means))
        thr     = float(np.exp(means[li] - nsigma * np.sqrt(covs[li])))
        method, gmm_model = "gmm", gmm
    except Exception:
        thr    = float(np.percentile(vals, 90))
        method = "percentile"

    mask = lss2d >= thr
    nsel = int(np.sum(mask))

    if nsel > max_voxels:
        sorted_flat = np.sort(vals)[::-1]
        thr    = float(sorted_flat[topN_fallback - 1]) if len(sorted_flat) >= topN_fallback else thr
        mask   = lss2d >= thr
        nsel   = int(np.sum(mask))
        method = "topN"

    if save_plots:
        from scipy.stats import norm as _norm
        logvals_all = np.log(np.maximum(vals, 1e-12))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(logvals_all, bins=80, density=True, alpha=0.6, color="C0")
        xs = np.linspace(logvals_all.min(), logvals_all.max(), 400)
        if gmm_model is not None:
            for mu, cov, w in zip(gmm_model.means_.ravel(),
                                   gmm_model.covariances_.ravel(),
                                   gmm_model.weights_.ravel()):
                ax1.plot(xs, w * _norm.pdf(xs, loc=mu, scale=np.sqrt(cov)), lw=2)
        ax1.axvline(np.log(thr), color="k", ls="--", label=f"thr log={np.log(thr):.3f}")
        ax1.set_xlabel("log(LSS)")
        ax1.legend()
        ax1.set_title(f"{title_prefix} GMM  method={method}  n={nsel}")
        im = ax2.imshow(lss2d, origin="lower", cmap="viridis")
        ax2.set_title(f"{title_prefix} map + selected mask")
        plt.colorbar(im, ax=ax2, fraction=0.046)
        for yy, xx in zip(*np.where(mask)):
            ax2.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                    edgecolor="red", facecolor="none", linewidth=1.2))
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, out_fname), dpi=120)
        plt.close(fig)

    return {"lipid_mask": mask, "threshold": thr, "n_selected": nsel, "method": method}
