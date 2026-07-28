# VULCAN vs Repeated-Scan MC Review Package

This folder is a lightweight, self-contained notebook package for reviewing the
current VULCAN posterior uncertainty versus repeated-scan empirical variance.

## Contents

- `vulcan_mc_review.ipynb`  
  Main notebook. It only needs `numpy` and `matplotlib`.

- `prepare_review_data.py`  
  Extracts data from the full SPICE project outputs into compact `.npy` files.

- `data/`  
  Generated data files:
  - `spice_fid_stack_<tag>.npy`: five repeat `SPICE_f` reconstructions, shape `(5, 64, 64, 300)`.
  - `mc_std_<tag>.npy`: repeated-scan std computed as `np.std(fid_to_spec(stack), axis=0)`.
  - `vulcan_std_<tag>.npy`: VULCAN posterior std from `Uncert_02`.
  - `vmhmv_diag_fid_<tag>.npy`: lightweight diagonal of `2*sigma_noise^2 * V @ mHm @ V^H` in FID domain.
  - `brain_mask.npy`, `wref_norm.npy`, `ppm_axis.npy`, `metadata.json`.

The full per-voxel `V @ mHm @ V^H` covariance would be very large, so this
package stores only the diagonal diagnostic plus the final posterior std used
for the plots.

## Regenerate Data

Run from the repository root:

```bash
python review_vulcan_mc_notebook/prepare_review_data.py
```

## Minimal Environment

```bash
pip install numpy matplotlib notebook
```
