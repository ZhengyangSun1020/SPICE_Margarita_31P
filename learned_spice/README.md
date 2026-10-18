# Learned SPICE weights (31P CRT MRSI)

A small CNN predicts the **regulariser of SPICE**. It outputs the edge weights of the spatial smoothing graph and a per-voxel
Tikhonov map, which plays the role of λ_out. It does not replace the reconstruction. The solve is still the SPICE normal
equation used in `scripts/05c_spice_recon_31p.py` and in CA-SPICE (`scripts/ca_spice_0724`):

```
(G + rel · (L(w) + diag(τ))) U = B,      x = U Vᴴ
G = Fᴴ D F,  B = Fᴴ D y V (= adj_fid @ V),  rel = mean(real(diag(G)))
```

- **w** (4 per voxel: E, S, SE, SW neighbours) and **τ** come from the network. CA-SPICE builds the same quantities by hand: an
  80-ring PCr guidance image, compartment labels and offsets, and the rule `λ_out = min(1, 0.1 d²)`.
- The reconstruction is **still convex and quadratic**. That means:
  - the analytic covariance applies (`model.voxel_noise_std`, given the predicted weights);
  - the result can be exported as an ordinary SPICE system (`WW`, `lout_map`, `lam`, see `model.export_system`), so the
    existing SPICE and uncertainty scripts can use it unchanged;
  - the data consistency is exact: the network cannot add signal that the data and the subspace do not support.
- **One model covers acceleration, low SNR and super-resolution.** Training randomises the CSI grid (8, 10, 12), drops rings
  (like `data_proc_06_undersample.py`), acquires on a coarser grid than the CSI grid (super-resolution), and sets the noise
  level.

No existing file in the repository is modified; this package only reads conventions from them, and copies the PK2PI
metabolite table and the CRT DCF from 05c (keep them in sync).

## Layout

| File | Contents |
|---|---|
| `crt.py` | CRT ring trajectory, ring DCF × Hamming (as `compute_crt_dcf_hamming`), explicit NUDFT encoding, Gram |
| `phantom.py` | PK2PI 31P spectral model, subspace (as `generate_training_data_pk2pi` + SVD), random calf phantom |
| `data.py` | On-the-fly training pairs (fine-grid truth, random CRT operator, noise), all in the rank-r subspace |
| `model.py` | Features, `EdgeWeightNet`, graph Laplacian (identical to `calc_Bmatrix_simple`), SPICE solve, uncertainty, export |
| `train.py` | Training + comparison with SPICE baselines on a simulated test set |
| `validate_ring_subsets.py` | Real-data check: ring subsets of an oversampled scan, scored by held-out k-space prediction |
| `apply.py` | Apply a trained model to real data (`system.npz`, `adj_fid.npy`, `VD.npy`), static or dynamic |
| `tests/` | Laplacian equals `calc_Bmatrix_simple`, export reproduces the solve, noise std matches Monte Carlo, CLI runs |

## Simulation

- **Phantom.** A random calf cross-section on a 4× finer grid than the CSI grid, so the truth is a voxel average and the forward
  model is not the reconstruction model. It contains:
  - skin/fat, tibia and fibula, all without 31P;
  - 2–5 Voronoi muscle compartments, each at rest or in exercise (PCr depleted, Pi raised and shifted acidic);
  - an optional focal lesion that the anatomy does not show;
  - a surface-coil receive profile, and smooth B0 and line-broadening variations.
- **Spectra.** The PK2PI table (PCr, Pi, PDE, γ/α/β-ATP multiplets) at 49.9 MHz, 256 points, 2.5 kHz bandwidth, rank 12.
  These are set in `SimConfig`.
- **Guidance image.** A blurred, noisy PCr map, standing in for the 80-ring prior. It is given in 70% of the samples; the leg
  mask is given in 70% of samples, independently. The same model therefore works with or without them.

## Usage

```bash
pip install -e .                    # torch, numpy, scipy, matplotlib (already in pyproject)
python -m learned_spice.train --out-dir output/learned_spice/run02_mid276     # ~30 min on 4 CPU cores

# real-data check on an oversampled CRT scan (needs pymapvbvd, mri-nufft for the 05c/06c loaders)
python -m learned_spice.validate_ring_subsets --dat data/raw/meas_MID00276_FID221481_240FOV_10x10_80rings.dat \
    --model output/learned_spice/run02_mid276/model.pt --rank 1 --out-dir output/learned_spice/mid276_validation
python -m pytest learned_spice/tests -q

# real data (static or dynamic adj_fid), e.g. a CA-SPICE study folder
python -m learned_spice.apply --model output/learned_spice/run01/model.pt \
    --system output_study_mid283/system.npz --use-prior \
    --adj-fid output_study_mid283/adj_fid.npy --V output_study_mid283/VD12.npy \
    --leg output_study_mid283/reg_TRUE.npy --noise-std-b <std of one adj_fid entry> \
    --out-dir output/learned_spice/mid283
```

The noise level matters, because the network sets the amount of regularisation from it. For white k-space noise of std σ,
one adjoint-FID entry has std `σ · sqrt(sum(dcf²))`; pass that value, or `--sigma-k` with `--dcf`. Without either, `apply.py`
falls back to a rough estimate from the FID tail and warns.

## Results

### Simulation

`train.py` with the defaults, which are the MID00276 acquisition parameters: 3000 steps, about 30 min on 4 CPU cores. The test
set is 192 held-out simulated slices (64 each of 8×8, 10×10, 12×12). Values are means; lower is better. *rel. err* is the
subspace FID error; *PCr* and *Pi* are the NRMSE of the peak-height maps inside the leg.

| Method | rel. err | PCr | Pi | PCr (full) | PCr (ring-dropped) | PCr (super-res) |
|---|---|---|---|---|---|---|
| SPICE, λ = 0 | 0.806 | 0.307 | 0.729 | 0.277 | 0.382 | 0.325 |
| SPICE, hand-crafted graph (λ tuned on validation) | 0.174 | 0.138 | 0.374 | 0.123 | 0.133 | 0.176 |
| SPICE, hand-crafted graph (best λ per sample) | 0.159 | 0.128 | 0.365 | 0.115 | 0.123 | 0.162 |
| **Learned SPICE** | **0.083** | **0.077** | **0.344** | **0.063** | **0.081** | **0.107** |

These numbers come from the same generator used for training.

### Real data: MID00276, 80-ring 10×10 scan

The 80 rings cover |k| ≤ 0.44 cycles/voxel, so the scan is heavily oversampled for a 10×10 grid.
`validate_ring_subsets.py`:
1. reconstructs from 4 disjoint ring subsets per condition;
2. uses a spectral subspace from the SVD of the held-out rings' adjoint FIDs;
3. scores each reconstruction by how well it predicts the held-out rings' measured k-space (DCF-weighted NMSE).

A perfect reconstruction would score the noise floor; an all-zero reconstruction scores 1.0. The object shows only PCr, so
its signal is essentially rank 1. The model is `run02_mid276`; there is no prior and no leg mask.

| Rank 1, held-out NMSE (mean ± sd, 4 subsets) | 5 rings | 4 rings (4th dropped) | 3 rings (super-res) |
|---|---|---|---|
| noise floor | 0.299 | 0.299 | 0.300 |
| SPICE, λ = 0 | 0.752 ± 0.034 | 11.4 ± 3.7 | 37.5 ± 10.5 |
| SPICE, hand-crafted graph (λ from simulation) | 0.555 ± 0.039 | 0.569 ± 0.041 | 0.572 ± 0.034 |
| **Learned SPICE** | **0.537 ± 0.041** | **0.557 ± 0.050** | **0.541 ± 0.040** |
| SPICE, hand-crafted (λ chosen on the held-out data, optimistic) | 0.519 ± 0.029 | 0.522 ± 0.032 | 0.524 ± 0.030 |

- **Rank 1.** The learned weights are slightly better than the hand-crafted graph with its simulation-tuned λ, but by less
  than one standard deviation. They are worse than the hand-crafted graph with λ tuned on the held-out data itself.
- **Rank 4** (three extra components that are pure noise). The learned weights are worse: 0.939 against 0.911 for 5 rings.
  They under-regularise, and the maps show residual signal at the FOV edges where there is no tissue.

The large advantage in simulation therefore does **not** carry over to this scan yet. Likely reasons:
- the object (a compact, PCr-only phantom) is nothing like the simulated calves;
- training used a PK2PI subspace, not a data-derived one;
- no leg mask was given, so λ_out was not learned against a known support.

## Before trusting it on real data

1. **Match the simulation to the scans.** Set `n_spec`, the dwell time, the Larmor frequency, the points per ring and the
   ring radii (`first_circle` from the twix header) in `SimConfig` / `crt.py`, and retrain.
2. **Use your own subspace.** `apply.py` accepts any V, and the features do not depend on the rank. Still, train with the V you
   reconstruct with, by replacing `spectral_subspace(...)` in `train.py` with `np.load("VD12.npy")`.
3. **Validate on real data with a reference.**
   - Undersample a fully sampled scan retrospectively (`data_proc_06_undersample.py`), and compare the learned and
     hand-crafted reconstructions against the full-ring or rep-averaged reference.
   - Check repeatability across bouts, as in `fig_repeat.py`.
4. **Check the uncertainty.** Compare `voxel_noise_std` with the pseudo-replica SNR (`analysis/pseudo_replica.py`). The
   formula holds for fixed weights. The weights are predicted from the noisy data, so it is a conditional estimate.

## Next steps

- **Stage 2.** Unrolled SPICE with a learned prior on U (MoDL-style), reusing `spice_solve` as the data-consistency step.
- **Dynamic series.** Temporal low-rank or shared weights over reps, which `apply.py` already does with one graph per series.
- **3D stack-of-rings** (`05d`, `06c`, `07b`). An FFT along kz gives one 2D problem per partition.
- **Guided super-resolution.** Use the interleaved 1H FLASH (`07b`) as the guidance image.
