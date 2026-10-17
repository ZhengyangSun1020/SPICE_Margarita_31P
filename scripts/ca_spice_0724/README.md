# CA-SPICE — compartment-aware SPICE for dynamic ³¹P MRSI of the calf (24-07 study)

Dynamic ³¹P CRT MRSI (5 rings, 3T) of the calf, four bouts of the same leg with different protocols:

| Bout | Matrix / voxel | Time / rep | Rest / exercise / recovery reps |
|---|---|---|---|
| MID00283 | 10×10, 20 mm | 14 s | 1–4 / 5–26 / 27–52 |
| MID00277 | 10×10, 24 mm | 12 s | 1–5 / 6–30 / 31–60 |
| MID00281 | 8×8, 25 mm | 12 s | 1–5 / 6–30 / 31–60 |
| MID00279 | 8×8, 30 mm | 10 s | 1–6 / 7–36 / 37–72 |

**CA-SPICE** = SPICE (rank-12 spectral subspace + spatial regularisation) on the CSI grid, with a compartment-aware smoothing graph:
CSI voxels are labelled gastrocnemius / soleus / other muscle from masks drawn on the localizer; graph edges are weighted by an
80-ring PCr guidance image and cut across compartment boundaries (hard boundary); λ_out penalises signal outside the leg.

> Scripts were developed interactively and contain **hard-coded paths** (`/home/zsun/SPICE_Margarita/output_study_mid<ds>`,
> `/mnt/e/claude_tmp/study/...`, raw data in `/mnt/e/claude_tmp/study/raw`). Adjust them before running elsewhere.
> They import the repo pipeline (`scripts/05c_spice_recon_31p.py`, `scripts/recon_concept_twix_31p.py`).

## Pipeline

1. **Per-bout setup** (`output_study_mid<ds>/`: `meta.json`, `reg_TRUE.npy` leg/phantom labels, `system.npz` with the CSI system and
   80-ring PCr prior, `VD.npy` spectral subspace, `adj_fid.npy` NUFFT voxel FIDs, `kspace_fs.npy`).
   The original 24-07 setup script is not preserved; `recon/prep644.py` does the same steps for MID00644 and is the template.
   `recon/rebuild_study.py` re-does the FOV-shift-dependent parts (80-ring prior, NUFFT adjoint); `recon/system11.py` builds the
   soft-boundary SPICE system (λ, λ_out map); `recon/fix_prior.py` rebuilds the 80-ring guidance image keeping only k-samples inside the
   bout grid's Nyquist limit (needed for MID00279, 30 mm: without it the prior aliases into a ramp).
2. **Masks**: `masks/outline_to_mask.py`, `masks/make_grids.py` (drawn GM / soleus pixel masks → CSI voxel labels
   `masks_hr/mask_labels.json`), `masks/phantom_mask.py` (reference phantom on the localizer).
3. **Reconstruction**: `recon/hybrid_slam_spice.py <ds>` runs NUFFT, CA-SPICE ("SPICE (compartment, CSI grid)") and, for reference,
   CC-SPICE (fine-grid sub-voxel model, "hybrid μ"). Final CA-SPICE settings (`recon/run_sp_final.sh`):
   `COMP_BOUND=hard SAVE_SP=1 VFILE=VD12.npy PIXMASK=1 LABELS=mask` → `output_study_mid<ds>/spice_final/fid_spice.npy`.
   Other options: `COMP_DELTA` (compartment offset in the guidance image), `SP_LAMS` (λ sweep), `TLR` (low-rank-in-time test),
   `SAVE_COMP` / `SAVE_FID` (CC-SPICE outputs). `recon/run_r12.sh` makes the rank-12 subspace `VD12.npy`.
4. **Noise model / quantification**: `recon/save_linops.py <ds>` saves the exact linear operators (NUFFT `PN`, CA-SPICE `PS`) and the
   measured k-t noise variance; `analysis/lin_quant.py` gives the exact noise variance per voxel / ROI / frequency and the
   noise-power-subtracted magnitude spectrum.
5. **Figures** (`figures/`, all read `figures/hybrid_fig.py` for geometry / localizer / masks):
   - `recon_overview_ca_fig.py` — Fig 1 pipeline
   - `fig2_spectra.py` (`SET=a`: MID00283 + 279; `SET=b`: 281 + 277) — Fig 2 / S1 spectra
   - `fig3_maps.py` — Fig 3 PCr depletion and Pi increase maps (MID00283)
   - `fig_repeat.py` — Fig 4 repeatability across the four bouts
   - `compare_bouts_fig.py` — four-bout maps + voxel dynamics (supplementary)
   Captions: `figure_captions.md`.
6. **MATLAB / AMARES export**: `matlab_export/export_hybrid_matlab.py` + `build_spice_unfitted.m` write `obj_dynamic` files for
   `Leg_eval_twix.m`; `export_fitted.m` reads AMARES fits back.

## Quantification conventions (used in all final figures)

- Change metrics use **peak heights of averaged spectra**: rest = mean of the rest reps, end of exercise = mean of the last 4 reps fully
  inside exercise; PCr depletion = 100·(1 − PCr_ex/PCr_rest), Pi increase = 100·(Pi_ex − Pi_rest)/PCr_rest (Pi peak 3.8–5.4 ppm).
  Per-rep "strongest Pi line" estimates are biased upward at rest by noise and are not used for changes.
- **Voxel exclusion** (`analysis/edge_rule.py`, same rule in Figs 3 and 4): CA-SPICE PCr rep-to-rep noise in rest + late recovery ≥ 25 %,
  muscle area fraction < 50 %, or adjacent to the reference phantom.
- The subspace shapes noise into metabolite-like peaks (largest in the Pi band), so absolute resting Pi in low-SNR (soleus) voxels is
  inflated in magnitude spectra; report soleus Pi as a change from rest (`analysis/noise_pi.py` shows this).

## Supporting analyses (`analysis/`)

`pseudo_replica.py` (pseudo-multiple-replica SNR), `srf_kweights.py` (spatial response / leakage of each soleus estimate),
`gtm_correct.py` (ROI unmixing test — unstable, not used), `sim_ktshare.py` (alternating outer rings + low-rank-in-time simulation),
`diag279.py` (MID00279 guidance-image diagnosis), `noise_pi.py` (noise-only recon: Pi bias).
