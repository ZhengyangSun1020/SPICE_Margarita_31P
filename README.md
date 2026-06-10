# SPICE-MARGARITA

A pipeline for *in vivo* brain 2D-MRSI reconstruction and metabolite concentration uncertainty quantification.

## Overview

This codebase implements a full processing pipeline from raw k-space MRSI data to quantified metabolite maps with analytical uncertainty estimates. The key contributions are:

1. **SPICE reconstruction** with spatially-regularized CG solver (Toeplitz and finufft backends)
2. **B0-corrected iterative NUFFT reconstruction** for high-fidelity spatial encoding
3. **Analytical uncertainty quantification** via Laplacian approximation of the posterior covariance
4. **Monte Carlo uncertainty cross-validation** for spectral fitting uncertainty

## Repository Structure

```
SPICE_MARGARITA/
├── scripts/               # Numbered pipeline scripts (01–12)
│   ├── 01_coil_correction.py  # Coil sensitivity (MORSE-PI default; RNI with --method rni)
│   ├── 02_b0_correction.py    # B0 map estimation and phase correction
│   ├── 03_lipid_removal.py    # L2-penalized lipid suppression
│   ├── 04_run_spice.py        # SPICE reconstruction with spatial constraint
│   ├── 05_adjoint_recon.py    # Adjoint NUFFT reconstruction
│   ├── 06_iterative_nufft_recon.py  # Iterative NUFFT reconstruction
│   ├── 07_spectral_fitting.py # FSL-MRS spectral fitting
│   ├── 08_prefitting_uncertainty.py # Hessian-based uncertainty (per-voxel)
│   ├── 09_uncertainty_postproc.py   # Uncertainty post-processing
│   ├── 10_uncertainty_lobpcg.py     # LOBPCG eigenvalue solver for Hessian
│   ├── 11_laplacian_conc_uncertainty.py  # Laplacian approximation uncertainty
│   └── 12_analytical_conc_uncertainty.py # Analytical concentration uncertainty
├── utils/                 # Core Python package
│   ├── signal.py          # FID signal generation and phantom construction
│   ├── simulation.py      # Synthetic B0 map and phantom simulation
│   ├── graph.py           # Spatial graph construction and Laplacian regularization
│   ├── recon.py           # NUFFT operators, SPICE solver, B0 correction, phase correction
│   ├── fitting.py         # Nonlinear spectral fitting and MC basis fitting
│   ├── uncertainty.py     # Posterior covariance and uncertainty sampling
│   ├── plotting.py        # Visualization utilities
│   ├── io.py              # NIfTI / CSV I/O and logging
│   ├── coil_sens.py       # MORSE-PI coil sensitivity estimation
│   ├── xcorr.py           # Cross-correlation frequency alignment
│   └── utils.py           # Backward-compatibility re-export shim
├── basis/                 # Basis set (JSON metabolite definitions)
├── environment.yml        # Conda environment specification
└── pyproject.toml         # Package metadata and pip dependencies
```

> **Data directories** (`data/`, `output/`, `save_iter*/`) are excluded from version control. Download or generate them separately.

## Pipeline

Run scripts in order from the project root:

```bash
# 01. Coil sensitivity (MORSE-PI default)
python scripts/01_coil_correction.py \
    --data-dir ./data/ --out-dir ./output \
    --n-ref 6 --max-iter 50 --calib-width 16

# 02. B0 map estimation and phase correction
python scripts/02_b0_correction.py \
    --data-dir ./data/ --out-dir ./output

# 03. Lipid suppression
python scripts/03_lipid_removal.py \
    --data-dir ./data/ --out-dir ./output

# 04. SPICE reconstruction with spatial constraint
python scripts/04_run_spice.py \
    --data-dir ./data/ --basis-dir ./basis/ \
    --out-dir ./output --rank 20 --lambda1 1e-4

# 05. Adjoint NUFFT reconstruction
python scripts/05_adjoint_recon.py \
    --data-dir ./data/ --out-dir ./output

# 06. Iterative NUFFT reconstruction
python scripts/06_iterative_nufft_recon.py \
    --data-dir ./data/ --out-dir ./output

# 07. FSL-MRS spectral fitting
python scripts/07_spectral_fitting.py \
    --data-dir ./data/ --out-dir ./output

# 08-10. Hessian uncertainty (parallelisable over voxels)
python scripts/08_prefitting_uncertainty.py \
    --out-dir ./output --hess-dir ./output/Hess_1e4 ...
python scripts/10_uncertainty_lobpcg.py \
    --hess-dir ./output/Hess_1e4 --rank 20 ...

# 11-12. Concentration uncertainty maps
python scripts/11_laplacian_conc_uncertainty.py --out-dir ./output ...
python scripts/12_analytical_conc_uncertainty.py \
    --data-dir ./data/ --basis-dir ./basis/ \
    --out-dir ./output --hess-dir ./output/Hess_1e4 --rank 20
```

## Installation

**Recommended — conda (includes FSL-MRS):**

```bash
git clone https://github.com/JasonLvernex/SPICE_Margarita.git
cd SPICE_Margarita
conda env create -f environment.yml
conda activate SPICE_MARGARITA
pip install -e .
```

**pip only (FSL-MRS must be installed separately via conda):**

```bash
pip install git+https://github.com/JasonLvernex/SPICE_Margarita.git
```

> FSL-MRS requires a dedicated conda channel and cannot be installed via pip alone.
> See `environment.yml` for the full conda setup.

### Dependencies

- Python ≥ 3.12
- NumPy, SciPy, Matplotlib
- PyTorch + [torchkbnufft](https://github.com/mmuckley/torchkbnufft)
- [finufft](https://finufft.readthedocs.io) + [mri-nufft](https://github.com/mind-inria/mri-nufft)
- [FSL-MRS](https://open.win.ox.ac.uk/pages/fsl/fsl_mrs/) (conda install)
- [NIfTI-MRS](https://github.com/wtclarke/nifti_mrs)
- networkx, psutil, tqdm, nibabel

## Citation

If you use this code, please cite the associated paper (in preparation) and acknowledge this repository.
