# SPICE-MARGARITA

A pipeline for *in vivo* brain 2D-MRSI reconstruction and metabolite pre-spectral-fitting and concentration uncertainty quantification.

## Overview

This codebase implements a full processing pipeline from raw k-space MRSI data to quantified metabolite maps with analytical uncertainty estimates. The key functionalities are:

MRSI reconstruction methods available:
1. **SPICE reconstruction** with spatially-regularized low-rank MRSI reconstruction (Toeplitz and finufft backends), refer:https://pubmed.ncbi.nlm.nih.gov/31483526/
2. **B0-corrected iterative NUFFT reconstruction**
3. **B0-corrected adjoint NUFFT reconstruction**

Spectral fitting:
5. [**FSL_MRS MRSI fitting** ](https://open.oxcin.ox.ac.uk/pages/fsl/fsl_mrs/)

Uncertainty Quantification:
5. **Analytical uncertainty quantification** voxel-wise Laplacian approximation of the posterior covariance (default)
5. **LOBPCG uncertainty quantification** fast low-rank appriximation covariance
6. **Monte Carlo uncertainty cross-validation** for spectral fitting uncertainty

## Repository Structure

```
SPICE_MARGARITA/
├── scripts/               # Numbered pipeline scripts (01–12)
│   ├── 01_coil_correction.py  # [Optional] Coil sensitivity correction for phase-pole artefacts (MORSE-PI default; RNI with --method rni)
│   ├── 02_b0_correction.py    # B0 map estimation
│   ├── 03_lipid_removal.py    # L2-penalized lipid suppression
│   ├── 04_run_spice.py        # SPICE reconstruction with spatial constraint
│   ├── 05_adjoint_recon.py    # Adjoint NUFFT reconstruction
│   ├── 06_iterative_nufft_recon.py  # Iterative NUFFT reconstruction
│   ├── 07_spectral_fitting.py # FSL-MRS spectral fitting
│   ├── 08_prefitting_uncertainty.py # voxel-wise  covariance matrix calculation via laplacian approximation
│   ├── 09_uncertainty_postproc.py   # Voxel-wise pre-fitting uncertainty (dependent on running 08)
│   ├── 10_uncertainty_lobpcg.py     # [Optional] faster pre-fitting uncertainty via LOBPCG low-rank approximation of the covariance matrix
│   ├── 11_laplacian_conc_uncertainty.py  # [Optional] Monte-Carlo concentration uncertainty
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
├── data/                  # Raw input data (gitignored; see data/README.md)
├── environment.yml        # Conda environment specification
└── pyproject.toml         # Package metadata and pip dependencies
```

> **Data directories** (`data/`, `output/`, `save_iter*/`) are excluded from version control. Download or generate them separately.


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

If you use this code, please cite the associated paper (Journal paper in preparation) and acknowledge this repository.

see also: Lyu T, Jbabdi S, Clarke W, Finney S. Pipeline for Quantifying Uncertainty for SPICE Reconstructed MRSI. In: Proceedings of the Cape Town - 2026 ISMRM-ISMRT Annual Meeting and Exhibition, Cape Town, South Africa. Program #402-03-003.
