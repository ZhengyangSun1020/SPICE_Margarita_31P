# SPICE-MARGARITA

[![License](https://img.shields.io/badge/License-Oxford%20Non--Commercial-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://www.python.org/downloads/)

A pipeline for *in vivo* brain 2D-MRSI reconstruction, spectral fitting, and metabolite concentration uncertainty quantification.

## Overview

This codebase implements a full processing pipeline from raw k-space MRSI data to quantified metabolite maps with analytical uncertainty estimates.

![Pipeline overview](pipeline_overview.png)

### MRSI Reconstruction

1. **SPICE** — spatially-regularized low-rank MRSI reconstruction (Toeplitz and finufft backends). Ref: [Liang 2020](https://pubmed.ncbi.nlm.nih.gov/31483526/)
2. **Iterative NUFFT reconstruction** — CG solver with B0 correction
3. **Adjoint NUFFT reconstruction** — fast diagnostic reconstruction

### Spectral Fitting

- [**FSL-MRS**](https://open.win.ox.ac.uk/pages/fsl/fsl_mrs/) spectral fitting for quantified metabolite concentrations

### Uncertainty Quantification

- **Laplacian covariance** — voxel-wise Hessian-based posterior covariance (default, exact)
- **LOBPCG** — fast low-rank approximation of the posterior covariance
- **Monte Carlo** — concentration uncertainty via repeated spectral fitting over posterior samples

## Repository Structure

```
SPICE_MARGARITA/
├── scripts/
│   ├── data_preproc/                   # Raw data → pre-processed arrays
│   │   ├── data_proc_01_twix2npy.py    # Siemens twix (.dat) → .npy; MORSE-PI wref_o; saves affine.npy
│   │   ├── data_proc_02_coil_correction.py   # [Optional] Coil sensitivity (MORSE-PI / RNI)
│   │   ├── data_proc_03_B0_map_estimation.py # B0 field map estimation
│   │   └── data_proc_04_lipid_removal.py     # L2-lipid suppression
│   ├── recon_method/                   # MRSI reconstruction
│   │   ├── recon_01_run_spice.py       # SPICE reconstruction with spatial regularization
│   │   ├── recon_02_adjoint_recon.py   # [Optional] Adjoint NUFFT reconstruction (diagnostic)
│   │   └── recon_03_iterative_nufft_recon.py # [Optional] Iterative NUFFT (CG + B0 correction)
│   ├── specfitting/                    # Spectral fitting
│   │   └── specfit_01_fsl_mrsi_fit.py  # xcorr alignment + FSL-MRS spectral fitting
│   └── uncertainty/                    # Uncertainty quantification
│       ├── analytical/                 # Laplacian / analytical methods
│       │   ├── Uncert_01_Laplacian_Covariance.py            # Per-voxel Hessian mHm (HPC)
│       │   ├── Uncert_02_prefitting_uncertainty_laplacian.py # Pre-fitting std (Laplacian / LOBPCG)
│       │   ├── Uncert_03_prefitting_uncertainty_lobpcg.py   # Pre-fitting std (LOBPCG fast path)
│       │   └── Uncert_04_analytical_conc_uncertainty.py     # Analytical concentration uncertainty
│       └── MC/                         # Monte Carlo methods
│           ├── Uncert_05_MC_conc_uncertainty.py   # MC conc. uncertainty (posterior samples)
│           ├── Uncert_06_pair_conc_correlation.py # Pairwise concentration correlation
│           └── Uncert_07_group_uncertainty.py     # Cross-subject group std (pre-fitting + conc)
├── utils/                 # Core Python package
│   ├── recon.py           # NUFFT operators, SPICE solver, B0 correction, phase correction
│   ├── fitting.py         # Nonlinear spectral fitting and MC basis fitting
│   ├── uncertainty.py     # Posterior covariance and uncertainty sampling
│   ├── graph.py           # Spatial graph construction and Laplacian regularization
│   ├── plotting.py        # Visualization utilities
│   ├── io.py              # NIfTI / CSV I/O and logging
│   ├── signal.py          # FID signal generation and phantom construction
│   ├── simulation.py      # Synthetic B0 map and phantom simulation
│   ├── coil_sens.py       # MORSE-PI coil sensitivity estimation
│   ├── xcorr.py           # Cross-correlation frequency alignment
│   ├── pipeline_utils.py  # Shared pipeline helpers (brain mask, symlink logic)
│   ├── scan_params.py     # Scan parameter save/load utilities
│   └── utils.py           # Backward-compatibility re-export shim
├── basis/                 # Basis set (JSON metabolite definitions + SS_training.csv)
├── data/                  # Raw input data (gitignored; see data/README.md)
├── environment.yml        # Conda environment specification
└── pyproject.toml         # Package metadata and pip dependencies
```

> **Data directories** (`data/`, `output/`, `save_iter*/`) are excluded from version control. See `data/README.md` for expected input files.

## Installation

**Recommended — conda (includes FSL-MRS):**

```bash
git clone https://github.com/JasonLvernex/SPICE-VULCAN.git
cd SPICE-VULCAN
conda env create -f environment.yml
conda activate VULCAN
pip install -e .
```

**pip only (FSL-MRS must be installed separately via conda):**

```bash
pip install git+https://github.com/JasonLvernex/SPICE-VULCAN.git
```

> **FSL-MRS / FSL platform note**
> FSL-MRS requires a dedicated conda channel and cannot be installed via pip alone.
> See `environment.yml` for the full conda setup.
>
> - **macOS / Linux**: the conda installation above works out of the box.
> - **Windows**: FSL requires WSL2 or a compatibility layer — follow the official guide at
>   https://fsl.fmrib.ox.ac.uk/fsl/docs/install/windows.html before running `conda env create`.

### Dependencies

- Python ≥ 3.12
- NumPy, SciPy, Matplotlib, scikit-learn
- PyTorch + [torchkbnufft](https://github.com/mmuckley/torchkbnufft)
- [finufft](https://finufft.readthedocs.io) + [mri-nufft](https://github.com/mind-inria/mri-nufft)
- [FSL-MRS](https://open.win.ox.ac.uk/pages/fsl/fsl_mrs/) ≥ 2.4.15 (conda install)
- [FSLeyes](https://open.win.ox.ac.uk/pages/fsl/fsleyes/fsleyes/) ≥ 1.19.0 (conda install) + [fsleyes-plugin-mrs](https://git.fmrib.ox.ac.uk/wclarke/fsleyes-plugin-mrs) ≥ 0.1.10
- [NIfTI-MRS](https://github.com/wtclarke/nifti_mrs) ≥ 1.4.1
- networkx, psutil, tqdm, nibabel

## Citation

If you use this code, please cite the associated paper (in preparation) and acknowledge this repository.

See also: Lyu T, Jbabdi S, Clarke W, Finney S. Pipeline for Quantifying Uncertainty for SPICE Reconstructed MRSI. In: Proceedings of the 2026 ISMRM-ISMRT Annual Meeting and Exhibition, Cape Town, South Africa. Program #402-03-003.
