# SPICE-MARGARITA

A pipeline for *in vivo* brain 2D-MRSI reconstruction, spectral fitting, and metabolite concentration uncertainty quantification.

## Overview

This codebase implements a full processing pipeline from raw k-space MRSI data to quantified metabolite maps with analytical uncertainty estimates.

### MRSI Reconstruction

1. **SPICE** — spatially-regularized low-rank MRSI reconstruction (Toeplitz and finufft backends). Ref: [Liang 2007](https://pubmed.ncbi.nlm.nih.gov/31483526/)
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
├── scripts/               # Numbered pipeline scripts (01–12)
│   ├── 01_coil_correction.py        # [Optional] Coil sensitivity (MORSE-PI default; RNI with --method rni)
│   ├── 02_B0_map_estimation.py      # B0 field map estimation
│   ├── 03_lipid_removal.py          # L2-lipid suppression
│   ├── 04_run_spice.py              # SPICE reconstruction with spatial regularization
│   ├── 05_adjoint_recon.py          # [Optional] Adjoint NUFFT reconstruction (diagnostic)
│   ├── 06_iterative_nufft_recon.py  # [Optional] Iterative NUFFT reconstruction (CG + B0 correction)
│   ├── 07_spectral_fitting.py       # FSL-MRS spectral fitting
│   ├── 08_Laplacian_Covariance.py   # Per-voxel Laplacian covariance matrix computation
│   ├── 09_prefitting_uncertainty_laplacian.py  # Pre-fitting uncertainty (Laplacian)
│   ├── 10_prefitting_uncertainty_lobpcg.py     # [Optional] Pre-fitting uncertainty (LOBPCG, optional)
│   ├── 11_analytical_conc_uncertainty.py       # Analytical concentration uncertainty
│   └── 12_MC_conc_uncertainty.py               # [Optional] Monte Carlo concentration uncertainty
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
│   └── utils.py           # Backward-compatibility re-export shim
├── basis/                 # Basis set (JSON metabolite definitions)
├── data/                  # Raw input data (gitignored; see data/README.md)
├── environment.yml        # Conda environment specification
└── pyproject.toml         # Package metadata and pip dependencies
```

> **Data directories** (`data/`, `output/`, `save_iter*/`) are excluded from version control. See `data/README.md` for expected input files.

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

See also: Lyu T, Jbabdi S, Clarke W, Finney S. Pipeline for Quantifying Uncertainty for SPICE Reconstructed MRSI. In: Proceedings of the 2026 ISMRM-ISMRT Annual Meeting and Exhibition, Cape Town, South Africa. Program #402-03-003.
