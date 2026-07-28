# Data Directory

Data files are not included in version control due to file size.

## Directory Structure

```
data/
├── raw/
│   └── invivo_YYMMDD/           # One folder per session
│       ├── meas_MID*_MRSI_*.dat # MRSI acquisitions (one per scan/subject)
│       ├── meas_MID*_wref_*.dat # Water-reference acquisition (shared across scans)
│       ├── *_MRSI_*.seq         # PyPulseq sequence file for MRSI
│       └── *_wref_*.seq         # PyPulseq sequence file for wref
└── processed/
    └── invivo_YYMMDD_01/        # Per-subject output of data_proc_01_twix2npy.py
        ├── mrsi_data.npy        # (1, K_mrsi, N_shot, N_coils)  complex64
        ├── mrsi_ksp.npy         # (3, K_mrsi, N_shot)  float64   kx, ky, t ∈ [0, 2π]
        ├── wref_data.npy        # (1, K_wref, N_shot, N_coils)  complex64   [symlink for _02…_NN]
        ├── wref_ksp.npy         # (3, K_wref, N_shot)  float64              [symlink for _02…_NN]
        ├── wref_o.npy           # (Ny, Nx, 1)  complex64  sensitivity-weighted adjoint wref image
        │                        #   used as brain mask reference in steps 01–08  [symlink for _02…_NN]
        ├── sigma_noise.npy      # scalar float32  noise σ                   [symlink for _02…_NN]
        ├── wref_o_check.png     # Quick-check magnitude/phase plot of wref_o [symlink for _02…_NN]
        ├── scan_params.json     # Scan parameters auto-loaded by pipeline scripts
        └── ecalib.npy           # (Ny, Nx, N_coils)  optional; only needed for --method rni in step 01
```

> `wref_data`, `wref_ksp`, `wref_o`, `sigma_noise`, and `wref_o_check.png` are written once to `_01` and symlinked from `_02` … `_NN` since all scans in a session share the same water reference.

## Generating processed files from raw twix

Run `scripts/data_proc_01_twix2npy.py` to convert `.dat` → `.npy`:

```bash
python scripts/data_proc_01_twix2npy.py \
    --raw-dir data/raw/invivo_260623 \
    --out-prefix data/processed/invivo_260623 \
    --mrsi-dat meas_MID00066_FID92027_MRSI_64_cr_1.dat \
               meas_MID00067_FID92028_MRSI_64_cr_2.dat \
               meas_MID00068_FID92029_MRSI_64_cr_3.dat \
    --wref-dat meas_MID00071_FID92032_wref_64_cr_300.dat \
    --mrsi-seq 260622_SMF_MRSI_cr_64.seq \
    --wref-seq 260622_SMF_wref_cr_64.seq \
    --wref-start-index 238 --mrsi-start-index 252 --fid-trunc 500
```

This creates `data/processed/invivo_260623_01/`, `_02/`, `_03/` with wref files symlinked from `_01`.

## Running the pipeline

Pass a processed subject directory to each pipeline script:

```bash
python scripts/01_coil_correction.py \
    --data-dir data/processed/invivo_260623_01 \
    --basis-dir basis/ --save-plots
```
