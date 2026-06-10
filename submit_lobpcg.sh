#!/bin/bash
# ============================================================
# submit_lobpcg.sh
# Submits ONE GPU job for step 10 (LOBPCG uncertainty).
#
# Usage:
#   bash submit_lobpcg.sh
#
# fsl_sub GPU submission on FMRIB cluster:
#   --coprocessor cuda  → request a CUDA GPU node
#   -q gpu              → GPU queue (may vary; check with: fsl_sub --list_queues)
# ============================================================

# ── User config ───────────────────────────────────────────────
CONDA_INIT="/home/fs0/fcj757/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="torch_env"

SCRIPT_DIR="/home/fs0/fcj757/scratch/SPICE_MARGARITA/scripts"
DATA_DIR="/home/fs0/fcj757/scratch/SPICE_MARGARITA/data/"
OUT_DIR="/home/fs0/fcj757/scratch/SPICE_MARGARITA/output/"

RANK=20
LAMBDA=1e-4
LAMBDA_WE_MAX=5000
BRAIN_THRESHOLD=0.08
CSV_NAME="SS_training"

# LOBPCG settings
K_EIG=5000            # number of smallest eigenpairs
LOBPCG_MAXITER=200
LOBPCG_TOL=1e-3
DAMP=0.0
N_SAMPLES=1000
SIGMA2=1.0          # overridden automatically if sigma_noise.npy exists

# Plot voxel
VOXEL_X=38
VOXEL_Y=20
THRESHOLD=5e-5

# GPU queue: gpu_short (4h) or gpu_long (2.5d)
# GPU class:  A30-12G (12GB)  H100-10G (10GB)  H100-80G (80GB)
GPU_QUEUE="gpu_long"
GPU_CLASS="H100-80G"
# ──────────────────────────────────────────────────────────────

WRAPPER="/home/fs0/fcj757/scratch/SPICE_MARGARITA/tmp/lobpcg_job_$$.sh"
cat > "$WRAPPER" << EOF
#!/bin/bash
source ${CONDA_INIT}
conda activate ${CONDA_ENV}

echo "[submit_lobpcg] CUDA device: \$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"

python ${SCRIPT_DIR}/10_uncertainty_lobpcg.py \\
    --data-dir        ${DATA_DIR} \\
    --out-dir         ${OUT_DIR} \\
    --rank            ${RANK} \\
    --lambda          ${LAMBDA} \\
    --lambda-we-max   ${LAMBDA_WE_MAX} \\
    --brain-threshold ${BRAIN_THRESHOLD} \\
    --csv-name        ${CSV_NAME} \\
    --k-eig           ${K_EIG} \\
    --lobpcg-maxiter  ${LOBPCG_MAXITER} \\
    --lobpcg-tol      ${LOBPCG_TOL} \\
    --damp            ${DAMP} \\
    --n-samples       ${N_SAMPLES} \\
    --sigma2          ${SIGMA2} \\
    --voxel-x         ${VOXEL_X} \\
    --voxel-y         ${VOXEL_Y} \\
    --threshold       ${THRESHOLD} \\
    --dark-mode
EOF
chmod +x "$WRAPPER"

echo "Submitting LOBPCG job to queue '${GPU_QUEUE}'  class='${GPU_CLASS}' …"
JOB_ID=$(fsl_sub -q ${GPU_QUEUE} -c cuda --coprocessor_class ${GPU_CLASS} "$WRAPPER")
echo "  job_id=${JOB_ID}"
echo "  wrapper=${WRAPPER}"
echo ""
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Output:   ${OUT_DIR}/lobpcg/"
