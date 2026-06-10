#!/bin/bash
# ============================================================
# submit_hessian.sh
# Submits multiple 08_uncertainty.py jobs to the cluster,
# each processing a different slice of brain voxels.
#
# Usage:
#   bash submit_hessian.sh
#
# Key variable to adjust:
#   CHUNK_SIZE — how many voxels each job handles
# ============================================================

# ── User config ───────────────────────────────────────────────
CHUNK_SIZE=60               # voxels per job  ← change this
TOTAL_VOXELS=1250           # safe upper bound; extra jobs exit immediately

QUEUE="short"               # fsl_sub queue
MAX_WORKERS=2               # parallel workers inside each job (OMP x workers = total threads)

CONDA_INIT="/home/fs0/fcj757/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="torch_env"

SCRIPT_DIR="/home/fs0/fcj757/scratch/SPICE_MARGARITA/scripts"
DATA_DIR="/home/fs0/fcj757/scratch/SPICE_MARGARITA/invivo_260305/cr/"
OUT_DIR="/home/fs0/fcj757/scratch/SPICE_MARGARITA/output"
HESS_DIR="/home/fs0/fcj757/scratch/SPICE_MARGARITA/Hess_vox_batch"

RANK=20
LAMBDA=1e-4
LAMBDA_WE_MAX=5000
CG_MAXITER=300
CG_RTOL=1e-3
BRAIN_THRESHOLD=0.08
CSV_NAME="SS_training"

# tmp dir for per-job wrapper scripts
TMP_DIR="/home/fs0/fcj757/scratch/SPICE_MARGARITA/tmp/hess_jobs_$$"
mkdir -p "$TMP_DIR"
# ──────────────────────────────────────────────────────────────

echo "Submitting jobs: CHUNK_SIZE=${CHUNK_SIZE}  TOTAL_VOXELS=${TOTAL_VOXELS}"
echo "Queue: ${QUEUE}  MAX_WORKERS=${MAX_WORKERS}"
echo "------------------------------------------------------------"

START=0
JOB_COUNT=0

while [ $START -lt $TOTAL_VOXELS ]; do
    END=$((START + CHUNK_SIZE))

    # create a wrapper script for this chunk
    WRAPPER="${TMP_DIR}/hess_${START}_${END}.sh"
    cat > "$WRAPPER" << EOF
#!/bin/bash
source ${CONDA_INIT}
conda activate ${CONDA_ENV}

python ${SCRIPT_DIR}/08_uncertainty.py \\
    --data-dir        ${DATA_DIR} \\
    --out-dir         ${OUT_DIR} \\
    --hess-dir        ${HESS_DIR} \\
    --rank            ${RANK} \\
    --lambda          ${LAMBDA} \\
    --lambda-we-max   ${LAMBDA_WE_MAX} \\
    --brain-threshold ${BRAIN_THRESHOLD} \\
    --csv-name        ${CSV_NAME} \\
    --max-workers     ${MAX_WORKERS} \\
    --cg-maxiter      ${CG_MAXITER} \\
    --cg-rtol         ${CG_RTOL} \\
    --vox-start       ${START} \\
    --vox-end         ${END}
EOF
    chmod +x "$WRAPPER"

    JOB_ID=$(fsl_sub -q ${QUEUE} "$WRAPPER")
    echo "  Submitted voxels [${START}:${END})  job_id=${JOB_ID}"

    START=$END
    JOB_COUNT=$((JOB_COUNT + 1))
done

echo "------------------------------------------------------------"
echo "Total jobs submitted: ${JOB_COUNT}"
echo "Wrapper scripts in:   ${TMP_DIR}"
