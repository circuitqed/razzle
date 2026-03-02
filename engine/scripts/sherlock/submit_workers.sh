#!/bin/bash
# Submit self-play worker jobs to Sherlock SLURM
#
# Usage:
#   ./submit_workers.sh [NUM_JOBS] [GPUS_PER_JOB]
#
# Each job requests 1 GPU and runs multiple worker processes
# (one per GPU). Default: 4 jobs x 1 GPU = 4 workers.

set -e

NUM_JOBS=${1:-4}
GPUS_PER_JOB=${2:-1}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ${NUM_JOBS} worker jobs (${GPUS_PER_JOB} GPU each)..."

for i in $(seq 0 $((NUM_JOBS - 1))); do
    JOB_NAME="razzle-w${i}"
    sbatch \
        --job-name="$JOB_NAME" \
        --export="WORKER_BASE_ID=${i}" \
        "${SCRIPT_DIR}/worker.sbatch"
    echo "  Submitted $JOB_NAME"
done

echo "Done. Check status with: squeue -u \$USER"
