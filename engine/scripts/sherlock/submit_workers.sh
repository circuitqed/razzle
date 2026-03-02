#!/bin/bash
# Submit self-play worker jobs to Sherlock SLURM
#
# Usage:
#   ./submit_workers.sh TOTAL_GAMES [NUM_GPUS] [WORKERS_PER_GPU]
#
# Examples:
#   ./submit_workers.sh 1000           # 1000 games across 8 GPUs x 3 workers
#   ./submit_workers.sh 5000 16        # 5000 games across 16 GPUs x 3 workers
#   ./submit_workers.sh 5000 16 4      # 5000 games across 16 GPUs x 4 workers
#
# Each GPU runs WORKERS_PER_GPU processes. MAX_GAMES is per-worker-process,
# so each GPU produces WORKERS_PER_GPU * MAX_GAMES games total.
#
# At ~17 games/hr per worker process (2000 sims, medium net):
#   1000 games / (8 GPUs * 3 workers) = 42 games/worker ≈ 2.5 hours
#   5000 games / (16 GPUs * 3 workers) = 104 games/worker ≈ 6 hours

set -e

TOTAL_GAMES=${1:?Usage: $0 TOTAL_GAMES [NUM_GPUS] [WORKERS_PER_GPU]}
NUM_GPUS=${2:-8}
WORKERS_PER_GPU=${3:-3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
GAMES_PER_WORKER=$(( (TOTAL_GAMES + TOTAL_WORKERS - 1) / TOTAL_WORKERS ))

# Estimate wall time: ~17 games/hr per worker process, add 50% buffer
EST_HOURS=$(python3 -c "import math; print(max(1, math.ceil($GAMES_PER_WORKER / 17 * 1.5)))")
WALL_TIME="${EST_HOURS}:00:00"
# Cap at 48h (partition max)
if [ "$EST_HOURS" -gt 48 ]; then
    WALL_TIME="48:00:00"
fi

echo "=== Razzle Training Batch ==="
echo "Total games:        $TOTAL_GAMES"
echo "GPU jobs:           $NUM_GPUS"
echo "Workers per GPU:    $WORKERS_PER_GPU"
echo "Total workers:      $TOTAL_WORKERS"
echo "Games per worker:   $GAMES_PER_WORKER"
echo "Games per GPU:      $((GAMES_PER_WORKER * WORKERS_PER_GPU))"
echo "Est. wall time:     $WALL_TIME"
echo "============================="
echo ""

mkdir -p "$SCRIPT_DIR/logs"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    JOB_NAME="razzle-w${i}"
    JOB_ID=$(sbatch \
        --job-name="$JOB_NAME" \
        --time="$WALL_TIME" \
        --export="WORKER_BASE_ID=${i},MAX_GAMES=${GAMES_PER_WORKER},WORKERS_PER_GPU=${WORKERS_PER_GPU}" \
        --parsable \
        "${SCRIPT_DIR}/worker.sbatch")
    echo "  Submitted $JOB_NAME (job $JOB_ID, ${WORKERS_PER_GPU}x${GAMES_PER_WORKER} games)"
done

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER -n razzle"
echo "  tail -f $SCRIPT_DIR/logs/worker_*.out"
