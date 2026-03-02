#!/bin/bash
# Submit self-play worker jobs to Sherlock SLURM
#
# Usage:
#   ./submit_workers.sh TOTAL_GAMES [NUM_GPUS]
#
# Examples:
#   ./submit_workers.sh 1000        # 1000 games across 8 GPUs (default)
#   ./submit_workers.sh 5000 16     # 5000 games across 16 GPUs
#
# Each GPU runs 1 worker (Sherlock GPUs use Exclusive_Process mode).
# At ~17 games/hr per GPU (2000 sims, medium net):
#   1000 games / 16 GPUs ≈ 63 games each ≈ 4 hours
#   5000 games / 16 GPUs ≈ 312 games each ≈ 19 hours

set -e

TOTAL_GAMES=${1:?Usage: $0 TOTAL_GAMES [NUM_GPUS]}
NUM_GPUS=${2:-8}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GAMES_PER_GPU=$(( (TOTAL_GAMES + NUM_GPUS - 1) / NUM_GPUS ))

# Estimate wall time: ~17 games/hr per GPU, add 50% buffer
EST_HOURS=$(python3 -c "import math; print(max(1, math.ceil($GAMES_PER_GPU / 17 * 1.5)))")
WALL_TIME="${EST_HOURS}:00:00"
# Cap at 48h (partition max)
if [ "$EST_HOURS" -gt 48 ]; then
    WALL_TIME="48:00:00"
fi

echo "=== Razzle Training Batch ==="
echo "Total games:      $TOTAL_GAMES"
echo "GPU jobs:         $NUM_GPUS"
echo "Games per GPU:    $GAMES_PER_GPU"
echo "Est. wall time:   $WALL_TIME"
echo "============================="
echo ""

mkdir -p "$SCRIPT_DIR/logs"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    JOB_NAME="razzle-w${i}"
    JOB_ID=$(sbatch \
        --job-name="$JOB_NAME" \
        --time="$WALL_TIME" \
        --export="WORKER_BASE_ID=${i},MAX_GAMES=${GAMES_PER_GPU}" \
        --parsable \
        "${SCRIPT_DIR}/worker.sbatch")
    echo "  Submitted $JOB_NAME (job $JOB_ID, $GAMES_PER_GPU games)"
done

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER -n razzle"
echo "  tail -f $SCRIPT_DIR/logs/worker_*.out"
