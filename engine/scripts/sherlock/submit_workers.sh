#!/bin/bash
# Submit self-play worker jobs to Sherlock SLURM
#
# Usage:
#   ./submit_workers.sh TOTAL_GAMES [NUM_WORKERS]
#
# Examples:
#   ./submit_workers.sh 1000        # 1000 games across 8 workers (default)
#   ./submit_workers.sh 5000 16     # 5000 games across 16 workers
#
# Each worker gets TOTAL_GAMES/NUM_WORKERS games and exits when done.
# At ~17 games/hr per GPU (2000 sims):
#   1000 games / 16 workers = 63 games each ≈ 3.7 hours
#   5000 games / 16 workers = 312 games each ≈ 18 hours

set -e

TOTAL_GAMES=${1:?Usage: $0 TOTAL_GAMES [NUM_WORKERS]}
NUM_WORKERS=${2:-8}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GAMES_PER_WORKER=$(( (TOTAL_GAMES + NUM_WORKERS - 1) / NUM_WORKERS ))

# Estimate wall time: ~85 games/hr on Titan Xp, add 50% buffer for queue/startup
EST_HOURS=$(python3 -c "import math; print(max(1, math.ceil($GAMES_PER_WORKER / 17 * 1.5)))")
WALL_TIME="${EST_HOURS}:00:00"
# Cap at 48h (partition max)
if [ "$EST_HOURS" -gt 48 ]; then
    WALL_TIME="48:00:00"
fi

echo "=== Razzle Training Batch ==="
echo "Total games:      $TOTAL_GAMES"
echo "Workers:          $NUM_WORKERS"
echo "Games per worker: $GAMES_PER_WORKER"
echo "Est. wall time:   $WALL_TIME"
echo "============================="
echo ""

mkdir -p "$SCRIPT_DIR/logs"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    JOB_NAME="razzle-w${i}"
    JOB_ID=$(sbatch \
        --job-name="$JOB_NAME" \
        --time="$WALL_TIME" \
        --export="WORKER_BASE_ID=${i},MAX_GAMES=${GAMES_PER_WORKER}" \
        --parsable \
        "${SCRIPT_DIR}/worker.sbatch")
    echo "  Submitted $JOB_NAME (job $JOB_ID, $GAMES_PER_WORKER games)"
done

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER -n razzle"
echo "  tail -f $SCRIPT_DIR/logs/worker_*.out"
