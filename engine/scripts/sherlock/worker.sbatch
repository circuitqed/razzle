#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/worker_%j.out
#SBATCH --error=logs/worker_%j.err
#
# Razzle Dazzle self-play worker for Sherlock
#
# Launches multiple self-play worker processes sharing one GPU.
# CPU-bound MCTS tree traversal is the bottleneck, so multiple
# workers keep the GPU better utilized with overlapping inference.

set -e

# --- Configuration ---
API_URL="https://razzledazzle.lazybrains.com/api"
SIMULATIONS=2000
BATCH_SIZE=32
NETWORK_SIZE="medium"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
WORKER_BASE_ID="${WORKER_BASE_ID:-0}"
MAX_GAMES="${MAX_GAMES:-0}"

# --- Environment setup ---
export PYTHONUNBUFFERED=1
module load py-pytorch/2.4.1_py312

# Project paths
PROJECT_DIR="$SCRATCH/razzle/engine"
mkdir -p "$PROJECT_DIR/scripts/sherlock/logs"

# Load API key
if [ -f "$SCRATCH/razzle/.env" ]; then
    export $(grep -v '^#' "$SCRATCH/razzle/.env" | xargs)
fi

if [ -z "$TRAINING_API_KEY" ]; then
    echo "ERROR: TRAINING_API_KEY not set. Create $SCRATCH/razzle/.env with TRAINING_API_KEY=..."
    exit 1
fi

# --- GPU info ---
echo "=== Sherlock Job $SLURM_JOB_ID ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA visible: $CUDA_VISIBLE_DEVICES"
echo "API: $API_URL"
echo "Simulations: $SIMULATIONS"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Max games per worker: ${MAX_GAMES:-unlimited}"
echo "Started: $(date)"
echo "==================================="

# --- Launch workers ---
cd "$PROJECT_DIR"

PIDS=()
for i in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    WORKER_ID="sherlock_${SLURM_JOB_ID}_${WORKER_BASE_ID}_${i}"
    WORKSPACE="$SCRATCH/razzle-workers/${WORKER_ID}"
    mkdir -p "$WORKSPACE"

    echo "Starting worker $WORKER_ID..."

    python3 scripts/worker_selfplay.py \
        --worker-id "$WORKER_ID" \
        --api-url "$API_URL" \
        --workspace "$WORKSPACE" \
        --device cuda \
        --simulations "$SIMULATIONS" \
        --batch-size "$BATCH_SIZE" \
        --network-size "$NETWORK_SIZE" \
        --temperature-moves 15 \
        --random-opening-moves 8 \
        --random-opening-fraction 0.3 \
        --arena-fraction 0.1 \
        --model-check-interval 5 \
        --max-games "$MAX_GAMES" &

    PIDS+=($!)

    # Stagger launches to avoid CUDA initialization race
    if [ "$i" -lt $((WORKERS_PER_GPU - 1)) ]; then
        echo "Waiting 30s before next worker..."
        sleep 30
    fi
done

echo "All $WORKERS_PER_GPU workers launched (PIDs: ${PIDS[*]})"

# Wait for all workers to finish
wait
echo "All workers finished at $(date)"
