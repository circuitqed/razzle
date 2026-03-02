#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/worker_%j.out
#SBATCH --error=logs/worker_%j.err
#
# Razzle Dazzle self-play worker for Sherlock
#
# Runs one self-play worker per GPU. The worker plays games using
# MCTS + neural network and submits them to the training API.

set -e

# --- Configuration ---
API_URL="https://razzledazzle.lazybrains.com/api"
SIMULATIONS=400
BATCH_SIZE=32
NETWORK_SIZE="medium"
WORKER_BASE_ID="${WORKER_BASE_ID:-0}"

# Worker ID: sherlock_<SLURM_JOB_ID>_<base>
WORKER_ID="sherlock_${SLURM_JOB_ID}_${WORKER_BASE_ID}"

# --- Environment setup ---
export PYTHONUNBUFFERED=1
module load py-pytorch/2.4.1_py312

# Project paths
PROJECT_DIR="$SCRATCH/razzle/engine"
WORKSPACE="$SCRATCH/razzle-workers/${WORKER_ID}"
mkdir -p "$WORKSPACE" "$PROJECT_DIR/scripts/sherlock/logs"

# Load API key
if [ -f "$SCRATCH/razzle/.env" ]; then
    export $(grep -v '^#' "$SCRATCH/razzle/.env" | xargs)
fi

if [ -z "$TRAINING_API_KEY" ]; then
    echo "ERROR: TRAINING_API_KEY not set. Create $SCRATCH/razzle/.env with TRAINING_API_KEY=..."
    exit 1
fi

# --- GPU info ---
echo "=== Worker $WORKER_ID ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA visible: $CUDA_VISIBLE_DEVICES"
echo "API: $API_URL"
echo "Simulations: $SIMULATIONS"
echo "Started: $(date)"
echo "========================="

# --- Run worker ---
cd "$PROJECT_DIR"

python3 scripts/worker_selfplay.py \
    --worker-id "$WORKER_ID" \
    --api-url "$API_URL" \
    --workspace "$WORKSPACE" \
    --device cuda \
    --simulations "$SIMULATIONS" \
    --batch-size "$BATCH_SIZE" \
    --network-size "$NETWORK_SIZE" \
    --temperature-moves 30 \
    --random-opening-fraction 0.3 \
    --arena-fraction 0.1 \
    --model-check-interval 5
