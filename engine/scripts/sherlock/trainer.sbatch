#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/trainer_%j.out
#SBATCH --error=logs/trainer_%j.err
#SBATCH --job-name=razzle-trainer
#
# Razzle Dazzle trainer for Sherlock
#
# Polls the training API for pending games, trains the neural network,
# and uploads new model checkpoints. Runs continuously until wall time
# or all games are consumed.

set -e

# --- Configuration ---
API_URL="https://razzledazzle.lazybrains.com/api"
NETWORK_SIZE="medium"
THRESHOLD=512         # Games before training iteration
BATCH_SIZE=512        # Training batch size
EPOCHS=5              # Epochs per iteration
REPLAY_BUFFER=100000  # Replay buffer capacity (positions)
GAMMA=0.99
TD_LAMBDA=0.95
POLL_INTERVAL=60      # Seconds between checking for new games

# --- Environment setup ---
export PYTHONUNBUFFERED=1
module load py-pytorch/2.4.1_py312

# Project paths
PROJECT_DIR="$SCRATCH/razzle/engine"
OUTPUT_DIR="$SCRATCH/razzle-trainer/output"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR/scripts/sherlock/logs"

# Load API key
if [ -f "$SCRATCH/razzle/.env" ]; then
    export $(grep -v '^#' "$SCRATCH/razzle/.env" | xargs)
fi

if [ -z "$TRAINING_API_KEY" ]; then
    echo "ERROR: TRAINING_API_KEY not set. Create $SCRATCH/razzle/.env with TRAINING_API_KEY=..."
    exit 1
fi

# --- Info ---
echo "=== Razzle Trainer ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA visible: $CUDA_VISIBLE_DEVICES"
echo "API: $API_URL"
echo "Threshold: $THRESHOLD games"
echo "Batch size: $BATCH_SIZE"
echo "Replay buffer: $REPLAY_BUFFER positions"
echo "Started: $(date)"
echo "======================="

# --- Run trainer ---
cd "$PROJECT_DIR"

python3 scripts/trainer.py \
    --api-url "$API_URL" \
    --device cuda \
    --threshold "$THRESHOLD" \
    --poll-interval "$POLL_INTERVAL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate 0.001 \
    --replay-buffer-size "$REPLAY_BUFFER" \
    --network-size "$NETWORK_SIZE" \
    --output "$OUTPUT_DIR" \
    --gamma "$GAMMA" \
    --td-lambda "$TD_LAMBDA"
