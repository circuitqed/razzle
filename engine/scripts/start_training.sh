#!/bin/bash
#
# Start distributed training in the background
#
# Usage:
#   ./scripts/start_training.sh              # Use defaults
#   ./scripts/start_training.sh --workers 8  # Override workers
#
# The script runs train_distributed.py with nohup so it persists
# after the terminal closes. Output goes to /tmp/training.log
#
# To stop training: pkill -f train_distributed.py
# To monitor: tail -f /tmp/training.log
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/training.log"

# Default arguments (can be overridden by passing arguments to this script)
DEFAULT_ARGS=(
    --workers 9
    --workers-per-instance 3
    --network-size medium
    --threshold 512
    --simulations 800
    --batch-size 512
    --gpu RTX_3060
    --max-price 0.15
    --random-opening-moves 8
    --random-opening-fraction 0.3
    --output output/distributed
    --gamma 0.99
    --td-lambda 0.95
)

# Check if already running
if pgrep -f "train_distributed.py" > /dev/null; then
    echo "Training is already running!"
    echo "PID: $(pgrep -f train_distributed.py)"
    echo "Log: $LOG_FILE"
    echo ""
    echo "To stop: pkill -f train_distributed.py"
    exit 1
fi

cd "$ENGINE_DIR"

echo "Starting distributed training..."
echo "Log file: $LOG_FILE"
echo ""

# Use provided args or defaults
if [ $# -gt 0 ]; then
    ARGS=("$@")
else
    ARGS=("${DEFAULT_ARGS[@]}")
fi

echo "Arguments: ${ARGS[*]}"
echo ""

# Start with nohup, unbuffered Python output
nohup python3 -u scripts/train_distributed.py "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Started with PID: $PID"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop with:"
echo "  pkill -f train_distributed.py"
echo "  # or: kill $PID"
echo ""

# Wait a moment and show initial output
sleep 5
echo "--- Initial output ---"
head -30 "$LOG_FILE"
