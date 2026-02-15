#!/bin/bash
#
# Stop distributed training and clean up Vast.ai instances
#

set -e

echo "Stopping distributed training..."

# Kill the orchestrator process
if pgrep -f "train_distributed.py" > /dev/null; then
    pkill -f "train_distributed.py"
    echo "Killed orchestrator process"
else
    echo "Orchestrator not running"
fi

# Clean up any orphaned Vast.ai instances
echo ""
echo "Checking for Vast.ai instances..."
INSTANCES=$(vastai show instances --raw 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); print(' '.join(str(i['id']) for i in data))" 2>/dev/null || echo "")

if [ -n "$INSTANCES" ]; then
    echo "Found instances: $INSTANCES"
    echo "Destroying..."
    for id in $INSTANCES; do
        vastai destroy instance "$id" 2>/dev/null && echo "  Destroyed $id" || echo "  Failed to destroy $id"
    done
else
    echo "No instances running"
fi

echo ""
echo "Training stopped."
