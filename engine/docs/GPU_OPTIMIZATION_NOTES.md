# GPU Utilization Optimizations (2024-01)

## Changes Made

### 1. Increased MCTS Batch Size (16 → 32)
- **Files**: `razzle/ai/mcts.py`, `scripts/worker_selfplay.py`
- Larger batch size means more positions evaluated per GPU call
- Can be tuned further with `--batch-size` argument

### 2. Multiple Workers Per Instance Support
- **File**: `scripts/train_distributed.py`
- New argument: `--workers-per-instance N`
- Launches N worker processes sharing the same GPU
- Each worker gets a separate workspace directory
- Example: `--workers 5 --workers-per-instance 2` creates 5 instances running 10 total workers

### 3. Network Size Presets
- **Files**: `scripts/train_distributed.py`, `scripts/worker_selfplay.py`, `scripts/trainer.py`
- New argument: `--network-size {small,medium,large}`
  - **small**: 64 filters, 6 blocks (~900K params) - fast, current default
  - **medium**: 128 filters, 10 blocks (~3.5M params) - balanced
  - **large**: 256 filters, 15 blocks (~15M params) - strongest but slower
- Default is now **medium** to better utilize GPU compute
- Can still override with explicit `--filters` and `--blocks`

## Recommended Usage for Training

```bash
python3 scripts/train_distributed.py \
    --workers 10 \
    --workers-per-instance 2 \
    --network-size medium \
    --batch-size 32 \
    --simulations 800 \
    --threshold 100
```

This configuration:
- Creates 10 GPU instances running 20 worker processes total
- Uses a larger network (128 filters, 10 blocks) for better learning capacity
- Uses batch size 32 for efficient GPU utilization
- Runs 800 simulations per move for stronger play
- Trains every 100 games for more stable updates
