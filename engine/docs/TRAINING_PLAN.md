# Training Plan - Overnight Run Configuration

## Pre-Launch Parameter Confirmation

Before starting any training run, confirm all parameters with the user. Present the configuration in tables and ask for adjustments.

### Parameters to Confirm

**Network Architecture:**
- Size preset (small/medium/large)
- Custom filters/blocks if needed

**Self-Play:**
- Number of worker instances
- Workers per instance
- MCTS simulations (align with GUI: 256, 512, 1024, 2048)
- Random opening settings

**Trainer:**
- Training threshold (games per iteration)
- Training batch size
- Replay buffer size
- Epochs per iteration

**Infrastructure:**
- GPU type and max price
- Fresh start vs continue existing

### Example Confirmation Dialog

```
## Confirmed Configuration

| Parameter | Value |
|-----------|-------|
| Network | medium (96f/12b, ~2.4M params, AZ-style) |
| Simulations | 1024 |
| Worker instances | 7 |
| Training batch size | 512 |
| Training threshold | 512 games |
| Replay buffer | 100,000 positions |

Ready to start fresh?
```

---

## Current Configuration (February 2026)

### Network
| Parameter | Value | Notes |
|-----------|-------|-------|
| Size | **medium** | 96 filters, 12 residual blocks (AZ-style) |
| Parameters | ~2.4M | 84% in tower, AZ head design (2 policy filters, 1 value filter) |
| Input | 7 planes × 8×7 | Pieces, balls, touched, player, has_passed |
| Output | Policy (3137) + Value + Difficulty | |

### Self-Play Workers (7 instances)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Worker instances | 7 | High throughput |
| Workers per instance | 3 | 21 total worker processes |
| Simulations | 1024 | Aligned with GUI dropdown |
| MCTS batch size | 32 | GPU utilization |
| Temperature moves | 30 | Temp=1.0 for first 30 moves |
| Random opening moves | 8 | Opening diversity |
| Random opening fraction | 30% | Balance exploration/exploitation |
| Arena fraction | 10% | Model vs model matches |
| Dirichlet noise | alpha=0.3, epsilon=0.25 | Standard exploration |

### Trainer (1 instance)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training threshold | 512 games | Games before training iteration |
| Training batch size | 512 | Matched to threshold |
| Epochs per iteration | 10 | Standard |
| Replay buffer | 100,000 positions | Prevent catastrophic forgetting |
| Replay mix ratio | 50% new / 50% buffer | Balance new learning with retention |
| Policy weight | 1.0 | Standard |
| Value weight | 20.0 | Increased to balance with TD(λ) softer targets |
| Difficulty weight | 0.5 | Auxiliary task |
| Illegal penalty | 1.0 | Lagrange multiplier |

### Learning Rate Schedule
| Games | Learning Rate | Phase |
|-------|---------------|-------|
| 0-5k | 0.001 | Warmup |
| 5k-20k | 0.001 | Initial training |
| 20k-50k | 0.0005 | Main training |
| 50k-80k | 0.0002 | Late training |
| 80k+ | 0.0001 | Fine-tuning |

### Infrastructure
| Parameter | Value |
|-----------|-------|
| Workers | 9 |
| Workers per instance | 3 |
| GPU type | RTX 3060 |
| Max price | $0.15/hr |
| Total instances | 8 (7 workers + 1 trainer) |
| Estimated cost | ~$0.40/hr |

---

## Pre-Flight Checklist

1. **Confirm parameters** with user (see above)

2. **Clear existing training data** (if fresh start):
   ```bash
   curl -X DELETE https://razzledazzle.lazybrains.com/api/training/clear
   ```

3. **Verify API server is healthy**:
   ```bash
   curl https://razzledazzle.lazybrains.com/api/health
   ```

4. **Check no existing Vast.ai instances**:
   ```bash
   vastai show instances
   ```

---

## Launch Command

```bash
cd /home/projects/razzle/engine

nohup python3 -u scripts/train_distributed.py \
  --workers 7 \
  --workers-per-instance 3 \
  --api-url https://razzledazzle.lazybrains.com/api \
  --gpu RTX_3060 \
  --max-price 0.15 \
  --simulations 1024 \
  --network-size medium \
  --threshold 512 \
  --batch-size 32 \
  --trainer-batch-size 512 \
  --replay-buffer-size 100000 \
  --random-opening-moves 8 \
  --random-opening-fraction 0.3 \
  --output output/overnight > /tmp/training.log 2>&1 &

echo "Training started. Monitor with: tail -f /tmp/training.log"
```

**CLI Parameters:**
- `--threshold`: Games per training iteration (512)
- `--batch-size`: MCTS batch size for GPU parallelism (32)
- `--trainer-batch-size`: Training batch size (512)
- `--replay-buffer-size`: Replay buffer capacity (100000)

---

## Monitoring

### Dashboard
Press **T** in the webapp to open the training dashboard.

### CLI
```bash
# Watch training log
tail -f /tmp/training.log

# Check API status
curl -s https://razzledazzle.lazybrains.com/api/training/dashboard | jq

# Check latest metrics
curl -s https://razzledazzle.lazybrains.com/api/training/metrics/latest | jq

# List Vast.ai instances
vastai show instances
```

### Key Metrics to Watch

**Early Training (iterations 1-10):**
| Metric | Target | Red Flag |
|--------|--------|----------|
| illegal_penalty | < 0.3 | > 0.8 |
| policy_loss | < 5.0 | > 6.0 |
| legal_mass | > 80% | < 50% |

**Mid Training (iterations 10-50):**
| Metric | Target | Red Flag |
|--------|--------|----------|
| policy_top1_accuracy | > 25% | < 15% |
| value_std | > 0.3 | < 0.2 |

**Late Training (iterations 50+):**
| Metric | Target | Red Flag |
|--------|--------|----------|
| policy_top1_accuracy | > 40% | < 30% |
| policy_loss | < 2.5 | > 3.5 |

---

## Shutdown

```bash
# Kill training process
pkill -f train_distributed

# Destroy all Vast.ai instances
vastai show instances  # Get IDs
vastai destroy instance <ID>  # For each instance

# Or destroy all at once
for id in $(vastai show instances --raw | python3 -c "import sys,json; [print(i['id']) for i in json.load(sys.stdin)]"); do
  vastai destroy instance $id
done
```

---

## Troubleshooting

### Training not starting
- Check Vast.ai instances are running: `vastai show instances`
- Check trainer logs via SSH to trainer instance

### Games not being submitted
- Check worker logs via SSH to worker instances
- Verify API is reachable from workers

### Trainer stuck waiting for games
- Workers may still be setting up (first games take time)
- Check worker GPU utilization in `vastai show instances`

### Instance setup failures
- SSH connection issues are common, usually retry works
- Check max price isn't too low for available GPUs

---

## Post-Training

```bash
# Download final model
curl -o final_model.pt https://razzledazzle.lazybrains.com/api/training/models/iter_XXX/download

# Play against it
python3 cli/play.py --model final_model.pt --simulations 1024
```
