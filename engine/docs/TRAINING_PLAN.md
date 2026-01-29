# Training Plan - Fresh Run with Enhanced Value Calibration

## Goals

1. Train a medium-sized network (128 filters, 10 blocks) from scratch
2. Test new **quartic value loss** for better calibration
3. Test new **dashboard metrics tracking** end-to-end
4. Validate small-weight initialization prevents tanh saturation

## New Features in This Run

### 1. Quartic Value Loss
Added a 4th-power term to the value loss to penalize overconfident wrong predictions:
```
value_loss = 1.0 * MSE + 0.25 * mean((pred - target)^4)
```

| Error | Quadratic | Quartic (0.25) | Total |
|-------|-----------|----------------|-------|
| 0.1   | 0.01      | 0.000025       | 0.01  |
| 0.5   | 0.25      | 0.016          | 0.27  |
| 1.0   | 1.0       | 0.25           | 1.25  |
| 2.0   | 4.0       | 4.0            | 8.0   |

### 2. Small Weight Initialization
Final layers of value and difficulty heads initialized with `N(0, 0.01)` to prevent tanh/sigmoid saturation and vanishing gradients.

### 3. Comprehensive Metrics Dashboard
New `/training/metrics` endpoints store and serve historical metrics. Dashboard shows:
- Policy: accuracy, entropy, EBF, confidence, legal mass
- Value: mean, std, extremity, calibration error
- Pass: decision rate
- Loss: all components including quartic

## Configuration

### Network
- **Size**: Medium (128 filters, 10 blocks, ~3.5M parameters)
- **Heads**: Policy, Value (with quartic loss), Difficulty

### Self-Play Settings
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Simulations | 800 | Balanced speed/quality |
| Temperature | 1.0 (first 15 moves), 0.3 (after) | Exploration early |
| Batch size | 32 | Good GPU utilization |
| Dirichlet noise | alpha=0.3, epsilon=0.25 | Standard exploration |

### Training Settings
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 512 | Larger batches for stability |
| Learning rate | 0.001 | Adam default |
| Epochs per iteration | 10 | Standard |
| Games per iteration | 50 | Train frequently |
| Policy weight | 1.0 | Standard |
| Value weight (quadratic) | 1.0 | Standard MSE |
| Value weight (quartic) | 0| Calibration penalty |
| Difficulty weight | 0.5 | Auxiliary task |
| Illegal penalty | 1.0 | Lagrange multiplier |

### Infrastructure
| Parameter | Value |
|-----------|-------|
| Workers | 4-8 |
| Workers per instance | 3 |
| GPU type | RTX 3060  |
| Max price | $0.15/hr |

## Pre-Flight Checklist

1. **Clear existing training data**:
   ```bash
   curl -X DELETE http://localhost:8000/training/clear
   ```

2. **Verify API server has new code** (restart if needed):
   ```bash
   docker restart razzle-engine
   curl http://localhost:8000/health
   ```

3. **Verify metrics endpoint works**:
   ```bash
   curl http://localhost:8000/training/metrics
   # Should return {"metrics":[],"total":0,...}
   ```

4. **Open dashboard** to verify it loads (will show "No metrics available")

## Launch Commands

### Option A: Local Training (for testing)
```bash
cd /home/projects/razzle/engine

python3 scripts/train_local.py \
    --iterations 10 \
    --games-per-iter 20 \
    --simulations 400 \
    --device cuda \
    --network-size medium
```

### Option B: Distributed Training (Vast.ai)
```bash
cd /home/projects/razzle/engine

# Start workers
python3 scripts/train_distributed.py \
    --api-url https://razzledazzle.lazybrains.com/api \
    --workers 4 \
    --workers-per-instance 3 \
    --network-size medium \
    --simulations 800 \
    --batch-size 32 \
    --threshold 50 \
    --gpu RTX_3060 \
    --max-price 0.15

# In another terminal, start trainer
python3 scripts/trainer.py \
    --api-url https://razzledazzle.lazybrains.com/api \
    --device cuda \
    --threshold 50 \
    --network-size medium
```

## Monitoring

### New Dashboard
Press **T** in the webapp to open the training dashboard. It now shows:
- **Overview tab**: Loss trend, accuracy trend, EBF, calibration error
- **Policy tab**: All policy metrics over time
- **Value tab**: Value mean/std/extremity, calibration error
- **Pass tab**: Pass decision rate, game lengths
- **Loss tab**: All loss components including quartic

### CLI Monitoring
```bash
# Check training status
curl -s http://localhost:8000/training/dashboard | jq

# Check latest metrics
curl -s http://localhost:8000/training/metrics/latest | jq

# Watch progress
watch -n 30 'curl -s http://localhost:8000/training/metrics/latest | jq "{iteration, loss_total, policy_top1_accuracy, value_calibration_error}"'
```

## Key Metrics to Watch

### Early Training (iterations 1-10)
| Metric | Target | Red Flag |
|--------|--------|----------|
| illegal_penalty | < 0.3 | > 0.8 |
| policy_loss | < 5.0 | > 6.0 |
| legal_mass | > 80% | < 50% |

### Mid Training (iterations 10-50)
| Metric | Target | Red Flag |
|--------|--------|----------|
| policy_top1_accuracy | > 25% | < 15% |
| value_std | > 0.3 | < 0.2 |
| value_calibration_error | < 0.3 | > 0.4 |

### Late Training (iterations 50+)
| Metric | Target | Red Flag |
|--------|--------|----------|
| policy_top1_accuracy | > 40% | < 30% |
| policy_loss | < 2.5 | > 3.5 |
| value_calibration_error | < 0.15 | > 0.25 |

## Experiment Tracking

### Hypothesis
The quartic loss term will improve value calibration (lower calibration error) compared to pure MSE, without significantly impacting convergence speed.

### What to Compare
1. **value_calibration_error** over iterations
2. **value_std** - should still develop appropriate spread
3. **value_loss** (quadratic) vs **value_loss_quartic** - monitor both

### Success Criteria
- Calibration error < 0.15 by iteration 100
- Policy top-1 accuracy > 40% by iteration 100
- No regression in policy learning speed

## Troubleshooting

### Dashboard shows "No metrics available"
- Trainer hasn't run yet, or
- Engine server needs restart to pick up new code

### Metrics not updating
- Check trainer is running: look for `[Trainer] Metrics submitted to API` logs
- Check API endpoint: `curl http://localhost:8000/training/metrics/latest`

### High value_loss_quartic but normal value_loss
- Expected early in training (large errors)
- Should decrease as calibration improves

### Vanishing gradients (value loss stuck)
- Check if value predictions are all near -1 or +1
- The small init should prevent this, but monitor value_std

## Post-Training Validation

```bash
# Download final model
curl -o final_model.pt http://localhost:8000/training/models/iter_XXX/download

# Run calibration analysis
python3 scripts/analyze_calibration.py --model final_model.pt

# Play against random
python3 cli/play.py --model final_model.pt --watch --simulations 400
```
