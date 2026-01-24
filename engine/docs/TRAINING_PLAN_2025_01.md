# Training Plan - January 2025

## Goals

1. Train a medium-sized network (128 filters, 10 blocks) from scratch
2. Validate the new difficulty head learns meaningful predictions
3. Budget ~$10 on Vast.ai, stop early if not improving

## Configuration

### Network
- **Size**: Medium (128 filters, 10 blocks, ~3.5M parameters)
- **New feature**: Difficulty head predicting search difficulty [0-1]

### Self-Play Settings
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Simulations | 400 | Faster games, more data per hour |
| Temperature | 1.0 (first 15 moves), 0.3 (after) | Exploration early, exploitation late |
| Batch size | 32 | Good GPU utilization |
| Dirichlet noise | alpha=0.3, epsilon=0.25 | Standard exploration |

### Training Settings
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 256 | Standard for stability |
| Learning rate | 0.001 | Adam default, stable for training |
| Epochs per iteration | 5 | Faster iterations, more frequent updates |
| Games per iteration | 50 | Train frequently early on |
| Policy weight | 1.0 | Standard |
| Value weight | 1.0 | Standard |
| Difficulty weight | 0.5 | Lower weight since it's auxiliary |
| Illegal penalty | 1.0 | Standard Lagrange multiplier |

### Infrastructure
| Parameter | Value |
|-----------|-------|
| Workers | 4 |
| Workers per instance | 2 |
| GPU type | RTX 3060 or better |
| Max price | $0.15/hr |
| Estimated cost | ~$2-3/hr total |

## Pre-Flight Checklist

1. **Clear existing training data**:
   ```bash
   curl -X DELETE https://razzledazzle.lazybrains.com/api/training/clear
   ```

2. **Verify API server is running**:
   ```bash
   curl https://razzledazzle.lazybrains.com/api/health
   ```

3. **Check no existing workers**:
   ```bash
   curl https://razzledazzle.lazybrains.com/api/training/dashboard
   ```

## Launch Command

```bash
cd /home/projects/razzle/engine

python3 scripts/train_distributed.py \
    --api-url https://razzledazzle.lazybrains.com/api \
    --workers 4 \
    --workers-per-instance 2 \
    --network-size medium \
    --simulations 400 \
    --batch-size 32 \
    --threshold 50 \
    --epochs 5 \
    --gpu RTX_3060 \
    --max-price 0.15
```

## Monitoring Progress

### Key Metrics to Watch

1. **Policy Loss**: Should decrease from ~7.0 to <3.0
   - Initial (random): ~7-8 (log(3137) ≈ 8)
   - Good progress: <4.0 after 10 iterations
   - Target: <2.5

2. **Value Loss**: Should decrease from ~0.5 to <0.2
   - Initial (random): ~0.5-1.0
   - Good progress: <0.3 after 10 iterations
   - Target: <0.15

3. **Difficulty Loss**: Should decrease from ~0.7 to <0.5
   - Initial: ~0.69 (BCE with 0.5 predictions)
   - Good progress: <0.6 after 10 iterations
   - This is auxiliary - don't worry if it's slower

4. **Illegal Move Penalty**: Should decrease to near zero
   - Initial: ~0.99 (almost all mass on illegal)
   - Good progress: <0.3 after 5 iterations
   - Target: <0.05

5. **Game Length**: Should be reasonable
   - Too short (<20 moves): Possible bug
   - Normal: 30-60 moves
   - Too long (>100 moves): Draws, may indicate weak play

### Dashboard Commands

```bash
# Check training dashboard
curl -s https://razzledazzle.lazybrains.com/api/training/dashboard | jq

# Check latest model
curl -s https://razzledazzle.lazybrains.com/api/training/models/latest | jq

# Watch training progress
watch -n 30 'curl -s https://razzledazzle.lazybrains.com/api/training/dashboard | jq ".games_pending, .games_total, .latest_model.final_loss"'
```

### Local Analysis

```bash
# Plot training curves (after downloading games log)
python3 scripts/plot_training.py --log output/trainer/games_log.jsonl

# Analyze policy quality
python3 scripts/analyze_training.py --api-url https://razzledazzle.lazybrains.com/api
```

## Decision Points

### After 5 iterations (~25 games each = 125 games total)
- **Continue if**: Policy loss < 5.0, illegal penalty < 0.5
- **Stop if**: Losses not decreasing or illegal penalty still > 0.8

### After 10 iterations (~500 games total, ~1-2 hours)
- **Continue if**: Policy loss < 4.0, value loss < 0.4
- **Stop if**: Losses plateaued for 3+ iterations

### After 20 iterations (~1000 games total, ~3-4 hours)
- **Continue if**: Still improving, budget allows
- **Stop if**: Losses plateaued or budget exhausted

## Validation Tests

After training, validate the model:

### 1. Play against random
```bash
# Should win >90% after good training
python3 cli/play.py --model output/models/iter_XXX.pt --watch --simulations 200
```

### 2. Check difficulty predictions
```bash
python3 -c "
from razzle.ai.network import RazzleNet
from razzle.ai.evaluator import BatchedEvaluator
from razzle.core.state import GameState

net = RazzleNet.load('output/models/iter_XXX.pt')
ev = BatchedEvaluator(net)

# Check difficulty varies by position
state = GameState.new_game()
for _ in range(10):
    p, v, d = ev.evaluate_with_difficulty(state)
    print(f'Difficulty: {d:.3f}, Value: {v:.3f}')
    # Make a move
    from razzle.core.moves import get_legal_moves
    state.apply_move(get_legal_moves(state)[0])
"
```

### 3. Test timed game
```bash
python3 cli/play.py --model output/models/iter_XXX.pt --time 60 --simulations 1000
```

## Troubleshooting

### Workers not producing games
- Check Vast.ai instances are running
- Check worker logs: `vastai logs INSTANCE_ID`
- Verify API connectivity from workers

### Loss not decreasing
- Check game lengths (too short = bug)
- Check visit counts in games (should have variety)
- Verify player perspective tracking in training data

### High illegal penalty persists
- This is expected initially
- Should drop quickly in first 5 iterations
- If stuck, check legal mask computation

## Post-Training

1. **Download final model**:
   ```bash
   curl -o final_model.pt https://razzledazzle.lazybrains.com/api/training/models/iter_XXX/download
   ```

2. **Copy to webapp** (if deploying):
   ```bash
   cp final_model.pt /path/to/webapp/public/models/
   ```

3. **Update CLAUDE.md** with new model location and training results
