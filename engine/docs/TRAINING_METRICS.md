# Training Metrics Reference

This document defines the metrics used to evaluate Razzle Dazzle model training quality.

## Overview

Training metrics are organized into categories:
1. **Policy Metrics** - How well the network predicts MCTS move choices
2. **Value Metrics** - How well the network predicts game outcomes
3. **Game Quality Metrics** - Self-play diversity and health indicators

---

## 1. Policy Metrics

### 1.1 Policy Top-1 Accuracy

**What it measures**: The percentage of positions where the network's highest-probability move matches MCTS's most-visited move.

**Definition**:
- `target_best` = argmax of MCTS visit count distribution (most-visited move)
- `pred_best` = argmax of network's policy output (after masking illegal moves)
- Top-1 = percentage where `pred_best == target_best`

**Why it matters**: This tells us if the network has learned to predict MCTS's top choice. High accuracy means the network can serve as a good prior for MCTS, reducing the search needed to find good moves.

**Note**: This compares against MCTS's top choice, not the move actually played (which may differ due to temperature sampling during self-play).

**Calculation**:
```
top1_accuracy = count(argmax(pred) == argmax(mcts_visits)) / total_positions
```

**Expected values**:
- Untrained: ~3-5% (random among ~20-30 legal moves)
- Early training: 20-30%
- Well-trained: 40-60%
- Strong model: >60%

**Implementation**: `razzle/training/metrics.py:compute_policy_accuracy()`

---

### 1.2 Policy Top-3 Accuracy

**What it measures**: The percentage of positions where MCTS's most-visited move is among the network's top 3 predictions.

**Definition**:
- `target_best` = argmax of MCTS visit count distribution (most-visited move)
- `pred_top3` = top 3 moves by network probability (after masking illegal moves)
- Top-3 = percentage where `target_best in pred_top3`

**Why it matters**: More forgiving than top-1. Even if the network's #1 isn't perfect, if MCTS's preferred move is in the top 3, search can still find it quickly.

**Calculation**:
```
top3_accuracy = count(argmax(mcts_visits) in top3(pred)) / total_positions
```

**Expected values**:
- Untrained: ~10-15%
- Early training: 35-45%
- Well-trained: 55-70%
- Strong model: >70%

**Implementation**: `razzle/training/metrics.py:compute_policy_accuracy()`

---

### 1.3 Policy Entropy

**What it measures**: Average entropy of the predicted policy distribution, computed only over legal moves.

**Definition**:
1. Mask network output to legal moves only
2. Renormalize to sum to 1
3. Compute entropy: `-sum(p * log(p))`

**Why it matters**: Indicates how "spread out" the network's predictions are among legal moves:
- High entropy: Network is uncertain, distributing probability across many moves
- Low entropy: Network is confident, concentrating probability on few moves

**Calculation**:
```
legal_probs = pred[legal_moves] / sum(pred[legal_moves])  # renormalize
entropy = -sum(legal_probs * log(legal_probs))
```

**Expected values** (with ~15 legal moves on average):
- Maximum possible: log(15) ≈ 2.7 (uniform over legal moves)
- Untrained: ~2.0-2.5 (76%+ of max)
- Well-trained: 1.5-1.8 (60-65% of max)
- Strong model: 1.0-1.5 (more confident)

**Notes**: Should decrease during training as the network learns. Very low entropy (<0.5) may indicate overconfidence or mode collapse.

**Implementation**: `razzle/training/metrics.py:compute_policy_metrics()`

---

### 1.4 Legal Move Mass

**What it measures**: Average probability mass assigned to legal moves (not renormalized).

**Definition**:
```
legal_mass = sum(pred_probs[legal_moves])
```

Where `pred_probs` is the network's softmax output (probabilities sum to 1 over all 3137 actions).

**Why it matters**: Critical diagnostic. The network outputs 3137 action probabilities but only ~10-30 are legal in any position. An untrained network wastes ~99.5% of probability on illegal moves. The illegal move penalty in training drives this up.

**Expected values**:
- Untrained: ~0.5% (uniform over 3137, ~15 legal)
- After a few iterations: 70-80%
- Well-trained: >95%
- Target: >99%

**Notes**: This is the most important early training signal. If legal_mass isn't improving, the illegal move penalty may be misconfigured.

**Implementation**: `razzle/training/metrics.py:compute_policy_metrics()`

---

## 2. Value Metrics

### 2.1 Value Mean

**What it measures**: Average predicted value across all positions.

**Definition**:
```
value_mean = mean(predicted_values)
```

Where predictions are in [-1, +1] from the current player's perspective.

**Why it matters**: Detects systematic bias. Should be close to 0 for balanced self-play (equal P0/P1 wins). A positive bias means the network thinks the current player is usually winning; negative means usually losing.

**Expected values**:
- Balanced training: -0.1 to +0.1
- Slight bias: |mean| 0.1-0.2 (acceptable)
- Problematic bias: |mean| > 0.2

**Implementation**: `razzle/training/metrics.py:compute_value_metrics()`

---

### 2.2 Value Std

**What it measures**: Standard deviation of predicted values.

**Definition**:
```
value_std = std(predicted_values)
```

**Why it matters**: Indicates spread of confidence levels:
- Low std: All predictions cluster near 0 (underconfident)
- High std: Wide range of predictions (appropriate spread)

**Expected values**:
- Underconfident: <0.3
- Appropriate: 0.3-0.6
- High confidence: >0.6

**Implementation**: `razzle/training/metrics.py:compute_value_metrics()`

---

### 2.3 Value Extremity

**What it measures**: Average absolute value of predictions (how confident the network is).

**Definition**:
```
extremity = mean(|predicted_values|)
```

**Why it matters**: Complementary to std. High extremity means predictions are often near -1 or +1. Unlike std, this measures distance from 0 rather than spread around the mean.

**Expected values**:
- Underconfident: <0.3
- Moderate: 0.3-0.5
- Confident: 0.5-0.7
- Very confident: >0.7

**Implementation**: `razzle/training/metrics.py:compute_value_metrics()`

---

## 3. Calibration Metrics

### 3.1 Value Calibration Error (MAE)

**What it measures**: Weighted mean absolute error between predicted values and actual outcomes, bucketed by prediction confidence.

**Definition**:
1. Bucket positions by predicted value (10 bins from -1 to +1)
2. For each bucket:
   - `predicted_mean` = mean of predictions in bucket
   - `actual_mean` = mean of actual game outcomes in bucket
   - `bucket_error` = |predicted_mean - actual_mean|
3. Calibration error = weighted average of bucket errors (weighted by bucket size)

**Actual outcome**: The game result from the current player's perspective at that position:
- +1 if current player won
- -1 if current player lost
- 0 for draw

**Why it matters**: The gold standard for value head quality. A well-calibrated model's predictions match actual win rates. If the model predicts +0.6 for a set of positions, those positions should have ~80% wins for the current player (actual mean ≈ +0.6).

**Expected values**:
- Untrained: ~0.5 (predictions uncorrelated with outcomes)
- Partially trained: 0.2-0.4
- Well-calibrated: <0.15
- Excellent: <0.10

**Implementation**: `razzle/training/metrics.py:compute_value_calibration()`, `compute_calibration_error()`

---

### 3.2 Confidence Accuracy

**What it measures**: When the network is confident (|pred| > 0.5), how often is the prediction directionally correct?

**Definition**:
```
positive_accuracy = mean(actual > 0) where pred > 0.5
negative_accuracy = mean(actual < 0) where pred < -0.5
```

"Correct" means the prediction and outcome have the same sign:
- pred > 0.5 (confident win) → did current player win? (actual > 0)
- pred < -0.5 (confident loss) → did current player lose? (actual < 0)

**Why it matters**: Ensures confident predictions are trustworthy. A network that's often confident but wrong is worse than one that's uncertain.

**Expected values**:
- Random: 50%
- Useful confidence: >65%
- Well-calibrated: >75%

**Implementation**: `scripts/analyze_calibration.py:diagnose_calibration_issues()`

---

## 4. Game Quality Metrics

### 4.1 Game Length Statistics

**What it measures**: Average and standard deviation of game lengths in **turns** (not atomic actions).

**Definition**:
- A "turn" ends when the player switches (after a knight move or END_TURN)
- Pass chains within a turn count as one turn
- `game_length = count(knight_moves) + count(END_TURN)`

**Why it matters**:
- Very short games: May indicate forced wins/bugs
- Very long games: May indicate draws, stalemates, or weak play
- High variance: Diverse game outcomes

**Expected values** (Razzle Dazzle specific):
- Average: 30-60 turns
- Std: 10-25 turns

**Implementation**: `razzle/training/metrics.py:compute_game_diversity()`

---

### 4.2 Opening Diversity

**What it measures**: Ratio of unique opening sequences to total games.

**Definition**:
- Extract the moves comprising the first N **turns** (not atomic actions)
- A turn includes all passes plus the ending knight move or END_TURN
- Diversity = unique openings / total games

**Why it matters**: Low diversity suggests the network has collapsed to playing the same opening repeatedly ("echo chamber"). Training needs exploration.

**Calculation**:
```
opening = moves_in_first_N_turns(game)  # includes passes within those turns
diversity = count(unique_openings) / total_games
```
(Default N=5 turns)

**Expected values**:
- Collapsed: <0.1
- Low diversity: 0.1-0.3
- Healthy: 0.3-0.7
- High diversity: >0.7

**Implementation**: `razzle/training/metrics.py:compute_game_diversity()`

---

### 4.3 Win Rate Balance

**What it measures**: Player 0 vs Player 1 win rates in self-play.

**Definition**:
```
p0_win_rate = P0_wins / (P0_wins + P1_wins)
```

Draws are excluded (though current rules don't produce draws - move limit causes current player to lose).

**Why it matters**: Should be near 50/50 for balanced self-play. Persistent imbalance suggests a first-mover advantage the network hasn't learned to counter.

**Expected values**:
- Balanced: 45-55% P0 win rate
- Slight first-mover advantage: 50-55%
- Concerning imbalance: >60% or <40%

**Note**: Game results are stored as +1 (P0 win) or -1 (P1 win). No draws in current rules.

**Implementation**: Computed in training loop, reported in analysis scripts.

---

## 5. Pass Efficiency Metrics

### 5.1 Pass Eval Ratio

**What it measures**: Fraction of neural network evaluations spent on pass positions (where `has_passed=True`).

**Definition**:
```
pass_eval_ratio = pass_evals / total_evals
```

Where:
- `pass_evals` = evaluations on positions where player is mid-turn (after a pass)
- `total_evals` = all neural network evaluations during MCTS

**Why it matters**: If MCTS spends disproportionate effort evaluating pass sequences, it may be inefficient. The pass quiescence search helps by evaluating pass chains together, but we want to monitor this.

**Expected values**:
- Healthy: 10-25%
- High: >30% (may indicate pass sequences consuming too much search)

**Implementation**: `razzle/ai/mcts.py:MCTSStats.pass_eval_ratio`

---

### 5.2 Quiescence Eval Ratio

**What it measures**: Fraction of evaluations that occur inside pass quiescence search.

**Definition**:
```
quiescence_ratio = quiescence_evals / total_evals
```

**Why it matters**: Quiescence search evaluates all pass continuations when hitting a mid-pass leaf. High ratio means many evals are "extra" work to handle pass chains.

**Expected values**:
- Typical: 20-40%
- High: >50% (pass quiescence may be expensive)

**Implementation**: `razzle/ai/mcts.py:MCTSStats.quiescence_evals`

---

### 5.3 Pass Decision Rate (Move-level)

**What it measures**: Fraction of player decisions that are pass chains (vs knight moves).

**Definition**:
- A "pass decision" = any turn that includes one or more passes (chain treated as one decision)
- A "knight decision" = a turn with no passes (just a knight move)
- `pass_decision_rate = pass_decisions / (pass_decisions + knight_decisions)`

**Why it matters**: Baseline for comparing against pass_eval_ratio. If pass_decision_rate is 15% but pass_eval_ratio is 40%, MCTS is spending disproportionate effort on passes.

**Expected values**:
- Typical: 10-20% (most turns are knight moves)

**Implementation**: `razzle/training/metrics.py:compute_pass_stats()`

---

## 6. Search Complexity Metrics

### 6.1 Effective Branching Factor (EBF)

**What it measures**: How many moves the network's probability is effectively spread across.

**Definition**:
```
EBF = exp(entropy)
```

Where entropy is computed over legal moves (see Policy Entropy).

**Why it matters**: Indicates how much MCTS needs to explore:
- Low EBF (~2-3): Network is confident, few moves need exploration
- High EBF (~8-10): Network is uncertain, many moves seem reasonable

**Expected values**:
- Untrained: 6-8 (spread across many legal moves)
- Well-trained: 4-6 (more focused)
- Very confident: 2-4

**Implementation**: `razzle/training/metrics.py:PolicyMetrics.effective_branching_factor`

---

### 6.2 Policy Confidence

**What it measures**: Average probability assigned to the top legal move.

**Definition**:
```
confidence = max(policy[legal_moves])
```

After renormalizing policy to legal moves only.

**Why it matters**:
- High confidence + high accuracy = good
- High confidence + low accuracy = overconfident (bad)
- Low confidence = uncertain, needs more search

**Expected values**:
- Untrained: 20-35%
- Well-trained: 40-55%
- Very confident: >60%

**Implementation**: `razzle/training/metrics.py:PolicyMetrics.policy_confidence`

---

## 7. Difficulty Metric (Experimental)

### 7.1 Position Difficulty

**What it measures**: KL divergence between the network's raw policy and MCTS's policy. High divergence means MCTS found a much better move than the network predicted.

**Definition**:
```
kl_div = sum(mcts_policy * log(mcts_policy / raw_policy))
difficulty = min(1.0, kl_div / 2.0)
```

Where:
- `raw_policy` = network's output before MCTS search
- `mcts_policy` = visit count distribution after MCTS search

**Why it matters**: Identifies "hard" positions where search significantly improves move selection. Could be used to weight training examples or allocate more search time.

**Expected values**:
- Easy position: <0.3 (network already knows the move)
- Moderate: 0.3-0.6
- Hard position: >0.6 (search was critical)

**Note**: This measures "surprise" - how much MCTS changed the network's opinion. A perfectly trained network would have difficulty ≈ 0 everywhere.

**Implementation**: `scripts/trainer.py:compute_difficulty_target()`

---

## Using These Metrics

### During Training
Log every iteration:
- Policy: legal_mass, top1_accuracy
- Value: calibration_error, mean
- Losses: policy_loss, value_loss, illegal_penalty

### Periodic Evaluation
Every N iterations, run full analysis:
- All policy metrics
- Full calibration table
- Game diversity

### Diagnosis Guide

| Symptom | Likely Cause | Check |
|---------|-------------|-------|
| legal_mass not improving | Illegal penalty too low | Increase `illegal_penalty_weight` |
| top1_accuracy low | Insufficient training | More games, more epochs |
| value_mean far from 0 | Player perspective bug | Check player tracking in data prep |
| calibration_error high | Value head undertrained | More training, check loss weighting |
| opening_diversity low | Temperature too low | Increase exploration temperature |

---

## Implementation Status

| Metric | Implemented | Tested | In Training Loop |
|--------|-------------|--------|------------------|
| Policy top-1 | ✅ | ✅ | ✅ |
| Policy top-3 | ✅ | ✅ | ✅ |
| Policy entropy | ✅ | ✅ | ✅ |
| Legal mass | ✅ | ✅ | ✅ |
| Value mean | ✅ | ✅ | ✅ |
| Value std | ✅ | ✅ | ✅ |
| Value extremity | ✅ | ✅ | ✅ |
| Calibration error | ✅ | ❓ | ❌ |
| Confidence accuracy | ✅ | ❓ | ❌ |
| Game length (turns) | ✅ | ✅ | ❌ |
| Opening diversity | ✅ | ❓ | ❌ |
| Pass eval ratio | ✅ | ✅ | ❌ |
| Quiescence eval ratio | ✅ | ✅ | ❌ |
| Pass decision rate | ✅ | ✅ | ❌ |
| Effective branching factor | ✅ | ✅ | ❌ |
| Policy confidence | ✅ | ✅ | ❌ |
| Position difficulty | ✅ | ❓ | ✅ |
