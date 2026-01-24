# Training Architecture

This document explains the AlphaZero-style training pipeline for Razzle Dazzle, including critical implementation details and the reasoning behind design decisions.

## Overview

The training pipeline follows the AlphaZero approach:
1. **Self-play**: Neural network plays against itself using MCTS
2. **Data collection**: Game states, MCTS policies, and outcomes are recorded
3. **Training**: Network learns to predict MCTS policies (policy head) and game outcomes (value head)
4. **Iteration**: New network generates better self-play data

## Neural Network Architecture

### Input
- 7 planes of 8x7 (board is 8 rows × 7 columns = 56 squares)
- Planes:
  - Plane 0: Current player's pieces
  - Plane 1: Current player's ball
  - Plane 2: Opponent's pieces
  - Plane 3: Opponent's ball
  - Plane 4: Touched mask (pieces that can't receive passes)
  - Plane 5: Current player indicator (all 1s if player 0)
  - Plane 6: Has passed indicator (all 1s if mid-pass turn)

### Output
- **Policy head**: 3137 logits (56×56 = 3136 possible src→dst moves + 1 for END_TURN)
- **Value head**: Single scalar in [-1, 1] predicting game outcome

### Action Space

Moves are encoded as `src * 56 + dst` where:
- Knight moves: piece at `src` moves to `dst`
- Ball passes: ball at `src` passes to piece at `dst`
- END_TURN: Special action at index 3136 (internally represented as -1)

## Critical Training Details

### 1. Player Perspective Tracking

**Problem**: In Razzle Dazzle, turns don't strictly alternate. Ball passes keep the same player, only knight moves and END_TURN switch players.

**Incorrect approach** (bug that was fixed):
```python
player_to_move = i % 2  # WRONG! Assumes strict alternation
```

**Correct approach**: Replay the game to track actual `state.current_player`:
```python
state = GameState.new_game()
for move, visit_counts in zip(game.moves, game.visit_counts):
    player_to_move = state.current_player  # Correct!
    # ... record training example
    state.apply_move(move)
```

**Why this matters**: Value targets must be from the correct player's perspective. If player tracking is wrong, the network learns contradictory values—the same position gets +1 and -1 targets depending on whether ball passes occurred. This makes learning impossible.

### 2. Legal Move Handling

**Problem**: The policy network outputs 3137 logits, but only ~10-20 moves are legal in any position. An untrained network assigns roughly uniform probability, meaning ~99.5% of probability mass goes to illegal moves.

**Two-part solution**:

#### A. MCTS Masking (at inference)
MCTS only creates child nodes for legal moves and renormalizes the policy:
```python
legal_moves = get_legal_moves(state)
for move in legal_moves:
    prior = policy[move] / sum(policy[legal_moves])
    children[move] = Node(prior=prior)
```

#### B. Training Loss with Illegal Move Penalty

Standard cross-entropy has a problem: if `target[illegal] = 0`, then `0 * log(pred) = 0`, so illegal moves contribute **zero gradient**. The network never learns to avoid them.

**Solution**: Masked cross-entropy + Lagrange multiplier constraint:

```python
# Masked cross-entropy: only compute loss on legal moves
legal_ce = -sum(target[legal] * log(pred[legal]))

# Illegal move penalty: Lagrange multiplier enforcing "prob on illegal = 0"
illegal_penalty = λ * sum(pred[illegal])

# Total policy loss
policy_loss = legal_ce + illegal_penalty
```

**Interpretation**:
- First term: "Among legal moves, learn which is best"
- Second term: "Don't waste probability on illegal moves" (acts as Lagrange multiplier)

Since `sum(pred[illegal]) = 1 - sum(pred[legal])`, minimizing the penalty maximizes probability mass on legal moves.

**Why not just mask?** If we only mask during training (ignore illegal moves), the network never gets a direct signal that illegal moves should have zero probability. The probability mass "wasted" on illegal moves dilutes the network's ability to express preferences among legal moves—effectively raising the temperature.

### 3. Temperature Handling

MCTS uses temperature to control exploration vs exploitation:
- **High temperature** (1.0): Sample moves proportionally to visit counts (exploration)
- **Low temperature** (0.0): Always pick highest visit count (exploitation)

**Important**: Training targets must use the same temperature as self-play:
```python
if temperature != 1.0:
    visits = visits ** (1.0 / temperature)
policy = visits / visits.sum()
```

During self-play, early moves use temperature=1.0 for exploration, later moves use temperature=0.0. The training targets should reflect this.

### 4. Pass Quiescence Search

**Problem**: During a turn, a player can make multiple ball passes before ending their turn. If MCTS evaluates a mid-pass position with the neural network, it may miss winning pass chains.

**Solution**: Exhaustive pass quiescence search. When MCTS reaches a leaf node that's mid-pass (`has_passed=True`), it:
1. Expands all possible pass continuations
2. Recursively searches each branch
3. Takes the **maximum** value (since the same player is moving)

```python
def _quiescence_search(self, node: Node, depth: int = 0) -> float:
    if node.state.is_terminal():
        return node.state.get_result(node.state.current_player)
    if not node.state.has_passed:
        # Turn ended - evaluate here
        _, value = self.evaluator.evaluate(node.state)
        return value
    # Mid-pass: search all children, take max
    best = float('-inf')
    for child in node.children.values():
        best = max(best, self._quiescence_search(child, depth + 1))
    return best
```

**Why max, not min?** During a pass sequence, the same player is moving the entire time. We want the best outcome for that player.

**Config options**:
- `pass_quiescence: bool = True` - Enable/disable
- `pass_quiescence_max_depth: int = 10` - Safety limit

### 5. END_TURN Action

END_TURN is a special move (internally -1) that ends the current player's turn after ball passes. It needs explicit handling:

- **Policy index**: END_TURN maps to index 3136 (NUM_ACTIONS - 1)
- **MCTS expansion**: `policy[END_TURN_ACTION]` is used as prior for END_TURN
- **Training**: END_TURN is included in legal masks when available

## Training Data Flow

### Distributed Training (API-based)

```
Workers (self-play) → API Server → Trainer
     ↑                    ↓
     └──── New Model ─────┘
```

1. **Workers** generate games via MCTS, POST to API with:
   - Move sequence
   - Visit counts per move
   - Game result

2. **Trainer** polls API, reconstructs training data:
   - Replays game to get board states
   - Tracks actual current_player (not i % 2)
   - Generates legal move masks
   - Computes value targets from correct perspective

3. **Training** uses:
   - Masked cross-entropy on legal moves
   - Illegal move penalty
   - MSE for value head

### Local Training (selfplay.py)

Self-play directly records:
- Board state tensors
- MCTS policy outputs (already includes temperature)
- Player at each state
- Legal move masks
- Game result

## Loss Function

```
L = policy_weight * L_policy + value_weight * L_value

L_policy = CE(target[legal], pred[legal]) + λ * Σ pred[illegal]
L_value = MSE(target_value, pred_value)
```

Default weights:
- `policy_weight = 1.0`
- `value_weight = 1.0`
- `illegal_penalty_weight (λ) = 1.0`

## Diagnostics

### Policy Analysis (`scripts/diagnose_policy.py`)

Analyzes whether the network has learned move legality:
```bash
# Check baseline (untrained network)
python scripts/diagnose_policy.py --baseline

# Analyze trained model
python scripts/diagnose_policy.py output/models/iter_100.pt
```

Expected results:
- **Untrained**: ~0.5% probability on legal moves (uniform over 3137)
- **Well-trained**: >90% probability on legal moves

### What to Check if Training Fails

1. **Loss not decreasing**: Check player perspective tracking
2. **High probability on illegal moves**: Check illegal penalty weight, verify masks
3. **Network plays randomly**: Check temperature handling, verify MCTS is searching
4. **Contradictory values**: Check player tracking (the i % 2 bug)

## Configuration

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 0.001  # Adam default, stable for training
    weight_decay: float = 1e-4
    epochs: int = 10
    policy_weight: float = 1.0
    value_weight: float = 1.0
    illegal_penalty_weight: float = 1.0  # λ for illegal move constraint
    device: str = 'cuda'
```

### MCTSConfig

```python
@dataclass
class MCTSConfig:
    num_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    pass_quiescence: bool = True  # Search through pass sequences
    pass_quiescence_max_depth: int = 10
```

## Advanced Training Features

### Replay Buffer

Prevents catastrophic forgetting by mixing old and new positions during training.

```python
from razzle.training.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(max_positions=100_000)
buffer.add(states, policies, values, legal_masks)

# Mix 50% new data with 50% from buffer
buf_states, buf_policies, buf_values, buf_masks = buffer.sample(num_samples)
```

**Config options** (train_local.py):
- `--replay-buffer-size 100000` - Maximum positions to store
- `--replay-mix-ratio 0.5` - Fraction of training batch from buffer

### Checkpoint Gating

Only promotes models that beat the previous best by a threshold.

```python
# New model must win >55% of games against previous best
if win_rate >= args.gating_threshold:
    best_model_path = checkpoint_path  # Promoted!
else:
    print("Model rejected")  # Keep using previous best
```

**Config options** (train_local.py):
- `--gating-games 20` - Games to play for validation
- `--gating-threshold 0.55` - Win rate needed to promote
- `--gating-simulations 100` - MCTS sims for gating games

### Random Opening Moves

Increases opening diversity to break self-play echo chambers.

```python
selfplay = SelfPlay(
    random_opening_moves=4,     # First N moves can be random
    random_opening_fraction=0.3  # 30% of games use random opening
)
```

## Files Reference

| File | Purpose |
|------|---------|
| `razzle/training/trainer.py` | Training loop with masked CE + illegal penalty |
| `razzle/training/selfplay.py` | Self-play game generation |
| `razzle/training/replay_buffer.py` | Replay buffer for preventing forgetting |
| `scripts/trainer.py` | Distributed trainer (fetches from API) |
| `scripts/train_local.py` | Local training with replay buffer + gating |
| `scripts/train_distributed.py` | Distributed training orchestrator |
| `scripts/diagnose_policy.py` | Policy analysis tool |
| `razzle/ai/mcts.py` | Monte Carlo Tree Search with pass quiescence |
| `razzle/ai/network.py` | Neural network architecture (7 input planes) |
