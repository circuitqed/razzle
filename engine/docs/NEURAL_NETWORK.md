# Razzle Dazzle Neural Network Architecture

This document explains the neural network used in the Razzle Dazzle AI engine.

## Overview

The network follows the AlphaZero architecture: a residual convolutional neural network with separate policy and value heads. Given a board position, it outputs:
1. **Policy**: Probability distribution over all possible moves
2. **Value**: Estimated probability of winning from this position

## Input Representation

The board state is encoded as a tensor of shape `(7, 8, 7)`:
- 8 rows × 7 columns = 56 squares (fits in a 64-bit integer for bitboard ops)
- 7 input planes (channels)

### Input Planes

| Plane | Description |
|-------|-------------|
| 0 | Current player's pieces (1 where pieces exist, 0 elsewhere) |
| 1 | Current player's ball position |
| 2 | Opponent's pieces |
| 3 | Opponent's ball position |
| 4 | Touched mask - pieces that cannot receive passes this turn |
| 5 | Player indicator (all 1s if Player 1, all 0s if Player 2) |

The representation is **canonical** - it always shows the position from the current player's perspective, so the network learns one strategy that applies to both players.

## Architecture

```
Input (7, 8, 7)
    │
    ▼
┌─────────────────────┐
│  Input Convolution  │  3×3 conv, 64 filters, BatchNorm, ReLU
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Residual Tower     │  6 residual blocks (configurable)
│  ┌─────────────┐    │
│  │ Conv 3×3    │    │
│  │ BatchNorm   │    │
│  │ ReLU        │    │
│  │ Conv 3×3    │    │
│  │ BatchNorm   │───┐│
│  │     +       │◄──┘│  Skip connection
│  │ ReLU        │    │
│  └─────────────┘    │
└─────────────────────┘
    │
    ├─────────────────────┐
    ▼                     ▼
┌─────────────┐    ┌─────────────┐
│ Policy Head │    │ Value Head  │
│             │    │             │
│ Conv 1×1    │    │ Conv 1×1    │
│ 2 filters   │    │ 1 filter    │
│ BatchNorm   │    │ BatchNorm   │
│ ReLU        │    │ ReLU        │
│ Flatten     │    │ Flatten     │
│ Linear→3136│    │ Linear→64  │
│ LogSoftmax  │    │ ReLU        │
│             │    │ Linear→1   │
│             │    │ Tanh        │
└─────────────┘    └─────────────┘
    │                     │
    ▼                     ▼
 Policy               Value
(3136 log-probs)    (scalar in [-1, 1])
```

### Key Components

**Input Convolution**
- 3×3 convolution transforming 7 input planes → 64 feature maps
- Batch normalization + ReLU activation
- No bias (BatchNorm handles the shift)

**Residual Blocks** (default: 6)
- Two 3×3 convolutions with BatchNorm
- Skip connection adds input to output before final ReLU
- All convolutions preserve spatial dimensions (padding=1)

**Policy Head**
- 1×1 convolution reducing to 2 feature maps
- Flattens to 2 × 8 × 7 = 112 features
- Fully connected layer outputs 3136 logits (56 × 56 possible moves)
- Log-softmax for numerical stability during training

**Value Head**
- 1×1 convolution reducing to 1 feature map
- Flattens to 1 × 8 × 7 = 56 features
- Hidden layer with 64 units
- Output is tanh (bounded to [-1, 1])

## Output Encoding

### Policy (3136 dimensions)
Moves are encoded as `src * 56 + dst` where:
- `src` = source square (0-55)
- `dst` = destination square (0-55)

The policy is a probability distribution over all 3136 possible (src, dst) pairs. During MCTS, only legal moves are considered - illegal move probabilities are masked out and the distribution is renormalized.

### Value (1 dimension)
- +1.0 = current player wins
- -1.0 = opponent wins
- 0.0 = draw

## Configuration

The default configuration creates a network with ~340K parameters:

```python
NetworkConfig(
    num_input_planes=7,    # Fixed by board representation
    num_filters=64,        # Width of residual tower
    num_blocks=6,          # Depth of residual tower
    policy_filters=2,      # Channels before policy FC
    value_filters=1,       # Channels before value FC
    value_hidden=64        # Hidden layer size in value head
)
```

### Scaling

| Config | Filters | Blocks | Parameters | Notes |
|--------|---------|--------|------------|-------|
| Tiny | 32 | 4 | ~90K | Fast, for testing |
| Default | 64 | 6 | ~340K | Good balance |
| Large | 128 | 10 | ~1.5M | Stronger play |
| Full | 256 | 19 | ~25M | AlphaZero-scale |

## Training

The network is trained via self-play using the AlphaZero algorithm:

1. **Self-play**: Games are played using MCTS with the current network
2. **Data collection**: Each position records (state, MCTS policy, game outcome)
3. **Training**: Network is trained to predict the MCTS policy and game outcome

### Loss Function

```
L = L_policy + L_value

L_policy = -Σ π_i * log(p_i)    # Cross-entropy with MCTS policy
L_value  = (z - v)²              # MSE with game outcome
```

Where:
- `π` = MCTS visit distribution (target policy)
- `p` = network policy output
- `z` = actual game outcome (+1, -1, or 0)
- `v` = network value output

### Regularization
- L2 weight decay (1e-4)
- Batch normalization provides implicit regularization
- No dropout (residual networks typically don't need it)

## Integration with MCTS

During search, the network is called once per node expansion:

```python
policy, value = network.predict(state.to_tensor())
```

The policy guides tree exploration (high-probability moves are searched more), while the value provides leaf node evaluations without requiring rollouts.

### PUCT Formula

MCTS uses the PUCT (Polynomial Upper Confidence Trees) formula for node selection:

```
score = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
```

Where:
- `Q` = average value from simulations through this node
- `P` = prior probability from network policy
- `N_parent`, `N_child` = visit counts
- `c_puct` = exploration constant (default 1.5)

## Files

- `razzle/ai/network.py` - Network architecture and config
- `razzle/core/state.py` - Board representation and `to_tensor()` method
- `razzle/ai/mcts.py` - MCTS implementation using the network
- `razzle/training/trainer.py` - Training loop and loss computation
