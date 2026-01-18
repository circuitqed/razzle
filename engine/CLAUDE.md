# Razzle Dazzle Engine

This is the AI/engine component of the Razzle Dazzle project.

## Project Overview

Razzle Dazzle is a two-player abstract strategy board game played on an 8x7 board. Each player has 5 pieces that move like chess knights, plus a ball. The goal is to get your ball to the opponent's back row.

This engine implements:
- **Game logic** using efficient bitboards (56 squares fit in 64 bits)
- **AlphaZero-style AI** with MCTS and neural network evaluation
- **Training pipeline** with self-play and Vast.ai cloud GPU integration

## Architecture

```
razzle/
├── core/           # Game logic
│   ├── bitboard.py # Bitboard utilities, precomputed tables
│   ├── state.py    # GameState class
│   └── moves.py    # Move generation
├── ai/             # AI components
│   ├── mcts.py     # Monte Carlo Tree Search
│   ├── network.py  # PyTorch neural network
│   └── evaluator.py# Batched inference
└── training/       # Training pipeline
    ├── selfplay.py # Self-play game generation
    ├── trainer.py  # Network training
    └── vastai.py   # Cloud GPU integration
```

## Key Design Decisions

### Bitboards
- Board is 8x7 = 56 squares, fits in uint64
- Precomputed knight attack tables for fast move generation
- State is 4 integers: p1_pieces, p1_ball, p2_pieces, p2_ball

### Atomic Moves
- Each action is a single move: either a knight move OR a ball pass
- Encoded as: `src * 56 + dst`
- Total action space: 56 * 56 = 3136 possible moves

### Neural Network
- Input: 6 planes of 8x7 (pieces, balls, touched mask, player indicator)
- Architecture: Residual CNN (configurable depth/width)
- Output: Policy (3136 logits) + Value (scalar)

### MCTS
- PUCT selection (exploration vs exploitation)
- Dirichlet noise at root for exploration during training
- Temperature-based move selection

## API Contract

This engine exposes functionality via:
1. **Python library** - Direct import for CLI and training
2. **FastAPI server** - REST/WebSocket for webapp (see `server/`)

The webapp communicates with the engine through the server API defined in `docs/ENGINE_API.md`.

## Development Workflow

```bash
# Install in dev mode
cd engine
pip install -e ".[dev]"

# Run tests
pytest

# Play in terminal
python cli/play.py --simulations 400

# Watch AI vs AI
python cli/play.py --watch --simulations 200

# Local training
python scripts/train_local.py --iterations 5 --games-per-iter 50
```

## Cloud Training (Vast.ai)

```bash
# Install vastai CLI
pip install vastai
vastai set api-key YOUR_KEY

# Run cloud training
python scripts/train_cloud.py --gpu RTX_3090 --max-price 0.30 --data games/
```

## Current Status

- [x] Core game engine with bitboards
- [x] MCTS implementation
- [x] Neural network architecture
- [x] Self-play generation
- [x] Training loop
- [x] Vast.ai integration
- [x] Terminal CLI client
- [ ] FastAPI server
- [ ] Unit tests
- [ ] Trained model

## Next Steps

1. Write unit tests for core game logic
2. Implement FastAPI server for webapp integration
3. Run initial training to verify pipeline works
4. Optimize batched inference for faster self-play
