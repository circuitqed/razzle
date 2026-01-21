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
    ├── selfplay.py   # Self-play game generation
    ├── trainer.py    # Network training
    ├── vastai.py     # Cloud GPU integration
    └── api_client.py # HTTP client for training API
```

## Key Design Decisions

### Bitboards
- Board is 8x7 = 56 squares, fits in uint64
- Precomputed knight attack tables for fast move generation
- State is 4 integers: p1_pieces, p1_ball, p2_pieces, p2_ball

### Atomic Moves
- Each action is a single move: either a knight move OR a ball pass OR end turn
- Encoded as: `src * 56 + dst` for moves, index 3136 for END_TURN
- Total action space: 56 * 56 + 1 = 3137 possible actions

### Neural Network
- Input: 6 planes of 8x7 (pieces, balls, touched mask, player indicator)
- Architecture: Residual CNN (configurable depth/width)
- Output: Policy (3137 logits) + Value (scalar)

### Training
- See `docs/TRAINING.md` for detailed training architecture documentation
- Key features:
  - Correct player perspective tracking (turns don't always alternate due to ball passes)
  - Masked cross-entropy loss on legal moves only
  - Illegal move penalty (Lagrange multiplier) to focus probability on legal moves
  - Temperature-aware policy target generation

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

## Distributed Training

The training pipeline uses a REST API architecture for distributed self-play:

```
┌─────────────────────────────────────────────────────────────┐
│                     VAST.AI CLOUD                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │Worker 0 │  │Worker 1 │  │Worker N │  │ Trainer │       │
│  │selfplay │  │selfplay │  │selfplay │  │  train  │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│       └────────────┴─────┬──────┴────────────┘             │
│                          │ HTTPS                           │
└──────────────────────────┼──────────────────────────────────┘
                           ▼
                    API Server
              (razzledazzle.lazybrains.com)
```

**Components:**
- **Workers**: Generate self-play games via MCTS, POST to API
- **Trainer**: Polls API for games, trains network, uploads new model
- **API Server**: Stores games in SQLite, serves models to workers

```bash
# Install vastai CLI
pip install vastai
vastai set api-key YOUR_KEY

# Start distributed training (creates workers + trainer on Vast.ai)
python scripts/train_distributed.py --workers 4

# Or with custom settings
python scripts/train_distributed.py \
    --workers 8 \
    --api-url https://razzledazzle.lazybrains.com \
    --gpu RTX_3060 \
    --max-price 0.10 \
    --threshold 100
```

**Training API Endpoints:**
- `POST /training/games` - Workers submit completed games
- `GET /training/games` - Trainer fetches pending games
- `POST /training/models` - Trainer uploads new model
- `GET /training/models/latest` - Workers check for updates
- `GET /training/dashboard` - Monitor training progress

## Current Status

- [x] Core game engine with bitboards
- [x] MCTS implementation
- [x] Neural network architecture
- [x] Self-play generation
- [x] Training loop with illegal move penalty
- [x] Vast.ai integration
- [x] Terminal CLI client
- [x] FastAPI server (REST + WebSocket)
- [x] Distributed training API
- [x] Unit tests (229+ tests)
- [x] Training bug fixes (player perspective, END_TURN handling)
- [ ] Trained model

## Next Steps

1. Run distributed training to generate trained model (requires fresh start due to architecture changes)
2. Validate with policy diagnostics (`scripts/diagnose_policy.py`)
3. Increase default simulations for better tactical play (currently 800, consider 1600-2000)

## Future Optimizations

### Parallel MCTS (High Priority)
Currently MCTS runs single-threaded. Implementing parallel tree search would significantly improve performance:
- **Virtual loss**: When a thread selects a node, add a temporary "loss" to discourage other threads from selecting the same path
- **Lock-free tree updates**: Use atomic operations for visit counts and value updates
- **Batch leaf evaluation**: Collect multiple leaf nodes across threads, evaluate together on GPU
- Expected speedup: 4-8x on multi-core CPU, more with GPU batching

This is important because:
- Current neural network MCTS: ~500 sims/s (single-threaded)
- With parallel search: potentially 2000-4000 sims/s
- Enables deeper tactical search without increasing wall-clock time

### Other Potential Optimizations
- **Transposition tables**: Cache evaluations for repeated positions
- **Progressive widening**: Limit branching factor early in search, expand as visits increase
- **Move ordering in tree**: Prioritize forced responses (moves that limit opponent options)

## Multi-Agent Development

This project uses multiple AI agents working in parallel:
- **Engine agent** (this codebase) - game logic, AI, server API
- **Webapp agent** - React frontend in `/webapp`

### Communication
- Agents communicate via **GitHub Issues** for bugs and feature requests
- Check issues regularly: `gh issue list`
- After completing tasks, check for new issues from the webapp agent

### Workflow
1. Complete your task
2. Run `gh issue list` to check for new issues
3. Address any bugs or feature requests from the webapp agent
4. Create issues for the webapp agent if you need frontend changes
