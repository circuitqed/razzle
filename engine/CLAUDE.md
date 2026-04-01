# KnightBall Engine

This is the AI/engine component of the KnightBall project.

## Project Overview

KnightBall is a two-player abstract strategy board game played on an 8x7 board. Each player has 5 pieces that move like chess knights, plus a ball. The goal is to get your ball to the opponent's back row.

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
- Input: 7 planes of 8x7 (pieces, balls, touched mask, player indicator, has_passed)
- Architecture: AlphaZero-style residual CNN with small projection heads
- Output: Policy (3137 logits) + Value (scalar) + Difficulty (scalar)
- Presets: `small` (~236K), `medium` (~2.4M), `large` (~24M, = AlphaZero chess)
- Use `create_network(preset='medium')` to create networks

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
              (knightball.org)
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
    --api-url https://knightball.org \
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

## Training Data Management

**IMPORTANT**: Training data requires careful lifecycle management. Multiple training runs can coexist but share the same DB.

### Training Runs
Each run is named (e.g. `pegasus`, `gryphon`) and models are versioned as `{run_name}_iter_{N:03d}`. Use `--run-name` when launching `train_distributed.py`.

### What lives where
- **Training games DB** (`server/data/games.db`): Temporary storage. Games are consumed by the trainer and can be cleared between runs. The DB grows large (4+ GB) and should be VACUUMed periodically.
- **Model .pt files** (`output/models/`): Persist on disk even after DB records are deleted. The webapp difficulty tiers reference specific .pt files by name — do NOT delete model files that are referenced in `webapp/src/utils/autoMatch.ts` or `webapp/src/components/NewGameDialog.tsx`.
- **ONNX exports** (`output/models/*.onnx`): Auto-generated on demand from .pt files. Safe to delete (will be regenerated).
- **Training metrics** (DB `training_metrics` table): Dashboard chart data. Lost when DB is cleared between runs.
- **Trainer state** (DB `trainer_state` table): Optimizer state + replay buffer. Lost when DB is cleared.

### Before starting a new run
1. Upload a seed model: create a fresh network with the correct architecture and upload via `TrainingAPIClient.upload_model()` with `{run_name}_iter_000`
2. Clear old training data if changing architectures: delete old models from `training_models` table (NOT the .pt files on disk), clear `training_games`, `training_metrics`, and `trainer_state` tables
3. The `training/models/latest` endpoint returns the highest iteration — old run models with higher iteration numbers will shadow new run's seed model if not cleared

### Completed training runs
- **pegasus**: Medium net (96f/12b, 2.4M params), 635 iterations, 422k games. Best strength at iter_250 (Elo 1219). Models used for webapp difficulty tiers. Old rules models archived in `output/old_rules_models.zip`.
- **gryphon**: Large net (256f/20b, 24M params), in progress.

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
- [x] Trained model (262 iterations via distributed training)
- [x] On-demand ONNX export for browser inference
- [x] Model arena for comparing model strengths

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
