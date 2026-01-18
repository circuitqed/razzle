# Razzle Dazzle

A two-player abstract strategy board game with AlphaZero-style AI.

## Game Overview

Razzle Dazzle is played on an 8x7 board. Each player has 5 pieces that move like chess knights, plus a ball. The goal is to get your ball to the opponent's back row.

## Project Structure

```
razzle/
├── engine/          # Python game engine + AI
│   ├── razzle/      # Core game logic and AI
│   ├── server/      # FastAPI REST/WebSocket server
│   ├── cli/         # Terminal client
│   └── tests/       # Test suite
├── webapp/          # Frontend web application
├── docs/            # API documentation
└── docker-compose.yml
```

## Quick Start

### With Docker

```bash
docker-compose up
```

- Engine API: http://localhost:8000
- Webapp: http://localhost:7492

### Manual Setup

**Engine:**
```bash
cd engine
pip install -e ".[dev,server]"
pytest  # Run tests
uvicorn server.main:app --reload  # Start server
```

**Webapp:**
```bash
cd webapp
npm install
npm run dev
```

## Components

### Engine (`/engine`)

Python-based game engine with:
- Efficient bitboard representation (56 squares in uint64)
- AlphaZero-style MCTS with neural network evaluation
- FastAPI server with REST and WebSocket APIs
- Self-play training pipeline

### Webapp (`/webapp`)

React/TypeScript frontend for playing the game.

## Documentation

- [Engine API Specification](docs/ENGINE_API.md)

## Development

```bash
# Run engine tests
cd engine && pytest -v

# Run with coverage
pytest --cov=razzle --cov=server
```

## License

MIT
