# KnightBall Web Application

This is the web frontend for the KnightBall project.

## Project Overview

A React-based web interface for playing KnightBall against the AI or other players. Supports both server-side and client-side (in-browser) AI inference.

## Responsibilities

This component handles:
- **Game board visualization** - Interactive 8x7 SVG board with drag-and-drop
- **Move input** - Click/drag pieces, highlight legal moves
- **Game state management** - React state, server sync
- **Client-side AI** - ONNX neural net inference + MCTS in a Web Worker
- **Multiplayer** - Real-time online games via WebSocket
- **UI/UX** - Responsive design, sound effects, evaluation meter

## Architecture

```
webapp/
├── src/
│   ├── components/        # React components
│   │   ├── Board.tsx      # Game board (SVG)
│   │   ├── Piece.tsx      # Game piece rendering
│   │   ├── EvaluationMeter.tsx
│   │   ├── MoveHistory.tsx
│   │   ├── OnlineLobby.tsx
│   │   └── ...
│   ├── engine/            # Client-side game engine (TypeScript port)
│   │   ├── bitboard.ts    # BigInt bitboard ops, KNIGHT_ATTACKS table
│   │   ├── state.ts       # EngineState interface, applyMove
│   │   ├── moves.ts       # getLegalMoves, getPassMoves, getKnightMoves
│   │   ├── tensor.ts      # stateToTensor for ONNX inference
│   │   ├── symmetry.ts    # MOVE_ROTATION_MAP, rotatePolicy180
│   │   ├── mcts.ts        # PUCT search with pass quiescence
│   │   ├── evaluator.ts   # OnnxEvaluator, RandomEvaluator
│   │   └── modelCache.ts  # IndexedDB cache for ONNX models
│   ├── workers/
│   │   └── ai.worker.ts   # Web Worker for off-thread AI search
│   ├── hooks/
│   │   ├── useGame.ts     # Game state management + AI orchestration
│   │   └── useAIWorker.ts # React hook for worker lifecycle
│   ├── api/
│   │   ├── engine.ts      # REST API client (games, models, ONNX)
│   │   └── online.ts      # Online multiplayer API
│   ├── contexts/
│   │   └── AuthContext.tsx # Auth state
│   └── types/             # TypeScript types
├── package.json
└── Dockerfile
```

## Client-Side AI

The webapp includes a full TypeScript port of the game engine and MCTS, enabling AI inference directly in the browser without server round-trips.

### How it works
1. On game start, the webapp fetches ONNX model info from `/api/models/onnx/latest` (or `/api/models/onnx/by-name/{name}` for a specific model)
2. The ONNX model is downloaded and cached in IndexedDB
3. A Web Worker runs MCTS using onnxruntime-web for neural net evaluation
4. When the user switches models, the new ONNX is fetched (exported on-demand server-side if needed)

### Model loading
- Models are loaded reactively when `aiModel` changes in useGame
- `random_weights` uses a RandomEvaluator (no ONNX needed)
- Falls back to server-side AI if client model isn't loaded yet

## Engine API

The webapp communicates with the engine via REST/WebSocket API through an nginx proxy (`/api` -> engine:8000).

See `docs/ENGINE_API.md` for the full contract.

### Key Endpoints

```typescript
POST /games                           // Create game
GET  /games/{id}                      // Get state
POST /games/{id}/move                 // Make move
POST /games/{id}/ai                   // Server-side AI move
POST /games/{id}/resign               // Resign
GET  /models                          // List models (with has_onnx flag)
GET  /models/onnx/latest              // Latest ONNX (auto-exports if needed)
GET  /models/onnx/by-name/{name}      // Specific ONNX (on-demand export)
GET  /models/onnx/{filename}          // Download ONNX file
```

## Development

```bash
# Install dependencies (include dev for vitest, tsc, etc.)
npm install --include=dev

# Start dev server
npm run dev

# Build for production
npm run build

# Type check
./node_modules/.bin/tsc --noEmit

# Run engine tests
./node_modules/.bin/vitest run src/engine/__tests__/engine.test.ts
```

## Current Status

- [x] Project scaffolding
- [x] Board component (SVG with drag-and-drop)
- [x] API client
- [x] Single-player vs AI (server-side and client-side)
- [x] Client-side ONNX inference with Web Worker
- [x] On-demand ONNX export (any model playable client-side)
- [x] Online multiplayer via WebSocket
- [x] Mobile responsiveness
- [x] Sound effects
- [x] Evaluation meter
- [x] Game history browser and replay viewer
- [x] Analysis board

## Dependencies

- React 18 + React Router
- TypeScript
- Vite (build tool)
- TailwindCSS
- onnxruntime-web (browser ONNX inference)
- recharts (training dashboard charts)
