# Razzle Dazzle Web Application

This is the web frontend for the Razzle Dazzle project.

## Project Overview

A React-based web interface for playing Razzle Dazzle against the AI or other players.

## Responsibilities

This component handles:
- **Game board visualization** - Interactive 8x7 board
- **Move input** - Click/drag pieces, highlight legal moves
- **Game state management** - React state, WebSocket sync
- **Multiplayer** - Real-time games via WebSocket
- **UI/UX** - Responsive design, animations, themes

## Architecture

```
webapp/
├── src/
│   ├── components/    # React components
│   │   ├── Board.tsx  # Game board
│   │   ├── Piece.tsx  # Game piece
│   │   └── ...
│   ├── hooks/         # Custom hooks
│   │   ├── useGame.ts # Game state management
│   │   └── useApi.ts  # Engine API client
│   ├── api/           # API client
│   │   └── engine.ts  # Calls to engine server
│   └── types/         # TypeScript types
├── package.json
└── Dockerfile
```

## Engine API

The webapp communicates with the engine via REST/WebSocket API.

See `docs/ENGINE_API.md` for the full contract.

### Key Endpoints

```typescript
// Create new game
POST /games → { game_id: string }

// Get game state
GET /games/{id} → GameState

// Make a move
POST /games/{id}/move { move: number } → GameState

// Get AI move
POST /games/{id}/ai { simulations?: number } → { move: number, ... }

// Real-time updates
WS /games/{id}/ws
```

### Types

```typescript
interface GameState {
  board: {
    p1_pieces: number;  // Bitboard
    p1_ball: number;
    p2_pieces: number;
    p2_ball: number;
  };
  current_player: 0 | 1;
  legal_moves: number[];
  status: 'playing' | 'won' | 'draw';
  winner: 0 | 1 | null;
}
```

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

## Design Notes

### Board Rendering
- SVG-based board for crisp scaling
- Pieces rendered as distinct shapes (X vs O style)
- Legal move highlighting on piece selection

### State Management
- Local state for UI responsiveness
- Sync with server via WebSocket for multiplayer
- Optimistic updates with rollback on server rejection

### Styling
- TailwindCSS for utility-first styling
- Dark/light theme support
- Mobile-responsive layout

## Current Status

- [ ] Project scaffolding
- [ ] Board component
- [ ] API client
- [ ] Single-player vs AI
- [ ] Multiplayer support
- [ ] Mobile responsiveness

## Dependencies

- React 18+
- TypeScript
- Vite (build tool)
- TailwindCSS
- Socket.io-client (WebSocket)
