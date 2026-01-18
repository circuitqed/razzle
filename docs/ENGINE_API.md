# Engine API Specification

This document defines the API contract between the Razzle Dazzle engine server and its clients (webapp, CLI, etc.).

## Base URL

- Development: `http://localhost:8000`
- Production: Configured via environment

## Data Types

### Move Encoding

Moves are encoded as integers: `src * 56 + dst`

Where:
- `src` = source square (0-55)
- `dst` = destination square (0-55)
- Square index = `row * 7 + col` (row 0 = rank 1, col 0 = file a)

Example: Knight on d1 (square 3) moving to c3 (square 16) = `3 * 56 + 16 = 184`

### Bitboards

Board state uses bitboards (64-bit integers). Each bit represents a square:
- Bit 0 = a1, Bit 1 = b1, ... Bit 6 = g1
- Bit 7 = a2, ... Bit 55 = g8

### GameState Object

```json
{
  "game_id": "abc123",
  "board": {
    "p1_pieces": 62,
    "p1_ball": 8,
    "p2_pieces": 4432676798464,
    "p2_ball": 562949953421312
  },
  "current_player": 0,
  "legal_moves": [184, 185, 240, ...],
  "status": "playing",
  "winner": null,
  "ply": 0
}
```

## REST Endpoints

### Create Game

```
POST /games
Content-Type: application/json

Request body (optional):
{
  "player1_type": "human",    // "human" | "ai"
  "player2_type": "ai",
  "ai_simulations": 800
}

Response:
{
  "game_id": "abc123"
}
```

### Get Game State

```
GET /games/{game_id}

Response:
{
  "game_id": "abc123",
  "board": { ... },
  "current_player": 0,
  "legal_moves": [184, 185, ...],
  "status": "playing",
  "winner": null,
  "ply": 5
}
```

### Make Move

```
POST /games/{game_id}/move
Content-Type: application/json

Request:
{
  "move": 184
}

Response: GameState object (updated)

Errors:
- 400: Invalid move
- 404: Game not found
- 409: Not your turn / Game already finished
```

### Get AI Move

Request AI to calculate and play a move.

```
POST /games/{game_id}/ai
Content-Type: application/json

Request:
{
  "simulations": 800,     // Optional, default 800
  "temperature": 0.0      // Optional, default 0 (deterministic)
}

Response:
{
  "move": 184,
  "algebraic": "d1-c3",
  "policy": [0.001, 0.002, ..., 0.45, ...],  // 3136 values
  "value": 0.23,
  "visits": 800,
  "time_ms": 1234,
  "top_moves": [
    {"move": 184, "algebraic": "d1-c3", "visits": 450, "value": 0.25},
    {"move": 240, "algebraic": "d1-e3", "visits": 200, "value": 0.18},
    ...
  ]
}
```

### Get Legal Moves (Human Readable)

```
GET /games/{game_id}/legal-moves

Response:
{
  "moves": [
    {"move": 184, "algebraic": "d1-c3", "type": "knight"},
    {"move": 186, "algebraic": "d1-e3", "type": "knight"},
    {"move": 59, "algebraic": "d1-c1", "type": "pass"},
    ...
  ]
}
```

### Undo Move

```
POST /games/{game_id}/undo

Response: GameState object (previous state)

Errors:
- 400: Nothing to undo
- 404: Game not found
```

## WebSocket API

For real-time games (multiplayer or AI with live updates).

### Connection

```
WS /games/{game_id}/ws
```

### Messages (Server → Client)

**State Update**
```json
{
  "type": "state",
  "data": { /* GameState object */ }
}
```

**AI Thinking**
```json
{
  "type": "thinking",
  "data": {
    "simulations_done": 400,
    "simulations_total": 800,
    "current_best": "d1-c3",
    "value": 0.21
  }
}
```

**Game Over**
```json
{
  "type": "game_over",
  "data": {
    "winner": 0,
    "reason": "ball_reached_goal"
  }
}
```

**Error**
```json
{
  "type": "error",
  "data": {
    "message": "Invalid move",
    "code": "INVALID_MOVE"
  }
}
```

### Messages (Client → Server)

**Make Move**
```json
{
  "type": "move",
  "data": {
    "move": 184
  }
}
```

**Request AI Move**
```json
{
  "type": "ai_move",
  "data": {
    "simulations": 800
  }
}
```

**Undo**
```json
{
  "type": "undo"
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| GAME_NOT_FOUND | 404 | Game ID doesn't exist |
| INVALID_MOVE | 400 | Move is not legal |
| NOT_YOUR_TURN | 409 | Not current player's turn |
| GAME_FINISHED | 409 | Game is already over |
| INTERNAL_ERROR | 500 | Server error |

## Utility Endpoints

### Convert Move Format

```
GET /util/move?encoded=184
Response: {"algebraic": "d1-c3", "src": 3, "dst": 16}

GET /util/move?algebraic=d1-c3
Response: {"encoded": 184, "src": 3, "dst": 16}
```

### Health Check

```
GET /health
Response: {"status": "ok", "version": "0.1.0"}
```
