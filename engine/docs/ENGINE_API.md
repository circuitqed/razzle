# Engine API Documentation

This document describes the REST and WebSocket APIs provided by the Razzle Dazzle game engine server.

## Base URL

Development: `http://localhost:8000`
Production: `https://razzledazzle.lazybrains.com`

## Bot Types

The engine supports multiple AI bot types:

| Bot Type | Description | Use Case |
|----------|-------------|----------|
| `neural` | Uses trained neural network for position evaluation | Strongest (if trained well) |
| `mcts` | Pure MCTS with uniform priors over legal moves | Good baseline for testing |
| `random` | Picks random legal moves | Weakest, useful for testing |

**Default:** `mcts` (vanilla MCTS) is the default until we have a well-trained neural model.

## Game Endpoints

### Create Game

`POST /games`

Creates a new game session.

**Request Body:**
```json
{
  "player1_type": "human",
  "player2_type": "ai",
  "ai_simulations": 800,
  "bot_type": "mcts"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `player1_type` | string | `"human"` | Type of player 1 (`"human"` or `"ai"`) |
| `player2_type` | string | `"ai"` | Type of player 2 (`"human"` or `"ai"`) |
| `ai_simulations` | int | `800` | Number of MCTS simulations per move |
| `bot_type` | string | `"mcts"` | AI bot type: `"neural"`, `"mcts"`, or `"random"` |

**Response:**
```json
{
  "game_id": "abc12345"
}
```

### Get Game State

`GET /games/{game_id}`

Returns the current game state.

**Response:**
```json
{
  "game_id": "abc12345",
  "board": {
    "p1_pieces": "20971524",
    "p1_ball": "1",
    "p2_pieces": "1441151880758558720",
    "p2_ball": "72057594037927936"
  },
  "current_player": 0,
  "legal_moves": [261, 279, 318, 372, ...],
  "status": "playing",
  "winner": null,
  "ply": 0,
  "touched_mask": "0",
  "has_passed": false
}
```

**Note:** Bitboards are returned as strings to preserve precision in JavaScript (which loses precision for integers > 2^53).

### Make Move

`POST /games/{game_id}/move`

Make a human move.

**Request Body:**
```json
{
  "move": 261
}
```

**Response:** Same as Get Game State.

### Get AI Move

`POST /games/{game_id}/ai`

Get the AI to calculate and play a move.

**Request Body:**
```json
{
  "simulations": 800,
  "temperature": 0.0,
  "bot_type": "mcts"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `simulations` | int | `800` | Number of MCTS simulations |
| `temperature` | float | `0.0` | Move selection temperature (0=best, 1=exploratory) |
| `bot_type` | string | *game default* | Override bot type for this move (optional) |

**Response:**
```json
{
  "move": 3612,
  "algebraic": "e8-d6",
  "policy": [...],
  "value": 0.15,
  "visits": 800,
  "time_ms": 1234,
  "top_moves": [
    {"move": 3612, "algebraic": "e8-d6", "visits": 450, "value": 0.15},
    {"move": 3668, "algebraic": "e8-f6", "visits": 200, "value": 0.12}
  ],
  "game_state": {...}
}
```

### Get Legal Moves

`GET /games/{game_id}/legal-moves`

Get all legal moves with their types.

**Response:**
```json
{
  "moves": [
    {"move": 261, "algebraic": "c1-d3", "type": "knight"},
    {"move": 279, "algebraic": "c1-b3", "type": "knight"},
    {"move": -1, "algebraic": "END", "type": "end_turn"}
  ]
}
```

Move types: `"knight"`, `"pass"`, `"end_turn"`

### Undo Move

`POST /games/{game_id}/undo`

Undo the last move.

**Response:** Same as Get Game State.

## Analysis Endpoints

### Analyze Position

`POST /analyze`

Analyze a specific board position without making a move.

**Request Body:**
```json
{
  "pieces": ["20971524", "1441151880758558720"],
  "balls": ["1", "72057594037927936"],
  "current_player": 0,
  "touched_mask": "0",
  "has_passed": false,
  "last_knight_dst": -1,
  "simulations": 200
}
```

**Response:**
```json
{
  "value": 0.05,
  "legal_moves": [261, 279, ...],
  "top_moves": [
    {"move": 261, "algebraic": "c1-d3", "visits": 80, "value": 0.08, "policy": 0.12}
  ],
  "time_ms": 450
}
```

### Analyze Game

`POST /games/{game_id}/analyze`

Analyze an entire game, classifying each move.

**Query Parameters:**
- `simulations_per_position` (int, default 200): MCTS simulations per position

**Response:**
```json
{
  "game_id": "abc12345",
  "move_analyses": [
    {
      "move": 261,
      "algebraic": "c1-d3",
      "value_before": 0.0,
      "value_after": 0.05,
      "best_move": 261,
      "best_move_algebraic": "c1-d3",
      "best_value": 0.05,
      "delta": 0.0,
      "classification": "best"
    }
  ],
  "summary": {
    "best": 5,
    "good": 3,
    "inaccuracy": 2,
    "mistake": 1,
    "blunder": 0
  }
}
```

Classifications:
- `best`: delta >= -0.02
- `good`: delta >= -0.08
- `inaccuracy`: delta >= -0.15
- `mistake`: delta >= -0.30
- `blunder`: delta < -0.30

## Game Browser Endpoints

### List Games

`GET /games`

List games with filtering and pagination.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `player_id` | string | Filter by player user ID |
| `status` | string | Filter by status: `"playing"` or `"finished"` |
| `winner` | int | Filter by winner: `0`, `1`, or `null` |
| `date_from` | string | ISO date filter (>=) |
| `date_to` | string | ISO date filter (<=) |
| `page` | int | Page number (default 1) |
| `per_page` | int | Items per page (default 20) |

**Response:**
```json
{
  "games": [
    {
      "game_id": "abc12345",
      "player1_type": "human",
      "player2_type": "ai",
      "status": "finished",
      "winner": 0,
      "move_count": 24,
      "ply": 24,
      "created_at": "2024-01-15T10:30:00",
      "updated_at": "2024-01-15T10:45:00"
    }
  ],
  "total": 100,
  "page": 1,
  "per_page": 20,
  "total_pages": 5
}
```

### Get Full Game

`GET /games/{game_id}/full`

Get full game data including all moves for replay.

**Response:**
```json
{
  "game_id": "abc12345",
  "player1_type": "human",
  "player2_type": "ai",
  "status": "finished",
  "winner": 0,
  "ply": 24,
  "moves": [261, 3612, 318, ...],
  "moves_algebraic": ["c1-d3", "e8-d6", "e1-f3", ...],
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:45:00"
}
```

## WebSocket API

### Connect

`WS /games/{game_id}/ws`

Connect to a game for real-time updates.

### Messages from Server

**State Update:**
```json
{
  "type": "state",
  "data": { /* GameStateResponse */ }
}
```

**Game Over:**
```json
{
  "type": "game_over",
  "data": {
    "winner": 0,
    "reason": "ball_reached_goal"
  }
}
```

**AI Thinking (progress):**
```json
{
  "type": "thinking",
  "data": {
    "simulations_done": 400,
    "simulations_total": 800,
    "current_best": "c1-d3",
    "value": 0.12
  }
}
```

**Error:**
```json
{
  "type": "error",
  "data": {
    "message": "Invalid move",
    "code": "INVALID_MOVE"
  }
}
```

### Messages to Server

**Make Move:**
```json
{
  "type": "move",
  "data": {"move": 261}
}
```

**Request AI Move:**
```json
{
  "type": "ai_move",
  "data": {"simulations": 800}
}
```

**Undo:**
```json
{
  "type": "undo"
}
```

## Authentication Endpoints

### Register

`POST /auth/register`

```json
{
  "username": "player1",
  "password": "secret123",
  "display_name": "Player One"
}
```

### Login

`POST /auth/login`

```json
{
  "username": "player1",
  "password": "secret123"
}
```

### Logout

`POST /auth/logout`

### Get Current User

`GET /auth/me`

Requires authentication cookie.

## Utility Endpoints

### Health Check

`GET /health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model": "output/models/iter_100.pt"
}
```

### Convert Move Format

`GET /util/move?encoded=261` or `GET /util/move?algebraic=c1-d3`

```json
{
  "encoded": 261,
  "algebraic": "c1-d3",
  "src": 4,
  "dst": 17
}
```

## Move Encoding

Moves are encoded as integers:
- **Knight/Pass moves:** `src * 56 + dst` where `src` and `dst` are square indices (0-55)
- **END_TURN:** `-1`

Square indices: a1=0, b1=1, ..., g1=6, a2=7, ..., g8=55

Algebraic notation: `{src_square}-{dst_square}` (e.g., "c1-d3") or "END" for end turn.
