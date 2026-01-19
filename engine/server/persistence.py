"""
SQLite persistence for game storage.

Provides durable storage for games so they survive server restarts.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from razzle.core.state import GameState


# Default database location
DEFAULT_DB_PATH = Path(__file__).parent / "games.db"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initialize the database schema."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                player1_type TEXT NOT NULL DEFAULT 'human',
                player2_type TEXT NOT NULL DEFAULT 'ai',
                ai_simulations INTEGER NOT NULL DEFAULT 800,
                state_json TEXT NOT NULL,
                moves_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Add moves_json column if it doesn't exist (migration for existing DBs)
        try:
            conn.execute("ALTER TABLE games ADD COLUMN moves_json TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH):
    """Get a database connection with proper cleanup."""
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def state_to_json(state: GameState) -> str:
    """Serialize GameState to JSON string."""
    data = {
        "pieces": state.pieces,
        "balls": state.balls,
        "current_player": state.current_player,
        "touched_mask": state.touched_mask,
        "has_passed": state.has_passed,
        "last_knight_dst": state.last_knight_dst,
        "ply": state.ply,
        # Don't persist history - it's for in-session undo only
    }
    return json.dumps(data)


def state_from_json(json_str: str) -> GameState:
    """Deserialize GameState from JSON string."""
    data = json.loads(json_str)
    state = GameState(
        pieces=data["pieces"],
        balls=data["balls"],
        current_player=data["current_player"],
        touched_mask=data["touched_mask"],
        has_passed=data["has_passed"],
        last_knight_dst=data.get("last_knight_dst", -1),
        ply=data["ply"],
        history=[]  # Fresh history for loaded game
    )
    return state


def save_game(
    game_id: str,
    state: GameState,
    player1_type: str = "human",
    player2_type: str = "ai",
    ai_simulations: int = 800,
    moves: Optional[list[int]] = None,
    db_path: Path = DEFAULT_DB_PATH
) -> None:
    """Save or update a game in the database."""
    now = datetime.utcnow().isoformat()
    state_json = state_to_json(state)
    moves_json = json.dumps(moves if moves is not None else [])

    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO games (game_id, player1_type, player2_type, ai_simulations, state_json, moves_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                state_json = excluded.state_json,
                moves_json = excluded.moves_json,
                updated_at = excluded.updated_at
        """, (game_id, player1_type, player2_type, ai_simulations, state_json, moves_json, now, now))
        conn.commit()


def append_move(game_id: str, move: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Append a move to a game's move history."""
    now = datetime.utcnow().isoformat()

    with get_connection(db_path) as conn:
        # Get current moves
        row = conn.execute("SELECT moves_json FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if row is None:
            return

        moves = json.loads(row["moves_json"]) if row["moves_json"] else []
        moves.append(move)

        conn.execute(
            "UPDATE games SET moves_json = ?, updated_at = ? WHERE game_id = ?",
            (json.dumps(moves), now, game_id)
        )
        conn.commit()


def pop_move(game_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[int]:
    """Remove and return the last move from a game's history (for undo)."""
    now = datetime.utcnow().isoformat()

    with get_connection(db_path) as conn:
        row = conn.execute("SELECT moves_json FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if row is None:
            return None

        moves = json.loads(row["moves_json"]) if row["moves_json"] else []
        if not moves:
            return None

        removed_move = moves.pop()

        conn.execute(
            "UPDATE games SET moves_json = ?, updated_at = ? WHERE game_id = ?",
            (json.dumps(moves), now, game_id)
        )
        conn.commit()
        return removed_move


def load_game(game_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[dict]:
    """
    Load a game from the database.

    Returns dict with keys: game_id, player1_type, player2_type, ai_simulations, state, moves
    Or None if not found.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()

        if row is None:
            return None

        # Handle missing moves_json column (old databases)
        moves_json = row["moves_json"] if "moves_json" in row.keys() else "[]"

        return {
            "game_id": row["game_id"],
            "player1_type": row["player1_type"],
            "player2_type": row["player2_type"],
            "ai_simulations": row["ai_simulations"],
            "state": state_from_json(row["state_json"]),
            "moves": json.loads(moves_json) if moves_json else [],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def load_all_games(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Load all games from the database."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM games ORDER BY updated_at DESC").fetchall()

        games = []
        for row in rows:
            # Handle missing moves_json column (old databases)
            moves_json = row["moves_json"] if "moves_json" in row.keys() else "[]"

            games.append({
                "game_id": row["game_id"],
                "player1_type": row["player1_type"],
                "player2_type": row["player2_type"],
                "ai_simulations": row["ai_simulations"],
                "state": state_from_json(row["state_json"]),
                "moves": json.loads(moves_json) if moves_json else [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return games


def delete_game(game_id: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Delete a game from the database. Returns True if deleted."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
        conn.commit()
        return cursor.rowcount > 0


def cleanup_old_games(max_age_days: int = 7, db_path: Path = DEFAULT_DB_PATH) -> int:
    """
    Delete games older than max_age_days.

    Returns number of games deleted.
    """
    from datetime import timedelta
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM games WHERE updated_at < ?", (cutoff,)
        )
        conn.commit()
        return cursor.rowcount
