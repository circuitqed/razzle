"""
SQLite persistence for game and user storage.

Provides durable storage for games and user accounts.
"""

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from razzle.core.state import GameState


# Default database location
DEFAULT_DB_PATH = Path(__file__).parent / "games.db"

# Password hashing configuration
HASH_ITERATIONS = 100000
HASH_ALGORITHM = 'sha256'


def init_db(db_path: Path = None) -> None:
    """Initialize the database schema."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    with get_connection(db_path) as conn:
        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at TEXT NOT NULL,
                last_login_at TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

        # Games table
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
        # Migrations for existing databases
        migrations = [
            ("ALTER TABLE games ADD COLUMN moves_json TEXT NOT NULL DEFAULT '[]'", None),
            ("ALTER TABLE games ADD COLUMN player1_user_id TEXT", None),
            ("ALTER TABLE games ADD COLUMN player2_user_id TEXT", None),
            ("ALTER TABLE games ADD COLUMN ai_model_version TEXT", None),
            ("ALTER TABLE games ADD COLUMN bot_type TEXT DEFAULT 'mcts'", None),
        ]
        for sql, _ in migrations:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Indexes for user lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_player1 ON games(player1_user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_player2 ON games(player2_user_id)")

        # Training games from self-play workers
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT NOT NULL,
                moves TEXT NOT NULL,
                result REAL NOT NULL,
                visit_counts TEXT NOT NULL,
                model_version TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_games_status ON training_games(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_games_created ON training_games(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_games_worker ON training_games(worker_id)")

        # Model checkpoints
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE NOT NULL,
                iteration INTEGER NOT NULL,
                games_trained_on INTEGER,
                final_loss REAL,
                final_policy_loss REAL,
                final_value_loss REAL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_models_iteration ON training_models(iteration)")

        # Training metrics history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                -- Policy metrics
                policy_top1_accuracy REAL,
                policy_top3_accuracy REAL,
                policy_entropy REAL,
                policy_legal_mass REAL,
                policy_ebf REAL,
                policy_confidence REAL,
                -- Value metrics
                value_mean REAL,
                value_std REAL,
                value_extremity REAL,
                value_calibration_error REAL,
                -- Pass metrics
                pass_decision_rate REAL,
                -- Loss metrics
                loss_total REAL,
                loss_policy REAL,
                loss_value REAL,
                loss_difficulty REAL,
                loss_illegal_penalty REAL,
                -- Game stats
                num_games INTEGER,
                num_examples INTEGER,
                avg_game_length REAL,
                -- Meta
                learning_rate REAL,
                model_version TEXT,
                train_time_sec REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_metrics_iteration ON training_metrics(iteration)")

        conn.commit()


@contextmanager
def get_connection(db_path: Path = None):
    """Get a database connection with proper cleanup."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
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
    player1_user_id: Optional[str] = None,
    player2_user_id: Optional[str] = None,
    ai_model_version: Optional[str] = None,
    bot_type: Optional[str] = None,
    db_path: Path = None
) -> None:
    """Save or update a game in the database.

    If moves is None, only updates state_json (doesn't overwrite moves_json).
    If moves is provided, updates both state_json and moves_json.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat()
    state_json = state_to_json(state)
    # Default bot_type for backward compatibility
    if bot_type is None:
        bot_type = "mcts"

    with get_connection(db_path) as conn:
        if moves is not None:
            # Full insert/update including moves
            moves_json = json.dumps(moves)
            conn.execute("""
                INSERT INTO games (game_id, player1_type, player2_type, ai_simulations, state_json,
                                 moves_json, player1_user_id, player2_user_id, ai_model_version,
                                 bot_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    moves_json = excluded.moves_json,
                    updated_at = excluded.updated_at
            """, (game_id, player1_type, player2_type, ai_simulations, state_json, moves_json,
                  player1_user_id, player2_user_id, ai_model_version, bot_type, now, now))
        else:
            # Update state only, don't touch moves_json (for in-game state updates)
            conn.execute("""
                INSERT INTO games (game_id, player1_type, player2_type, ai_simulations, state_json,
                                 moves_json, player1_user_id, player2_user_id, ai_model_version,
                                 bot_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
            """, (game_id, player1_type, player2_type, ai_simulations, state_json,
                  player1_user_id, player2_user_id, ai_model_version, bot_type, now, now))
        conn.commit()


def append_move(game_id: str, move: int, db_path: Path = None) -> None:
    """Append a move to a game's move history."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
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


def associate_user_with_game(game_id: str, user_id: str, db_path: Path = None) -> bool:
    """
    Associate a user with a game if not already associated.

    This allows users who log in after starting a game to still have
    the game counted as theirs.

    Returns True if the association was made, False if already associated.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        # Check if game already has a player1_user_id
        row = conn.execute(
            "SELECT player1_user_id FROM games WHERE game_id = ?",
            (game_id,)
        ).fetchone()

        if row is None:
            return False

        # Only update if not already set
        if row["player1_user_id"] is None:
            conn.execute(
                "UPDATE games SET player1_user_id = ? WHERE game_id = ?",
                (user_id, game_id)
            )
            conn.commit()
            return True

        return False


def pop_move(game_id: str, db_path: Path = None) -> Optional[int]:
    """Remove and return the last move from a game's history (for undo)."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
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


def load_game(game_id: str, db_path: Path = None) -> Optional[dict]:
    """
    Load a game from the database.

    Returns dict with keys: game_id, player1_type, player2_type, ai_simulations, state, moves, bot_type
    Or None if not found.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()

        if row is None:
            return None

        # Handle missing columns (old databases)
        moves_json = row["moves_json"] if "moves_json" in row.keys() else "[]"
        bot_type = row["bot_type"] if "bot_type" in row.keys() else "mcts"

        return {
            "game_id": row["game_id"],
            "player1_type": row["player1_type"],
            "player2_type": row["player2_type"],
            "ai_simulations": row["ai_simulations"],
            "state": state_from_json(row["state_json"]),
            "moves": json.loads(moves_json) if moves_json else [],
            "bot_type": bot_type or "mcts",  # Default for null values
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def load_all_games(db_path: Path = None) -> list[dict]:
    """Load all games from the database."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM games ORDER BY updated_at DESC").fetchall()

        games = []
        for row in rows:
            # Handle missing columns (old databases)
            moves_json = row["moves_json"] if "moves_json" in row.keys() else "[]"
            bot_type = row["bot_type"] if "bot_type" in row.keys() else "mcts"

            games.append({
                "game_id": row["game_id"],
                "player1_type": row["player1_type"],
                "player2_type": row["player2_type"],
                "ai_simulations": row["ai_simulations"],
                "state": state_from_json(row["state_json"]),
                "moves": json.loads(moves_json) if moves_json else [],
                "bot_type": bot_type or "mcts",  # Default for null values
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return games


def delete_game(game_id: str, db_path: Path = None) -> bool:
    """Delete a game from the database. Returns True if deleted."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
        conn.commit()
        return cursor.rowcount > 0


def cleanup_old_games(max_age_days: int = 7, empty_game_max_age_hours: int = 1, db_path: Path = None) -> int:
    """
    Delete old games and abandoned empty games.

    - Games older than max_age_days are deleted
    - Games with no moves older than empty_game_max_age_hours are deleted

    Returns number of games deleted.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    from datetime import timedelta

    old_cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
    empty_cutoff = (datetime.utcnow() - timedelta(hours=empty_game_max_age_hours)).isoformat()

    with get_connection(db_path) as conn:
        # Delete old games
        cursor1 = conn.execute(
            "DELETE FROM games WHERE updated_at < ?", (old_cutoff,)
        )
        old_deleted = cursor1.rowcount

        # Delete empty games (no moves) older than the empty game cutoff
        cursor2 = conn.execute(
            "DELETE FROM games WHERE moves_json = '[]' AND updated_at < ?", (empty_cutoff,)
        )
        empty_deleted = cursor2.rowcount

        conn.commit()
        return old_deleted + empty_deleted


# --- Password Hashing ---

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        password.encode('utf-8'),
        salt.encode('utf-8'),
        HASH_ITERATIONS
    )
    return f"{salt}${key.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        salt, key_hex = password_hash.split('$')
        key = hashlib.pbkdf2_hmac(
            HASH_ALGORITHM,
            password.encode('utf-8'),
            salt.encode('utf-8'),
            HASH_ITERATIONS
        )
        return secrets.compare_digest(key.hex(), key_hex)
    except (ValueError, AttributeError):
        return False


# --- User Management ---

def create_user(
    username: str,
    password: str,
    display_name: Optional[str] = None,
    db_path: Path = None
) -> Optional[dict]:
    """
    Create a new user account.

    Returns user dict on success, None if username already exists.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    user_id = secrets.token_hex(8)
    password_hash = hash_password(password)
    now = datetime.utcnow().isoformat()

    with get_connection(db_path) as conn:
        try:
            conn.execute("""
                INSERT INTO users (user_id, username, password_hash, display_name, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, username.lower(), password_hash, display_name or username, now))
            conn.commit()
            return {
                "user_id": user_id,
                "username": username.lower(),
                "display_name": display_name or username,
                "created_at": now,
            }
        except sqlite3.IntegrityError:
            return None  # Username already exists


def authenticate_user(
    username: str,
    password: str,
    db_path: Path = None
) -> Optional[dict]:
    """
    Authenticate a user by username and password.

    Returns user dict on success, None on failure.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1",
            (username.lower(),)
        ).fetchone()

        if row is None:
            return None

        if not verify_password(password, row["password_hash"]):
            return None

        # Update last login
        now = datetime.utcnow().isoformat()
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE user_id = ?",
            (now, row["user_id"])
        )
        conn.commit()

        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "created_at": row["created_at"],
            "last_login_at": now,
        }


def get_user_by_id(
    user_id: str,
    db_path: Path = None
) -> Optional[dict]:
    """Get a user by their ID."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ? AND is_active = 1",
            (user_id,)
        ).fetchone()

        if row is None:
            return None

        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
        }


def get_user_by_username(
    username: str,
    db_path: Path = None
) -> Optional[dict]:
    """Get a user by their username."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1",
            (username.lower(),)
        ).fetchone()

        if row is None:
            return None

        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
        }


# --- Game Queries for Browser ---

def list_games(
    player_id: Optional[str] = None,
    status: Optional[str] = None,
    winner: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
    db_path: Path = None
) -> dict:
    """
    List games with filtering and pagination.

    Returns dict with games list and pagination info.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conditions = []
    params = []

    if player_id:
        conditions.append("(player1_user_id = ? OR player2_user_id = ?)")
        params.extend([player_id, player_id])

    if date_from:
        conditions.append("created_at >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("created_at <= ?")
        params.append(date_to)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    with get_connection(db_path) as conn:
        # Get total count
        count_row = conn.execute(
            f"SELECT COUNT(*) as count FROM games WHERE {where_clause}",
            params
        ).fetchone()
        total = count_row["count"]

        # Get page of games with usernames from users table
        offset = (page - 1) * per_page
        rows = conn.execute(
            f"""SELECT g.game_id, g.player1_type, g.player2_type, g.player1_user_id, g.player2_user_id,
                       g.state_json, g.moves_json, g.created_at, g.updated_at, g.ai_model_version,
                       u1.username as player1_username, u2.username as player2_username
                FROM games g
                LEFT JOIN users u1 ON g.player1_user_id = u1.user_id
                LEFT JOIN users u2 ON g.player2_user_id = u2.user_id
                WHERE {where_clause.replace('player1_user_id', 'g.player1_user_id').replace('player2_user_id', 'g.player2_user_id').replace('created_at', 'g.created_at')}
                ORDER BY g.updated_at DESC
                LIMIT ? OFFSET ?""",
            params + [per_page, offset]
        ).fetchall()

        games = []
        for row in rows:
            state = state_from_json(row["state_json"])
            moves_json = row["moves_json"] if row["moves_json"] else "[]"
            moves = json.loads(moves_json)

            # Determine game status and winner from state
            game_status = "finished" if state.is_terminal() else "playing"
            game_winner = state.get_winner()

            # Skip in-progress games with no moves (abandoned/empty games)
            if game_status == "playing" and len(moves) == 0:
                continue

            # Apply status/winner filters (post-fetch since they depend on state)
            if status and game_status != status:
                continue
            if winner is not None and game_winner != winner:
                continue

            games.append({
                "game_id": row["game_id"],
                "player1_type": row["player1_type"],
                "player2_type": row["player2_type"],
                "player1_user_id": row["player1_user_id"],
                "player2_user_id": row["player2_user_id"],
                "player1_username": row["player1_username"],
                "player2_username": row["player2_username"],
                "status": game_status,
                "winner": game_winner,
                "move_count": len(moves),
                "ply": state.ply,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "ai_model_version": row["ai_model_version"],
            })

        return {
            "games": games,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }


def get_game_full(game_id: str, db_path: Path = None) -> Optional[dict]:
    """
    Get full game data for replay including all moves.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT game_id, player1_type, player2_type, player1_user_id, player2_user_id,
                      state_json, moves_json, created_at, updated_at, ai_model_version
               FROM games WHERE game_id = ?""",
            (game_id,)
        ).fetchone()

        if row is None:
            return None

        state = state_from_json(row["state_json"])
        moves_json = row["moves_json"] if row["moves_json"] else "[]"
        moves = json.loads(moves_json)

        # Convert moves to algebraic notation
        from razzle.core.moves import move_to_algebraic
        moves_algebraic = [move_to_algebraic(m) for m in moves]

        return {
            "game_id": row["game_id"],
            "player1_type": row["player1_type"],
            "player2_type": row["player2_type"],
            "player1_user_id": row["player1_user_id"],
            "player2_user_id": row["player2_user_id"],
            "status": "finished" if state.is_terminal() else "playing",
            "winner": state.get_winner(),
            "ply": state.ply,
            "moves": moves,
            "moves_algebraic": moves_algebraic,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "ai_model_version": row["ai_model_version"],
            "state": state,
        }


# --- Training Data Management ---

# Default models directory
DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "output" / "models"


def save_training_game(
    worker_id: str,
    moves: list[int],
    result: float,
    visit_counts: list[dict[int, int]],
    model_version: Optional[str] = None,
    db_path: Path = None
) -> int:
    """
    Save a training game from a self-play worker.

    Returns the game ID.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO training_games (worker_id, moves, result, visit_counts, model_version, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
        """, (worker_id, json.dumps(moves), result, json.dumps(visit_counts), model_version, now))
        conn.commit()
        return cursor.lastrowid


def get_pending_training_games(
    limit: int = 100,
    mark_used: bool = True,
    db_path: Path = None
) -> tuple[list[dict], int]:
    """
    Fetch pending training games.

    Args:
        limit: Maximum number of games to return
        mark_used: If True, atomically mark returned games as 'used'

    Returns:
        Tuple of (list of games, total pending count)
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        # Get total pending count
        count_row = conn.execute(
            "SELECT COUNT(*) as count FROM training_games WHERE status = 'pending'"
        ).fetchone()
        total_pending = count_row["count"]

        # Fetch games
        rows = conn.execute("""
            SELECT id, worker_id, moves, result, visit_counts, model_version, created_at
            FROM training_games
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,)).fetchall()

        games = []
        game_ids = []
        for row in rows:
            games.append({
                "id": row["id"],
                "worker_id": row["worker_id"],
                "moves": json.loads(row["moves"]),
                "result": row["result"],
                "visit_counts": json.loads(row["visit_counts"]),
                "model_version": row["model_version"],
                "created_at": row["created_at"],
            })
            game_ids.append(row["id"])

        # Mark as used if requested
        if mark_used and game_ids:
            placeholders = ",".join("?" * len(game_ids))
            conn.execute(
                f"UPDATE training_games SET status = 'used' WHERE id IN ({placeholders})",
                game_ids
            )
            conn.commit()

        return games, total_pending


def get_all_training_games(
    limit: int = 500,
    offset: int = 0,
    db_path: Path = None
) -> tuple[list[dict], int]:
    """
    Fetch all training games for analysis (both pending and used).

    Args:
        limit: Maximum number of games to return
        offset: Number of games to skip

    Returns:
        Tuple of (list of games, total count)
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        # Get total count
        total = conn.execute("SELECT COUNT(*) FROM training_games").fetchone()[0]

        # Fetch games
        rows = conn.execute("""
            SELECT id, worker_id, moves, result, visit_counts, model_version, created_at
            FROM training_games
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

        games = []
        for row in rows:
            games.append({
                "id": row["id"],
                "worker_id": row["worker_id"],
                "moves": json.loads(row["moves"]),
                "result": row["result"],
                "visit_counts": json.loads(row["visit_counts"]),
                "model_version": row["model_version"],
                "created_at": row["created_at"],
            })

        return games, total


def get_training_games_stats(db_path: Path = None) -> dict:
    """Get statistics about training games."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        # Count by status
        rows = conn.execute("""
            SELECT status, COUNT(*) as count FROM training_games GROUP BY status
        """).fetchall()
        status_counts = {row["status"]: row["count"] for row in rows}

        # Count by worker
        rows = conn.execute("""
            SELECT worker_id, COUNT(*) as count FROM training_games GROUP BY worker_id
        """).fetchall()
        worker_counts = {row["worker_id"]: row["count"] for row in rows}

        # Get last activity per worker
        rows = conn.execute("""
            SELECT worker_id, MAX(created_at) as last_seen FROM training_games GROUP BY worker_id
        """).fetchall()
        worker_last_seen = {row["worker_id"]: row["last_seen"] for row in rows}

        return {
            "total": sum(status_counts.values()),
            "pending": status_counts.get("pending", 0),
            "used": status_counts.get("used", 0),
            "by_worker": worker_counts,
            "worker_last_seen": worker_last_seen,
        }


def save_training_model(
    version: str,
    iteration: int,
    file_path: str,
    games_trained_on: Optional[int] = None,
    final_loss: Optional[float] = None,
    final_policy_loss: Optional[float] = None,
    final_value_loss: Optional[float] = None,
    db_path: Path = None
) -> int:
    """
    Save a training model checkpoint record.

    Returns the model ID.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO training_models
                (version, iteration, games_trained_on, final_loss, final_policy_loss, final_value_loss, file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version) DO UPDATE SET
                games_trained_on = excluded.games_trained_on,
                final_loss = excluded.final_loss,
                final_policy_loss = excluded.final_policy_loss,
                final_value_loss = excluded.final_value_loss,
                file_path = excluded.file_path
        """, (version, iteration, games_trained_on, final_loss, final_policy_loss, final_value_loss, file_path, now))
        conn.commit()
        return cursor.lastrowid


def get_latest_training_model(db_path: Path = None) -> Optional[dict]:
    """Get the most recent training model."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM training_models ORDER BY iteration DESC LIMIT 1
        """).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "version": row["version"],
            "iteration": row["iteration"],
            "games_trained_on": row["games_trained_on"],
            "final_loss": row["final_loss"],
            "final_policy_loss": row["final_policy_loss"],
            "final_value_loss": row["final_value_loss"],
            "file_path": row["file_path"],
            "created_at": row["created_at"],
        }


def get_training_model_by_version(version: str, db_path: Path = None) -> Optional[dict]:
    """Get a specific training model by version."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM training_models WHERE version = ?
        """, (version,)).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "version": row["version"],
            "iteration": row["iteration"],
            "games_trained_on": row["games_trained_on"],
            "final_loss": row["final_loss"],
            "final_policy_loss": row["final_policy_loss"],
            "final_value_loss": row["final_value_loss"],
            "file_path": row["file_path"],
            "created_at": row["created_at"],
        }


def list_training_models(limit: int = 50, db_path: Path = None) -> list[dict]:
    """List training models, most recent first."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM training_models ORDER BY iteration DESC LIMIT ?
        """, (limit,)).fetchall()

        return [{
            "id": row["id"],
            "version": row["version"],
            "iteration": row["iteration"],
            "games_trained_on": row["games_trained_on"],
            "final_loss": row["final_loss"],
            "final_policy_loss": row["final_policy_loss"],
            "final_value_loss": row["final_value_loss"],
            "file_path": row["file_path"],
            "created_at": row["created_at"],
        } for row in rows]


def clear_training_data(db_path: Path = None) -> dict:
    """Clear all training games and models. Returns counts of deleted items."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        # Get counts before deletion
        games_count = conn.execute("SELECT COUNT(*) FROM training_games").fetchone()[0]
        models_count = conn.execute("SELECT COUNT(*) FROM training_models").fetchone()[0]
        metrics_count = conn.execute("SELECT COUNT(*) FROM training_metrics").fetchone()[0]

        # Delete all training games
        conn.execute("DELETE FROM training_games")

        # Get file paths before deleting models
        model_rows = conn.execute("SELECT file_path FROM training_models").fetchall()
        model_files = [row["file_path"] for row in model_rows]

        # Delete all training models from DB
        conn.execute("DELETE FROM training_models")

        # Delete all training metrics
        conn.execute("DELETE FROM training_metrics")

        conn.commit()

        # Delete model files from disk
        deleted_files = 0
        for file_path in model_files:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    deleted_files += 1
            except Exception:
                pass  # Ignore file deletion errors

        return {
            "games_deleted": games_count,
            "models_deleted": models_count,
            "metrics_deleted": metrics_count,
            "files_deleted": deleted_files,
        }


# --- Training Metrics Management ---

def save_training_metrics(
    iteration: int,
    metrics: dict,
    db_path: Path = None
) -> int:
    """
    Save training metrics for an iteration.

    Args:
        iteration: Training iteration number
        metrics: Dictionary of metric values

    Returns:
        The metrics record ID.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO training_metrics (
                iteration, timestamp,
                policy_top1_accuracy, policy_top3_accuracy, policy_entropy,
                policy_legal_mass, policy_ebf, policy_confidence,
                value_mean, value_std, value_extremity, value_calibration_error,
                pass_decision_rate,
                loss_total, loss_policy, loss_value, loss_difficulty, loss_illegal_penalty,
                num_games, num_examples, avg_game_length,
                learning_rate, model_version, train_time_sec
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            iteration, now,
            metrics.get('policy_top1_accuracy'),
            metrics.get('policy_top3_accuracy'),
            metrics.get('policy_entropy'),
            metrics.get('policy_legal_mass'),
            metrics.get('policy_ebf'),
            metrics.get('policy_confidence'),
            metrics.get('value_mean'),
            metrics.get('value_std'),
            metrics.get('value_extremity'),
            metrics.get('value_calibration_error'),
            metrics.get('pass_decision_rate'),
            metrics.get('loss_total') or metrics.get('loss'),
            metrics.get('loss_policy') or metrics.get('policy_loss'),
            metrics.get('loss_value') or metrics.get('value_loss'),
            metrics.get('loss_difficulty') or metrics.get('difficulty_loss'),
            metrics.get('loss_illegal_penalty') or metrics.get('illegal_penalty'),
            metrics.get('num_games') or metrics.get('games'),
            metrics.get('num_examples') or metrics.get('examples'),
            metrics.get('avg_game_length'),
            metrics.get('learning_rate'),
            metrics.get('model_version'),
            metrics.get('train_time_sec') or metrics.get('train_time'),
        ))
        conn.commit()
        return cursor.lastrowid


def get_training_metrics(
    limit: int = 100,
    offset: int = 0,
    db_path: Path = None
) -> tuple[list[dict], int]:
    """
    Get training metrics history.

    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip

    Returns:
        Tuple of (list of metrics dicts, total count)
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        # Get total count
        total = conn.execute("SELECT COUNT(*) FROM training_metrics").fetchone()[0]

        # Fetch metrics
        rows = conn.execute("""
            SELECT * FROM training_metrics
            ORDER BY iteration ASC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

        metrics = []
        for row in rows:
            metrics.append({
                "id": row["id"],
                "iteration": row["iteration"],
                "timestamp": row["timestamp"],
                "policy_top1_accuracy": row["policy_top1_accuracy"],
                "policy_top3_accuracy": row["policy_top3_accuracy"],
                "policy_entropy": row["policy_entropy"],
                "policy_legal_mass": row["policy_legal_mass"],
                "policy_ebf": row["policy_ebf"],
                "policy_confidence": row["policy_confidence"],
                "value_mean": row["value_mean"],
                "value_std": row["value_std"],
                "value_extremity": row["value_extremity"],
                "value_calibration_error": row["value_calibration_error"],
                "pass_decision_rate": row["pass_decision_rate"],
                "loss_total": row["loss_total"],
                "loss_policy": row["loss_policy"],
                "loss_value": row["loss_value"],
                "loss_difficulty": row["loss_difficulty"],
                "loss_illegal_penalty": row["loss_illegal_penalty"],
                "num_games": row["num_games"],
                "num_examples": row["num_examples"],
                "avg_game_length": row["avg_game_length"],
                "learning_rate": row["learning_rate"],
                "model_version": row["model_version"],
                "train_time_sec": row["train_time_sec"],
            })

        return metrics, total


def get_latest_training_metrics(db_path: Path = None) -> Optional[dict]:
    """Get the most recent training metrics."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM training_metrics ORDER BY iteration DESC LIMIT 1
        """).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "iteration": row["iteration"],
            "timestamp": row["timestamp"],
            "policy_top1_accuracy": row["policy_top1_accuracy"],
            "policy_top3_accuracy": row["policy_top3_accuracy"],
            "policy_entropy": row["policy_entropy"],
            "policy_legal_mass": row["policy_legal_mass"],
            "policy_ebf": row["policy_ebf"],
            "policy_confidence": row["policy_confidence"],
            "value_mean": row["value_mean"],
            "value_std": row["value_std"],
            "value_extremity": row["value_extremity"],
            "value_calibration_error": row["value_calibration_error"],
            "pass_decision_rate": row["pass_decision_rate"],
            "loss_total": row["loss_total"],
            "loss_policy": row["loss_policy"],
            "loss_value": row["loss_value"],
            "loss_difficulty": row["loss_difficulty"],
            "loss_illegal_penalty": row["loss_illegal_penalty"],
            "num_games": row["num_games"],
            "num_examples": row["num_examples"],
            "avg_game_length": row["avg_game_length"],
            "learning_rate": row["learning_rate"],
            "model_version": row["model_version"],
            "train_time_sec": row["train_time_sec"],
        }
