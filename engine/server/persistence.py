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
                -- Value metrics (validation - computed before training)
                value_mean REAL,
                value_std REAL,
                value_extremity REAL,
                value_mse REAL,
                value_sign_accuracy REAL,
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

        # Arena matches table - results of model vs model matches
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arena_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model1_version TEXT NOT NULL,
                model2_version TEXT NOT NULL,
                model1_wins INTEGER NOT NULL DEFAULT 0,
                model2_wins INTEGER NOT NULL DEFAULT 0,
                draws INTEGER NOT NULL DEFAULT 0,
                simulations INTEGER NOT NULL DEFAULT 200,
                created_at TEXT NOT NULL,
                UNIQUE(model1_version, model2_version, simulations)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_arena_matches_model1 ON arena_matches(model1_version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_arena_matches_model2 ON arena_matches(model2_version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_arena_matches_sims ON arena_matches(simulations)")

        # Arena ratings table - ELO ratings per model per simulation count
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arena_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                simulations INTEGER NOT NULL DEFAULT 800,
                elo_rating REAL NOT NULL DEFAULT 1000.0,
                games_played INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL,
                UNIQUE(model_version, simulations)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_arena_ratings_version ON arena_ratings(model_version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_arena_ratings_sims ON arena_ratings(simulations)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_arena_ratings_elo ON arena_ratings(elo_rating DESC)")

        # Trainer state - optimizer state, replay buffer persisted across runs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trainer_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                iteration INTEGER NOT NULL DEFAULT 0,
                total_games_trained INTEGER NOT NULL DEFAULT 0,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trainer_state_key ON trainer_state(key)")

        # Players table - unified humans and AI models with ELO ratings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                player_type TEXT NOT NULL,
                user_id TEXT,
                model_version TEXT,
                simulations INTEGER,
                display_name TEXT NOT NULL,
                elo_rating REAL NOT NULL DEFAULT 1000.0,
                elo_games_played INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(player_type, model_version, simulations)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_players_type ON players(player_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_players_user ON players(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_players_elo ON players(elo_rating DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_players_model ON players(model_version)")

        # Add player_id columns to games table (migration)
        games_migrations = [
            "ALTER TABLE games ADD COLUMN player1_player_id TEXT REFERENCES players(player_id)",
            "ALTER TABLE games ADD COLUMN player2_player_id TEXT REFERENCES players(player_id)",
            "ALTER TABLE games ADD COLUMN winner INTEGER",  # 0, 1, or NULL for draw/ongoing
        ]
        for sql in games_migrations:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_player1_pid ON games(player1_player_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_player2_pid ON games(player2_player_id)")

        # Online multiplayer columns
        online_migrations = [
            "ALTER TABLE games ADD COLUMN join_code TEXT",
            "ALTER TABLE games ADD COLUMN online_status TEXT DEFAULT 'local'",
            # Values: 'local', 'waiting', 'playing', 'finished', 'abandoned'
        ]
        for sql in online_migrations:
            try:
                conn.execute(sql)
                conn.commit()  # Commit each migration separately
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    pass  # Column already exists
                else:
                    # Log but continue - might be other issues
                    print(f"Migration warning: {e}")

        # Create indexes for online columns (wrapped in try/except in case columns don't exist)
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_games_join_code ON games(join_code)")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_games_online_status ON games(online_status)")
        except sqlite3.OperationalError:
            pass

        # Migration: add new columns if they don't exist
        try:
            conn.execute("ALTER TABLE training_metrics ADD COLUMN value_mse REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute("ALTER TABLE training_metrics ADD COLUMN value_sign_accuracy REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

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
    # Convert all values to Python native types (numpy types aren't JSON serializable)
    data = {
        "pieces": [int(p) for p in state.pieces],
        "balls": [int(b) for b in state.balls],
        "current_player": int(state.current_player),
        "touched_mask": int(state.touched_mask),
        "has_passed": bool(state.has_passed),
        "last_knight_dst": int(state.last_knight_dst) if state.last_knight_dst is not None else -1,
        "ply": int(state.ply),
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
    now = datetime.utcnow().isoformat() + 'Z'
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
    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        # Get current moves
        row = conn.execute("SELECT moves_json FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if row is None:
            return

        moves = json.loads(row["moves_json"]) if row["moves_json"] else []
        moves.append(int(move))  # Convert numpy int64 to Python int

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
    now = datetime.utcnow().isoformat() + 'Z'

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

    old_cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat() + 'Z'
    empty_cutoff = (datetime.utcnow() - timedelta(hours=empty_game_max_age_hours)).isoformat() + 'Z'

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
    now = datetime.utcnow().isoformat() + 'Z'

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
        now = datetime.utcnow().isoformat() + 'Z'
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

    # Always exclude games with no moves (never started)
    conditions.append("moves_json IS NOT NULL AND moves_json != '[]' AND length(moves_json) > 2")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    with get_connection(db_path) as conn:
        # Fetch all matching games (we need to filter by status/winner in Python)
        rows = conn.execute(
            f"""SELECT g.game_id, g.player1_type, g.player2_type, g.player1_user_id, g.player2_user_id,
                       g.state_json, g.moves_json, g.created_at, g.updated_at, g.ai_model_version,
                       u1.username as player1_username, u2.username as player2_username
                FROM games g
                LEFT JOIN users u1 ON g.player1_user_id = u1.user_id
                LEFT JOIN users u2 ON g.player2_user_id = u2.user_id
                WHERE {where_clause.replace('player1_user_id', 'g.player1_user_id').replace('player2_user_id', 'g.player2_user_id').replace('created_at', 'g.created_at')}
                ORDER BY g.updated_at DESC""",
            params
        ).fetchall()

        # Process and filter games
        all_games = []
        for row in rows:
            state = state_from_json(row["state_json"])
            moves_json = row["moves_json"] if row["moves_json"] else "[]"
            moves = json.loads(moves_json)

            # Determine game status and winner from state
            game_status = "finished" if state.is_terminal() else "playing"
            game_winner = state.get_winner()

            # Skip games with no moves (never started) - backup check
            if len(moves) == 0:
                continue

            # Apply status/winner filters
            if status and game_status != status:
                continue
            if winner is not None and game_winner != winner:
                continue

            all_games.append({
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

        # Paginate results
        total = len(all_games)
        offset = (page - 1) * per_page
        games = all_games[offset:offset + per_page]

        return {
            "games": games,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page if total > 0 else 1,
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
    now = datetime.utcnow().isoformat() + 'Z'

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
    now = datetime.utcnow().isoformat() + 'Z'

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

        # Delete trainer state records
        trainer_state_count = conn.execute("SELECT COUNT(*) FROM trainer_state").fetchone()[0]
        state_rows = conn.execute("SELECT file_path FROM trainer_state").fetchall()
        state_files = [row["file_path"] for row in state_rows]
        conn.execute("DELETE FROM trainer_state")

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

        # Delete trainer state files from disk
        for file_path in state_files:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    deleted_files += 1
            except Exception:
                pass

        return {
            "games_deleted": games_count,
            "models_deleted": models_count,
            "metrics_deleted": metrics_count,
            "trainer_states_deleted": trainer_state_count,
            "files_deleted": deleted_files,
        }


# --- Trainer State Management ---

def save_trainer_state_record(
    key: str,
    iteration: int,
    total_games_trained: int,
    file_path: str,
    file_size: int,
    db_path: Path = None
) -> int:
    """
    Save or update a trainer state record.

    Uses INSERT...ON CONFLICT(key) DO UPDATE to upsert.

    Returns the record ID.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO trainer_state (key, iteration, total_games_trained, file_path, file_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                iteration = excluded.iteration,
                total_games_trained = excluded.total_games_trained,
                file_path = excluded.file_path,
                file_size = excluded.file_size,
                created_at = excluded.created_at
        """, (key, iteration, total_games_trained, file_path, file_size, now))
        conn.commit()
        return cursor.lastrowid


def get_trainer_state_record(key: str, db_path: Path = None) -> Optional[dict]:
    """Get a trainer state record by key. Returns dict or None."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM trainer_state WHERE key = ?", (key,)
        ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "key": row["key"],
            "iteration": row["iteration"],
            "total_games_trained": row["total_games_trained"],
            "file_path": row["file_path"],
            "file_size": row["file_size"],
            "created_at": row["created_at"],
        }


def delete_trainer_state_records(db_path: Path = None) -> int:
    """Delete all trainer state records. Returns count of deleted records."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM trainer_state").fetchone()[0]
        conn.execute("DELETE FROM trainer_state")
        conn.commit()
        return count


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
    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO training_metrics (
                iteration, timestamp,
                policy_top1_accuracy, policy_top3_accuracy, policy_entropy,
                policy_legal_mass, policy_ebf, policy_confidence,
                value_mean, value_std, value_extremity, value_mse, value_sign_accuracy, value_calibration_error,
                pass_decision_rate,
                loss_total, loss_policy, loss_value, loss_difficulty, loss_illegal_penalty,
                num_games, num_examples, avg_game_length,
                learning_rate, model_version, train_time_sec
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            metrics.get('value_mse'),
            metrics.get('value_sign_accuracy'),
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
                "value_mse": row["value_mse"] if "value_mse" in row.keys() else None,
                "value_sign_accuracy": row["value_sign_accuracy"] if "value_sign_accuracy" in row.keys() else None,
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
            "value_mse": row["value_mse"] if "value_mse" in row.keys() else None,
            "value_sign_accuracy": row["value_sign_accuracy"] if "value_sign_accuracy" in row.keys() else None,
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


# --- Arena Management ---

def save_arena_match(
    model1_version: str,
    model2_version: str,
    model1_wins: int,
    model2_wins: int,
    draws: int,
    simulations: int = 800,
    db_path: Path = None
) -> int:
    """
    Save or update an arena match result.

    If a match between these two models at this simulation count already exists,
    the results are added to the existing record.

    Returns:
        The match record ID.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        # Check if match already exists
        existing = conn.execute("""
            SELECT id, model1_wins, model2_wins, draws FROM arena_matches
            WHERE model1_version = ? AND model2_version = ? AND simulations = ?
        """, (model1_version, model2_version, simulations)).fetchone()

        if existing:
            # Update existing match
            cursor = conn.execute("""
                UPDATE arena_matches SET
                    model1_wins = model1_wins + ?,
                    model2_wins = model2_wins + ?,
                    draws = draws + ?,
                    created_at = ?
                WHERE id = ?
            """, (model1_wins, model2_wins, draws, now, existing["id"]))
            conn.commit()
            return existing["id"]
        else:
            # Insert new match
            cursor = conn.execute("""
                INSERT INTO arena_matches
                    (model1_version, model2_version, model1_wins, model2_wins, draws, simulations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model1_version, model2_version, model1_wins, model2_wins, draws, simulations, now))
            conn.commit()
            return cursor.lastrowid


def get_arena_matches(
    model_version: Optional[str] = None,
    simulations: Optional[int] = None,
    limit: int = 100,
    db_path: Path = None
) -> list[dict]:
    """
    Get arena match history.

    Args:
        model_version: Filter to matches involving this model
        simulations: Filter by simulation count
        limit: Maximum number of matches to return

    Returns:
        List of match dictionaries.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conditions = []
    params = []

    if model_version:
        conditions.append("(model1_version = ? OR model2_version = ?)")
        params.extend([model_version, model_version])

    if simulations is not None:
        conditions.append("simulations = ?")
        params.append(simulations)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    with get_connection(db_path) as conn:
        rows = conn.execute(f"""
            SELECT * FROM arena_matches
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """, params + [limit]).fetchall()

        return [{
            "id": row["id"],
            "model1_version": row["model1_version"],
            "model2_version": row["model2_version"],
            "model1_wins": row["model1_wins"],
            "model2_wins": row["model2_wins"],
            "draws": row["draws"],
            "simulations": row["simulations"],
            "created_at": row["created_at"],
        } for row in rows]


def get_all_arena_matches(
    simulations: Optional[int] = None,
    db_path: Path = None
) -> list[dict]:
    """
    Get all arena matches for ELO computation.

    Args:
        simulations: Filter by simulation count (None = all)

    Returns:
        List of all match dictionaries.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        if simulations is not None:
            rows = conn.execute("""
                SELECT * FROM arena_matches
                WHERE simulations = ?
                ORDER BY created_at ASC
            """, (simulations,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM arena_matches
                ORDER BY created_at ASC
            """).fetchall()

        return [{
            "id": row["id"],
            "model1_version": row["model1_version"],
            "model2_version": row["model2_version"],
            "model1_wins": row["model1_wins"],
            "model2_wins": row["model2_wins"],
            "draws": row["draws"],
            "simulations": row["simulations"],
            "created_at": row["created_at"],
        } for row in rows]


def save_arena_rating(
    model_version: str,
    iteration: int,
    elo_rating: float,
    games_played: int,
    simulations: int = 800,
    db_path: Path = None
) -> int:
    """
    Save or update an arena ELO rating.

    Returns:
        The rating record ID.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO arena_ratings
                (model_version, iteration, simulations, elo_rating, games_played, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_version, simulations) DO UPDATE SET
                elo_rating = excluded.elo_rating,
                games_played = excluded.games_played,
                last_updated = excluded.last_updated
        """, (model_version, iteration, simulations, elo_rating, games_played, now))
        conn.commit()
        return cursor.lastrowid


def get_arena_ratings(
    simulations: Optional[int] = None,
    limit: int = 100,
    db_path: Path = None
) -> list[dict]:
    """
    Get arena ELO ratings sorted by rating (descending).

    Args:
        simulations: Filter by simulation count (None = all)
        limit: Maximum number of ratings to return

    Returns:
        List of rating dictionaries.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        if simulations is not None:
            rows = conn.execute("""
                SELECT * FROM arena_ratings
                WHERE simulations = ?
                ORDER BY elo_rating DESC
                LIMIT ?
            """, (simulations, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM arena_ratings
                ORDER BY elo_rating DESC
                LIMIT ?
            """, (limit,)).fetchall()

        return [{
            "id": row["id"],
            "model_version": row["model_version"],
            "iteration": row["iteration"],
            "simulations": row["simulations"],
            "elo_rating": row["elo_rating"],
            "games_played": row["games_played"],
            "last_updated": row["last_updated"],
        } for row in rows]


def get_arena_rating_by_version(
    model_version: str,
    simulations: int = 800,
    db_path: Path = None
) -> Optional[dict]:
    """
    Get the arena rating for a specific model version at a given simulation count.

    Returns:
        Rating dictionary or None if not found.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM arena_ratings
            WHERE model_version = ? AND simulations = ?
        """, (model_version, simulations)).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "model_version": row["model_version"],
            "iteration": row["iteration"],
            "simulations": row["simulations"],
            "elo_rating": row["elo_rating"],
            "games_played": row["games_played"],
            "last_updated": row["last_updated"],
        }


def clear_arena_data(db_path: Path = None) -> dict:
    """
    Clear all arena matches and ratings.

    Returns:
        Dictionary with counts of deleted items.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        matches_count = conn.execute("SELECT COUNT(*) FROM arena_matches").fetchone()[0]
        ratings_count = conn.execute("SELECT COUNT(*) FROM arena_ratings").fetchone()[0]

        conn.execute("DELETE FROM arena_matches")
        conn.execute("DELETE FROM arena_ratings")
        conn.commit()

        return {
            "matches_deleted": matches_count,
            "ratings_deleted": ratings_count,
        }


# --- Online Multiplayer Game Management ---

# Characters to use for join codes (avoid ambiguous ones like O/0, I/1/l)
JOIN_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_join_code(length: int = 6, db_path: Path = None) -> str:
    """
    Generate a unique 6-character alphanumeric join code.

    Uses characters that are unambiguous (no O/0, I/1/l).
    Keeps generating until a unique code is found.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        for _ in range(100):  # Max attempts to find unique code
            code = ''.join(secrets.choice(JOIN_CODE_CHARS) for _ in range(length))
            existing = conn.execute(
                "SELECT 1 FROM games WHERE join_code = ?", (code,)
            ).fetchone()
            if not existing:
                return code
        raise RuntimeError("Failed to generate unique join code")


def create_online_game(
    host_user_id: str,
    host_color: int = 0,
    db_path: Path = None
) -> dict:
    """
    Create a new online game with a join code.

    Args:
        host_user_id: User ID of the host
        host_color: 0 for blue (player 1), 1 for red (player 2)

    Returns:
        Dict with game_id, join_code, host_color, status
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    game_id = secrets.token_hex(4)
    join_code = generate_join_code(db_path=db_path)
    now = datetime.utcnow().isoformat() + 'Z'

    state = GameState.new_game()
    state_json = state_to_json(state)

    # Host is player1 if blue, player2 if red
    player1_user_id = host_user_id if host_color == 0 else None
    player2_user_id = host_user_id if host_color == 1 else None

    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO games (
                game_id, player1_type, player2_type, ai_simulations,
                state_json, moves_json, player1_user_id, player2_user_id,
                join_code, online_status, created_at, updated_at
            )
            VALUES (?, 'human', 'human', 0, ?, '[]', ?, ?, ?, 'waiting', ?, ?)
        """, (game_id, state_json, player1_user_id, player2_user_id,
              join_code, now, now))
        conn.commit()

    return {
        "game_id": game_id,
        "join_code": join_code,
        "host_color": host_color,
        "status": "waiting",
    }


def join_online_game(
    join_code: str,
    guest_user_id: str,
    db_path: Path = None
) -> Optional[dict]:
    """
    Join an existing online game using a join code.

    Args:
        join_code: The 6-character join code
        guest_user_id: User ID of the joining player

    Returns:
        Dict with game_id, guest_color, opponent info, or None if not found/invalid
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    now = datetime.utcnow().isoformat() + 'Z'
    join_code = join_code.upper().strip()

    with get_connection(db_path) as conn:
        # Find the waiting game with this code
        row = conn.execute("""
            SELECT game_id, player1_user_id, player2_user_id, online_status
            FROM games
            WHERE join_code = ?
        """, (join_code,)).fetchone()

        if row is None:
            return None  # Game not found

        if row["online_status"] != "waiting":
            return {"error": "game_not_waiting", "status": row["online_status"]}

        game_id = row["game_id"]
        player1_user_id = row["player1_user_id"]
        player2_user_id = row["player2_user_id"]

        # Determine which slot is for the guest
        if player1_user_id is None:
            # Guest is player 1 (blue)
            guest_color = 0
            host_user_id = player2_user_id
            conn.execute("""
                UPDATE games
                SET player1_user_id = ?, online_status = 'playing', updated_at = ?
                WHERE game_id = ?
            """, (guest_user_id, now, game_id))
        elif player2_user_id is None:
            # Guest is player 2 (red)
            guest_color = 1
            host_user_id = player1_user_id
            conn.execute("""
                UPDATE games
                SET player2_user_id = ?, online_status = 'playing', updated_at = ?
                WHERE game_id = ?
            """, (guest_user_id, now, game_id))
        else:
            # Both slots filled (shouldn't happen for waiting games)
            return {"error": "game_full"}

        conn.commit()

        # Get opponent info
        opponent = None
        if host_user_id:
            opponent_row = conn.execute("""
                SELECT u.user_id, u.display_name, p.elo_rating
                FROM users u
                LEFT JOIN players p ON p.user_id = u.user_id AND p.player_type = 'human'
                WHERE u.user_id = ?
            """, (host_user_id,)).fetchone()
            if opponent_row:
                opponent = {
                    "user_id": opponent_row["user_id"],
                    "display_name": opponent_row["display_name"],
                    "elo_rating": opponent_row["elo_rating"] or 1000.0,
                }

    return {
        "game_id": game_id,
        "your_color": guest_color,
        "opponent": opponent,
        "status": "playing",
    }


def get_game_by_code(
    join_code: str,
    db_path: Path = None
) -> Optional[dict]:
    """
    Look up a game by its join code.

    Returns game data or None if not found.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    join_code = join_code.upper().strip()

    with get_connection(db_path) as conn:
        row = conn.execute("""
            SELECT game_id, player1_user_id, player2_user_id,
                   join_code, online_status, state_json, created_at, updated_at
            FROM games
            WHERE join_code = ?
        """, (join_code,)).fetchone()

        if row is None:
            return None

        return {
            "game_id": row["game_id"],
            "player1_user_id": row["player1_user_id"],
            "player2_user_id": row["player2_user_id"],
            "join_code": row["join_code"],
            "online_status": row["online_status"],
            "state": state_from_json(row["state_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def get_online_game_status(
    game_id: str,
    user_id: str,
    db_path: Path = None
) -> Optional[dict]:
    """
    Get the status of an online game from a specific user's perspective.

    Returns dict with game info and user's color, or None if not found/unauthorized.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        row = conn.execute("""
            SELECT g.game_id, g.player1_user_id, g.player2_user_id,
                   g.join_code, g.online_status, g.state_json, g.winner,
                   u1.display_name as p1_name, u2.display_name as p2_name,
                   p1.elo_rating as p1_elo, p2.elo_rating as p2_elo
            FROM games g
            LEFT JOIN users u1 ON g.player1_user_id = u1.user_id
            LEFT JOIN users u2 ON g.player2_user_id = u2.user_id
            LEFT JOIN players p1 ON p1.user_id = g.player1_user_id AND p1.player_type = 'human'
            LEFT JOIN players p2 ON p2.user_id = g.player2_user_id AND p2.player_type = 'human'
            WHERE g.game_id = ?
        """, (game_id,)).fetchone()

        if row is None:
            return None

        # Check if user is part of this game
        if user_id not in (row["player1_user_id"], row["player2_user_id"]):
            return {"error": "not_authorized"}

        your_color = 0 if user_id == row["player1_user_id"] else 1
        state = state_from_json(row["state_json"])
        is_your_turn = state.current_player == your_color

        # Opponent info
        opponent = None
        if your_color == 0 and row["player2_user_id"]:
            opponent = {
                "user_id": row["player2_user_id"],
                "display_name": row["p2_name"],
                "elo_rating": row["p2_elo"] or 1000.0,
            }
        elif your_color == 1 and row["player1_user_id"]:
            opponent = {
                "user_id": row["player1_user_id"],
                "display_name": row["p1_name"],
                "elo_rating": row["p1_elo"] or 1000.0,
            }

        return {
            "game_id": row["game_id"],
            "join_code": row["join_code"],
            "status": row["online_status"],
            "your_color": your_color,
            "opponent": opponent,
            "is_your_turn": is_your_turn,
            "winner": row["winner"],
            "state": state,
        }


def abandon_online_game(
    game_id: str,
    abandoning_user_id: str,
    db_path: Path = None
) -> Optional[dict]:
    """
    Abandon/forfeit an online game.

    The abandoning player loses, opponent wins.

    Returns dict with winner info, or None if not found.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        row = conn.execute("""
            SELECT game_id, player1_user_id, player2_user_id, online_status
            FROM games
            WHERE game_id = ?
        """, (game_id,)).fetchone()

        if row is None:
            return None

        # Check if user is part of this game
        if abandoning_user_id not in (row["player1_user_id"], row["player2_user_id"]):
            return {"error": "not_authorized"}

        # Can only abandon playing or waiting games
        if row["online_status"] not in ("waiting", "playing"):
            return {"error": "cannot_abandon", "status": row["online_status"]}

        # Determine winner (opponent of abandoning player)
        if abandoning_user_id == row["player1_user_id"]:
            winner = 1  # Player 2 wins
        else:
            winner = 0  # Player 1 wins

        # If waiting, just delete the game
        if row["online_status"] == "waiting":
            conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
            conn.commit()
            return {"status": "cancelled", "winner": None}

        # Mark as abandoned with winner
        conn.execute("""
            UPDATE games
            SET online_status = 'abandoned', winner = ?, updated_at = ?
            WHERE game_id = ?
        """, (winner, now, game_id))
        conn.commit()

        return {
            "status": "abandoned",
            "winner": winner,
        }


def get_user_online_games(
    user_id: str,
    db_path: Path = None
) -> dict:
    """
    Get all online games for a user, separated by status.

    Returns dict with 'active' (playing) and 'waiting' games.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT g.game_id, g.player1_user_id, g.player2_user_id,
                   g.join_code, g.online_status, g.state_json, g.winner,
                   g.created_at, g.updated_at,
                   u1.display_name as p1_name, u2.display_name as p2_name
            FROM games g
            LEFT JOIN users u1 ON g.player1_user_id = u1.user_id
            LEFT JOIN users u2 ON g.player2_user_id = u2.user_id
            WHERE (g.player1_user_id = ? OR g.player2_user_id = ?)
              AND g.online_status IN ('waiting', 'playing')
            ORDER BY g.updated_at DESC
        """, (user_id, user_id)).fetchall()

        waiting = []
        active = []

        for row in rows:
            your_color = 0 if user_id == row["player1_user_id"] else 1
            state = state_from_json(row["state_json"])

            game_info = {
                "game_id": row["game_id"],
                "join_code": row["join_code"],
                "status": row["online_status"],
                "your_color": your_color,
                "is_your_turn": state.current_player == your_color,
                "ply": state.ply,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

            # Add opponent info if present
            if your_color == 0 and row["player2_user_id"]:
                game_info["opponent_name"] = row["p2_name"]
            elif your_color == 1 and row["player1_user_id"]:
                game_info["opponent_name"] = row["p1_name"]

            if row["online_status"] == "waiting":
                waiting.append(game_info)
            else:
                active.append(game_info)

        return {
            "waiting": waiting,
            "active": active,
        }


def update_online_game_status(
    game_id: str,
    status: str,
    winner: Optional[int] = None,
    db_path: Path = None
) -> bool:
    """
    Update the online status of a game.

    Returns True if updated successfully.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        if winner is not None:
            cursor = conn.execute("""
                UPDATE games
                SET online_status = ?, winner = ?, updated_at = ?
                WHERE game_id = ?
            """, (status, winner, now, game_id))
        else:
            cursor = conn.execute("""
                UPDATE games
                SET online_status = ?, updated_at = ?
                WHERE game_id = ?
            """, (status, now, game_id))
        conn.commit()
        return cursor.rowcount > 0


# --- Player Management ---

def generate_ai_player_id(model_version: str, simulations: int) -> str:
    """Generate a consistent player ID for an AI model+sims combo."""
    return f"ai_{model_version}_{simulations}"


def generate_human_player_id(user_id: str) -> str:
    """Generate a player ID for a human user."""
    return f"human_{user_id}"


def create_player(
    player_id: str,
    player_type: str,
    display_name: str,
    user_id: Optional[str] = None,
    model_version: Optional[str] = None,
    simulations: Optional[int] = None,
    elo_rating: float = 1000.0,
    db_path: Path = None
) -> Optional[dict]:
    """
    Create a new player (human or AI).

    Returns player dict on success, None if already exists.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        try:
            conn.execute("""
                INSERT INTO players
                    (player_id, player_type, user_id, model_version, simulations,
                     display_name, elo_rating, elo_games_played, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (player_id, player_type, user_id, model_version, simulations,
                  display_name, elo_rating, now))
            conn.commit()
            return {
                "player_id": player_id,
                "player_type": player_type,
                "user_id": user_id,
                "model_version": model_version,
                "simulations": simulations,
                "display_name": display_name,
                "elo_rating": elo_rating,
                "elo_games_played": 0,
                "created_at": now,
            }
        except sqlite3.IntegrityError:
            return None  # Player already exists


def get_player(player_id: str, db_path: Path = None) -> Optional[dict]:
    """Get a player by ID."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM players WHERE player_id = ?",
            (player_id,)
        ).fetchone()

        if row is None:
            return None

        return {
            "player_id": row["player_id"],
            "player_type": row["player_type"],
            "user_id": row["user_id"],
            "model_version": row["model_version"],
            "simulations": row["simulations"],
            "display_name": row["display_name"],
            "elo_rating": row["elo_rating"],
            "elo_games_played": row["elo_games_played"],
            "created_at": row["created_at"],
        }


def get_or_create_ai_player(
    model_version: str,
    simulations: int,
    db_path: Path = None
) -> dict:
    """
    Get or create an AI player for a model+sims combo.

    Always returns a player dict.
    """
    player_id = generate_ai_player_id(model_version, simulations)
    player = get_player(player_id, db_path)

    if player is not None:
        return player

    # Create display name
    display_name = f"{model_version} ({simulations} sims)"

    player = create_player(
        player_id=player_id,
        player_type="ai",
        display_name=display_name,
        model_version=model_version,
        simulations=simulations,
        db_path=db_path,
    )

    # If creation failed (race condition), fetch again
    if player is None:
        player = get_player(player_id, db_path)

    return player


def get_or_create_human_player(
    user_id: str,
    display_name: str,
    db_path: Path = None
) -> dict:
    """
    Get or create a human player for a user.

    Always returns a player dict.
    """
    player_id = generate_human_player_id(user_id)
    player = get_player(player_id, db_path)

    if player is not None:
        return player

    player = create_player(
        player_id=player_id,
        player_type="human",
        display_name=display_name,
        user_id=user_id,
        db_path=db_path,
    )

    # If creation failed (race condition), fetch again
    if player is None:
        player = get_player(player_id, db_path)

    return player


def update_player_elo(
    player_id: str,
    new_rating: float,
    games_increment: int = 1,
    db_path: Path = None
) -> bool:
    """
    Update a player's ELO rating.

    Returns True if updated, False if player not found.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            UPDATE players
            SET elo_rating = ?,
                elo_games_played = elo_games_played + ?
            WHERE player_id = ?
        """, (new_rating, games_increment, player_id))
        conn.commit()
        return cursor.rowcount > 0


def update_game_result(
    game_id: str,
    winner: Optional[int],
    player1_player_id: Optional[str] = None,
    player2_player_id: Optional[str] = None,
    db_path: Path = None
) -> bool:
    """
    Update a game's result and player references.

    Args:
        game_id: The game ID
        winner: 0 for player1 win, 1 for player2 win, None for draw
        player1_player_id: Player ID for player 1
        player2_player_id: Player ID for player 2

    Returns True if updated.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    now = datetime.utcnow().isoformat() + 'Z'

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            UPDATE games
            SET winner = ?,
                player1_player_id = COALESCE(?, player1_player_id),
                player2_player_id = COALESCE(?, player2_player_id),
                updated_at = ?
            WHERE game_id = ?
        """, (winner, player1_player_id, player2_player_id, now, game_id))
        conn.commit()
        return cursor.rowcount > 0


def get_leaderboard(
    player_type: Optional[str] = None,
    limit: int = 100,
    min_games: int = 0,
    db_path: Path = None
) -> list[dict]:
    """
    Get leaderboard sorted by ELO rating.

    Args:
        player_type: Filter by 'human' or 'ai' (None = all)
        limit: Maximum number of players to return
        min_games: Minimum games played to be included

    Returns:
        List of player dicts sorted by ELO descending.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conditions = ["elo_games_played >= ?"]
    params = [min_games]

    if player_type:
        conditions.append("player_type = ?")
        params.append(player_type)

    where_clause = " AND ".join(conditions)

    with get_connection(db_path) as conn:
        rows = conn.execute(f"""
            SELECT * FROM players
            WHERE {where_clause}
            ORDER BY elo_rating DESC
            LIMIT ?
        """, params + [limit]).fetchall()

        return [{
            "player_id": row["player_id"],
            "player_type": row["player_type"],
            "user_id": row["user_id"],
            "model_version": row["model_version"],
            "simulations": row["simulations"],
            "display_name": row["display_name"],
            "elo_rating": row["elo_rating"],
            "elo_games_played": row["elo_games_played"],
            "created_at": row["created_at"],
        } for row in rows]


def get_player_recent_games(
    player_id: str,
    limit: int = 20,
    db_path: Path = None
) -> list[dict]:
    """
    Get a player's recent games.

    Returns list of game summaries.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT g.game_id, g.player1_player_id, g.player2_player_id, g.winner,
                   g.created_at, g.updated_at,
                   p1.display_name as player1_name, p1.elo_rating as player1_elo,
                   p2.display_name as player2_name, p2.elo_rating as player2_elo
            FROM games g
            LEFT JOIN players p1 ON g.player1_player_id = p1.player_id
            LEFT JOIN players p2 ON g.player2_player_id = p2.player_id
            WHERE g.player1_player_id = ? OR g.player2_player_id = ?
            ORDER BY g.updated_at DESC
            LIMIT ?
        """, (player_id, player_id, limit)).fetchall()

        return [{
            "game_id": row["game_id"],
            "player1_player_id": row["player1_player_id"],
            "player2_player_id": row["player2_player_id"],
            "player1_name": row["player1_name"],
            "player2_name": row["player2_name"],
            "player1_elo": row["player1_elo"],
            "player2_elo": row["player2_elo"],
            "winner": row["winner"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        } for row in rows]


def apply_elo_update(
    player1_id: str,
    player2_id: str,
    winner: Optional[int],
    db_path: Path = None
) -> tuple[float, float]:
    """
    Apply ELO rating updates after a game.

    Args:
        player1_id: Player 1's ID
        player2_id: Player 2's ID
        winner: 0 if player1 won, 1 if player2 won, None for draw

    Returns:
        Tuple of (player1_new_rating, player2_new_rating)
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Import here to avoid circular dependency
    from razzle.training.elo import expected_score, EloRating

    p1 = get_player(player1_id, db_path)
    p2 = get_player(player2_id, db_path)

    if not p1 or not p2:
        raise ValueError("Player not found")

    # Same player = no rating change
    if player1_id == player2_id:
        return p1["elo_rating"], p2["elo_rating"]

    r1 = EloRating(rating=p1["elo_rating"], games_played=p1["elo_games_played"])
    r2 = EloRating(rating=p2["elo_rating"], games_played=p2["elo_games_played"])

    # Determine scores
    if winner == 0:
        score1, score2 = 1.0, 0.0
    elif winner == 1:
        score1, score2 = 0.0, 1.0
    else:
        score1, score2 = 0.5, 0.5

    # Calculate expected scores
    exp1 = expected_score(r1.rating, r2.rating)
    exp2 = expected_score(r2.rating, r1.rating)

    # Update ratings
    new_r1 = r1.rating + r1.k_factor * (score1 - exp1)
    new_r2 = r2.rating + r2.k_factor * (score2 - exp2)

    # Save updates
    update_player_elo(player1_id, new_r1, games_increment=1, db_path=db_path)
    update_player_elo(player2_id, new_r2, games_increment=1, db_path=db_path)

    return new_r1, new_r2
