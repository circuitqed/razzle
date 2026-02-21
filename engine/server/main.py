"""
FastAPI server for Razzle Dazzle game engine.

Provides REST and WebSocket APIs for game management and AI play.
"""

from __future__ import annotations
import asyncio
import gzip
import logging
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import jwt
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Response, Depends, Cookie, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import shutil

from razzle.core.state import GameState
from razzle.core.moves import (
    get_legal_moves, move_to_algebraic, algebraic_to_move,
    decode_move, encode_move
)
from razzle.core.bitboard import algebraic_to_sq, sq_to_algebraic
from razzle.ai.mcts import MCTS, MCTSConfig, play_move_timed
from razzle.ai.evaluator import BatchedEvaluator, DummyEvaluator
from razzle.ai.network import RazzleNet
from razzle.ai.time_manager import create_time_manager, TimeManager
import random as py_random

from . import persistence


# --- Bot Types ---
# Available AI bot types:
# - "neural": Uses trained neural network for evaluation (strongest if trained well)
# - "mcts": Pure MCTS with uniform priors over legal moves (good baseline)
# - "random": Picks random legal moves (weakest, useful for testing)
BOT_TYPES = ["neural", "mcts", "random"]
DEFAULT_BOT_TYPE = "neural"  # Use trained neural network model


# --- Pydantic Models ---

class CreateGameRequest(BaseModel):
    player1_type: str = "human"
    player2_type: str = "ai"
    ai_simulations: int = 1024
    bot_type: str = Field(default=DEFAULT_BOT_TYPE, description="AI bot type: 'neural', 'mcts', or 'random'")
    time_control: Optional[float] = Field(default=None, description="Total time per player in seconds. If set, enables dynamic time management.")


class CreateGameResponse(BaseModel):
    game_id: str


class BoardState(BaseModel):
    """Board state with bitboards as strings to preserve precision in JavaScript.

    JavaScript's Number type loses precision for integers > 2^53.
    Board uses 56 squares (8x7), so bitboards can exceed this limit.
    """
    p1_pieces: str  # Bitboard as string
    p1_ball: str
    p2_pieces: str
    p2_ball: str


class GameStateResponse(BaseModel):
    game_id: str
    board: BoardState
    current_player: int
    legal_moves: list[int]
    status: str
    winner: Optional[int]
    ply: int
    touched_mask: str  # Bitboard of ineligible receivers (string for JS precision)
    has_passed: bool  # Whether a pass has been made this turn
    last_knight_dst: int = -1  # Destination of opponent's last knight move (-1 if none)
    time_control: Optional[float] = None  # Total time per player in seconds
    time_remaining: Optional[list[float]] = None  # Remaining time [p1, p2] if time controlled


class MakeMoveRequest(BaseModel):
    move: int


class AIMoveRequest(BaseModel):
    simulations: int = 800
    temperature: float = 0.0
    bot_type: Optional[str] = Field(default=None, description="AI bot type: 'neural', 'mcts', or 'random'. If None, uses game's default.")
    model: Optional[str] = Field(default=None, description="Model to use: 'random_weights' or path to model file. If None, uses latest model.")
    use_time_control: bool = Field(default=True, description="If game has time control, use dynamic time management for this move.")


class TopMove(BaseModel):
    move: int
    algebraic: str
    visits: int
    value: float


class AIMoveResponse(BaseModel):
    move: int
    algebraic: str
    policy: list[float]
    value: float
    visits: int
    time_ms: int
    top_moves: list[TopMove]
    game_state: GameStateResponse
    difficulty: Optional[float] = Field(default=None, description="Position difficulty [0-1] if using time control")
    remaining_time: Optional[float] = Field(default=None, description="Remaining time in seconds if using time control")


class LegalMove(BaseModel):
    move: int
    algebraic: str
    type: str


class LegalMovesResponse(BaseModel):
    moves: list[LegalMove]


class MoveConversionResponse(BaseModel):
    encoded: Optional[int] = None
    algebraic: Optional[str] = None
    src: int
    dst: int


class HealthResponse(BaseModel):
    status: str
    version: str
    model: Optional[str] = None


# --- Auth Models ---

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, pattern=r'^[a-zA-Z0-9_]+$')
    password: str = Field(..., min_length=6, max_length=128)
    display_name: Optional[str] = Field(None, max_length=64)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    display_name: Optional[str]
    created_at: str
    last_login_at: Optional[str] = None


class AuthResponse(BaseModel):
    user: UserResponse
    message: str


# --- JWT Configuration ---

JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24
AUTH_COOKIE_NAME = "razzle_auth"
ANON_COOKIE_NAME = "razzle_anon"


class TrainingIterationData(BaseModel):
    iteration: int
    timestamp: str
    num_games: int
    p1_wins: int
    p2_wins: int
    draws: int
    avg_game_length: float
    min_game_length: int = 0
    max_game_length: int = 0
    std_game_length: float = 0.0
    training_examples: int
    selfplay_time_sec: float = 0.0
    training_time_sec: float = 0.0
    total_time_sec: float = 0.0
    final_loss: float = 0.0
    final_policy_loss: float = 0.0
    final_value_loss: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    cpu_percent: float = 0.0
    device: str = "cpu"
    win_rate_vs_random: Optional[float] = None
    elo_rating: Optional[float] = None


class TrainingStatusResponse(BaseModel):
    status: str  # "no_training", "training", "completed"
    run_id: Optional[str] = None
    start_time: Optional[str] = None
    total_games: int = 0
    total_examples: int = 0
    total_time_sec: float = 0.0
    iterations: list[TrainingIterationData] = []
    config: dict = {}


# --- Training API Models ---

class SubmitGameRequest(BaseModel):
    """Request to submit a training game from a worker."""
    worker_id: str
    moves: list[int]
    result: float  # 1.0 (P0 wins), -1.0 (P1 wins), 0.0 (draw)
    visit_counts: list[dict[str, int]]  # Sparse MCTS visit counts per position
    model_version: Optional[str] = None


class SubmitGameResponse(BaseModel):
    """Response after submitting a training game."""
    id: int
    status: str = "accepted"


class TrainingGameData(BaseModel):
    """A training game returned to the trainer."""
    id: int
    worker_id: str
    moves: list[int]
    result: float
    visit_counts: list[dict[str, int]]
    model_version: Optional[str]
    created_at: str


class FetchGamesResponse(BaseModel):
    """Response with pending training games."""
    games: list[TrainingGameData]
    count: int
    total_pending: int


class ModelInfo(BaseModel):
    """Information about a training model."""
    version: str
    iteration: int
    games_trained_on: Optional[int] = None
    final_loss: Optional[float] = None
    final_policy_loss: Optional[float] = None
    final_value_loss: Optional[float] = None
    download_url: str
    created_at: str


class LatestModelResponse(BaseModel):
    """Response with latest model info, or null if none exists."""
    model: Optional[ModelInfo] = None


class UploadModelResponse(BaseModel):
    """Response after uploading a model."""
    version: str
    status: str = "uploaded"


# --- Trainer State Models ---

class TrainerStateInfo(BaseModel):
    """Metadata about a trainer state file."""
    key: str
    iteration: int
    total_games_trained: int
    file_size: int
    download_url: str
    created_at: str


class TrainerStateResponse(BaseModel):
    """Response with trainer state info, or null if none exists."""
    state: Optional[TrainerStateInfo] = None


class UploadTrainerStateResponse(BaseModel):
    """Response after uploading trainer state."""
    key: str
    status: str = "uploaded"


class WorkerStats(BaseModel):
    """Statistics for a single worker."""
    games: int
    last_seen: Optional[str]


class TrainingDashboardResponse(BaseModel):
    """Enhanced training status for dashboard."""
    status: str
    games_pending: int
    games_total: int
    latest_model: Optional[ModelInfo] = None
    workers: dict[str, WorkerStats]
    models: list[ModelInfo]


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    data: Optional[dict] = None


class LogRequest(BaseModel):
    entries: list[LogEntry]
    session_id: Optional[str] = None


# --- Training Metrics Models ---

class TrainingMetricsData(BaseModel):
    """Training metrics for a single iteration."""
    id: Optional[int] = None
    iteration: int
    timestamp: Optional[str] = None
    # Policy metrics
    policy_top1_accuracy: Optional[float] = None
    policy_top3_accuracy: Optional[float] = None
    policy_entropy: Optional[float] = None
    policy_legal_mass: Optional[float] = None
    policy_ebf: Optional[float] = None
    policy_confidence: Optional[float] = None
    # Value metrics (validation - computed before training on unseen games)
    value_mean: Optional[float] = None
    value_std: Optional[float] = None
    value_extremity: Optional[float] = None
    value_mse: Optional[float] = None  # True validation MSE
    value_sign_accuracy: Optional[float] = None  # Sign match rate
    value_calibration_error: Optional[float] = None
    # Pass metrics
    pass_decision_rate: Optional[float] = None
    # Loss metrics
    loss_total: Optional[float] = None
    loss_policy: Optional[float] = None
    loss_value: Optional[float] = None
    loss_difficulty: Optional[float] = None
    loss_illegal_penalty: Optional[float] = None
    # Game stats
    num_games: Optional[int] = None
    num_examples: Optional[int] = None
    avg_game_length: Optional[float] = None
    # Meta
    learning_rate: Optional[float] = None
    model_version: Optional[str] = None
    train_time_sec: Optional[float] = None


class SubmitMetricsRequest(BaseModel):
    """Request to submit training metrics."""
    iteration: int
    metrics: dict


class SubmitMetricsResponse(BaseModel):
    """Response after submitting metrics."""
    id: int
    status: str = "accepted"


class TrainingMetricsHistoryResponse(BaseModel):
    """Response with training metrics history."""
    metrics: list[TrainingMetricsData]
    total: int
    limit: int
    offset: int


class LatestMetricsResponse(BaseModel):
    """Response with latest training metrics."""
    metrics: Optional[TrainingMetricsData] = None


# --- Game Storage ---

class Game:
    """Represents an active game session."""

    def __init__(
        self,
        game_id: str,
        player1_type: str = "human",
        player2_type: str = "ai",
        ai_simulations: int = 1024,
        bot_type: str = DEFAULT_BOT_TYPE,
        time_control: Optional[float] = None,
        online_status: str = "local",
        join_code: Optional[str] = None
    ):
        self.game_id = game_id
        self.state = GameState.new_game()
        self.player_types = [player1_type, player2_type]
        self.ai_simulations = ai_simulations
        self.bot_type = bot_type
        self.time_control = time_control
        self.websockets: list[WebSocket] = []

        # Player IDs for ELO tracking (set when game starts)
        self.player_ids: list[Optional[str]] = [None, None]
        self.elo_updated: bool = False  # Track if ELO has been updated for this game

        # Resign tracking
        self.resigned_by: Optional[int] = None  # Player who resigned (0 or 1), None if no resignation

        # Online multiplayer state
        self.online_status = online_status  # 'local', 'waiting', 'playing', 'finished', 'abandoned'
        self.join_code = join_code
        self.player_user_ids: list[Optional[str]] = [None, None]  # User IDs for players
        self.player_websockets: dict[str, WebSocket] = {}  # user_id -> WebSocket
        self.player_connected: list[bool] = [False, False]  # Connection status per player
        self.disconnect_timers: dict[str, asyncio.Task] = {}  # user_id -> disconnect grace timer

        # Create time managers if time control is enabled
        self.time_managers: list[Optional[TimeManager]] = [None, None]
        if time_control is not None:
            for i in range(2):
                self.time_managers[i] = create_time_manager(
                    total_time=time_control,
                    min_sims=100,
                    max_sims=ai_simulations,
                    default_sims=ai_simulations
                )

    def is_online_game(self) -> bool:
        """Check if this is an online multiplayer game."""
        return self.online_status != "local"

    def get_player_color(self, user_id: str) -> Optional[int]:
        """Get the player color (0 or 1) for a user ID, or None if not a player."""
        if user_id == self.player_user_ids[0]:
            return 0
        elif user_id == self.player_user_ids[1]:
            return 1
        return None

    def can_user_move(self, user_id: str) -> bool:
        """Check if the user is allowed to make a move."""
        if not self.is_online_game():
            return True  # Local games allow any client to move
        player_color = self.get_player_color(user_id)
        if player_color is None:
            return False  # User is not a player
        return player_color == self.state.current_player

    def is_game_over(self) -> bool:
        """Check if the game is over (terminal state or resignation)."""
        return self.state.is_terminal() or self.resigned_by is not None

    def get_winner(self) -> Optional[int]:
        """Get the winner, accounting for resignation."""
        if self.resigned_by is not None:
            return 1 - self.resigned_by  # Opponent wins
        return self.state.get_winner()

    def to_response(self) -> GameStateResponse:
        """Convert to API response."""
        if self.is_game_over():
            status = "finished"
        else:
            status = "playing"

        # Get time remaining if time controlled
        time_remaining = None
        if self.time_control is not None:
            time_remaining = [
                self.time_managers[0].remaining_time if self.time_managers[0] else 0.0,
                self.time_managers[1].remaining_time if self.time_managers[1] else 0.0,
            ]

        return GameStateResponse(
            game_id=self.game_id,
            board=BoardState(
                p1_pieces=str(self.state.pieces[0]),
                p1_ball=str(self.state.balls[0]),
                p2_pieces=str(self.state.pieces[1]),
                p2_ball=str(self.state.balls[1])
            ),
            current_player=self.state.current_player,
            legal_moves=get_legal_moves(self.state),
            status=status,
            winner=self.get_winner(),
            ply=self.state.ply,
            touched_mask=str(self.state.touched_mask),
            has_passed=self.state.has_passed,
            last_knight_dst=self.state.last_knight_dst,
            time_control=self.time_control,
            time_remaining=time_remaining
        )

    def check_and_update_elo(self) -> Optional[tuple[float, float]]:
        """
        Check if game is over and update ELO ratings.

        Returns tuple of (p1_new_rating, p2_new_rating) if updated, None otherwise.
        """
        if self.elo_updated:
            return None

        if not self.is_game_over():
            return None

        # Need both player IDs to update ELO
        if not self.player_ids[0] or not self.player_ids[1]:
            return None

        winner = self.get_winner()

        try:
            new_r1, new_r2 = persistence.apply_elo_update(
                self.player_ids[0],
                self.player_ids[1],
                winner
            )
            self.elo_updated = True

            # Update game record with winner
            persistence.update_game_result(
                self.game_id,
                winner=winner,
                player1_player_id=self.player_ids[0],
                player2_player_id=self.player_ids[1]
            )

            logging.info(f"Game {self.game_id} ended. Winner: {winner}. "
                        f"ELO: {self.player_ids[0]}={new_r1:.0f}, {self.player_ids[1]}={new_r2:.0f}")
            return (new_r1, new_r2)
        except Exception as e:
            logging.error(f"Failed to update ELO for game {self.game_id}: {e}")
            return None


def setup_game_players(
    game: Game,
    user: Optional[dict],
    model_version: Optional[str]
) -> None:
    """
    Set up player IDs for a game based on player types.

    Args:
        game: The game to set up
        user: Current authenticated user (for human player 1)
        model_version: AI model version being used
    """
    # Player 1 (usually human)
    if game.player_types[0] == "human":
        if user:
            player1 = persistence.get_or_create_human_player(
                user["user_id"],
                user.get("display_name") or user.get("username")
            )
            game.player_ids[0] = player1["player_id"]
        # Else: anonymous human, no ELO tracking
    elif game.player_types[0] == "ai":
        if model_version:
            player1 = persistence.get_or_create_ai_player(model_version, game.ai_simulations)
            game.player_ids[0] = player1["player_id"]

    # Player 2 (usually AI)
    if game.player_types[1] == "ai":
        if model_version:
            player2 = persistence.get_or_create_ai_player(model_version, game.ai_simulations)
            game.player_ids[1] = player2["player_id"]
    elif game.player_types[1] == "human":
        # Human vs human - would need second user to join
        pass


# Global game storage (in production, use Redis or database)
games: dict[str, Game] = {}

# Global AI evaluator (lazy loaded)
evaluator: Optional[BatchedEvaluator] = None
_model_path_used: Optional[str] = None  # Track which model is loaded
_model_mtime: float = 0  # Track model file modification time

# Directories to search for models (in priority order)
MODEL_SEARCH_DIRS = [
    Path("models"),  # Docker volume mount (/app/models)
    Path("/app/models"),  # Docker container path (absolute)
    Path("output/models"),  # New training pipeline models
    Path("/app/output/models"),  # Docker container path
    Path("/home/projects/razzle/engine/output/new_rules_500"),
    Path("output/new_rules_500"),
    Path("/home/projects/razzle/engine/output/output"),
    Path("output/output"),
    Path("/home/projects/razzle/engine/output"),
    Path("output"),
]


def find_latest_model() -> Optional[tuple[Path, float]]:
    """Find the latest model file by modification time.

    Returns tuple of (path, mtime) or None if no model found.
    """
    latest_model = None
    latest_mtime = 0

    for search_dir in MODEL_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        # Look for model files (iter_*.pt, model_iter_*.pt, or trained_model.pt)
        for pattern in ["iter_*.pt", "model_iter_*.pt", "trained_model.pt"]:
            for model_path in search_dir.glob(pattern):
                try:
                    mtime = model_path.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_model = model_path
                except OSError:
                    continue

    if latest_model:
        return (latest_model, latest_mtime)
    return None


def load_model(model_path: Path) -> Optional[RazzleNet]:
    """Load a model from the given path."""
    try:
        net = RazzleNet.load(model_path, device='cpu')
        logging.info(f"Loaded trained model from {model_path}")
        return net
    except Exception as e:
        logging.warning(f"Failed to load model from {model_path}: {e}")
        return None


def get_evaluator(check_for_updates: bool = False) -> BatchedEvaluator:
    """Get or create the AI evaluator.

    Args:
        check_for_updates: If True, check if a newer model is available and reload if so.
    """
    global evaluator, _model_path_used, _model_mtime

    # Check for newer model if requested
    if check_for_updates and evaluator is not None:
        latest = find_latest_model()
        if latest:
            latest_path, latest_mtime = latest
            if latest_mtime > _model_mtime:
                logging.info(f"Newer model found: {latest_path} (mtime: {latest_mtime} > {_model_mtime})")
                net = load_model(latest_path)
                if net:
                    _model_path_used = str(latest_path)
                    _model_mtime = latest_mtime
                    try:
                        evaluator = BatchedEvaluator(net, device='cpu')
                        logging.info(f"Reloaded evaluator with new model: {latest_path}")
                    except Exception as e:
                        logging.error(f"Failed to create evaluator: {e}")

    # Initial load
    if evaluator is None:
        latest = find_latest_model()
        net = None

        if latest:
            latest_path, latest_mtime = latest
            net = load_model(latest_path)
            if net:
                _model_path_used = str(latest_path)
                _model_mtime = latest_mtime

        if net is None:
            logging.info("No trained model found, using random weights")
            _model_path_used = "random_weights"
            _model_mtime = 0
            net = RazzleNet()

        try:
            evaluator = BatchedEvaluator(net, device='cpu')
        except Exception:
            evaluator = DummyEvaluator()

    return evaluator


def get_model_info() -> str:
    """Get info about which model is loaded."""
    global _model_path_used
    if _model_path_used is None:
        return "not_loaded"
    return _model_path_used


# Cache for per-model evaluators
_model_evaluators: dict[str, BatchedEvaluator] = {}


def get_evaluator_for_model(model_path: Optional[str]) -> BatchedEvaluator:
    """Get an evaluator for a specific model.

    Args:
        model_path: Path to model file, 'random_weights', or None for default (latest).

    Returns:
        BatchedEvaluator for the specified model.
    """
    global _model_evaluators

    # None means use the default (latest) model
    if model_path is None:
        return get_evaluator()

    # Check cache
    if model_path in _model_evaluators:
        return _model_evaluators[model_path]

    # Load the specified model
    if model_path == "random_weights":
        logging.info("Using random weights model")
        net = RazzleNet()
    else:
        path = Path(model_path)
        if not path.exists():
            logging.warning(f"Model not found: {model_path}, using random weights")
            net = RazzleNet()
        else:
            net = load_model(path)
            if net is None:
                logging.warning(f"Failed to load model: {model_path}, using random weights")
                net = RazzleNet()

    try:
        ev = BatchedEvaluator(net, device='cpu')
        _model_evaluators[model_path] = ev
        return ev
    except Exception as e:
        logging.error(f"Failed to create evaluator for {model_path}: {e}")
        return DummyEvaluator()


# --- App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan handler."""
    # Startup: initialize database and load existing games
    persistence.init_db()
    persistence.cleanup_old_games(max_age_days=7)  # Clean up stale games

    # Load persisted games into memory
    for game_data in persistence.load_all_games():
        game = Game(
            game_id=game_data["game_id"],
            player1_type=game_data["player1_type"],
            player2_type=game_data["player2_type"],
            ai_simulations=game_data["ai_simulations"],
            bot_type=game_data.get("bot_type", DEFAULT_BOT_TYPE)  # Default for old games
        )
        game.state = game_data["state"]
        games[game.game_id] = game
        logging.info(f"Loaded game {game.game_id} from database")

    logging.info(f"Loaded {len(games)} games from database")

    # Pre-load evaluator
    get_evaluator()

    yield

    # Shutdown: nothing to do (games are persisted on each change)
    games.clear()


app = FastAPI(
    title="Razzle Dazzle Engine",
    description="Game engine API for Razzle Dazzle board game",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth Utilities ---

def create_jwt_token(user_id: str) -> str:
    """Create a JWT token for a user."""
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Optional[str]:
    """Decode a JWT token and return the user_id, or None if invalid."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(
    request: Request,
    auth_cookie: Optional[str] = Cookie(None, alias=AUTH_COOKIE_NAME)
) -> Optional[dict]:
    """Get the current user from the auth cookie, or None if not authenticated."""
    token = auth_cookie
    if not token:
        return None

    user_id = decode_jwt_token(token)
    if not user_id:
        return None

    user = persistence.get_user_by_id(user_id)
    return user


async def require_auth(
    request: Request,
    auth_cookie: Optional[str] = Cookie(None, alias=AUTH_COOKIE_NAME)
) -> dict:
    """Require authentication, raise 401 if not authenticated."""
    user = await get_current_user(request, auth_cookie)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def get_user_or_anon(
    request: Request,
    response: Response,
    auth_cookie: Optional[str] = Cookie(None, alias=AUTH_COOKIE_NAME),
    anon_cookie: Optional[str] = Cookie(None, alias=ANON_COOKIE_NAME),
) -> dict:
    """Get authenticated user, or create/retrieve anonymous session."""
    # Try real auth first
    user = await get_current_user(request, auth_cookie)
    if user:
        return user

    # Fall back to anonymous session
    anon_id = anon_cookie
    if not anon_id:
        anon_id = secrets.token_hex(16)
        response.set_cookie(
            ANON_COOKIE_NAME, anon_id,
            max_age=86400 * 30,  # 30 days
            httponly=True,
            samesite="lax",
        )

    return {
        "user_id": f"anon_{anon_id}",
        "username": "anonymous",
        "display_name": "Anonymous",
        "is_anonymous": True,
    }


# --- Auth Endpoints ---

@app.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest, response: Response):
    """Register a new user account."""
    user = persistence.create_user(
        username=request.username,
        password=request.password,
        display_name=request.display_name,
    )

    if not user:
        raise HTTPException(status_code=409, detail="Username already exists")

    # Create token and set cookie
    token = create_jwt_token(user["user_id"])
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=JWT_EXPIRY_HOURS * 3600,
    )

    return AuthResponse(
        user=UserResponse(**user),
        message="Account created successfully"
    )


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, response: Response):
    """Login with username and password."""
    user = persistence.authenticate_user(
        username=request.username,
        password=request.password,
    )

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create token and set cookie
    token = create_jwt_token(user["user_id"])
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=JWT_EXPIRY_HOURS * 3600,
    )

    return AuthResponse(
        user=UserResponse(**user),
        message="Login successful"
    )


@app.post("/auth/logout")
async def logout(response: Response):
    """Logout by clearing the auth cookie."""
    response.delete_cookie(key=AUTH_COOKIE_NAME)
    return {"message": "Logged out successfully"}


@app.get("/auth/me", response_model=UserResponse)
async def get_me(user: dict = Depends(require_auth)):
    """Get the current authenticated user."""
    return UserResponse(**user)


# --- REST Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0", model=get_model_info())


class AvailableModelInfo(BaseModel):
    """Information about an available model file."""
    name: str  # Display name (filename)
    path: str  # Full path for selection
    mtime: float  # Modification time for sorting
    has_onnx: bool = False  # Whether an ONNX export exists alongside the .pt file


class ModelsResponse(BaseModel):
    """Response for listing available models."""
    models: list[AvailableModelInfo]
    current: str  # Currently loaded model


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all available models."""
    models = []
    seen_paths = set()

    for search_dir in MODEL_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for pattern in ["iter_*.pt", "model_iter_*.pt", "trained_model.pt"]:
            for model_path in search_dir.glob(pattern):
                try:
                    path_str = str(model_path.resolve())
                    if path_str in seen_paths:
                        continue
                    seen_paths.add(path_str)
                    mtime = model_path.stat().st_mtime
                    onnx_path = model_path.with_suffix(".onnx")
                    models.append(AvailableModelInfo(
                        name=model_path.name,
                        path=path_str,
                        mtime=mtime,
                        has_onnx=onnx_path.exists(),
                    ))
                except OSError:
                    continue

    # Sort by modification time (newest first)
    models.sort(key=lambda m: m.mtime, reverse=True)

    # Add "random_weights" as a special option at the beginning
    models.insert(0, AvailableModelInfo(name="random_weights", path="random_weights", mtime=0))

    return ModelsResponse(models=models, current=get_model_info())


# --- On-demand ONNX export ---
_onnx_export_locks: dict[str, threading.Lock] = {}
_onnx_export_locks_lock = threading.Lock()
_onnx_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="onnx-export")


def _get_export_lock(pt_path: str) -> threading.Lock:
    """Get or create a per-model lock for ONNX export."""
    with _onnx_export_locks_lock:
        if pt_path not in _onnx_export_locks:
            _onnx_export_locks[pt_path] = threading.Lock()
        return _onnx_export_locks[pt_path]


def _export_onnx_sync(pt_path: Path, onnx_path: Path) -> None:
    """Export a .pt model to .onnx (runs in thread pool). Uses legacy TorchScript exporter."""
    import torch
    from razzle.core.bitboard import ROWS, COLS

    lock = _get_export_lock(str(pt_path))
    with lock:
        # Double-check after acquiring lock
        if onnx_path.exists():
            return

        logging.info(f"Exporting ONNX: {pt_path} -> {onnx_path}")
        net = RazzleNet.load(pt_path, device="cpu")
        net.eval()
        dummy = torch.randn(1, 7, ROWS, COLS)
        torch.onnx.export(
            net,
            dummy,
            str(onnx_path),
            opset_version=17,
            input_names=["board_input"],
            output_names=["policy", "value", "difficulty"],
            dynamic_axes={
                "board_input": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
                "difficulty": {0: "batch"},
            },
            dynamo=False,  # Use legacy TorchScript exporter (no onnxscript dep)
        )
        size_kb = onnx_path.stat().st_size / 1024
        logging.info(f"ONNX export complete: {onnx_path} ({size_kb:.1f} KB)")


def _find_pt_model(model_name: str) -> Optional[Path]:
    """Find a .pt model file by name across MODEL_SEARCH_DIRS."""
    if not model_name.endswith(".pt"):
        return None
    if "/" in model_name or "\\" in model_name:
        return None
    for search_dir in MODEL_SEARCH_DIRS:
        candidate = search_dir / model_name
        if candidate.exists():
            return candidate
    return None


class OnnxModelInfo(BaseModel):
    """Info about available ONNX model."""
    version: str
    url: str
    size_bytes: int


@app.get("/models/onnx/latest")
async def get_latest_onnx_model():
    """Get info about the latest ONNX model available for browser inference.

    If no ONNX file exists, auto-exports from the latest .pt model.
    """
    # Look for existing ONNX files in model directories
    onnx_files = []
    search_dirs = list(MODEL_SEARCH_DIRS) + [Path("models"), Path(".")]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for onnx_path in search_dir.glob("*.onnx"):
            try:
                onnx_files.append(onnx_path)
            except OSError:
                continue

    if not onnx_files:
        # No ONNX files exist — try to auto-export from the latest .pt
        latest = find_latest_model()
        if not latest:
            raise HTTPException(status_code=404, detail="No model available")
        pt_path, _ = latest
        onnx_path = pt_path.with_suffix(".onnx")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_onnx_thread_pool, _export_onnx_sync, pt_path, onnx_path)
        return OnnxModelInfo(
            version=onnx_path.stem,
            url=f"/models/onnx/{onnx_path.name}",
            size_bytes=onnx_path.stat().st_size,
        )

    # Use most recent
    onnx_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best = onnx_files[0]

    return OnnxModelInfo(
        version=best.stem,
        url=f"/models/onnx/{best.name}",
        size_bytes=best.stat().st_size,
    )


@app.get("/models/onnx/by-name/{model_name}")
async def get_onnx_model_by_name(model_name: str):
    """Get or export an ONNX model for a specific .pt file.

    If the ONNX file doesn't exist alongside the .pt, it will be exported on demand.
    Export takes ~2-3 seconds and is protected by per-model locking.
    """
    pt_path = _find_pt_model(model_name)
    if pt_path is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    onnx_path = pt_path.with_suffix(".onnx")

    if not onnx_path.exists():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_onnx_thread_pool, _export_onnx_sync, pt_path, onnx_path)

    return OnnxModelInfo(
        version=onnx_path.stem,
        url=f"/models/onnx/{onnx_path.name}",
        size_bytes=onnx_path.stat().st_size,
    )


@app.get("/models/onnx/{filename}")
async def download_onnx_model(filename: str):
    """Download an ONNX model file."""
    # Security: only allow .onnx files, no path traversal
    if not filename.endswith(".onnx") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    search_dirs = list(MODEL_SEARCH_DIRS) + [Path("models"), Path(".")]
    for search_dir in search_dirs:
        candidate = search_dir / filename
        if candidate.exists():
            return FileResponse(
                path=candidate,
                filename=filename,
                media_type="application/octet-stream",
                headers={"Cache-Control": "public, max-age=86400"},
            )

    raise HTTPException(status_code=404, detail="Model file not found")


# Training output directories to check
TRAINING_OUTPUT_DIRS = [
    Path("output/new_rules_500"),  # Latest parallel training with no-draw rules
    Path("/home/projects/razzle/engine/output/new_rules_500"),
    Path("output"),
    Path("output/output"),  # Cloud training extracts here
    Path("output/cloud_run"),
    Path("output_shaped"),
    Path("/home/projects/razzle/engine/output"),
    Path("/home/projects/razzle/engine/output/output"),
    Path("/home/projects/razzle/engine/output/cloud_run"),
]

# Plot directories
PLOTS_DIRS = [
    Path("output/plots"),
    Path("/home/projects/razzle/engine/output/plots"),
]


@app.get("/training/plots/{filename}")
async def get_training_plot(filename: str):
    """Serve training plot images."""
    # Only allow specific filenames for security
    allowed_files = [
        "training_summary.png",
        "loss_curves.png",
        "win_rates.png",
        "game_lengths.png",
        "training_metrics.json",
    ]
    if filename not in allowed_files:
        raise HTTPException(status_code=404, detail="Plot not found")

    for plots_dir in PLOTS_DIRS:
        file_path = plots_dir / filename
        if file_path.exists():
            media_type = "application/json" if filename.endswith(".json") else "image/png"
            return FileResponse(file_path, media_type=media_type)

    raise HTTPException(status_code=404, detail="Plot not found")


@app.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get current training status and metrics for live dashboard."""
    import json

    # Find the training log
    log_path = None
    for output_dir in TRAINING_OUTPUT_DIRS:
        candidate = output_dir / "training_log.json"
        if candidate.exists():
            log_path = candidate
            break

    if log_path is None:
        return TrainingStatusResponse(status="no_training")

    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
    except (json.JSONDecodeError, IOError):
        return TrainingStatusResponse(status="no_training")

    # Convert iterations to response format
    iterations = []
    for it in log.get('iterations', []):
        iterations.append(TrainingIterationData(
            iteration=it['iteration'],
            timestamp=it.get('timestamp', ''),
            num_games=it['num_games'],
            p1_wins=it['p1_wins'],
            p2_wins=it['p2_wins'],
            draws=it['draws'],
            avg_game_length=it['avg_game_length'],
            min_game_length=it.get('min_game_length', 0),
            max_game_length=it.get('max_game_length', 0),
            std_game_length=it.get('std_game_length', 0.0),
            training_examples=it['training_examples'],
            selfplay_time_sec=it.get('selfplay_time_sec', 0.0),
            training_time_sec=it.get('training_time_sec', 0.0),
            total_time_sec=it.get('total_time_sec', 0.0),
            final_loss=it.get('final_loss', 0.0),
            final_policy_loss=it.get('final_policy_loss', 0.0),
            final_value_loss=it.get('final_value_loss', 0.0),
            gpu_memory_used_mb=it.get('gpu_memory_used_mb', 0.0),
            gpu_memory_total_mb=it.get('gpu_memory_total_mb', 0.0),
            gpu_utilization_pct=it.get('gpu_utilization_pct', 0.0),
            cpu_percent=it.get('cpu_percent', 0.0),
            device=it.get('device', 'cpu'),
            win_rate_vs_random=it.get('win_rate_vs_random'),
            elo_rating=it.get('elo_rating')
        ))

    return TrainingStatusResponse(
        status="training" if iterations else "no_training",
        run_id=log.get('run_id'),
        start_time=log.get('start_time'),
        total_games=log.get('total_games', 0),
        total_examples=log.get('total_examples', 0),
        total_time_sec=log.get('total_time_sec', 0.0),
        iterations=iterations,
        config=log.get('config', {})
    )


# --- Training API Endpoints ---

# Models directory for storing uploaded models
TRAINING_MODELS_DIR = Path(os.environ.get("RAZZLE_MODELS_DIR", "output/models"))
TRAINING_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Trainer state directory for storing optimizer state and replay buffers
TRAINING_STATE_DIR = Path(os.environ.get("RAZZLE_STATE_DIR", "output/trainer_state"))
TRAINING_STATE_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/training/games", response_model=SubmitGameResponse)
async def submit_training_game(request: SubmitGameRequest):
    """Submit a completed training game from a self-play worker."""
    # Convert visit_counts keys from string to int (JSON doesn't support int keys)
    visit_counts = [
        {int(k): v for k, v in vc.items()}
        for vc in request.visit_counts
    ]

    game_id = persistence.save_training_game(
        worker_id=request.worker_id,
        moves=request.moves,
        result=request.result,
        visit_counts=visit_counts,
        model_version=request.model_version,
    )

    return SubmitGameResponse(id=game_id, status="accepted")


@app.get("/training/games", response_model=FetchGamesResponse)
async def fetch_training_games(
    status: str = "pending",
    limit: int = 100,
    mark_used: bool = True
):
    """
    Fetch training games for the trainer.

    Args:
        status: Filter by status (default: "pending")
        limit: Maximum number of games to return
        mark_used: If true, atomically mark returned games as "used"
    """
    if status != "pending":
        raise HTTPException(status_code=400, detail="Only status='pending' is supported")

    games_data, total_pending = persistence.get_pending_training_games(
        limit=limit,
        mark_used=mark_used,
    )

    games = [
        TrainingGameData(
            id=g["id"],
            worker_id=g["worker_id"],
            moves=g["moves"],
            result=g["result"],
            visit_counts=[{str(k): v for k, v in vc.items()} for vc in g["visit_counts"]],
            model_version=g["model_version"],
            created_at=g["created_at"],
        )
        for g in games_data
    ]

    return FetchGamesResponse(
        games=games,
        count=len(games),
        total_pending=total_pending - len(games) if mark_used else total_pending,
    )


@app.get("/training/games/all")
async def get_all_training_games(
    limit: int = 500,
    offset: int = 0,
):
    """
    Fetch all training games for analysis (both pending and used).
    Does not mark games as used.
    """
    games, total = persistence.get_all_training_games(limit=limit, offset=offset)
    return {
        "games": games,
        "count": len(games),
        "total": total,
        "offset": offset,
    }


@app.get("/training/models/latest", response_model=LatestModelResponse)
async def get_latest_model():
    """Get information about the latest training model."""
    model = persistence.get_latest_training_model()

    if model is None:
        return LatestModelResponse(model=None)

    return LatestModelResponse(
        model=ModelInfo(
            version=model["version"],
            iteration=model["iteration"],
            games_trained_on=model["games_trained_on"],
            final_loss=model["final_loss"],
            final_policy_loss=model["final_policy_loss"],
            final_value_loss=model["final_value_loss"],
            download_url=f"/training/models/{model['version']}/download",
            created_at=model["created_at"],
        )
    )


class ModelsListResponse(BaseModel):
    """Response containing list of available models."""
    models: list[ModelInfo]
    total: int


@app.get("/training/models", response_model=ModelsListResponse)
async def list_training_models_endpoint(limit: int = 100):
    """List all available training models."""
    models = persistence.list_training_models(limit=limit)
    model_infos = [
        ModelInfo(
            version=m["version"],
            iteration=m["iteration"],
            games_trained_on=m["games_trained_on"],
            final_loss=m["final_loss"],
            final_policy_loss=m["final_policy_loss"],
            final_value_loss=m["final_value_loss"],
            download_url=f"/training/models/{m['version']}/download",
            created_at=m["created_at"],
        )
        for m in models
    ]
    return ModelsListResponse(models=model_infos, total=len(model_infos))


@app.post("/training/models", response_model=UploadModelResponse)
async def upload_training_model(
    version: str = Form(...),
    iteration: int = Form(...),
    games_trained_on: Optional[int] = Form(None),
    final_loss: Optional[float] = Form(None),
    final_policy_loss: Optional[float] = Form(None),
    final_value_loss: Optional[float] = Form(None),
    compressed: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Upload a new training model checkpoint."""
    # Validate filename
    is_compressed = compressed == "true" or file.filename.endswith(".gz")
    if not (file.filename.endswith(".pt") or file.filename.endswith(".pt.gz")):
        raise HTTPException(status_code=400, detail="Model file must be a .pt or .pt.gz file")

    # Read the file content
    content = await file.read()

    # Decompress if needed
    if is_compressed:
        try:
            content = gzip.decompress(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decompress: {e}")

    # Save file (always save uncompressed)
    file_path = TRAINING_MODELS_DIR / f"{version}.pt"
    with open(file_path, "wb") as f:
        f.write(content)

    # Save to database
    persistence.save_training_model(
        version=version,
        iteration=iteration,
        file_path=str(file_path),
        games_trained_on=games_trained_on,
        final_loss=final_loss,
        final_policy_loss=final_policy_loss,
        final_value_loss=final_value_loss,
    )

    return UploadModelResponse(version=version, status="uploaded")


@app.get("/training/models/{version}/download")
async def download_training_model(version: str):
    """Download a specific training model file."""
    model = persistence.get_training_model_by_version(version)

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    file_path = Path(model["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    return FileResponse(
        path=file_path,
        filename=f"{version}.pt",
        media_type="application/octet-stream",
    )


@app.get("/training/dashboard", response_model=TrainingDashboardResponse)
async def get_training_dashboard():
    """Get enhanced training status for the dashboard."""
    stats = persistence.get_training_games_stats()
    models = persistence.list_training_models(limit=10)
    latest = persistence.get_latest_training_model()

    # Build worker stats
    workers = {}
    for worker_id, count in stats.get("by_worker", {}).items():
        workers[worker_id] = WorkerStats(
            games=count,
            last_seen=stats.get("worker_last_seen", {}).get(worker_id),
        )

    # Build model info list
    model_infos = [
        ModelInfo(
            version=m["version"],
            iteration=m["iteration"],
            games_trained_on=m["games_trained_on"],
            final_loss=m["final_loss"],
            final_policy_loss=m["final_policy_loss"],
            final_value_loss=m["final_value_loss"],
            download_url=f"/training/models/{m['version']}/download",
            created_at=m["created_at"],
        )
        for m in models
    ]

    latest_model = None
    if latest:
        latest_model = ModelInfo(
            version=latest["version"],
            iteration=latest["iteration"],
            games_trained_on=latest["games_trained_on"],
            final_loss=latest["final_loss"],
            final_policy_loss=latest["final_policy_loss"],
            final_value_loss=latest["final_value_loss"],
            download_url=f"/training/models/{latest['version']}/download",
            created_at=latest["created_at"],
        )

    return TrainingDashboardResponse(
        status="active" if workers else "idle",
        games_pending=stats.get("pending", 0),
        games_total=stats.get("total", 0),
        latest_model=latest_model,
        workers=workers,
        models=model_infos,
    )


class ClearTrainingResponse(BaseModel):
    """Response after clearing training data."""
    games_deleted: int
    models_deleted: int
    metrics_deleted: int
    files_deleted: int


@app.delete("/training/clear")
async def clear_training_data():
    """Clear all training games, models, and metrics to start fresh."""
    result = persistence.clear_training_data()
    return ClearTrainingResponse(**result)


# --- Training Metrics API Endpoints ---

@app.post("/training/metrics", response_model=SubmitMetricsResponse)
async def submit_training_metrics(request: SubmitMetricsRequest):
    """
    Submit training metrics from the trainer.

    Stores metrics in the database for historical tracking and dashboard visualization.
    """
    metrics_id = persistence.save_training_metrics(
        iteration=request.iteration,
        metrics=request.metrics,
    )
    return SubmitMetricsResponse(id=metrics_id, status="accepted")


@app.get("/training/metrics", response_model=TrainingMetricsHistoryResponse)
async def get_training_metrics(
    limit: int = 100,
    offset: int = 0,
):
    """
    Get training metrics history.

    Returns metrics for all iterations, ordered by iteration number (ascending).
    """
    metrics, total = persistence.get_training_metrics(limit=limit, offset=offset)

    return TrainingMetricsHistoryResponse(
        metrics=[TrainingMetricsData(**m) for m in metrics],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/training/metrics/latest", response_model=LatestMetricsResponse)
async def get_latest_training_metrics():
    """
    Get the most recent training metrics.

    Returns null if no metrics have been recorded yet.
    """
    metrics = persistence.get_latest_training_metrics()

    if metrics is None:
        return LatestMetricsResponse(metrics=None)

    return LatestMetricsResponse(metrics=TrainingMetricsData(**metrics))


# --- Trainer State Endpoints ---

@app.post("/training/state/{key}", response_model=UploadTrainerStateResponse)
async def upload_trainer_state(
    key: str,
    iteration: int = Form(0),
    total_games_trained: int = Form(0),
    compressed: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Upload trainer state (optimizer state, replay buffer, etc.)."""
    content = await file.read()

    # Decompress if needed
    if compressed == "true":
        try:
            content = gzip.decompress(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decompress: {e}")

    # Save file
    file_path = TRAINING_STATE_DIR / f"{key}.dat"
    with open(file_path, "wb") as f:
        f.write(content)

    # Save metadata to DB
    persistence.save_trainer_state_record(
        key=key,
        iteration=iteration,
        total_games_trained=total_games_trained,
        file_path=str(file_path),
        file_size=len(content),
    )

    return UploadTrainerStateResponse(key=key, status="uploaded")


@app.get("/training/state/{key}", response_model=TrainerStateResponse)
async def get_trainer_state(key: str):
    """Get metadata about a trainer state file."""
    record = persistence.get_trainer_state_record(key)

    if record is None:
        return TrainerStateResponse(state=None)

    return TrainerStateResponse(
        state=TrainerStateInfo(
            key=record["key"],
            iteration=record["iteration"],
            total_games_trained=record["total_games_trained"],
            file_size=record["file_size"],
            download_url=f"/training/state/{record['key']}/download",
            created_at=record["created_at"],
        )
    )


@app.get("/training/state/{key}/download")
async def download_trainer_state(key: str):
    """Download a trainer state file."""
    record = persistence.get_trainer_state_record(key)

    if record is None:
        raise HTTPException(status_code=404, detail="Trainer state not found")

    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Trainer state file not found on disk")

    return FileResponse(
        path=file_path,
        filename=f"{key}.dat",
        media_type="application/octet-stream",
    )


class ResetTrainingResponse(BaseModel):
    """Response from training reset."""
    games_deleted: int
    models_deleted: int
    metrics_deleted: int
    trainer_states_deleted: int = 0
    files_deleted: int


@app.post("/training/reset", response_model=ResetTrainingResponse)
async def reset_training():
    """
    Reset all training data (games, models, metrics, trainer state).

    WARNING: This permanently deletes all training progress!
    Use this to start fresh training from scratch.
    """
    result = persistence.clear_training_data()
    return ResetTrainingResponse(**result)


# Setup client logging
LOG_DIR = Path("/tmp/razzle-logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure file logger for client logs
client_logger = logging.getLogger("razzle.client")
client_logger.setLevel(logging.DEBUG)
client_log_handler = logging.FileHandler(LOG_DIR / "client.log")
client_log_handler.setFormatter(logging.Formatter("%(message)s"))
client_logger.addHandler(client_log_handler)


@app.post("/logs")
async def receive_logs(request: LogRequest):
    """Receive logs from the client."""
    session_id = request.session_id or "unknown"
    for entry in request.entries:
        log_line = f"[{entry.timestamp}] [{session_id}] [{entry.level.upper()}] {entry.message}"
        if entry.data:
            log_line += f" | {entry.data}"
        client_logger.info(log_line)
    return {"received": len(request.entries)}


@app.post("/games", response_model=CreateGameResponse)
async def create_game(
    request: CreateGameRequest = None,
    auth_request: Request = None,
    auth_cookie: Optional[str] = Cookie(None, alias=AUTH_COOKIE_NAME)
):
    """Create a new game."""
    if request is None:
        request = CreateGameRequest()

    # Check for newer model on game creation
    get_evaluator(check_for_updates=True)

    # Get current user if authenticated
    user = await get_current_user(auth_request, auth_cookie) if auth_request else None
    player1_user_id = user["user_id"] if user else None

    # Validate bot type
    bot_type = request.bot_type
    if bot_type not in BOT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid bot_type: {bot_type}. Must be one of: {BOT_TYPES}")

    game_id = str(uuid.uuid4())[:8]
    game = Game(
        game_id=game_id,
        player1_type=request.player1_type,
        player2_type=request.player2_type,
        ai_simulations=request.ai_simulations,
        bot_type=bot_type,
        time_control=request.time_control
    )
    games[game_id] = game

    # Get model version for ELO tracking
    # Neural bots use actual model version, others use bot_type as identifier
    if bot_type == "neural":
        model_version = get_model_info()
    else:
        model_version = bot_type  # e.g., "mcts", "random"

    # Set up player IDs for ELO tracking
    setup_game_players(game, user, model_version)

    # Persist to database
    persistence.save_game(
        game_id=game_id,
        state=game.state,
        player1_type=request.player1_type,
        player2_type=request.player2_type,
        ai_simulations=request.ai_simulations,
        player1_user_id=player1_user_id,
        ai_model_version=model_version if request.player2_type == "ai" else None,
        bot_type=bot_type,
    )

    # Update game record with player IDs
    if game.player_ids[0] or game.player_ids[1]:
        persistence.update_game_result(
            game_id,
            winner=None,
            player1_player_id=game.player_ids[0],
            player2_player_id=game.player_ids[1]
        )

    return CreateGameResponse(game_id=game_id)


@app.get("/games/{game_id}", response_model=GameStateResponse)
async def get_game(game_id: str):
    """Get current game state."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    return games[game_id].to_response()


@app.post("/games/{game_id}/move", response_model=GameStateResponse)
async def make_move(
    game_id: str,
    request: MakeMoveRequest,
    auth_request: Request = None,
    auth_cookie: Optional[str] = Cookie(None, alias=AUTH_COOKIE_NAME)
):
    """Make a move in the game."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]

    if game.state.is_terminal():
        raise HTTPException(status_code=409, detail="Game already finished")

    legal_moves = get_legal_moves(game.state)
    if request.move not in legal_moves:
        raise HTTPException(status_code=400, detail="Invalid move")

    # Associate user with game if logged in (for users who log in mid-game)
    user = await get_current_user(auth_request, auth_cookie) if auth_request else None
    if user:
        persistence.associate_user_with_game(game_id, user["user_id"])

    game.state.apply_move(request.move)

    # Persist state and record move
    persistence.save_game(game_id, game.state)
    persistence.append_move(game_id, request.move)

    # Check for game over and update ELO
    game.check_and_update_elo()

    # Notify WebSocket clients
    response = game.to_response()
    await broadcast_state(game, response)

    return response


@app.post("/games/{game_id}/ai", response_model=AIMoveResponse)
async def get_ai_move(
    game_id: str,
    request: AIMoveRequest = None,
    auth_request: Request = None,
    auth_cookie: Optional[str] = Cookie(None, alias=AUTH_COOKIE_NAME)
):
    """Get AI to calculate and play a move.

    Bot types:
    - "neural": Uses trained neural network (strongest if trained well)
    - "mcts": Pure MCTS with uniform priors (good baseline for testing)
    - "random": Random legal moves (weakest)

    If the game has time control enabled and use_time_control is True,
    the AI will dynamically allocate simulations based on position difficulty.
    """
    if request is None:
        request = AIMoveRequest()

    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]

    # Use request bot_type if provided, otherwise use game's default
    bot_type = request.bot_type if request.bot_type else game.bot_type
    if bot_type not in BOT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid bot_type: {bot_type}. Must be one of: {BOT_TYPES}")

    # Associate user with game if logged in (for users who log in mid-game)
    user = await get_current_user(auth_request, auth_cookie) if auth_request else None
    if user:
        persistence.associate_user_with_game(game_id, user["user_id"])

    if game.state.is_terminal():
        raise HTTPException(status_code=409, detail="Game already finished")

    start_time = time.time()
    difficulty = None
    remaining_time = None

    # Check if we should use time control
    current_player = game.state.current_player
    time_manager = game.time_managers[current_player] if game.time_control else None
    use_time_control = request.use_time_control and time_manager is not None and bot_type == "neural"

    # Handle "random" bot type - just pick a random legal move
    if bot_type == "random":
        legal_moves = get_legal_moves(game.state)
        move = py_random.choice(legal_moves)
        # Create dummy policy and analysis for response
        from razzle.ai.network import NUM_ACTIONS
        policy = [0.0] * NUM_ACTIONS
        if move == -1:
            policy[NUM_ACTIONS - 1] = 1.0  # END_TURN
        else:
            policy[move] = 1.0
        elapsed_ms = int((time.time() - start_time) * 1000)
        top_moves = [TopMove(move=move, algebraic=move_to_algebraic(move), visits=1, value=0.0)]
        total_visits = 1
        value = 0.0
    elif use_time_control:
        # Use dynamic time management based on position difficulty
        ev = get_evaluator_for_model(request.model)

        move, root, info = play_move_timed(
            game.state, ev, time_manager,
            temperature=request.temperature,
            batch_size=16
        )

        policy = MCTS(ev, MCTSConfig(temperature=request.temperature)).get_policy(root).tolist()
        total_visits = info['simulations_done']
        difficulty = info['difficulty']
        remaining_time = time_manager.remaining_time

        elapsed_ms = int(info['elapsed_time'] * 1000)

        # Get analysis
        mcts = MCTS(ev, MCTSConfig(temperature=request.temperature))
        analysis = mcts.analyze(root, top_k=5)
        top_moves = [
            TopMove(
                move=m['move'],
                algebraic=m['algebraic'],
                visits=m['visits'],
                value=m['value']
            )
            for m in analysis
        ]
        value = root.value
    else:
        # Run MCTS with fixed simulations
        if bot_type == "mcts":
            # Pure MCTS with uniform priors over legal moves
            ev = DummyEvaluator()
        else:  # "neural"
            ev = get_evaluator_for_model(request.model)

        # Use temperature for opening diversity (first 2-3 moves = ~6 ply)
        temperature = request.temperature
        if game.state.ply < 6 and temperature == 0.0:
            temperature = 1.0  # Opening diversity

        # Mid-pass positions have narrow trees — use fewer sims
        sims = request.simulations
        if game.state.has_passed:
            sims = max(50, sims // 10)

        config = MCTSConfig(
            num_simulations=sims,
            temperature=temperature,
            batch_size=16 if bot_type == "neural" else 1  # Batching only helps with NN
        )
        mcts = MCTS(ev, config)

        # Run search (batched for neural, regular for mcts)
        if bot_type == "neural":
            root = mcts.search_batched(game.state, add_noise=False)
        else:
            root = mcts.search(game.state, add_noise=False)

        move = mcts.select_move(root)
        policy = mcts.get_policy(root).tolist()
        total_visits = root.visit_count

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Get analysis
        analysis = mcts.analyze(root, top_k=5)
        top_moves = [
            TopMove(
                move=m['move'],
                algebraic=m['algebraic'],
                visits=m['visits'],
                value=m['value']
            )
            for m in analysis
        ]
        value = root.value

    # Apply the move
    game.state.apply_move(move)

    # Persist state and record move
    persistence.save_game(game_id, game.state)
    persistence.append_move(game_id, move)

    # Check for game over and update ELO
    game.check_and_update_elo()

    # Notify WebSocket clients
    game_response = game.to_response()
    await broadcast_state(game, game_response)

    # Convert numpy types to Python native types for JSON serialization
    return AIMoveResponse(
        move=int(move),
        algebraic=move_to_algebraic(move),
        policy=policy if isinstance(policy, list) else policy.tolist(),
        value=float(value) if value is not None else None,
        visits=int(total_visits),
        time_ms=int(elapsed_ms),
        top_moves=[
            TopMove(
                move=int(tm.move),
                algebraic=tm.algebraic,
                visits=int(tm.visits),
                value=float(tm.value) if tm.value is not None else None
            )
            for tm in top_moves
        ],
        game_state=game_response,
        difficulty=float(difficulty) if difficulty is not None else None,
        remaining_time=float(remaining_time) if remaining_time is not None else None
    )


@app.get("/games/{game_id}/legal-moves", response_model=LegalMovesResponse)
async def get_legal_moves_endpoint(game_id: str):
    """Get all legal moves in human-readable format."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]
    moves = get_legal_moves(game.state)

    # Determine move types
    ball_sq = None
    for sq in range(56):
        if game.state.balls[game.state.current_player] & (1 << sq):
            ball_sq = sq
            break

    result = []
    for m in moves:
        if m == -1:
            move_type = "end_turn"
        else:
            src, dst = decode_move(m)
            move_type = "pass" if src == ball_sq else "knight"
        result.append(LegalMove(
            move=m,
            algebraic=move_to_algebraic(m),
            type=move_type
        ))

    return LegalMovesResponse(moves=result)


@app.post("/games/{game_id}/undo", response_model=GameStateResponse)
async def undo_move(game_id: str):
    """Undo the last move."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]

    if not game.state.history:
        raise HTTPException(status_code=400, detail="Nothing to undo")

    game.state.undo_move()

    # Persist state and remove move from history
    persistence.save_game(game_id, game.state)
    persistence.pop_move(game_id)

    response = game.to_response()
    await broadcast_state(game, response)

    return response


class ResignRequest(BaseModel):
    player: Optional[int] = Field(None, description="Player resigning (0 or 1). If not provided, uses current player.")


@app.post("/games/{game_id}/resign", response_model=GameStateResponse)
async def resign_game(
    game_id: str,
    request: ResignRequest = None,
    auth_request: Request = None,
    auth_cookie: Optional[str] = Cookie(None, alias=AUTH_COOKIE_NAME)
):
    """Resign from the game. The opponent wins."""
    if request is None:
        request = ResignRequest()

    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]

    if game.is_game_over():
        raise HTTPException(status_code=409, detail="Game already finished")

    # Determine which player is resigning
    resigning_player = request.player
    if resigning_player is None:
        # Default to current player (for AI games, this is the human)
        resigning_player = 0  # In AI games, player 0 is human

    if resigning_player not in (0, 1):
        raise HTTPException(status_code=400, detail="Invalid player (must be 0 or 1)")

    # For online games, verify the user is the one resigning
    user = await get_current_user(auth_request, auth_cookie) if auth_request else None
    if game.is_online_game() and user:
        player_color = game.get_player_color(user["user_id"])
        if player_color is not None:
            resigning_player = player_color

    # Mark resignation
    game.resigned_by = resigning_player

    # Update ELO ratings
    game.check_and_update_elo()

    # Persist the resignation
    persistence.update_game_result(
        game_id,
        winner=game.get_winner(),
        player1_player_id=game.player_ids[0],
        player2_player_id=game.player_ids[1]
    )

    response = game.to_response()
    await broadcast_state(game, response)

    return response


# --- Online Multiplayer API Endpoints ---

class CreateOnlineGameRequest(BaseModel):
    host_color: int = Field(0, description="0 for blue (player 1), 1 for red (player 2)")


class OnlineOpponentInfo(BaseModel):
    user_id: str
    display_name: Optional[str]
    elo_rating: float = 1000.0


class CreateOnlineGameResponse(BaseModel):
    game_id: str
    join_code: str
    host_color: int
    status: str


class JoinOnlineGameRequest(BaseModel):
    join_code: str = Field(..., min_length=4, max_length=8)


class JoinOnlineGameResponse(BaseModel):
    game_id: str
    your_color: int
    opponent: Optional[OnlineOpponentInfo]
    status: str


class OnlineGameStatusResponse(BaseModel):
    game_id: str
    join_code: str
    status: str
    your_color: int
    opponent: Optional[OnlineOpponentInfo]
    is_your_turn: bool
    winner: Optional[int]
    game_state: GameStateResponse


class LeaveOnlineGameResponse(BaseModel):
    status: str
    winner: Optional[int]


class OnlineGameSummary(BaseModel):
    game_id: str
    join_code: str
    status: str
    your_color: int
    is_your_turn: bool
    ply: int
    opponent_name: Optional[str] = None
    created_at: str
    updated_at: str


class MyOnlineGamesResponse(BaseModel):
    active: list[OnlineGameSummary]
    waiting: list[OnlineGameSummary]


@app.post("/games/online", response_model=CreateOnlineGameResponse)
async def create_online_game(
    request: CreateOnlineGameRequest,
    user: dict = Depends(get_user_or_anon)
):
    """
    Create a new online game and get a shareable join code.

    The host chooses their color (0=blue, 1=red).
    Anonymous users are supported.
    """
    if request.host_color not in (0, 1):
        raise HTTPException(status_code=400, detail="host_color must be 0 (blue) or 1 (red)")

    result = persistence.create_online_game(
        host_user_id=user["user_id"],
        host_color=request.host_color,
    )

    # Also create an in-memory Game object for WebSocket support
    game = Game(
        game_id=result["game_id"],
        player1_type="human",
        player2_type="human",
        ai_simulations=0,
        bot_type="mcts",
        online_status="waiting",
        join_code=result["join_code"]
    )
    # Set host's user ID in the appropriate slot
    if request.host_color == 0:
        game.player_user_ids[0] = user["user_id"]
    else:
        game.player_user_ids[1] = user["user_id"]

    games[result["game_id"]] = game

    return CreateOnlineGameResponse(**result)


@app.post("/games/online/join", response_model=JoinOnlineGameResponse)
async def join_online_game(
    request: JoinOnlineGameRequest,
    user: dict = Depends(get_user_or_anon)
):
    """
    Join an existing online game using a join code.

    Returns game info and opponent details. Anonymous users are supported.
    """
    result = persistence.join_online_game(
        join_code=request.join_code,
        guest_user_id=user["user_id"],
    )

    if result is None:
        raise HTTPException(status_code=404, detail="Game not found")

    if "error" in result:
        if result["error"] == "game_not_waiting":
            raise HTTPException(status_code=409, detail=f"Game is not waiting for players (status: {result.get('status')})")
        elif result["error"] == "game_full":
            raise HTTPException(status_code=409, detail="Game is already full")
        raise HTTPException(status_code=400, detail=result["error"])

    # Load game into memory if not already
    game_id = result["game_id"]
    if game_id not in games:
        game_data = persistence.load_game(game_id)
        if game_data:
            game = Game(
                game_id=game_id,
                player1_type="human",
                player2_type="human",
                ai_simulations=0,
                bot_type="mcts",
                online_status="playing",
                join_code=request.join_code.upper().strip()
            )
            game.state = game_data["state"]
            games[game_id] = game

    # Update the in-memory Game object with user IDs
    if game_id in games:
        game = games[game_id]
        game.online_status = "playing"
        # Set guest user ID
        game.player_user_ids[result["your_color"]] = user["user_id"]
        # Set host user ID (opponent)
        if result.get("opponent"):
            host_color = 1 - result["your_color"]
            game.player_user_ids[host_color] = result["opponent"]["user_id"]

        # Notify host via WebSocket
        await broadcast_online_event(game, "player_joined", {
            "user_id": user["user_id"],
            "display_name": user.get("display_name") or user.get("username"),
            "color": result["your_color"],
        })

    opponent = None
    if result.get("opponent"):
        opponent = OnlineOpponentInfo(**result["opponent"])

    return JoinOnlineGameResponse(
        game_id=result["game_id"],
        your_color=result["your_color"],
        opponent=opponent,
        status=result["status"],
    )


@app.get("/games/online/{game_id}", response_model=OnlineGameStatusResponse)
async def get_online_game_status(
    game_id: str,
    user: dict = Depends(get_user_or_anon)
):
    """
    Get the current status of an online game.

    User must be a participant in the game.
    """
    result = persistence.get_online_game_status(game_id, user["user_id"])

    if result is None:
        raise HTTPException(status_code=404, detail="Game not found")

    if "error" in result:
        if result["error"] == "not_authorized":
            raise HTTPException(status_code=403, detail="You are not a participant in this game")
        raise HTTPException(status_code=400, detail=result["error"])

    # Build game state response
    state = result["state"]

    # Load into memory if needed
    if game_id not in games:
        game = Game(
            game_id=game_id,
            player1_type="human",
            player2_type="human",
            ai_simulations=0,
            bot_type="mcts"
        )
        game.state = state
        games[game_id] = game
    else:
        game = games[game_id]

    opponent = None
    if result.get("opponent"):
        opponent = OnlineOpponentInfo(**result["opponent"])

    return OnlineGameStatusResponse(
        game_id=result["game_id"],
        join_code=result["join_code"],
        status=result["status"],
        your_color=result["your_color"],
        opponent=opponent,
        is_your_turn=result["is_your_turn"],
        winner=result["winner"],
        game_state=game.to_response(),
    )


@app.post("/games/online/{game_id}/leave", response_model=LeaveOnlineGameResponse)
async def leave_online_game(
    game_id: str,
    user: dict = Depends(get_user_or_anon)
):
    """
    Leave/abandon an online game. This forfeits the game.

    For waiting games, the game is simply cancelled.
    For playing games, the opponent wins by forfeit.
    """
    result = persistence.abandon_online_game(game_id, user["user_id"])

    if result is None:
        raise HTTPException(status_code=404, detail="Game not found")

    if "error" in result:
        if result["error"] == "not_authorized":
            raise HTTPException(status_code=403, detail="You are not a participant in this game")
        elif result["error"] == "cannot_abandon":
            raise HTTPException(status_code=409, detail=f"Cannot abandon game (status: {result.get('status')})")
        raise HTTPException(status_code=400, detail=result["error"])

    # Notify opponent via WebSocket
    if game_id in games:
        game = games[game_id]
        await broadcast_online_event(game, "game_abandoned", {
            "abandoning_user_id": user["user_id"],
            "winner": result.get("winner"),
        })

    return LeaveOnlineGameResponse(
        status=result["status"],
        winner=result.get("winner"),
    )


@app.get("/games/online/mine", response_model=MyOnlineGamesResponse)
async def get_my_online_games(
    user: dict = Depends(get_user_or_anon)
):
    """
    List all online games for the current user.

    Returns active (playing) and waiting games separately.
    """
    result = persistence.get_user_online_games(user["user_id"])

    return MyOnlineGamesResponse(
        active=[OnlineGameSummary(**g) for g in result["active"]],
        waiting=[OnlineGameSummary(**g) for g in result["waiting"]],
    )


async def broadcast_online_event(game: Game, event_type: str, data: dict):
    """Broadcast an online game event to all connected WebSocket clients."""
    message = {
        "type": event_type,
        "data": data
    }
    disconnected = []
    for ws in game.websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        game.websockets.remove(ws)


# --- Game Browser Endpoints ---

class GameSummary(BaseModel):
    game_id: str
    player1_type: str
    player2_type: str
    player1_user_id: Optional[str]
    player2_user_id: Optional[str]
    player1_username: Optional[str]
    player2_username: Optional[str]
    status: str
    winner: Optional[int]
    move_count: int
    ply: int
    created_at: str
    updated_at: str
    ai_model_version: Optional[str]


class GameListResponse(BaseModel):
    games: list[GameSummary]
    total: int
    page: int
    per_page: int
    total_pages: int


class GameFullResponse(BaseModel):
    game_id: str
    player1_type: str
    player2_type: str
    player1_user_id: Optional[str]
    player2_user_id: Optional[str]
    status: str
    winner: Optional[int]
    ply: int
    moves: list[int]
    moves_algebraic: list[str]
    created_at: str
    updated_at: str
    ai_model_version: Optional[str]


@app.get("/games", response_model=GameListResponse)
async def list_games(
    player_id: Optional[str] = None,
    status: Optional[str] = None,
    winner: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
):
    """List games with filtering and pagination."""
    result = persistence.list_games(
        player_id=player_id,
        status=status,
        winner=winner,
        date_from=date_from,
        date_to=date_to,
        page=page,
        per_page=per_page,
    )
    return GameListResponse(
        games=[GameSummary(**g) for g in result["games"]],
        total=result["total"],
        page=result["page"],
        per_page=result["per_page"],
        total_pages=result["total_pages"],
    )


@app.get("/games/{game_id}/full", response_model=GameFullResponse)
async def get_game_full(game_id: str):
    """Get full game data including all moves for replay."""
    data = persistence.get_game_full(game_id)
    if not data:
        raise HTTPException(status_code=404, detail="Game not found")

    return GameFullResponse(
        game_id=data["game_id"],
        player1_type=data["player1_type"],
        player2_type=data["player2_type"],
        player1_user_id=data["player1_user_id"],
        player2_user_id=data["player2_user_id"],
        status=data["status"],
        winner=data["winner"],
        ply=data["ply"],
        moves=data["moves"],
        moves_algebraic=data["moves_algebraic"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        ai_model_version=data["ai_model_version"],
    )


# --- Analysis Endpoints ---

class AnalyzePositionRequest(BaseModel):
    pieces: list[str]  # [p1_pieces, p2_pieces] as strings for JS precision
    balls: list[str]   # [p1_ball, p2_ball] as strings for JS precision
    current_player: int
    touched_mask: str = "0"  # String for JS precision
    has_passed: bool = False
    last_knight_dst: int = -1
    simulations: int = 200


class MoveAnalysis(BaseModel):
    move: int
    algebraic: str
    visits: int
    value: float
    policy: float


class AnalyzePositionResponse(BaseModel):
    value: float
    legal_moves: list[int]
    top_moves: list[MoveAnalysis]
    time_ms: int


class MoveClassification(BaseModel):
    move: int
    algebraic: str
    value_before: float
    value_after: float
    best_move: int
    best_move_algebraic: str
    best_value: float
    delta: float
    classification: str  # best, good, inaccuracy, mistake, blunder


class AnalyzeGameResponse(BaseModel):
    game_id: str
    move_analyses: list[MoveClassification]
    summary: dict  # counts by classification


def classify_move(delta: float) -> str:
    """Classify a move based on the difference from best."""
    if delta >= -0.02:
        return "best"
    elif delta >= -0.08:
        return "good"
    elif delta >= -0.15:
        return "inaccuracy"
    elif delta >= -0.30:
        return "mistake"
    else:
        return "blunder"


@app.post("/analyze", response_model=AnalyzePositionResponse)
async def analyze_position(request: AnalyzePositionRequest):
    """Analyze a single position without making a move."""
    start_time = time.time()

    # Reconstruct game state (convert strings to ints for internal use)
    state = GameState(
        pieces=[int(p) for p in request.pieces],
        balls=[int(b) for b in request.balls],
        current_player=request.current_player,
        touched_mask=int(request.touched_mask),
        has_passed=request.has_passed,
        last_knight_dst=request.last_knight_dst,
        ply=0,
        history=[],
    )

    if state.is_terminal():
        return AnalyzePositionResponse(
            value=1.0 if state.get_winner() == state.current_player else -1.0,
            legal_moves=[],
            top_moves=[],
            time_ms=0,
        )

    # Run MCTS analysis
    ev = get_evaluator()
    config = MCTSConfig(
        num_simulations=request.simulations,
        temperature=0.0,
        batch_size=16,
    )
    mcts = MCTS(ev, config)
    root = mcts.search_batched(state, add_noise=False)
    policy = mcts.get_policy(root)

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Get analysis
    analysis = mcts.analyze(root, top_k=10)
    top_moves = [
        MoveAnalysis(
            move=m["move"],
            algebraic=m["algebraic"],
            visits=m["visits"],
            value=m["value"],
            policy=float(policy[m["move"]]) if m["move"] >= 0 else 0.0,
        )
        for m in analysis
    ]

    return AnalyzePositionResponse(
        value=root.value,
        legal_moves=get_legal_moves(state),
        top_moves=top_moves,
        time_ms=elapsed_ms,
    )


@app.post("/games/{game_id}/analyze", response_model=AnalyzeGameResponse)
async def analyze_game(game_id: str, simulations_per_position: int = 200):
    """Analyze an entire game, classifying each move."""
    # Get full game data
    data = persistence.get_game_full(game_id)
    if not data:
        raise HTTPException(status_code=404, detail="Game not found")

    moves = data["moves"]
    if not moves:
        return AnalyzeGameResponse(
            game_id=game_id,
            move_analyses=[],
            summary={"best": 0, "good": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0},
        )

    # Replay through the game and analyze each position
    ev = get_evaluator()
    config = MCTSConfig(
        num_simulations=simulations_per_position,
        temperature=0.0,
        batch_size=16,
    )

    state = GameState.new_game()
    analyses = []
    summary = {"best": 0, "good": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0}

    for move in moves:
        if state.is_terminal():
            break

        # Analyze position before the move
        mcts = MCTS(ev, config)
        root = mcts.search_batched(state, add_noise=False)
        best_move = mcts.select_move(root)

        # Get values from MCTS analysis
        analysis = mcts.analyze(root, top_k=10)
        best_value = analysis[0]["value"] if analysis else root.value

        # Find the value of the played move
        played_value = best_value
        for m in analysis:
            if m["move"] == move:
                played_value = m["value"]
                break

        # Calculate delta (best - played, from perspective of current player)
        delta = played_value - best_value

        classification = classify_move(delta)
        summary[classification] += 1

        analyses.append(MoveClassification(
            move=move,
            algebraic=move_to_algebraic(move),
            value_before=root.value,
            value_after=played_value,
            best_move=best_move,
            best_move_algebraic=move_to_algebraic(best_move),
            best_value=best_value,
            delta=delta,
            classification=classification,
        ))

        # Apply the move
        state.apply_move(move)

    return AnalyzeGameResponse(
        game_id=game_id,
        move_analyses=analyses,
        summary=summary,
    )


@app.get("/util/move", response_model=MoveConversionResponse)
async def convert_move(encoded: Optional[int] = None, algebraic: Optional[str] = None):
    """Convert between move formats."""
    if encoded is not None:
        src, dst = decode_move(encoded)
        return MoveConversionResponse(
            encoded=encoded,
            algebraic=move_to_algebraic(encoded),
            src=src,
            dst=dst
        )
    elif algebraic is not None:
        try:
            move = algebraic_to_move(algebraic)
            src, dst = decode_move(move)
            return MoveConversionResponse(
                encoded=move,
                algebraic=algebraic,
                src=src,
                dst=dst
            )
        except (ValueError, IndexError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid algebraic notation: {algebraic}")
    else:
        raise HTTPException(status_code=400, detail="Provide either 'encoded' or 'algebraic' parameter")


# --- Arena API Models ---

class ArenaMatchData(BaseModel):
    """Arena match result data."""
    id: Optional[int] = None
    model1_version: str
    model2_version: str
    model1_wins: int
    model2_wins: int
    draws: int
    simulations: int = 800
    created_at: Optional[str] = None


class ArenaRatingData(BaseModel):
    """Arena ELO rating data."""
    id: Optional[int] = None
    model_version: str
    iteration: int
    simulations: int = 800
    elo_rating: float = 1000.0
    games_played: int = 0
    last_updated: Optional[str] = None


class ArenaMatchesResponse(BaseModel):
    """Response with arena matches."""
    matches: list[ArenaMatchData]
    count: int


class ArenaRatingsResponse(BaseModel):
    """Response with arena ratings."""
    ratings: list[ArenaRatingData]
    count: int


class ArenaSubmitMatchRequest(BaseModel):
    """Request to submit an arena match result."""
    model1_version: str
    model2_version: str
    model1_wins: int
    model2_wins: int
    draws: int
    simulations: int = 800


class ArenaSubmitMatchResponse(BaseModel):
    """Response after submitting arena match."""
    id: int
    status: str = "accepted"


class ArenaSubmitRatingRequest(BaseModel):
    """Request to submit/update an arena rating."""
    model_version: str
    iteration: int
    simulations: int = 800
    elo_rating: float
    games_played: int


class ArenaSubmitRatingResponse(BaseModel):
    """Response after submitting arena rating."""
    id: int
    status: str = "accepted"


class ArenaComputeRequest(BaseModel):
    """Request to compute ELO ratings."""
    anchor_model: str = "initial"
    simulations: Optional[int] = None


class ArenaClearResponse(BaseModel):
    """Response after clearing arena data."""
    matches_deleted: int
    ratings_deleted: int


# --- Arena API Endpoints ---

@app.get("/arena/matches", response_model=ArenaMatchesResponse)
async def get_arena_matches(
    model: Optional[str] = None,
    simulations: Optional[int] = None,
    limit: int = 100,
):
    """
    Get arena match history.

    Args:
        model: Filter to matches involving this model
        simulations: Filter by simulation count
        limit: Maximum number of matches to return
    """
    matches = persistence.get_arena_matches(
        model_version=model,
        simulations=simulations,
        limit=limit,
    )

    return ArenaMatchesResponse(
        matches=[ArenaMatchData(**m) for m in matches],
        count=len(matches),
    )


@app.post("/arena/matches", response_model=ArenaSubmitMatchResponse)
async def submit_arena_match(request: ArenaSubmitMatchRequest):
    """Submit an arena match result."""
    match_id = persistence.save_arena_match(
        model1_version=request.model1_version,
        model2_version=request.model2_version,
        model1_wins=request.model1_wins,
        model2_wins=request.model2_wins,
        draws=request.draws,
        simulations=request.simulations,
    )

    return ArenaSubmitMatchResponse(id=match_id, status="accepted")


@app.get("/arena/ratings", response_model=ArenaRatingsResponse)
async def get_arena_ratings(
    simulations: Optional[int] = None,
    limit: int = 100,
):
    """
    Get arena ELO ratings sorted by rating (descending).

    Args:
        simulations: Filter by simulation count (None = all)
        limit: Maximum number of ratings to return
    """
    ratings = persistence.get_arena_ratings(
        simulations=simulations,
        limit=limit,
    )

    return ArenaRatingsResponse(
        ratings=[ArenaRatingData(**r) for r in ratings],
        count=len(ratings),
    )


@app.post("/arena/ratings", response_model=ArenaSubmitRatingResponse)
async def submit_arena_rating(request: ArenaSubmitRatingRequest):
    """Submit or update an arena ELO rating."""
    rating_id = persistence.save_arena_rating(
        model_version=request.model_version,
        iteration=request.iteration,
        elo_rating=request.elo_rating,
        games_played=request.games_played,
        simulations=request.simulations,
    )

    return ArenaSubmitRatingResponse(id=rating_id, status="accepted")


@app.post("/arena/compute")
async def compute_arena_ratings(request: ArenaComputeRequest = None):
    """
    Recompute all ELO ratings from match history.

    Args:
        anchor_model: Model to anchor at 1000 ELO (default: "initial")
        simulations: Compute only for this simulation count (None = all)
    """
    if request is None:
        request = ArenaComputeRequest()

    from razzle.training.elo import compute_all_ratings, get_iteration_from_version

    # Get all simulation counts or just the specified one
    if request.simulations is not None:
        sim_counts = [request.simulations]
    else:
        all_matches = persistence.get_all_arena_matches()
        sim_counts = list(set(m["simulations"] for m in all_matches))
        sim_counts.sort()

    results = {}
    for sims in sim_counts:
        matches = persistence.get_all_arena_matches(simulations=sims)
        if not matches:
            continue

        ratings = compute_all_ratings(matches, anchor_model=request.anchor_model)

        # Save ratings to database
        for version, rating in ratings.items():
            iteration = get_iteration_from_version(version)
            persistence.save_arena_rating(
                model_version=version,
                iteration=iteration,
                elo_rating=rating.rating,
                games_played=rating.games_played,
                simulations=sims,
            )

        results[sims] = {
            version: {"elo": rating.rating, "games": rating.games_played}
            for version, rating in ratings.items()
        }

    return {
        "status": "computed",
        "simulation_counts": sim_counts,
        "ratings": results,
    }


@app.delete("/arena/clear", response_model=ArenaClearResponse)
async def clear_arena_data():
    """Clear all arena matches and ratings."""
    result = persistence.clear_arena_data()
    return ArenaClearResponse(**result)


# --- Player/Leaderboard API Models ---

class PlayerData(BaseModel):
    """Player data (human or AI)."""
    player_id: str
    player_type: str  # 'human' or 'ai'
    user_id: Optional[str] = None
    model_version: Optional[str] = None
    simulations: Optional[int] = None
    display_name: str
    elo_rating: float
    elo_games_played: int
    created_at: str


class PlayerWithGamesData(PlayerData):
    """Player data with recent games."""
    recent_games: list[dict] = []


class PlayersListResponse(BaseModel):
    """Response with list of players."""
    players: list[PlayerData]
    count: int


class LeaderboardResponse(BaseModel):
    """Leaderboard response."""
    players: list[PlayerData]
    count: int


# --- Player/Leaderboard API Endpoints ---

@app.get("/players", response_model=PlayersListResponse)
async def list_players(
    player_type: Optional[str] = None,
    limit: int = 100,
    min_games: int = 0,
):
    """
    List all players.

    Args:
        player_type: Filter by 'human' or 'ai' (None = all)
        limit: Maximum number of players to return
        min_games: Minimum games played to be included
    """
    if player_type and player_type not in ("human", "ai"):
        raise HTTPException(status_code=400, detail="player_type must be 'human' or 'ai'")

    players = persistence.get_leaderboard(
        player_type=player_type,
        limit=limit,
        min_games=min_games,
    )

    return PlayersListResponse(
        players=[PlayerData(**p) for p in players],
        count=len(players),
    )


@app.get("/players/{player_id}", response_model=PlayerWithGamesData)
async def get_player(player_id: str, include_games: bool = True, games_limit: int = 20):
    """
    Get a player by ID with optional recent games.

    Args:
        player_id: The player ID
        include_games: Whether to include recent games (default: True)
        games_limit: Maximum recent games to include (default: 20)
    """
    player = persistence.get_player(player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")

    recent_games = []
    if include_games:
        recent_games = persistence.get_player_recent_games(player_id, limit=games_limit)

    return PlayerWithGamesData(
        **player,
        recent_games=recent_games,
    )


@app.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    player_type: Optional[str] = None,
    limit: int = 50,
    min_games: int = 1,
):
    """
    Get the ELO leaderboard.

    Args:
        player_type: Filter by 'human' or 'ai' (None = all)
        limit: Maximum number of players to return (default: 50)
        min_games: Minimum games played to be included (default: 1)
    """
    if player_type and player_type not in ("human", "ai"):
        raise HTTPException(status_code=400, detail="player_type must be 'human' or 'ai'")

    players = persistence.get_leaderboard(
        player_type=player_type,
        limit=limit,
        min_games=min_games,
    )

    return LeaderboardResponse(
        players=[PlayerData(**p) for p in players],
        count=len(players),
    )


# --- WebSocket ---

async def broadcast_state(game: Game, state: GameStateResponse):
    """Broadcast state update to all connected clients."""
    message = {
        "type": "state",
        "data": state.model_dump()
    }
    disconnected = []
    for ws in game.websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        game.websockets.remove(ws)


async def broadcast_thinking(game: Game, simulations_done: int, total: int, best_move: str, value: float):
    """Broadcast AI thinking progress."""
    message = {
        "type": "thinking",
        "data": {
            "simulations_done": simulations_done,
            "simulations_total": total,
            "current_best": best_move,
            "value": value
        }
    }
    for ws in game.websockets:
        try:
            await ws.send_json(message)
        except Exception:
            pass


async def send_error(ws: WebSocket, message: str, code: str):
    """Send error message to client."""
    await ws.send_json({
        "type": "error",
        "data": {"message": message, "code": code}
    })


# Grace period for disconnect before auto-forfeit (in seconds)
DISCONNECT_GRACE_PERIOD = 120  # 2 minutes


async def handle_player_disconnect(game: Game, user_id: str):
    """Handle player disconnect with grace period."""
    player_color = game.get_player_color(user_id)
    if player_color is not None:
        game.player_connected[player_color] = False

        # Notify opponent
        await broadcast_online_event(game, "opponent_disconnected", {
            "user_id": user_id,
            "color": player_color,
            "grace_period": DISCONNECT_GRACE_PERIOD,
        })

        # Start grace period timer
        async def disconnect_timer():
            await asyncio.sleep(DISCONNECT_GRACE_PERIOD)
            # Check if still disconnected
            if not game.player_connected[player_color] and game.online_status == "playing":
                # Auto-forfeit
                winner = 1 - player_color  # Opponent wins
                game.online_status = "abandoned"
                persistence.update_online_game_status(game.game_id, "abandoned", winner)

                await broadcast_online_event(game, "game_abandoned", {
                    "reason": "disconnect_timeout",
                    "abandoning_user_id": user_id,
                    "winner": winner,
                })

        # Cancel existing timer if any
        if user_id in game.disconnect_timers:
            game.disconnect_timers[user_id].cancel()

        game.disconnect_timers[user_id] = asyncio.create_task(disconnect_timer())


async def handle_player_reconnect(game: Game, user_id: str):
    """Handle player reconnection."""
    player_color = game.get_player_color(user_id)
    if player_color is not None:
        game.player_connected[player_color] = True

        # Cancel disconnect timer
        if user_id in game.disconnect_timers:
            game.disconnect_timers[user_id].cancel()
            del game.disconnect_timers[user_id]

        # Notify opponent
        await broadcast_online_event(game, "opponent_reconnected", {
            "user_id": user_id,
            "color": player_color,
        })


def extract_user_from_websocket(websocket: WebSocket) -> Optional[str]:
    """Extract user_id from WebSocket cookies (JWT auth or anonymous session)."""
    cookies = websocket.cookies
    token = cookies.get(AUTH_COOKIE_NAME)
    if token:
        user_id = decode_jwt_token(token)
        if user_id:
            return user_id
    # Fall back to anonymous cookie
    anon_id = cookies.get(ANON_COOKIE_NAME)
    if anon_id:
        return f"anon_{anon_id}"
    return None


@app.websocket("/games/{game_id}/ws")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game updates.

    For online games, authentication is required and only the current player can make moves.
    For local games (AI/PvP), authentication is optional.
    """
    if game_id not in games:
        await websocket.close(code=4004, reason="Game not found")
        return

    game = games[game_id]
    user_id = None

    # For online games, require authentication
    if game.is_online_game():
        await websocket.accept()
        user_id = extract_user_from_websocket(websocket)

        if not user_id:
            await send_error(websocket, "Authentication required for online games", "AUTH_REQUIRED")
            await websocket.close(code=4001, reason="Authentication required")
            return

        # Check if user is a player in this game
        player_color = game.get_player_color(user_id)
        if player_color is None:
            await send_error(websocket, "You are not a participant in this game", "NOT_AUTHORIZED")
            await websocket.close(code=4003, reason="Not authorized")
            return

        # Register connection
        game.player_websockets[user_id] = websocket
        game.player_connected[player_color] = True
        await handle_player_reconnect(game, user_id)
    else:
        await websocket.accept()

    game.websockets.append(websocket)

    # Send initial state with online game info
    initial_data = game.to_response().model_dump()
    if game.is_online_game():
        initial_data["online_status"] = game.online_status
        initial_data["your_color"] = game.get_player_color(user_id) if user_id else None
        initial_data["opponent_connected"] = game.player_connected[1 - game.get_player_color(user_id)] if user_id else None

    await websocket.send_json({
        "type": "state",
        "data": initial_data
    })

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "move":
                move = data.get("data", {}).get("move")
                if move is None:
                    await send_error(websocket, "Move not specified", "INVALID_REQUEST")
                    continue

                # For online games, check if it's this player's turn
                if game.is_online_game() and user_id:
                    if not game.can_user_move(user_id):
                        await send_error(websocket, "Not your turn", "NOT_YOUR_TURN")
                        continue

                if game.state.is_terminal():
                    await send_error(websocket, "Game already finished", "GAME_FINISHED")
                    continue

                legal_moves = get_legal_moves(game.state)
                if move not in legal_moves:
                    await send_error(websocket, "Invalid move", "INVALID_MOVE")
                    continue

                game.state.apply_move(move)
                persistence.save_game(game_id, game.state)  # Persist state
                persistence.append_move(game_id, move)  # Record move

                # Check for game over and update ELO
                game.check_and_update_elo()

                # Update online status if game finished
                if game.is_online_game() and game.state.is_terminal():
                    game.online_status = "finished"
                    persistence.update_online_game_status(game.game_id, "finished", game.state.get_winner())

                response = game.to_response()
                await broadcast_state(game, response)

                # Check for game over
                if game.state.is_terminal():
                    winner = game.state.get_winner()
                    reason = "ball_reached_goal" if winner is not None else "draw"
                    await broadcast_online_event(game, "game_over", {
                        "winner": winner,
                        "reason": reason
                    })

            elif msg_type == "ai_move":
                # AI moves not allowed in online games
                if game.is_online_game():
                    await send_error(websocket, "AI moves not allowed in online games", "NOT_ALLOWED")
                    continue

                simulations = data.get("data", {}).get("simulations", 800)

                if game.state.is_terminal():
                    await send_error(websocket, "Game already finished", "GAME_FINISHED")
                    continue

                # Mid-pass positions have narrow trees — use fewer sims
                sims = simulations
                if game.state.has_passed:
                    sims = max(50, sims // 10)

                # Run AI with batched search
                ev = get_evaluator()
                # Use temperature for opening diversity (first 2-3 moves = ~6 ply)
                temperature = 1.0 if game.state.ply < 6 else 0.0
                config = MCTSConfig(num_simulations=sims, temperature=temperature, batch_size=16)
                mcts = MCTS(ev, config)
                root = mcts.search_batched(game.state, add_noise=False)
                move = mcts.select_move(root)

                game.state.apply_move(move)
                persistence.save_game(game_id, game.state)  # Persist state
                persistence.append_move(game_id, move)  # Record move

                # Check for game over and update ELO
                game.check_and_update_elo()

                response = game.to_response()
                await broadcast_state(game, response)

                # Check for game over
                if game.state.is_terminal():
                    winner = game.state.get_winner()
                    reason = "ball_reached_goal" if winner is not None else "draw"
                    await websocket.send_json({
                        "type": "game_over",
                        "data": {"winner": winner, "reason": reason}
                    })

            elif msg_type == "undo":
                # Undo not allowed in online games
                if game.is_online_game():
                    await send_error(websocket, "Undo not allowed in online games", "NOT_ALLOWED")
                    continue

                if not game.state.history:
                    await send_error(websocket, "Nothing to undo", "INVALID_REQUEST")
                    continue

                game.state.undo_move()
                persistence.save_game(game_id, game.state)  # Persist state
                persistence.pop_move(game_id)  # Remove move from history
                response = game.to_response()
                await broadcast_state(game, response)

            elif msg_type == "ping":
                # Keep-alive ping
                await websocket.send_json({"type": "pong"})

            else:
                await send_error(websocket, f"Unknown message type: {msg_type}", "INVALID_REQUEST")

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in game.websockets:
            game.websockets.remove(websocket)

        # Handle online game disconnect
        if game.is_online_game() and user_id:
            if user_id in game.player_websockets:
                del game.player_websockets[user_id]
            await handle_player_disconnect(game, user_id)


# --- Entry Point ---

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
