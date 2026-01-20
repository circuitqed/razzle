"""
FastAPI server for Razzle Dazzle game engine.

Provides REST and WebSocket APIs for game management and AI play.
"""

from __future__ import annotations
import asyncio
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import jwt
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Response, Depends, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from razzle.core.state import GameState
from razzle.core.moves import (
    get_legal_moves, move_to_algebraic, algebraic_to_move,
    decode_move, encode_move
)
from razzle.core.bitboard import algebraic_to_sq, sq_to_algebraic
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator, DummyEvaluator
from razzle.ai.network import RazzleNet

from . import persistence


# --- Pydantic Models ---

class CreateGameRequest(BaseModel):
    player1_type: str = "human"
    player2_type: str = "ai"
    ai_simulations: int = 800


class CreateGameResponse(BaseModel):
    game_id: str


class BoardState(BaseModel):
    p1_pieces: int
    p1_ball: int
    p2_pieces: int
    p2_ball: int


class GameStateResponse(BaseModel):
    game_id: str
    board: BoardState
    current_player: int
    legal_moves: list[int]
    status: str
    winner: Optional[int]
    ply: int
    touched_mask: int  # Bitboard of ineligible receivers
    has_passed: bool  # Whether a pass has been made this turn


class MakeMoveRequest(BaseModel):
    move: int


class AIMoveRequest(BaseModel):
    simulations: int = 800
    temperature: float = 0.0


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


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    data: Optional[dict] = None


class LogRequest(BaseModel):
    entries: list[LogEntry]
    session_id: Optional[str] = None


# --- Game Storage ---

class Game:
    """Represents an active game session."""

    def __init__(
        self,
        game_id: str,
        player1_type: str = "human",
        player2_type: str = "ai",
        ai_simulations: int = 800
    ):
        self.game_id = game_id
        self.state = GameState.new_game()
        self.player_types = [player1_type, player2_type]
        self.ai_simulations = ai_simulations
        self.websockets: list[WebSocket] = []

    def to_response(self) -> GameStateResponse:
        """Convert to API response."""
        if self.state.is_terminal():
            status = "finished"
        else:
            status = "playing"

        return GameStateResponse(
            game_id=self.game_id,
            board=BoardState(
                p1_pieces=self.state.pieces[0],
                p1_ball=self.state.balls[0],
                p2_pieces=self.state.pieces[1],
                p2_ball=self.state.balls[1]
            ),
            current_player=self.state.current_player,
            legal_moves=get_legal_moves(self.state),
            status=status,
            winner=self.state.get_winner(),
            ply=self.state.ply,
            touched_mask=self.state.touched_mask,
            has_passed=self.state.has_passed
        )


# Global game storage (in production, use Redis or database)
games: dict[str, Game] = {}

# Global AI evaluator (lazy loaded)
evaluator: Optional[BatchedEvaluator] = None
_model_path_used: Optional[str] = None  # Track which model is loaded
_model_mtime: float = 0  # Track model file modification time

# Directories to search for models (in priority order)
MODEL_SEARCH_DIRS = [
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
        # Look for model files (model_iter_*.pt or trained_model.pt)
        for pattern in ["model_iter_*.pt", "trained_model.pt"]:
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
            ai_simulations=game_data["ai_simulations"]
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

    game_id = str(uuid.uuid4())[:8]
    game = Game(
        game_id=game_id,
        player1_type=request.player1_type,
        player2_type=request.player2_type,
        ai_simulations=request.ai_simulations
    )
    games[game_id] = game

    # Get current model version
    model_version = get_model_info()

    # Persist to database
    persistence.save_game(
        game_id=game_id,
        state=game.state,
        player1_type=request.player1_type,
        player2_type=request.player2_type,
        ai_simulations=request.ai_simulations,
        player1_user_id=player1_user_id,
        ai_model_version=model_version if request.player2_type == "ai" else None,
    )

    return CreateGameResponse(game_id=game_id)


@app.get("/games/{game_id}", response_model=GameStateResponse)
async def get_game(game_id: str):
    """Get current game state."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    return games[game_id].to_response()


@app.post("/games/{game_id}/move", response_model=GameStateResponse)
async def make_move(game_id: str, request: MakeMoveRequest):
    """Make a move in the game."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]

    if game.state.is_terminal():
        raise HTTPException(status_code=409, detail="Game already finished")

    legal_moves = get_legal_moves(game.state)
    if request.move not in legal_moves:
        raise HTTPException(status_code=400, detail="Invalid move")

    game.state.apply_move(request.move)

    # Persist state and record move
    persistence.save_game(game_id, game.state)
    persistence.append_move(game_id, request.move)

    # Notify WebSocket clients
    response = game.to_response()
    await broadcast_state(game, response)

    return response


@app.post("/games/{game_id}/ai", response_model=AIMoveResponse)
async def get_ai_move(game_id: str, request: AIMoveRequest = None):
    """Get AI to calculate and play a move."""
    if request is None:
        request = AIMoveRequest()

    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]

    if game.state.is_terminal():
        raise HTTPException(status_code=409, detail="Game already finished")

    # Run MCTS with batched search for better performance
    start_time = time.time()

    ev = get_evaluator()
    config = MCTSConfig(
        num_simulations=request.simulations,
        temperature=request.temperature,
        batch_size=16  # Use batched inference
    )
    mcts = MCTS(ev, config)

    # Run batched search for better GPU utilization
    root = mcts.search_batched(game.state, add_noise=False)
    move = mcts.select_move(root)
    policy = mcts.get_policy(root)

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

    # Apply the move
    game.state.apply_move(move)

    # Persist state and record move
    persistence.save_game(game_id, game.state)
    persistence.append_move(game_id, move)

    # Notify WebSocket clients
    game_response = game.to_response()
    await broadcast_state(game, game_response)

    return AIMoveResponse(
        move=move,
        algebraic=move_to_algebraic(move),
        policy=policy.tolist(),
        value=root.value,
        visits=root.visit_count,
        time_ms=elapsed_ms,
        top_moves=top_moves,
        game_state=game_response
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
    pieces: list[int]  # [p1_pieces, p2_pieces]
    balls: list[int]   # [p1_ball, p2_ball]
    current_player: int
    touched_mask: int = 0
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

    # Reconstruct game state
    state = GameState(
        pieces=request.pieces,
        balls=request.balls,
        current_player=request.current_player,
        touched_mask=request.touched_mask,
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


@app.websocket("/games/{game_id}/ws")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game updates."""
    if game_id not in games:
        await websocket.close(code=4004, reason="Game not found")
        return

    game = games[game_id]
    await websocket.accept()
    game.websockets.append(websocket)

    # Send initial state
    await websocket.send_json({
        "type": "state",
        "data": game.to_response().model_dump()
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

            elif msg_type == "ai_move":
                simulations = data.get("data", {}).get("simulations", 800)

                if game.state.is_terminal():
                    await send_error(websocket, "Game already finished", "GAME_FINISHED")
                    continue

                # Run AI with batched search
                ev = get_evaluator()
                config = MCTSConfig(num_simulations=simulations, temperature=0.0, batch_size=16)
                mcts = MCTS(ev, config)
                root = mcts.search_batched(game.state, add_noise=False)
                move = mcts.select_move(root)

                game.state.apply_move(move)
                persistence.save_game(game_id, game.state)  # Persist state
                persistence.append_move(game_id, move)  # Record move
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
                if not game.state.history:
                    await send_error(websocket, "Nothing to undo", "INVALID_REQUEST")
                    continue

                game.state.undo_move()
                persistence.save_game(game_id, game.state)  # Persist state
                persistence.pop_move(game_id)  # Remove move from history
                response = game.to_response()
                await broadcast_state(game, response)

            else:
                await send_error(websocket, f"Unknown message type: {msg_type}", "INVALID_REQUEST")

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in game.websockets:
            game.websockets.remove(websocket)


# --- Entry Point ---

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
