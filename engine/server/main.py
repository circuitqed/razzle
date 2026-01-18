"""
FastAPI server for Razzle Dazzle game engine.

Provides REST and WebSocket APIs for game management and AI play.
"""

from __future__ import annotations
import asyncio
import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
            touched_mask=self.state.touched_mask
        )


# Global game storage (in production, use Redis or database)
games: dict[str, Game] = {}

# Global AI evaluator (lazy loaded)
evaluator: Optional[BatchedEvaluator] = None


def get_evaluator() -> BatchedEvaluator:
    """Get or create the AI evaluator."""
    global evaluator
    if evaluator is None:
        # Try to load trained model, fall back to random
        try:
            net = RazzleNet()
            evaluator = BatchedEvaluator(net, device='cpu')
        except Exception:
            evaluator = DummyEvaluator()
    return evaluator


# --- App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan handler."""
    # Startup: pre-load evaluator
    get_evaluator()
    yield
    # Shutdown: cleanup
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


# --- REST Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest = None):
    """Create a new game."""
    if request is None:
        request = CreateGameRequest()

    game_id = str(uuid.uuid4())[:8]
    game = Game(
        game_id=game_id,
        player1_type=request.player1_type,
        player2_type=request.player2_type,
        ai_simulations=request.ai_simulations
    )
    games[game_id] = game

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

    # Run MCTS
    start_time = time.time()

    ev = get_evaluator()
    config = MCTSConfig(
        num_simulations=request.simulations,
        temperature=request.temperature
    )
    mcts = MCTS(ev, config)

    # Run search (blocking for now, could be async)
    root = mcts.search(game.state, add_noise=False)
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

    response = game.to_response()
    await broadcast_state(game, response)

    return response


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

                # Run AI
                ev = get_evaluator()
                config = MCTSConfig(num_simulations=simulations, temperature=0.0)
                mcts = MCTS(ev, config)
                root = mcts.search(game.state, add_noise=False)
                move = mcts.select_move(root)

                game.state.apply_move(move)
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
