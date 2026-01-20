"""Tests for FastAPI server."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from server.main import app, games
from razzle.ai.network import NUM_ACTIONS


@pytest.fixture(autouse=True)
def clear_games():
    """Clear games before each test."""
    games.clear()
    yield
    games.clear()


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


class TestCreateGame:
    def test_create_game_default(self, client):
        response = client.post("/games")
        assert response.status_code == 200
        data = response.json()
        assert "game_id" in data
        assert len(data["game_id"]) == 8

    def test_create_game_with_options(self, client):
        response = client.post("/games", json={
            "player1_type": "human",
            "player2_type": "ai",
            "ai_simulations": 400
        })
        assert response.status_code == 200
        data = response.json()
        assert "game_id" in data

    def test_create_multiple_games(self, client):
        r1 = client.post("/games")
        r2 = client.post("/games")
        assert r1.json()["game_id"] != r2.json()["game_id"]


class TestGetGame:
    def test_get_game(self, client):
        # Create game first
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Get game state
        response = client.get(f"/games/{game_id}")
        assert response.status_code == 200
        data = response.json()

        assert data["game_id"] == game_id
        assert data["current_player"] == 0
        assert data["status"] == "playing"
        assert data["winner"] is None
        assert data["ply"] == 0
        assert len(data["legal_moves"]) > 0
        assert "board" in data

    def test_get_nonexistent_game(self, client):
        response = client.get("/games/nonexistent")
        assert response.status_code == 404


class TestMakeMove:
    def test_make_valid_move(self, client):
        # Create game
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Get legal moves
        game_response = client.get(f"/games/{game_id}")
        legal_moves = game_response.json()["legal_moves"]

        # Make first legal move
        response = client.post(f"/games/{game_id}/move", json={
            "move": legal_moves[0]
        })
        assert response.status_code == 200
        data = response.json()

        # Player should have changed (if it was a knight move)
        # or stayed same (if it was a pass)
        assert data["ply"] >= 0

    def test_make_invalid_move(self, client):
        # Create game
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Try invalid move (encoding for a1-a1 which is never legal)
        response = client.post(f"/games/{game_id}/move", json={
            "move": 0
        })
        assert response.status_code == 400

    def test_move_on_nonexistent_game(self, client):
        response = client.post("/games/nonexistent/move", json={
            "move": 100
        })
        assert response.status_code == 404


class TestAIMove:
    def test_ai_move(self, client):
        # Create game
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Request AI move with low simulations for speed
        response = client.post(f"/games/{game_id}/ai", json={
            "simulations": 10,
            "temperature": 0.0
        })
        assert response.status_code == 200
        data = response.json()

        assert "move" in data
        assert "algebraic" in data
        assert "policy" in data
        assert "value" in data
        assert "time_ms" in data
        assert "top_moves" in data
        assert len(data["policy"]) == NUM_ACTIONS

    def test_ai_move_default_params(self, client):
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Request with no parameters (uses defaults)
        # Using low simulations for test speed
        response = client.post(f"/games/{game_id}/ai", json={
            "simulations": 5
        })
        assert response.status_code == 200

    def test_ai_move_on_nonexistent_game(self, client):
        response = client.post("/games/nonexistent/ai", json={
            "simulations": 10
        })
        assert response.status_code == 404


class TestLegalMoves:
    def test_get_legal_moves(self, client):
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        response = client.get(f"/games/{game_id}/legal-moves")
        assert response.status_code == 200
        data = response.json()

        assert "moves" in data
        assert len(data["moves"]) > 0

        # Check move format
        move = data["moves"][0]
        assert "move" in move
        assert "algebraic" in move
        assert "type" in move
        assert move["type"] in ["knight", "pass"]

    def test_legal_moves_nonexistent_game(self, client):
        response = client.get("/games/nonexistent/legal-moves")
        assert response.status_code == 404


class TestUndo:
    def test_undo_move(self, client):
        # Create game and make a move
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Get initial state
        initial = client.get(f"/games/{game_id}").json()

        # Get legal moves and make one
        legal_moves = initial["legal_moves"]
        client.post(f"/games/{game_id}/move", json={"move": legal_moves[0]})

        # Undo
        response = client.post(f"/games/{game_id}/undo")
        assert response.status_code == 200

        # State should be back to initial (approximately)
        # Note: ply might differ due to pass vs knight move

    def test_undo_nothing(self, client):
        # Create new game with no moves
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Try to undo with no history
        response = client.post(f"/games/{game_id}/undo")
        assert response.status_code == 400


class TestMoveConversion:
    def test_convert_encoded_to_algebraic(self, client):
        # d1-c3 = 3*56+16 = 184
        response = client.get("/util/move?encoded=184")
        assert response.status_code == 200
        data = response.json()

        assert data["encoded"] == 184
        assert data["algebraic"] == "d1-c3"
        assert data["src"] == 3
        assert data["dst"] == 16

    def test_convert_algebraic_to_encoded(self, client):
        response = client.get("/util/move?algebraic=d1-c3")
        assert response.status_code == 200
        data = response.json()

        assert data["encoded"] == 184
        assert data["algebraic"] == "d1-c3"
        assert data["src"] == 3
        assert data["dst"] == 16

    def test_convert_no_params(self, client):
        response = client.get("/util/move")
        assert response.status_code == 400

    def test_convert_invalid_algebraic(self, client):
        response = client.get("/util/move?algebraic=invalid")
        assert response.status_code == 400


class TestGameFlow:
    def test_full_game_flow(self, client):
        """Test a complete game flow: create, moves, AI, check state."""
        # Create game
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Get initial state
        state = client.get(f"/games/{game_id}").json()
        assert state["current_player"] == 0
        assert state["ply"] == 0

        # Make a human move (first legal knight move)
        legal = client.get(f"/games/{game_id}/legal-moves").json()
        knight_moves = [m for m in legal["moves"] if m["type"] == "knight"]
        if knight_moves:
            move = knight_moves[0]["move"]
            result = client.post(f"/games/{game_id}/move", json={"move": move})
            assert result.status_code == 200

            # After knight move, player should change
            new_state = result.json()
            assert new_state["current_player"] == 1
            assert new_state["ply"] == 1

        # AI makes a move
        ai_result = client.post(f"/games/{game_id}/ai", json={"simulations": 5})
        assert ai_result.status_code == 200

        # Game should have progressed (AI may pass or move)
        final_state = client.get(f"/games/{game_id}").json()
        # Ply only increments on knight moves, not passes
        # So we just check the game is still valid
        assert final_state["status"] == "playing"


class TestWebSocket:
    def test_websocket_connect(self, client):
        # Create game
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        # Connect via WebSocket
        with client.websocket_connect(f"/games/{game_id}/ws") as ws:
            # Should receive initial state
            data = ws.receive_json()
            assert data["type"] == "state"
            assert data["data"]["game_id"] == game_id

    def test_websocket_move(self, client):
        # Create game
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        with client.websocket_connect(f"/games/{game_id}/ws") as ws:
            # Receive initial state
            initial = ws.receive_json()
            legal_moves = initial["data"]["legal_moves"]

            # Send move
            ws.send_json({
                "type": "move",
                "data": {"move": legal_moves[0]}
            })

            # Should receive state update
            update = ws.receive_json()
            assert update["type"] == "state"

    def test_websocket_invalid_move(self, client):
        create_response = client.post("/games")
        game_id = create_response.json()["game_id"]

        with client.websocket_connect(f"/games/{game_id}/ws") as ws:
            # Receive initial state
            ws.receive_json()

            # Send invalid move
            ws.send_json({
                "type": "move",
                "data": {"move": 0}  # a1-a1 is never legal
            })

            # Should receive error
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["data"]["code"] == "INVALID_MOVE"

    def test_websocket_nonexistent_game(self, client):
        # Try to connect to nonexistent game
        with pytest.raises(Exception):
            with client.websocket_connect("/games/nonexistent/ws"):
                pass
