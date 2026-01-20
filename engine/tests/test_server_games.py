"""
Tests for game browser and analysis endpoints.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import persistence
from server.main import app
from razzle.core.state import GameState


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        persistence.init_db(db_path)
        yield db_path


@pytest.fixture
def client(temp_db, monkeypatch):
    """Create a test client with temporary database."""
    # Override the default db path before any database operations
    monkeypatch.setattr(persistence, 'DEFAULT_DB_PATH', temp_db)

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_games(temp_db, monkeypatch):
    """Create some sample games in the database."""
    # Ensure persistence uses the temp_db
    monkeypatch.setattr(persistence, 'DEFAULT_DB_PATH', temp_db)

    state = GameState.new_game()

    # Create a few games
    games = []
    for i in range(5):
        game_id = f"game{i:03d}"
        player2_type = "ai" if i % 2 == 0 else "human"
        persistence.save_game(
            game_id=game_id,
            state=state,
            player1_type="human",
            player2_type=player2_type,
            moves=[100, 200, 300],  # Dummy moves
            player1_user_id=f"user{i % 2}" if i < 3 else None,
        )
        games.append(game_id)

    return games


class TestGameListing:
    """Tests for game listing endpoint."""

    def test_list_games_empty(self, client):
        """Test listing games when none exist."""
        response = client.get('/games')

        assert response.status_code == 200
        data = response.json()
        assert data['games'] == []
        assert data['total'] == 0
        assert data['page'] == 1

    def test_list_games_with_data(self, client, sample_games):
        """Test listing games with existing games."""
        response = client.get('/games')

        assert response.status_code == 200
        data = response.json()
        assert len(data['games']) == 5
        assert data['total'] == 5

    def test_list_games_pagination(self, client, sample_games):
        """Test game listing pagination."""
        response = client.get('/games?page=1&per_page=2')

        assert response.status_code == 200
        data = response.json()
        assert len(data['games']) == 2
        assert data['total'] == 5
        assert data['page'] == 1
        assert data['per_page'] == 2
        assert data['total_pages'] == 3

    def test_list_games_filter_by_status(self, client, sample_games, temp_db):
        """Test filtering games by status."""
        # All sample games are 'playing' (not terminal)
        response = client.get('/games?status=playing')

        assert response.status_code == 200
        data = response.json()
        assert len(data['games']) > 0
        for game in data['games']:
            assert game['status'] == 'playing'


class TestGameFull:
    """Tests for getting full game data."""

    def test_get_game_full(self, client, sample_games):
        """Test getting full game data."""
        response = client.get(f'/games/{sample_games[0]}/full')

        assert response.status_code == 200
        data = response.json()
        assert data['game_id'] == sample_games[0]
        assert 'moves' in data
        assert 'moves_algebraic' in data
        assert isinstance(data['moves'], list)
        assert len(data['moves']) == 3  # We saved 3 dummy moves

    def test_get_game_full_not_found(self, client):
        """Test getting non-existent game."""
        response = client.get('/games/nonexistent/full')

        assert response.status_code == 404


class TestGameCreationWithUser:
    """Tests for game creation with user association."""

    def test_create_game_anonymous(self, client):
        """Test creating game without authentication."""
        response = client.post('/games', json={
            'player1_type': 'human',
            'player2_type': 'ai',
            'ai_simulations': 400
        })

        assert response.status_code == 200
        data = response.json()
        assert 'game_id' in data

    def test_create_game_authenticated(self, client):
        """Test creating game with authentication."""
        # Register and login
        client.post('/auth/register', json={
            'username': 'gameuser',
            'password': 'password123'
        })

        # Create game
        response = client.post('/games', json={
            'player1_type': 'human',
            'player2_type': 'ai'
        })

        assert response.status_code == 200


class TestAnalyzePosition:
    """Tests for position analysis endpoint."""

    def test_analyze_starting_position(self, client):
        """Test analyzing the starting position."""
        state = GameState.new_game()

        response = client.post('/analyze', json={
            'pieces': list(state.pieces),
            'balls': list(state.balls),
            'current_player': state.current_player,
            'touched_mask': state.touched_mask,
            'has_passed': state.has_passed,
            'simulations': 50  # Low for fast testing
        })

        assert response.status_code == 200
        data = response.json()
        assert 'value' in data
        assert 'top_moves' in data
        assert 'legal_moves' in data
        assert isinstance(data['top_moves'], list)
        assert -1 <= data['value'] <= 1

    def test_analyze_with_low_simulations(self, client):
        """Test analysis with minimal simulations."""
        state = GameState.new_game()

        response = client.post('/analyze', json={
            'pieces': list(state.pieces),
            'balls': list(state.balls),
            'current_player': 0,
            'simulations': 10
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data['top_moves']) > 0


class TestAnalyzeGame:
    """Tests for game analysis endpoint."""

    def test_analyze_game_with_moves(self, client, temp_db):
        """Test analyzing a game with moves."""
        # Create a game with some moves
        from razzle.core.moves import get_legal_moves

        state = GameState.new_game()
        moves = []

        # Make a few moves
        for _ in range(3):
            legal = get_legal_moves(state)
            if not legal or state.is_terminal():
                break
            move = legal[0]  # Just pick first legal move
            moves.append(move)
            state.apply_move(move)

        # Save the game
        persistence.save_game(
            game_id="analyzed_game",
            state=state,
            moves=moves,
            db_path=temp_db
        )

        # Analyze it
        response = client.post('/games/analyzed_game/analyze?simulations_per_position=20')

        assert response.status_code == 200
        data = response.json()
        assert data['game_id'] == 'analyzed_game'
        assert 'move_analyses' in data
        assert 'summary' in data
        assert len(data['move_analyses']) == len(moves)

    def test_analyze_empty_game(self, client, temp_db):
        """Test analyzing a game with no moves."""
        state = GameState.new_game()
        persistence.save_game(
            game_id="empty_game",
            state=state,
            moves=[],
            db_path=temp_db
        )

        response = client.post('/games/empty_game/analyze')

        assert response.status_code == 200
        data = response.json()
        assert data['move_analyses'] == []

    def test_analyze_nonexistent_game(self, client):
        """Test analyzing a game that doesn't exist."""
        response = client.post('/games/nonexistent/analyze')

        assert response.status_code == 404


class TestMoveClassification:
    """Tests for move classification logic."""

    def test_classification_thresholds(self):
        """Test that classification thresholds are correct."""
        from server.main import classify_move

        assert classify_move(0.0) == 'best'
        assert classify_move(-0.01) == 'best'
        assert classify_move(-0.02) == 'best'
        assert classify_move(-0.03) == 'good'
        assert classify_move(-0.08) == 'good'
        assert classify_move(-0.09) == 'inaccuracy'
        assert classify_move(-0.15) == 'inaccuracy'
        assert classify_move(-0.16) == 'mistake'
        assert classify_move(-0.30) == 'mistake'
        assert classify_move(-0.31) == 'blunder'
        assert classify_move(-0.50) == 'blunder'
