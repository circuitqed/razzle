"""
Tests for training API endpoints.

These endpoints are used by:
- Self-play workers to submit games
- Trainer to fetch games and upload models
- Dashboard to monitor training progress
"""

import io
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import persistence
from server.main import app, TRAINING_MODELS_DIR


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        persistence.init_db(db_path)
        yield db_path


@pytest.fixture
def temp_models_dir(monkeypatch):
    """Create a temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir)
        monkeypatch.setattr("server.main.TRAINING_MODELS_DIR", models_dir)
        yield models_dir


@pytest.fixture
def client(temp_db, temp_models_dir, monkeypatch):
    """Create a test client with temporary database and models directory."""
    monkeypatch.setattr(persistence, 'DEFAULT_DB_PATH', temp_db)
    with TestClient(app) as client:
        yield client


class TestSubmitTrainingGame:
    """Tests for POST /training/games endpoint."""

    def test_submit_game_success(self, client):
        """Test submitting a valid training game."""
        response = client.post("/training/games", json={
            "worker_id": "worker_0",
            "moves": [100, 200, 300, 400],
            "result": 1.0,
            "visit_counts": [
                {"100": 50, "150": 30, "200": 20},
                {"200": 45, "250": 35, "300": 20},
                {"300": 60, "350": 25, "400": 15},
                {"400": 70, "450": 20, "500": 10},
            ],
            "model_version": "iter_001"
        })

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["id"] >= 1
        assert data["status"] == "accepted"

    def test_submit_game_minimal(self, client):
        """Test submitting a game with minimal data."""
        response = client.post("/training/games", json={
            "worker_id": "worker_1",
            "moves": [100],
            "result": 0.0,
            "visit_counts": [{"100": 100}],
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"

    def test_submit_game_negative_result(self, client):
        """Test submitting a game with P1 win (negative result)."""
        response = client.post("/training/games", json={
            "worker_id": "worker_0",
            "moves": [100, 200],
            "result": -1.0,
            "visit_counts": [{"100": 50}, {"200": 50}],
        })

        assert response.status_code == 200

    def test_submit_multiple_games(self, client):
        """Test submitting multiple games from same worker."""
        for i in range(5):
            response = client.post("/training/games", json={
                "worker_id": "worker_0",
                "moves": [100 + i],
                "result": 1.0 if i % 2 == 0 else -1.0,
                "visit_counts": [{"100": 50}],
            })
            assert response.status_code == 200

        # Verify all games are pending
        response = client.get("/training/games?mark_used=false")
        assert response.status_code == 200
        data = response.json()
        assert data["total_pending"] == 5

    def test_submit_games_different_workers(self, client):
        """Test submitting games from different workers."""
        for worker_id in ["worker_0", "worker_1", "worker_2"]:
            response = client.post("/training/games", json={
                "worker_id": worker_id,
                "moves": [100],
                "result": 1.0,
                "visit_counts": [{"100": 50}],
            })
            assert response.status_code == 200


class TestFetchTrainingGames:
    """Tests for GET /training/games endpoint."""

    def test_fetch_empty(self, client):
        """Test fetching when no games exist."""
        response = client.get("/training/games")

        assert response.status_code == 200
        data = response.json()
        assert data["games"] == []
        assert data["count"] == 0
        assert data["total_pending"] == 0

    def test_fetch_games(self, client):
        """Test fetching pending games."""
        # Submit some games first
        for i in range(3):
            client.post("/training/games", json={
                "worker_id": f"worker_{i}",
                "moves": [100, 200],
                "result": 1.0,
                "visit_counts": [{"100": 50}, {"200": 50}],
                "model_version": "iter_001"
            })

        # Fetch without marking used
        response = client.get("/training/games?mark_used=false")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert data["total_pending"] == 3
        assert len(data["games"]) == 3

        # Check game data structure
        game = data["games"][0]
        assert "id" in game
        assert "worker_id" in game
        assert "moves" in game
        assert "result" in game
        assert "visit_counts" in game
        assert "created_at" in game

    def test_fetch_marks_used(self, client):
        """Test that fetching marks games as used by default."""
        # Submit a game
        client.post("/training/games", json={
            "worker_id": "worker_0",
            "moves": [100],
            "result": 1.0,
            "visit_counts": [{"100": 50}],
        })

        # Fetch with mark_used=true (default)
        response1 = client.get("/training/games")
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["count"] == 1

        # Fetch again - should be empty now
        response2 = client.get("/training/games")
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["count"] == 0
        assert data2["total_pending"] == 0

    def test_fetch_with_limit(self, client):
        """Test fetching with a limit."""
        # Submit 5 games
        for i in range(5):
            client.post("/training/games", json={
                "worker_id": "worker_0",
                "moves": [100],
                "result": 1.0,
                "visit_counts": [{"100": 50}],
            })

        # Fetch only 2
        response = client.get("/training/games?limit=2&mark_used=false")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["total_pending"] == 5

    def test_fetch_invalid_status(self, client):
        """Test fetching with invalid status."""
        response = client.get("/training/games?status=invalid")
        assert response.status_code == 400


class TestTrainingModels:
    """Tests for model upload/download endpoints."""

    def test_get_latest_model_empty(self, client):
        """Test getting latest model when none exist."""
        response = client.get("/training/models/latest")

        assert response.status_code == 200
        data = response.json()
        assert data["model"] is None

    def test_upload_model(self, client, temp_models_dir):
        """Test uploading a model file."""
        # Create a dummy model file
        model_content = b"dummy pytorch model data"
        files = {"file": ("iter_001.pt", io.BytesIO(model_content), "application/octet-stream")}
        data = {
            "version": "iter_001",
            "iteration": "1",
            "games_trained_on": "100",
            "final_loss": "0.234",
        }

        response = client.post("/training/models", data=data, files=files)

        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["version"] == "iter_001"
        assert resp_data["status"] == "uploaded"

        # Verify file was saved
        assert (temp_models_dir / "iter_001.pt").exists()

    def test_upload_model_minimal(self, client, temp_models_dir):
        """Test uploading a model with minimal metadata."""
        model_content = b"dummy model"
        files = {"file": ("iter_002.pt", io.BytesIO(model_content), "application/octet-stream")}
        data = {
            "version": "iter_002",
            "iteration": "2",
        }

        response = client.post("/training/models", data=data, files=files)

        assert response.status_code == 200

    def test_upload_model_invalid_extension(self, client):
        """Test uploading a model with wrong file extension."""
        files = {"file": ("model.txt", io.BytesIO(b"not a model"), "text/plain")}
        data = {"version": "bad", "iteration": "1"}

        response = client.post("/training/models", data=data, files=files)

        assert response.status_code == 400
        assert "must be a .pt file" in response.json()["detail"]

    def test_get_latest_model_after_upload(self, client, temp_models_dir):
        """Test getting latest model after uploading."""
        # Upload a model
        files = {"file": ("iter_001.pt", io.BytesIO(b"model1"), "application/octet-stream")}
        data = {"version": "iter_001", "iteration": "1", "games_trained_on": "50"}
        client.post("/training/models", data=data, files=files)

        # Get latest
        response = client.get("/training/models/latest")

        assert response.status_code == 200
        model = response.json()["model"]
        assert model is not None
        assert model["version"] == "iter_001"
        assert model["iteration"] == 1
        assert model["games_trained_on"] == 50
        assert model["download_url"] == "/training/models/iter_001/download"

    def test_latest_model_is_highest_iteration(self, client, temp_models_dir):
        """Test that latest model returns highest iteration."""
        # Upload multiple models
        for i in [1, 3, 2]:  # Out of order
            files = {"file": (f"iter_{i:03d}.pt", io.BytesIO(b"model"), "application/octet-stream")}
            data = {"version": f"iter_{i:03d}", "iteration": str(i)}
            client.post("/training/models", data=data, files=files)

        # Latest should be iter_003
        response = client.get("/training/models/latest")
        model = response.json()["model"]
        assert model["version"] == "iter_003"
        assert model["iteration"] == 3

    def test_download_model(self, client, temp_models_dir):
        """Test downloading a model file."""
        # Upload a model
        model_content = b"pytorch model binary data here"
        files = {"file": ("iter_001.pt", io.BytesIO(model_content), "application/octet-stream")}
        data = {"version": "iter_001", "iteration": "1"}
        client.post("/training/models", data=data, files=files)

        # Download it
        response = client.get("/training/models/iter_001/download")

        assert response.status_code == 200
        assert response.content == model_content
        assert response.headers["content-type"] == "application/octet-stream"

    def test_download_nonexistent_model(self, client):
        """Test downloading a model that doesn't exist."""
        response = client.get("/training/models/nonexistent/download")
        assert response.status_code == 404


class TestTrainingDashboard:
    """Tests for GET /training/dashboard endpoint."""

    def test_dashboard_empty(self, client):
        """Test dashboard with no data."""
        response = client.get("/training/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
        assert data["games_pending"] == 0
        assert data["games_total"] == 0
        assert data["latest_model"] is None
        assert data["workers"] == {}
        assert data["models"] == []

    def test_dashboard_with_games(self, client):
        """Test dashboard after submitting games."""
        # Submit games from different workers
        for worker in ["worker_0", "worker_1"]:
            for i in range(3):
                client.post("/training/games", json={
                    "worker_id": worker,
                    "moves": [100],
                    "result": 1.0,
                    "visit_counts": [{"100": 50}],
                })

        response = client.get("/training/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert data["games_pending"] == 6
        assert data["games_total"] == 6
        assert len(data["workers"]) == 2
        assert data["workers"]["worker_0"]["games"] == 3
        assert data["workers"]["worker_1"]["games"] == 3

    def test_dashboard_with_models(self, client, temp_models_dir):
        """Test dashboard with uploaded models."""
        # Upload models
        for i in range(3):
            files = {"file": (f"iter_{i:03d}.pt", io.BytesIO(b"model"), "application/octet-stream")}
            data = {"version": f"iter_{i:03d}", "iteration": str(i), "final_loss": str(0.5 - i * 0.1)}
            client.post("/training/models", data=data, files=files)

        response = client.get("/training/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert data["latest_model"] is not None
        assert data["latest_model"]["version"] == "iter_002"
        assert len(data["models"]) == 3


class TestTrainingWorkflow:
    """Integration tests for the complete training workflow."""

    def test_worker_submits_trainer_fetches(self, client):
        """Test the worker -> API -> trainer workflow."""
        # Worker submits games
        game_ids = []
        for i in range(10):
            response = client.post("/training/games", json={
                "worker_id": "worker_0",
                "moves": [100 + i, 200 + i, 300 + i],
                "result": 1.0 if i % 2 == 0 else -1.0,
                "visit_counts": [
                    {str(100 + i): 50, str(150 + i): 30},
                    {str(200 + i): 40, str(250 + i): 40},
                    {str(300 + i): 60, str(350 + i): 20},
                ],
                "model_version": "iter_001"
            })
            game_ids.append(response.json()["id"])

        # Trainer checks for games
        response = client.get("/training/games?limit=5&mark_used=false")
        assert response.json()["total_pending"] == 10

        # Trainer fetches first batch
        response = client.get("/training/games?limit=5")
        data = response.json()
        assert data["count"] == 5
        assert data["total_pending"] == 5  # 5 remaining

        # Verify game data can be used for training
        for game in data["games"]:
            assert len(game["moves"]) == len(game["visit_counts"])
            assert game["result"] in [1.0, -1.0, 0.0]

        # Trainer fetches second batch
        response = client.get("/training/games?limit=5")
        data = response.json()
        assert data["count"] == 5
        assert data["total_pending"] == 0

    def test_model_upload_and_worker_download(self, client, temp_models_dir):
        """Test the trainer uploads model, worker downloads it."""
        # Trainer uploads new model
        model_data = b"trained neural network weights"
        files = {"file": ("iter_001.pt", io.BytesIO(model_data), "application/octet-stream")}
        data = {
            "version": "iter_001",
            "iteration": "1",
            "games_trained_on": "100",
            "final_loss": "0.234",
            "final_policy_loss": "0.180",
            "final_value_loss": "0.054",
        }
        response = client.post("/training/models", data=data, files=files)
        assert response.status_code == 200

        # Worker checks for new model
        response = client.get("/training/models/latest")
        model_info = response.json()["model"]
        assert model_info["version"] == "iter_001"
        assert model_info["final_loss"] == 0.234

        # Worker downloads the model
        response = client.get(model_info["download_url"])
        assert response.status_code == 200
        assert response.content == model_data

    def test_multiple_training_iterations(self, client, temp_models_dir):
        """Test multiple iterations of training."""
        for iteration in range(1, 4):
            # Workers submit games
            for i in range(5):
                client.post("/training/games", json={
                    "worker_id": f"worker_{i % 2}",
                    "moves": [100],
                    "result": 1.0,
                    "visit_counts": [{"100": 50}],
                    "model_version": f"iter_{iteration-1:03d}" if iteration > 1 else None,
                })

            # Trainer fetches and trains
            response = client.get("/training/games?limit=100")
            games = response.json()["games"]
            assert len(games) == 5

            # Trainer uploads new model
            files = {"file": (f"iter_{iteration:03d}.pt", io.BytesIO(b"model"), "application/octet-stream")}
            data = {
                "version": f"iter_{iteration:03d}",
                "iteration": str(iteration),
                "games_trained_on": "5",
                "final_loss": str(0.5 - iteration * 0.1),
            }
            client.post("/training/models", data=data, files=files)

        # Verify final state
        response = client.get("/training/dashboard")
        data = response.json()
        assert data["games_total"] == 15
        assert data["games_pending"] == 0  # All used
        assert len(data["models"]) == 3
        assert data["latest_model"]["version"] == "iter_003"
