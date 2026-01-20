"""
HTTP client for training API.

Used by workers to submit games and check for model updates,
and by trainer to fetch games and upload models.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a training model."""
    version: str
    iteration: int
    download_url: str
    created_at: str
    games_trained_on: Optional[int] = None
    final_loss: Optional[float] = None
    final_policy_loss: Optional[float] = None
    final_value_loss: Optional[float] = None


@dataclass
class TrainingGame:
    """A training game fetched from the server."""
    id: int
    worker_id: str
    moves: list[int]
    result: float
    visit_counts: list[dict[int, int]]
    model_version: Optional[str]
    created_at: str


class TrainingAPIClient:
    """Client for the training API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the API client.

        Args:
            base_url: Server URL (default: from RAZZLE_API_URL env var or http://localhost:8000)
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
        """
        self.base_url = base_url or os.environ.get("RAZZLE_API_URL", "http://localhost:8000")
        self.timeout = timeout

        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _url(self, path: str) -> str:
        """Build full URL for a path."""
        return urljoin(self.base_url, path)

    # --- Worker Methods ---

    def submit_game(
        self,
        worker_id: str,
        moves: list[int],
        result: float,
        visit_counts: list[dict[int, int]],
        model_version: Optional[str] = None,
    ) -> int:
        """
        Submit a completed training game.

        Args:
            worker_id: Identifier for this worker
            moves: List of moves played
            result: Game result (1.0 for P0 win, -1.0 for P1 win, 0.0 for draw)
            visit_counts: MCTS visit counts per position (sparse: {move: count})
            model_version: Version of model used to generate this game

        Returns:
            The game ID assigned by the server
        """
        # Convert int keys to strings for JSON
        visit_counts_json = [
            {str(k): v for k, v in vc.items()}
            for vc in visit_counts
        ]

        response = self.session.post(
            self._url("/training/games"),
            json={
                "worker_id": worker_id,
                "moves": moves,
                "result": result,
                "visit_counts": visit_counts_json,
                "model_version": model_version,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["id"]

    def get_latest_model(self) -> Optional[ModelInfo]:
        """
        Check for the latest model.

        Returns:
            ModelInfo if a model exists, None otherwise
        """
        response = self.session.get(
            self._url("/training/models/latest"),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("model") is None:
            return None

        m = data["model"]
        return ModelInfo(
            version=m["version"],
            iteration=m["iteration"],
            download_url=m["download_url"],
            created_at=m["created_at"],
            games_trained_on=m.get("games_trained_on"),
            final_loss=m.get("final_loss"),
            final_policy_loss=m.get("final_policy_loss"),
            final_value_loss=m.get("final_value_loss"),
        )

    def download_model(self, version: str, dest_path: Path) -> Path:
        """
        Download a model file.

        Args:
            version: Model version to download
            dest_path: Where to save the model file

        Returns:
            Path to the downloaded file
        """
        response = self.session.get(
            self._url(f"/training/models/{version}/download"),
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return dest_path

    # --- Trainer Methods ---

    def fetch_pending_games(
        self,
        limit: int = 100,
        mark_used: bool = True,
    ) -> tuple[list[TrainingGame], int]:
        """
        Fetch pending training games.

        Args:
            limit: Maximum number of games to fetch
            mark_used: If True, mark fetched games as used

        Returns:
            Tuple of (list of games, total pending count)
        """
        response = self.session.get(
            self._url("/training/games"),
            params={
                "status": "pending",
                "limit": limit,
                "mark_used": str(mark_used).lower(),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        games = []
        for g in data["games"]:
            # Convert string keys back to ints
            visit_counts = [
                {int(k): v for k, v in vc.items()}
                for vc in g["visit_counts"]
            ]
            games.append(TrainingGame(
                id=g["id"],
                worker_id=g["worker_id"],
                moves=g["moves"],
                result=g["result"],
                visit_counts=visit_counts,
                model_version=g.get("model_version"),
                created_at=g["created_at"],
            ))

        return games, data["total_pending"]

    def upload_model(
        self,
        version: str,
        iteration: int,
        file_path: Path,
        games_trained_on: Optional[int] = None,
        final_loss: Optional[float] = None,
        final_policy_loss: Optional[float] = None,
        final_value_loss: Optional[float] = None,
    ) -> None:
        """
        Upload a trained model.

        Args:
            version: Version string for this model
            iteration: Training iteration number
            file_path: Path to the .pt model file
            games_trained_on: Number of games used for training
            final_loss: Final training loss
            final_policy_loss: Final policy loss
            final_value_loss: Final value loss
        """
        with open(file_path, "rb") as f:
            files = {"file": (f"{version}.pt", f, "application/octet-stream")}
            data = {
                "version": version,
                "iteration": str(iteration),
            }
            if games_trained_on is not None:
                data["games_trained_on"] = str(games_trained_on)
            if final_loss is not None:
                data["final_loss"] = str(final_loss)
            if final_policy_loss is not None:
                data["final_policy_loss"] = str(final_policy_loss)
            if final_value_loss is not None:
                data["final_value_loss"] = str(final_value_loss)

            response = self.session.post(
                self._url("/training/models"),
                data=data,
                files=files,
                timeout=self.timeout * 2,  # Longer timeout for upload
            )
            response.raise_for_status()

    def get_dashboard(self) -> dict:
        """
        Get training dashboard data.

        Returns:
            Dashboard data dict
        """
        response = self.session.get(
            self._url("/training/dashboard"),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # --- Health Check ---

    def health_check(self) -> bool:
        """
        Check if the server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = self.session.get(
                self._url("/health"),
                timeout=5.0,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_for_server(self, timeout: float = 60.0, poll_interval: float = 2.0) -> bool:
        """
        Wait for the server to become available.

        Args:
            timeout: Maximum time to wait
            poll_interval: Time between checks

        Returns:
            True if server became available, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.health_check():
                return True
            time.sleep(poll_interval)
        return False
