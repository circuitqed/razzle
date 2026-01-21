#!/usr/bin/env python3
"""
Distributed self-play worker for Razzle Dazzle training.

This script runs on cloud GPU instances (Vast.ai) and:
1. Plays games continuously using MCTS + neural network
2. Submits completed games to the training API
3. Writes status updates to status.json for monitoring
4. Periodically checks for new model weights via API and hot-reloads

Usage:
    python worker_selfplay.py --worker-id 0 --api-url http://server:8000 --simulations 400

Status file (status.json):
    {
        "worker_id": 0,
        "status": "running",
        "games_completed": 42,
        "current_game_moves": 73,
        "model_version": "iter_002",
        "last_update": "2024-01-20T15:30:45",
        "uptime_sec": 3600,
        "games_per_hour": 12.5
    }
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet, create_network
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator
from razzle.core.state import GameState
from razzle.training.api_client import TrainingAPIClient


@dataclass
class WorkerStatus:
    """Status information for monitoring."""
    worker_id: int
    status: str  # starting, running, paused, stopped, error
    games_completed: int
    games_submitted: int
    current_game_moves: int
    model_version: str
    last_update: str
    start_time: str
    uptime_sec: float
    games_per_hour: float
    error_message: Optional[str] = None
    gpu_name: str = ""
    simulations: int = 0
    api_url: str = ""


class SelfPlayWorker:
    """
    Continuous self-play worker that submits games via API.

    Generates games independently and submits them to the training server.
    """

    def __init__(
        self,
        worker_id: int,
        api_url: str,
        workspace: Path,
        device: str = 'cuda',
        simulations: int = 400,
        temperature_moves: int = 30,
        filters: int = 64,
        blocks: int = 6,
        batch_size: int = 32,
        model_check_interval: int = 5,  # Check for new model every N games
    ):
        self.worker_id = worker_id
        self.api_url = api_url
        self.workspace = Path(workspace)
        self.device = device
        self.simulations = simulations
        self.temperature_moves = temperature_moves
        self.filters = filters
        self.blocks = blocks
        self.batch_size = batch_size
        self.model_check_interval = model_check_interval

        # Directories
        self.model_dir = self.workspace / "model"
        self.status_file = self.workspace / "status.json"

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # API client
        self.api_client = TrainingAPIClient(base_url=api_url)

        # State
        self.network: Optional[RazzleNet] = None
        self.evaluator = None
        self.model_version = "none"
        self.games_completed = 0
        self.games_submitted = 0
        self.start_time = datetime.now()
        self.current_game_moves = 0
        self.shutdown_event = Event()
        self.status = "starting"
        self.error_message = None

        # GPU info
        self.gpu_name = self._get_gpu_name()

        # Status writer thread
        self._status_thread: Optional[Thread] = None

    def _get_gpu_name(self) -> str:
        """Get GPU name if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except:
            pass
        return "unknown"

    def _download_latest_model(self) -> bool:
        """Download the latest model from the API."""
        try:
            model_info = self.api_client.get_latest_model()
            if model_info is None:
                return False

            if model_info.version == self.model_version:
                return False  # Already have this version

            # Download model
            model_path = self.model_dir / f"{model_info.version}.pt"
            print(f"[Worker {self.worker_id}] Downloading model: {model_info.version}")
            self.api_client.download_model(model_info.version, model_path)

            # Load model
            self.network = RazzleNet.load(model_path, device=self.device)
            self.evaluator = BatchedEvaluator(
                self.network,
                batch_size=self.batch_size,
                device=self.device
            )
            self.model_version = model_info.version
            print(f"[Worker {self.worker_id}] Loaded model: {self.model_version}")
            return True

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error downloading model: {e}")
            return False

    def _load_or_create_network(self) -> bool:
        """Load model from API or create new one."""
        try:
            # First try to download from API
            if self._download_latest_model():
                return True

            # Check local model directory
            model_files = sorted(self.model_dir.glob("*.pt"))
            if model_files:
                latest = model_files[-1]
                self.network = RazzleNet.load(latest, device=self.device)
                self.model_version = latest.stem
                print(f"[Worker {self.worker_id}] Loaded local model: {self.model_version}")
            else:
                # Create new network
                self.network = create_network(self.filters, self.blocks, self.device)
                self.model_version = "initial"
                print(f"[Worker {self.worker_id}] Created new network")

            # Create evaluator
            self.evaluator = BatchedEvaluator(
                self.network,
                batch_size=self.batch_size,
                device=self.device
            )
            return True

        except Exception as e:
            self.error_message = f"Failed to load network: {e}"
            print(f"[Worker {self.worker_id}] {self.error_message}")
            return False

    def _check_for_new_model(self) -> bool:
        """Check if a new model is available via API."""
        try:
            model_info = self.api_client.get_latest_model()
            if model_info is None:
                return False

            if model_info.version != self.model_version:
                return self._download_latest_model()

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error checking for new model: {e}")

        return False

    def _get_status(self) -> WorkerStatus:
        """Get current worker status."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        games_per_hour = (self.games_completed / uptime * 3600) if uptime > 0 else 0.0

        return WorkerStatus(
            worker_id=self.worker_id,
            status=self.status,
            games_completed=self.games_completed,
            games_submitted=self.games_submitted,
            current_game_moves=self.current_game_moves,
            model_version=self.model_version,
            last_update=datetime.now().isoformat(),
            start_time=self.start_time.isoformat(),
            uptime_sec=uptime,
            games_per_hour=games_per_hour,
            error_message=self.error_message,
            gpu_name=self.gpu_name,
            simulations=self.simulations,
            api_url=self.api_url,
        )

    def _write_status(self):
        """Write status to file."""
        status = self._get_status()

        # Write atomically
        tmp_path = self.status_file.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(asdict(status), f, indent=2)
        tmp_path.rename(self.status_file)

    def _status_writer_loop(self):
        """Background thread to periodically write status."""
        while not self.shutdown_event.is_set():
            try:
                self._write_status()
            except Exception as e:
                print(f"[Worker {self.worker_id}] Error writing status: {e}")

            # Wait 5 seconds or until shutdown
            self.shutdown_event.wait(5.0)

    def _get_ball_row(self, state: GameState, player: int) -> int:
        """Get the row (0-7) of the specified player's ball."""
        for sq in range(56):
            if state.balls[player] & (1 << sq):
                return sq // 7
        return 0

    def play_one_game(self) -> tuple[list[int], float, list[dict[int, int]]]:
        """
        Play a single self-play game.

        Returns:
            Tuple of (moves, result, visit_counts)
        """
        state = GameState.new_game()
        moves = []
        visit_counts = []

        move_count = 0
        self.current_game_moves = 0

        while not state.is_terminal() and move_count < 300:
            # Configure MCTS
            temp = 1.0 if move_count < self.temperature_moves else 0.0
            config = MCTSConfig(
                num_simulations=self.simulations,
                temperature=temp,
                batch_size=self.batch_size
            )
            mcts = MCTS(self.evaluator, config)

            # Search
            root = mcts.search_batched(state, add_noise=True)

            # Record visit counts (sparse - only visited moves)
            vc = {}
            for move, child in root.children.items():
                if child.visit_count > 0:
                    vc[move] = child.visit_count
            visit_counts.append(vc)

            # Select and apply move
            move = mcts.select_move(root)
            moves.append(move)
            state.apply_move(move)

            move_count += 1
            self.current_game_moves = move_count

            # Check for shutdown
            if self.shutdown_event.is_set():
                break

        # Determine result
        winner = state.get_winner()
        if winner == 0:
            result = 1.0
        elif winner == 1:
            result = -1.0
        else:
            result = 0.0

        return moves, result, visit_counts

    def submit_game(self, moves: list[int], result: float, visit_counts: list[dict[int, int]]) -> bool:
        """Submit a game to the API."""
        try:
            game_id = self.api_client.submit_game(
                worker_id=f"worker_{self.worker_id}",
                moves=moves,
                result=result,
                visit_counts=visit_counts,
                model_version=self.model_version,
            )
            self.games_submitted += 1
            return True
        except Exception as e:
            print(f"[Worker {self.worker_id}] Failed to submit game: {e}")
            return False

    def run(self):
        """Main worker loop."""
        print(f"[Worker {self.worker_id}] Starting on {self.device}")
        print(f"[Worker {self.worker_id}] API URL: {self.api_url}")
        print(f"[Worker {self.worker_id}] Simulations: {self.simulations}")

        # Wait for API to be available
        print(f"[Worker {self.worker_id}] Waiting for API server...")
        if not self.api_client.wait_for_server(timeout=120):
            self.status = "error"
            self.error_message = "API server not available"
            print(f"[Worker {self.worker_id}] {self.error_message}")
            self._write_status()
            return

        print(f"[Worker {self.worker_id}] API server connected")

        # Load network
        if not self._load_or_create_network():
            self.status = "error"
            self._write_status()
            return

        self.status = "running"

        # Start status writer thread
        self._status_thread = Thread(target=self._status_writer_loop, daemon=True)
        self._status_thread.start()

        print(f"[Worker {self.worker_id}] Starting self-play loop")

        try:
            while not self.shutdown_event.is_set():
                # Play a game
                moves, result, visit_counts = self.play_one_game()

                if self.shutdown_event.is_set():
                    break

                # Submit game to API
                if self.submit_game(moves, result, visit_counts):
                    self.games_completed += 1

                    # Log progress
                    winner_str = {1.0: "P1", -1.0: "P2", 0.0: "Draw"}.get(result, "?")
                    print(f"[Worker {self.worker_id}] Game {self.games_completed}: "
                          f"{len(moves)} moves, winner={winner_str}")

                # Check for new model periodically
                if self.games_completed % self.model_check_interval == 0:
                    self._check_for_new_model()

        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            print(f"[Worker {self.worker_id}] Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.status = "stopped"
            self._write_status()
            print(f"[Worker {self.worker_id}] Stopped. Total games: {self.games_completed}")

    def shutdown(self):
        """Signal the worker to stop."""
        print(f"[Worker {self.worker_id}] Shutdown requested")
        self.shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(description='Distributed self-play worker')
    parser.add_argument('--worker-id', type=int, required=True, help='Unique worker ID')
    parser.add_argument('--api-url', type=str, required=True,
                        help='Training API URL (e.g., http://server:8000)')
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'),
                        help='Workspace directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move')
    parser.add_argument('--temperature-moves', type=int, default=30,
                        help='Number of moves to use temperature')
    parser.add_argument('--filters', type=int, default=None,
                        help='Network filter count (overrides --network-size)')
    parser.add_argument('--blocks', type=int, default=None,
                        help='Network residual blocks (overrides --network-size)')
    parser.add_argument('--network-size', type=str, default='medium', choices=['small', 'medium', 'large'],
                        help='Network size preset: small (64f/6b), medium (128f/10b), large (256f/15b)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='MCTS batch size for GPU parallelism')
    parser.add_argument('--model-check-interval', type=int, default=5,
                        help='Check for new model every N games')

    args = parser.parse_args()

    # Resolve network size presets
    NETWORK_PRESETS = {
        'small': (64, 6),      # ~900K params, fast inference
        'medium': (128, 10),   # ~3.5M params, balanced
        'large': (256, 15),    # ~15M params, stronger but slower
    }

    if args.filters is not None and args.blocks is not None:
        filters, blocks = args.filters, args.blocks
    else:
        filters, blocks = NETWORK_PRESETS[args.network_size]
        if args.filters is not None:
            filters = args.filters
        if args.blocks is not None:
            blocks = args.blocks

    # Ensure unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'

    worker = SelfPlayWorker(
        worker_id=args.worker_id,
        api_url=args.api_url,
        workspace=args.workspace,
        device=args.device,
        simulations=args.simulations,
        temperature_moves=args.temperature_moves,
        filters=filters,
        blocks=blocks,
        batch_size=args.batch_size,
        model_check_interval=args.model_check_interval
    )

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        worker.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker.run()


if __name__ == '__main__':
    main()
