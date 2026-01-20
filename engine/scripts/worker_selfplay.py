#!/usr/bin/env python3
"""
Distributed self-play worker for Razzle Dazzle training.

This script runs on cloud GPU instances (Vast.ai) and:
1. Plays games continuously using MCTS + neural network
2. Saves completed games to a pending/ directory for collection
3. Writes status updates to status.json for monitoring
4. Periodically checks for new model weights and hot-reloads

The local collector (collector.py) polls workers via SCP to:
- Download games from pending/
- Upload new model weights to model/

Usage:
    python worker_selfplay.py --worker-id 0 --simulations 400

Status file (status.json):
    {
        "worker_id": 0,
        "status": "running",
        "games_completed": 42,
        "games_pending": 5,
        "current_game_moves": 73,
        "model_version": "model_iter_002.pt",
        "last_update": "2024-01-20T15:30:45",
        "uptime_sec": 3600,
        "games_per_hour": 12.5
    }
"""

import argparse
import json
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet, create_network
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator
from razzle.core.state import GameState
from razzle.training.selfplay import GameRecord


@dataclass
class WorkerStatus:
    """Status information for monitoring."""
    worker_id: int
    status: str  # starting, running, paused, stopped, error
    games_completed: int
    games_pending: int
    current_game_moves: int
    model_version: str
    last_update: str
    start_time: str
    uptime_sec: float
    games_per_hour: float
    error_message: Optional[str] = None
    gpu_name: str = ""
    simulations: int = 0


class SelfPlayWorker:
    """
    Continuous self-play worker.

    Generates games independently and saves them for collection.
    """

    def __init__(
        self,
        worker_id: int,
        workspace: Path,
        device: str = 'cuda',
        simulations: int = 400,
        temperature_moves: int = 30,
        filters: int = 64,
        blocks: int = 6,
        batch_size: int = 16,
        model_check_interval: int = 5,  # Check for new model every N games
    ):
        self.worker_id = worker_id
        self.workspace = Path(workspace)
        self.device = device
        self.simulations = simulations
        self.temperature_moves = temperature_moves
        self.filters = filters
        self.blocks = blocks
        self.batch_size = batch_size
        self.model_check_interval = model_check_interval

        # Directories
        self.pending_dir = self.workspace / "pending"
        self.model_dir = self.workspace / "model"
        self.status_file = self.workspace / "status.json"

        # Create directories
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.network: Optional[RazzleNet] = None
        self.evaluator = None
        self.model_version = "none"
        self.games_completed = 0
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

    def _load_or_create_network(self) -> bool:
        """Load model from model_dir or create new one."""
        try:
            # Look for model files
            model_files = sorted(self.model_dir.glob("*.pt"))

            if model_files:
                # Load latest model
                latest = model_files[-1]
                self.network = RazzleNet.load(latest, device=self.device)
                self.model_version = latest.name
                print(f"[Worker {self.worker_id}] Loaded model: {self.model_version}")
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
        """Check if a new model is available and load it."""
        try:
            model_files = sorted(self.model_dir.glob("*.pt"))
            if not model_files:
                return False

            latest = model_files[-1]
            if latest.name != self.model_version:
                print(f"[Worker {self.worker_id}] New model detected: {latest.name}")
                self.network = RazzleNet.load(latest, device=self.device)
                self.evaluator = BatchedEvaluator(
                    self.network,
                    batch_size=self.batch_size,
                    device=self.device
                )
                self.model_version = latest.name
                print(f"[Worker {self.worker_id}] Loaded new model: {self.model_version}")
                return True

        except Exception as e:
            print(f"[Worker {self.worker_id}] Error checking for new model: {e}")

        return False

    def _count_pending_games(self) -> int:
        """Count games in pending directory."""
        return len(list(self.pending_dir.glob("*.pkl")))

    def _get_status(self) -> WorkerStatus:
        """Get current worker status."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        games_per_hour = (self.games_completed / uptime * 3600) if uptime > 0 else 0.0

        return WorkerStatus(
            worker_id=self.worker_id,
            status=self.status,
            games_completed=self.games_completed,
            games_pending=self._count_pending_games(),
            current_game_moves=self.current_game_moves,
            model_version=self.model_version,
            last_update=datetime.now().isoformat(),
            start_time=self.start_time.isoformat(),
            uptime_sec=uptime,
            games_per_hour=games_per_hour,
            error_message=self.error_message,
            gpu_name=self.gpu_name,
            simulations=self.simulations
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

    def play_one_game(self) -> GameRecord:
        """Play a single self-play game."""
        state = GameState.new_game()
        states = []
        policies = []
        moves = []
        ball_progress = []

        move_count = 0
        self.current_game_moves = 0

        while not state.is_terminal() and move_count < 300:
            # Track ball positions for reward shaping
            p0_ball_row = self._get_ball_row(state, 0)
            p1_ball_row = self._get_ball_row(state, 1)
            p0_progress = p0_ball_row / 7.0
            p1_progress = (7 - p1_ball_row) / 7.0
            ball_progress.append((p0_progress, p1_progress))

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

            # Record state and policy
            states.append(state.to_tensor())
            policies.append(mcts.get_policy(root))

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

        return GameRecord(
            states=states,
            policies=policies,
            result=result,
            moves=moves,
            ball_progress=ball_progress
        )

    def save_game(self, game: GameRecord, game_id: int):
        """Save a game to the pending directory."""
        # Include worker_id and timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_w{self.worker_id:02d}_{game_id:06d}_{timestamp}.pkl"
        path = self.pending_dir / filename

        with open(path, 'wb') as f:
            pickle.dump(game, f)

    def run(self):
        """Main worker loop."""
        print(f"[Worker {self.worker_id}] Starting on {self.device}")
        print(f"[Worker {self.worker_id}] Workspace: {self.workspace}")
        print(f"[Worker {self.worker_id}] Simulations: {self.simulations}")

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

        game_id = 0

        try:
            while not self.shutdown_event.is_set():
                # Play a game
                game = self.play_one_game()

                if self.shutdown_event.is_set():
                    break

                # Save game
                self.save_game(game, game_id)
                self.games_completed += 1
                game_id += 1

                # Log progress
                winner_str = {1.0: "P1", -1.0: "P2", 0.0: "Draw"}.get(game.result, "?")
                print(f"[Worker {self.worker_id}] Game {self.games_completed}: "
                      f"{len(game.moves)} moves, winner={winner_str}")

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
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'),
                        help='Workspace directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move')
    parser.add_argument('--temperature-moves', type=int, default=30,
                        help='Number of moves to use temperature')
    parser.add_argument('--filters', type=int, default=64,
                        help='Network filter count')
    parser.add_argument('--blocks', type=int, default=6,
                        help='Network residual blocks')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='MCTS batch size')
    parser.add_argument('--model-check-interval', type=int, default=5,
                        help='Check for new model every N games')

    args = parser.parse_args()

    # Ensure unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'

    worker = SelfPlayWorker(
        worker_id=args.worker_id,
        workspace=args.workspace,
        device=args.device,
        simulations=args.simulations,
        temperature_moves=args.temperature_moves,
        filters=args.filters,
        blocks=args.blocks,
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
