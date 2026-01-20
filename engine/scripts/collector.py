#!/usr/bin/env python3
"""
Game collector for distributed Razzle Dazzle training.

This script runs locally and coordinates distributed training by:
1. Polling workers via SCP to collect completed games
2. Aggregating games into training batches
3. Triggering training when enough games are collected
4. Distributing new model weights to all workers

The collector controls the pace of training and provides a central
point for monitoring progress.

Usage:
    # As a standalone service (used by train_distributed.py)
    python collector.py --workers workers.json --output output/

    # workers.json format:
    [
        {"worker_id": 0, "host": "1.2.3.4", "port": 22},
        {"worker_id": 1, "host": "5.6.7.8", "port": 22}
    ]
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Thread, Event, Lock
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet, create_network
from razzle.training.selfplay import GameRecord, games_to_training_data, load_games
from razzle.training.trainer import Trainer, TrainingConfig
from razzle.training.logger import TrainingLogger


@dataclass
class WorkerInfo:
    """Information about a remote worker."""
    worker_id: int
    host: str
    port: int
    status: str = "unknown"  # unknown, running, stopped, unreachable
    games_collected: int = 0
    games_pending: int = 0
    last_seen: Optional[str] = None
    model_version: str = "unknown"
    games_per_hour: float = 0.0
    error_count: int = 0


@dataclass
class CollectorStatus:
    """Overall collector status for monitoring."""
    status: str  # starting, running, training, stopped
    iteration: int
    total_games_collected: int
    games_in_batch: int
    training_threshold: int
    workers: list[WorkerInfo]
    current_model: str
    last_training: Optional[str]
    training_count: int
    start_time: str
    cost_estimate_usd: float = 0.0


class Collector:
    """
    Collects games from workers and coordinates training.
    """

    # SSH/SCP options for non-interactive use
    SSH_OPTS = [
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'ConnectTimeout=30',
        '-o', 'BatchMode=yes',
        '-o', 'LogLevel=ERROR',
    ]

    def __init__(
        self,
        workers: list[WorkerInfo],
        output_dir: Path,
        device: str = 'cpu',
        training_threshold: int = 100,
        max_training_interval: int = 900,  # 15 minutes
        poll_interval: int = 30,
        epochs: int = 10,
        filters: int = 64,
        blocks: int = 6,
        initial_model: Optional[Path] = None,
    ):
        self.workers = workers
        self.output_dir = Path(output_dir)
        self.device = device
        self.training_threshold = training_threshold
        self.max_training_interval = max_training_interval
        self.poll_interval = poll_interval
        self.epochs = epochs
        self.filters = filters
        self.blocks = blocks

        # Directories
        self.collected_dir = self.output_dir / "collected"
        self.models_dir = self.output_dir / "models"
        self.status_file = self.output_dir / "collector_status.json"

        self.collected_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.games: list[GameRecord] = []
        self.games_lock = Lock()
        self.total_games_collected = 0
        self.iteration = 0
        self.training_count = 0
        self.last_training_time = time.time()
        self.start_time = datetime.now()
        self.current_model: Optional[Path] = initial_model
        self.current_model_version = initial_model.name if initial_model else "none"

        # Network for training
        self.network: Optional[RazzleNet] = None

        # Control
        self.shutdown_event = Event()
        self.status = "starting"

        # Logger
        self.logger = TrainingLogger(self.output_dir, config={
            'training_threshold': training_threshold,
            'poll_interval': poll_interval,
            'epochs': epochs,
            'filters': filters,
            'blocks': blocks,
            'num_workers': len(workers),
        }, device=device)

    def _ssh_run(self, worker: WorkerInfo, command: str, timeout: int = 60) -> tuple[bool, str]:
        """Run command on worker via SSH."""
        try:
            result = subprocess.run(
                ['ssh'] + self.SSH_OPTS + [
                    '-p', str(worker.port),
                    f'root@{worker.host}',
                    command
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)

    def _scp_from(self, worker: WorkerInfo, remote_path: str, local_path: Path,
                  timeout: int = 120) -> bool:
        """Copy file from worker via SCP."""
        try:
            result = subprocess.run(
                ['scp'] + self.SSH_OPTS + [
                    '-P', str(worker.port),
                    f'root@{worker.host}:{remote_path}',
                    str(local_path)
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0
        except:
            return False

    def _scp_to(self, worker: WorkerInfo, local_path: Path, remote_path: str,
                timeout: int = 120) -> bool:
        """Copy file to worker via SCP."""
        try:
            result = subprocess.run(
                ['scp'] + self.SSH_OPTS + [
                    '-P', str(worker.port),
                    str(local_path),
                    f'root@{worker.host}:{remote_path}'
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0
        except:
            return False

    def poll_worker(self, worker: WorkerInfo) -> int:
        """
        Poll a single worker for games.

        Returns number of games collected.
        """
        # Get worker status
        success, output = self._ssh_run(worker, 'cat /workspace/status.json')
        if success:
            try:
                status_data = json.loads(output)
                worker.status = status_data.get('status', 'unknown')
                worker.games_pending = status_data.get('games_pending', 0)
                worker.model_version = status_data.get('model_version', 'unknown')
                worker.games_per_hour = status_data.get('games_per_hour', 0.0)
                worker.last_seen = datetime.now().isoformat()
                worker.error_count = 0
            except json.JSONDecodeError:
                worker.error_count += 1
        else:
            worker.error_count += 1
            if worker.error_count >= 3:
                worker.status = "unreachable"
            return 0

        # List pending games
        success, output = self._ssh_run(worker, 'ls /workspace/pending/*.pkl 2>/dev/null || true')
        if not success or not output.strip():
            return 0

        game_files = output.strip().split('\n')
        if not game_files or game_files == ['']:
            return 0

        collected = 0
        worker_dir = self.collected_dir / f"worker_{worker.worker_id}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        for remote_path in game_files:
            remote_path = remote_path.strip()
            if not remote_path:
                continue

            filename = Path(remote_path).name
            local_path = worker_dir / filename

            # Download game
            if self._scp_from(worker, remote_path, local_path):
                # Load and add to collection
                try:
                    with open(local_path, 'rb') as f:
                        game = pickle.load(f)

                    with self.games_lock:
                        self.games.append(game)
                        self.total_games_collected += 1

                    collected += 1
                    worker.games_collected += 1

                    # Remove from remote pending directory
                    self._ssh_run(worker, f'rm {remote_path}')

                except Exception as e:
                    print(f"[Collector] Error loading game {filename}: {e}")

        return collected

    def poll_all_workers(self) -> int:
        """Poll all workers for games. Returns total collected."""
        total = 0
        for worker in self.workers:
            try:
                collected = self.poll_worker(worker)
                if collected > 0:
                    print(f"[Collector] Worker {worker.worker_id}: collected {collected} games")
                total += collected
            except Exception as e:
                print(f"[Collector] Error polling worker {worker.worker_id}: {e}")
                worker.error_count += 1

        return total

    def should_train(self) -> bool:
        """Check if we should trigger training."""
        with self.games_lock:
            game_count = len(self.games)

        # Train if we have enough games
        if game_count >= self.training_threshold:
            return True

        # Or if max interval has passed and we have some games
        elapsed = time.time() - self.last_training_time
        if elapsed >= self.max_training_interval and game_count >= 10:
            return True

        return False

    def run_training(self):
        """Train on collected games."""
        with self.games_lock:
            if not self.games:
                return

            training_games = self.games.copy()
            self.games = []

        num_games = len(training_games)
        print(f"\n[Collector] Training on {num_games} games (iteration {self.iteration})")

        self.status = "training"
        self._write_status()

        # Convert to training data
        states, policies, values = games_to_training_data(training_games)
        print(f"[Collector] Training examples: {len(states)}")

        # Load or create network
        if self.network is None:
            if self.current_model and self.current_model.exists():
                self.network = RazzleNet.load(self.current_model, device=self.device)
                print(f"[Collector] Loaded model: {self.current_model}")
            else:
                self.network = create_network(self.filters, self.blocks, self.device)
                print(f"[Collector] Created new network")

        # Train
        self.logger.start_iteration(self.iteration)
        self.logger.start_selfplay()

        # Fake selfplay metrics since games came from workers
        from razzle.training.logger import IterationMetrics
        iter_metrics = self.logger.end_selfplay(training_games)

        config = TrainingConfig(
            epochs=self.epochs,
            device=self.device
        )
        trainer = Trainer(self.network, config)

        self.logger.start_training()
        history = trainer.train(states, policies, values, verbose=True)
        self.logger.end_training(iter_metrics, history)

        # Save new model
        model_filename = f"model_iter_{self.iteration:03d}.pt"
        model_path = self.models_dir / model_filename
        self.network.save(model_path)
        self.current_model = model_path
        self.current_model_version = model_filename

        print(f"[Collector] Saved model: {model_path}")

        # Distribute to workers
        self.distribute_model(model_path)

        self.iteration += 1
        self.training_count += 1
        self.last_training_time = time.time()
        self.status = "running"

    def distribute_model(self, model_path: Path):
        """Upload new model to all workers."""
        print(f"[Collector] Distributing model to {len(self.workers)} workers")

        for worker in self.workers:
            if worker.status == "unreachable":
                continue

            try:
                # Upload to model directory
                if self._scp_to(worker, model_path, f'/workspace/model/{model_path.name}'):
                    print(f"[Collector] Uploaded to worker {worker.worker_id}")
                else:
                    print(f"[Collector] Failed to upload to worker {worker.worker_id}")
            except Exception as e:
                print(f"[Collector] Error uploading to worker {worker.worker_id}: {e}")

    def _get_status(self) -> CollectorStatus:
        """Get current collector status."""
        with self.games_lock:
            games_in_batch = len(self.games)

        # Estimate cost (rough: $0.05/hr per worker)
        hours = (datetime.now() - self.start_time).total_seconds() / 3600
        cost = hours * len(self.workers) * 0.05

        return CollectorStatus(
            status=self.status,
            iteration=self.iteration,
            total_games_collected=self.total_games_collected,
            games_in_batch=games_in_batch,
            training_threshold=self.training_threshold,
            workers=[w for w in self.workers],
            current_model=self.current_model_version,
            last_training=datetime.fromtimestamp(self.last_training_time).isoformat()
                if self.training_count > 0 else None,
            training_count=self.training_count,
            start_time=self.start_time.isoformat(),
            cost_estimate_usd=cost
        )

    def _write_status(self):
        """Write collector status to file."""
        status = self._get_status()

        # Convert to dict (handle nested dataclasses)
        data = asdict(status)
        data['workers'] = [asdict(w) for w in status.workers]

        tmp_path = self.status_file.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(self.status_file)

    def run(self):
        """Main collector loop."""
        print(f"[Collector] Starting with {len(self.workers)} workers")
        print(f"[Collector] Training threshold: {self.training_threshold} games")
        print(f"[Collector] Poll interval: {self.poll_interval}s")
        print(f"[Collector] Output: {self.output_dir}")

        self.status = "running"

        # Initial model distribution
        if self.current_model and self.current_model.exists():
            self.distribute_model(self.current_model)

        try:
            while not self.shutdown_event.is_set():
                # Poll workers
                collected = self.poll_all_workers()

                with self.games_lock:
                    batch_size = len(self.games)

                if collected > 0:
                    print(f"[Collector] Batch: {batch_size}/{self.training_threshold} games")

                # Check if we should train
                if self.should_train():
                    self.run_training()

                # Write status
                self._write_status()

                # Wait for next poll
                self.shutdown_event.wait(self.poll_interval)

        except Exception as e:
            self.status = "error"
            print(f"[Collector] Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.status = "stopped"
            self._write_status()
            print(f"[Collector] Stopped. Total games: {self.total_games_collected}")

    def shutdown(self):
        """Signal collector to stop."""
        print("[Collector] Shutdown requested")
        self.shutdown_event.set()


def load_workers(workers_file: Path) -> list[WorkerInfo]:
    """Load worker configuration from JSON file."""
    with open(workers_file, 'r') as f:
        data = json.load(f)

    workers = []
    for w in data:
        workers.append(WorkerInfo(
            worker_id=w['worker_id'],
            host=w['host'],
            port=w.get('port', 22)
        ))
    return workers


def main():
    parser = argparse.ArgumentParser(description='Game collector for distributed training')
    parser.add_argument('--workers', type=Path, required=True,
                        help='JSON file with worker information')
    parser.add_argument('--output', type=Path, default=Path('output/distributed'),
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for training (cuda or cpu)')
    parser.add_argument('--training-threshold', type=int, default=100,
                        help='Number of games before training')
    parser.add_argument('--max-training-interval', type=int, default=900,
                        help='Max seconds between training (even if threshold not met)')
    parser.add_argument('--poll-interval', type=int, default=30,
                        help='Seconds between polling workers')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs per iteration')
    parser.add_argument('--filters', type=int, default=64,
                        help='Network filters')
    parser.add_argument('--blocks', type=int, default=6,
                        help='Network residual blocks')
    parser.add_argument('--model', type=Path, help='Initial model checkpoint')

    args = parser.parse_args()

    # Load worker configuration
    workers = load_workers(args.workers)
    print(f"Loaded {len(workers)} workers from {args.workers}")

    collector = Collector(
        workers=workers,
        output_dir=args.output,
        device=args.device,
        training_threshold=args.training_threshold,
        max_training_interval=args.max_training_interval,
        poll_interval=args.poll_interval,
        epochs=args.epochs,
        filters=args.filters,
        blocks=args.blocks,
        initial_model=args.model
    )

    # Handle signals
    import signal
    def signal_handler(signum, frame):
        collector.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    collector.run()


if __name__ == '__main__':
    main()
