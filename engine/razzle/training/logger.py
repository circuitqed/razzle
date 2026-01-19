"""
Training metrics logger for Razzle Dazzle.

Provides structured logging of training progress to JSON files
for easy monitoring by humans and scripts.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import time


@dataclass
class GameMetrics:
    """Metrics for a single self-play game."""
    game_id: int
    winner: str  # "p1", "p2", "draw"
    length: int  # number of moves
    p1_ball_progress: float  # how far P1's ball advanced (0-1)
    p2_ball_progress: float  # how far P2's ball advanced (0-1)


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    loss: float
    policy_loss: float
    value_loss: float
    learning_rate: float


@dataclass
class IterationMetrics:
    """Metrics for a full training iteration."""
    iteration: int
    timestamp: str

    # Self-play metrics
    num_games: int
    p1_wins: int
    p2_wins: int
    draws: int
    avg_game_length: float
    training_examples: int
    selfplay_time_sec: float

    # Game length details (early learning indicator)
    min_game_length: int = 0
    max_game_length: int = 0
    std_game_length: float = 0.0

    # Training metrics
    epochs: list[EpochMetrics] = field(default_factory=list)
    training_time_sec: float = 0.0
    final_loss: float = 0.0
    final_policy_loss: float = 0.0
    final_value_loss: float = 0.0

    # Total time for iteration
    total_time_sec: float = 0.0

    # Hardware metrics
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    cpu_percent: float = 0.0
    device: str = "cpu"

    # Optional evaluation metrics
    win_rate_vs_random: Optional[float] = None
    elo_rating: Optional[float] = None


@dataclass
class TrainingLog:
    """Complete training log."""
    run_id: str
    start_time: str
    config: dict
    iterations: list[IterationMetrics] = field(default_factory=list)

    # Running totals
    total_games: int = 0
    total_examples: int = 0
    total_time_sec: float = 0.0


def get_hardware_metrics(device: str = "cpu") -> dict:
    """Get current CPU/GPU usage metrics."""
    metrics = {
        "gpu_memory_used_mb": 0.0,
        "gpu_memory_total_mb": 0.0,
        "gpu_utilization_pct": 0.0,
        "cpu_percent": 0.0,
        "device": device
    }

    # Get CPU usage
    try:
        import psutil
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass

    # Get GPU metrics if using CUDA
    if device != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                # Memory usage
                metrics["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                metrics["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

                # Try to get GPU utilization via nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        metrics["gpu_utilization_pct"] = float(result.stdout.strip().split('\n')[0])
                except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                    pass
        except ImportError:
            pass

    return metrics


class TrainingLogger:
    """
    Logs training metrics to JSON file.

    Supports incremental updates - safe to read while training is in progress.
    """

    def __init__(self, output_dir: Path, config: dict = None, device: str = "cpu"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "training_log.json"
        self.device = device

        # Try to load existing log or create new one
        if self.log_path.exists():
            self.log = self._load()
        else:
            self.log = TrainingLog(
                run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                start_time=datetime.now().isoformat(),
                config=config or {}
            )

        # Timing helpers
        self._iter_start: Optional[float] = None
        self._selfplay_start: Optional[float] = None
        self._training_start: Optional[float] = None

    def _load(self) -> TrainingLog:
        """Load existing log from file."""
        with open(self.log_path, 'r') as f:
            data = json.load(f)

        iterations = []
        for it in data.get('iterations', []):
            epochs = [EpochMetrics(**e) for e in it.pop('epochs', [])]
            iterations.append(IterationMetrics(**it, epochs=epochs))

        return TrainingLog(
            run_id=data.get('run_id', 'unknown'),
            start_time=data.get('start_time', ''),
            config=data.get('config', {}),
            iterations=iterations,
            total_games=data.get('total_games', 0),
            total_examples=data.get('total_examples', 0),
            total_time_sec=data.get('total_time_sec', 0.0)
        )

    def save(self):
        """Save log to file."""
        # Convert to dict for JSON serialization
        data = {
            'run_id': self.log.run_id,
            'start_time': self.log.start_time,
            'config': self.log.config,
            'total_games': self.log.total_games,
            'total_examples': self.log.total_examples,
            'total_time_sec': self.log.total_time_sec,
            'iterations': []
        }

        for it in self.log.iterations:
            it_dict = asdict(it)
            data['iterations'].append(it_dict)

        # Write atomically
        tmp_path = self.log_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(self.log_path)

    def start_iteration(self, iteration: int):
        """Mark start of a new iteration."""
        self._iter_start = time.time()

    def start_selfplay(self):
        """Mark start of self-play phase."""
        self._selfplay_start = time.time()

    def end_selfplay(self, games: list) -> IterationMetrics:
        """
        Record self-play results.

        Args:
            games: List of GameRecord objects from self-play

        Returns:
            IterationMetrics (partially filled)
        """
        selfplay_time = time.time() - self._selfplay_start if self._selfplay_start else 0.0

        # Compute game statistics
        p1_wins = sum(1 for g in games if g.result == 1.0)
        p2_wins = sum(1 for g in games if g.result == -1.0)
        draws = sum(1 for g in games if g.result == 0.0)
        training_examples = sum(len(g.states) for g in games)

        # Game length statistics (key early learning indicator)
        game_lengths = [len(g.moves) for g in games]
        if game_lengths:
            avg_length = sum(game_lengths) / len(game_lengths)
            min_length = min(game_lengths)
            max_length = max(game_lengths)
            # Compute standard deviation
            variance = sum((x - avg_length) ** 2 for x in game_lengths) / len(game_lengths)
            std_length = variance ** 0.5
        else:
            avg_length = min_length = max_length = std_length = 0.0

        iteration = len(self.log.iterations)
        metrics = IterationMetrics(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            num_games=len(games),
            p1_wins=p1_wins,
            p2_wins=p2_wins,
            draws=draws,
            avg_game_length=avg_length,
            min_game_length=min_length,
            max_game_length=max_length,
            std_game_length=std_length,
            training_examples=training_examples,
            selfplay_time_sec=selfplay_time
        )

        return metrics

    def start_training(self):
        """Mark start of training phase."""
        self._training_start = time.time()

    def end_training(self, metrics: IterationMetrics, history: list[dict]):
        """
        Record training results and finalize iteration.

        Args:
            metrics: The IterationMetrics from end_selfplay
            history: List of epoch metrics from Trainer.train()
        """
        training_time = time.time() - self._training_start if self._training_start else 0.0
        total_time = time.time() - self._iter_start if self._iter_start else 0.0

        # Convert epoch history
        epochs = []
        for h in history:
            epochs.append(EpochMetrics(
                epoch=h['epoch'],
                loss=h['loss'],
                policy_loss=h['policy_loss'],
                value_loss=h['value_loss'],
                learning_rate=h['lr']
            ))

        # Update metrics
        metrics.epochs = epochs
        metrics.training_time_sec = training_time
        metrics.total_time_sec = total_time

        if epochs:
            metrics.final_loss = epochs[-1].loss
            metrics.final_policy_loss = epochs[-1].policy_loss
            metrics.final_value_loss = epochs[-1].value_loss

        # Capture hardware metrics
        hw = get_hardware_metrics(self.device)
        metrics.gpu_memory_used_mb = hw["gpu_memory_used_mb"]
        metrics.gpu_memory_total_mb = hw["gpu_memory_total_mb"]
        metrics.gpu_utilization_pct = hw["gpu_utilization_pct"]
        metrics.cpu_percent = hw["cpu_percent"]
        metrics.device = hw["device"]

        # Add to log
        self.log.iterations.append(metrics)
        self.log.total_games += metrics.num_games
        self.log.total_examples += metrics.training_examples
        self.log.total_time_sec += total_time

        # Save after each iteration
        self.save()

    def log_evaluation(self, iteration: int, win_rate_vs_random: float = None, elo: float = None):
        """Add evaluation metrics to an iteration."""
        for it in self.log.iterations:
            if it.iteration == iteration:
                if win_rate_vs_random is not None:
                    it.win_rate_vs_random = win_rate_vs_random
                if elo is not None:
                    it.elo_rating = elo
                break
        self.save()

    def get_summary(self) -> dict:
        """Get summary statistics for monitoring."""
        if not self.log.iterations:
            return {
                'status': 'no_data',
                'message': 'No training iterations completed yet'
            }

        latest = self.log.iterations[-1]

        return {
            'status': 'training',
            'run_id': self.log.run_id,
            'iterations_completed': len(self.log.iterations),
            'total_games': self.log.total_games,
            'total_examples': self.log.total_examples,
            'total_time_sec': self.log.total_time_sec,
            'latest_iteration': {
                'iteration': latest.iteration,
                'p1_wins': latest.p1_wins,
                'p2_wins': latest.p2_wins,
                'draws': latest.draws,
                'avg_game_length': latest.avg_game_length,
                'final_loss': latest.final_loss,
                'final_policy_loss': latest.final_policy_loss,
                'final_value_loss': latest.final_value_loss,
                'win_rate_vs_random': latest.win_rate_vs_random
            }
        }

    def print_status(self):
        """Print human-readable status to console."""
        summary = self.get_summary()

        if summary['status'] == 'no_data':
            print(summary['message'])
            return

        print(f"\n{'='*60}")
        print(f"Training Run: {summary['run_id']}")
        print(f"{'='*60}")
        print(f"Iterations: {summary['iterations_completed']}")
        print(f"Total games: {summary['total_games']}")
        print(f"Total examples: {summary['total_examples']:,}")
        print(f"Total time: {summary['total_time_sec']/60:.1f} min")

        latest = summary['latest_iteration']
        print(f"\nLatest iteration ({latest['iteration']}):")
        print(f"  Games: P1 {latest['p1_wins']} / P2 {latest['p2_wins']} / Draw {latest['draws']}")
        print(f"  Avg length: {latest['avg_game_length']:.1f} moves")
        print(f"  Loss: {latest['final_loss']:.4f} (policy={latest['final_policy_loss']:.4f}, value={latest['final_value_loss']:.4f})")

        if latest['win_rate_vs_random'] is not None:
            print(f"  Win rate vs random: {latest['win_rate_vs_random']:.1%}")


def print_training_monitor(output_dir: Path, watch: bool = False, interval: float = 5.0):
    """
    Print training progress. Useful for monitoring from another terminal.

    Args:
        output_dir: Directory containing training_log.json
        watch: If True, continuously monitor and refresh
        interval: Refresh interval in seconds (only used if watch=True)
    """
    log_path = Path(output_dir) / "training_log.json"

    while True:
        if not log_path.exists():
            print(f"Waiting for training to start ({log_path})...")
        else:
            logger = TrainingLogger(output_dir)
            logger.print_status()

            # Print loss trend if we have enough data
            if len(logger.log.iterations) >= 2:
                print("\nLoss trend (last 5):")
                for it in logger.log.iterations[-5:]:
                    bar_len = int(it.final_loss * 20)  # Scale for display
                    bar = '#' * min(bar_len, 40)
                    print(f"  Iter {it.iteration:3d}: {it.final_loss:.4f} {bar}")

        if not watch:
            break

        time.sleep(interval)
        print("\033[H\033[J", end="")  # Clear screen
