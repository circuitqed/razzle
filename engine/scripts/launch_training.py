#!/usr/bin/env python3
"""
Unified training launcher for Razzle Dazzle.

This script provides a single command to launch distributed training with
sensible defaults and pre-configured profiles. It handles:
1. Optionally clearing old training data
2. Verifying the API server is running
3. Searching for and provisioning GPU instances on Vast.ai
4. Starting self-play workers
5. Starting the trainer (locally or on cloud)
6. Monitoring progress

Usage:
    # Quick start with defaults (large network, 4 workers on RTX 3060)
    python scripts/launch_training.py

    # Fresh start (clear old data first)
    python scripts/launch_training.py --fresh

    # Use a specific profile
    python scripts/launch_training.py --profile alphazero

    # Custom configuration
    python scripts/launch_training.py --workers 8 --gpu RTX_4090 --network-size large

Profiles:
    default:    Large network (256f/15b), 4 workers, RTX 3060, 800 sims
    fast:       Medium network (128f/10b), 6 workers, RTX 3060, 400 sims
    alphazero:  AlphaZero-scale (256f/20b), 4 workers, RTX 4090, 800 sims
    cheap:      Medium network, 2 workers, RTX 3060, 400 sims
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Training Profiles
# =============================================================================

@dataclass
class TrainingProfile:
    """Pre-configured training settings."""
    name: str
    network_size: str  # small, medium, large, alphazero
    workers: int
    workers_per_instance: int
    gpu: str
    max_price: float
    simulations: int
    batch_size: int
    threshold: int  # games before training
    random_opening_moves: int
    random_opening_fraction: float
    description: str


PROFILES = {
    'default': TrainingProfile(
        name='default',
        network_size='large',
        workers=4,
        workers_per_instance=3,
        gpu='RTX_3060',
        max_price=0.15,
        simulations=800,
        batch_size=32,
        threshold=50,
        random_opening_moves=8,
        random_opening_fraction=0.3,
        description='Large network (18M params), balanced cost/performance',
    ),
    'fast': TrainingProfile(
        name='fast',
        network_size='medium',
        workers=6,
        workers_per_instance=3,
        gpu='RTX_3060',
        max_price=0.15,
        simulations=400,
        batch_size=32,
        threshold=40,
        random_opening_moves=6,
        random_opening_fraction=0.3,
        description='Medium network (3.3M params), faster iterations',
    ),
    'alphazero': TrainingProfile(
        name='alphazero',
        network_size='alphazero',
        workers=4,
        workers_per_instance=2,
        gpu='RTX_4090',
        max_price=0.50,
        simulations=800,
        batch_size=32,
        threshold=50,
        random_opening_moves=8,
        random_opening_fraction=0.3,
        description='AlphaZero-scale (24M params), maximum quality',
    ),
    'cheap': TrainingProfile(
        name='cheap',
        network_size='medium',
        workers=2,
        workers_per_instance=3,
        gpu='RTX_3060',
        max_price=0.10,
        simulations=400,
        batch_size=32,
        threshold=30,
        random_opening_moves=6,
        random_opening_fraction=0.3,
        description='Budget option, slower but cheap',
    ),
}

NETWORK_PRESETS = {
    'small': (64, 6),      # ~0.8M params
    'medium': (128, 10),   # ~3.3M params
    'large': (256, 15),    # ~18M params
    'alphazero': (256, 20),  # ~24M params
}


# =============================================================================
# API Utilities
# =============================================================================

def check_api_server(api_url: str) -> bool:
    """Check if the API server is running and healthy."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def clear_training_data(api_url: str) -> dict:
    """Clear all training data (games, models, metrics)."""
    try:
        resp = requests.delete(f"{api_url}/training/clear", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to clear training data: {e}")


def get_training_status(api_url: str) -> dict:
    """Get current training status."""
    try:
        resp = requests.get(f"{api_url}/training/dashboard", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def get_latest_metrics(api_url: str) -> Optional[dict]:
    """Get the latest training metrics."""
    try:
        resp = requests.get(f"{api_url}/training/metrics/latest", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get('metrics')
    except Exception:
        return None


# =============================================================================
# Training Launcher
# =============================================================================

class TrainingLauncher:
    """Manages the training launch process."""

    def __init__(
        self,
        api_url: str,
        profile: TrainingProfile,
        fresh_start: bool = False,
        output_dir: Path = Path('output/training'),
    ):
        self.api_url = api_url
        self.profile = profile
        self.fresh_start = fresh_start
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.worker_process: Optional[subprocess.Popen] = None
        self.shutdown_requested = False

    def run(self) -> int:
        """Run the complete training launch sequence."""
        print("=" * 60)
        print("RAZZLE DAZZLE TRAINING LAUNCHER")
        print("=" * 60)
        print(f"\nProfile: {self.profile.name}")
        print(f"  {self.profile.description}")
        print(f"\nConfiguration:")
        print(f"  Network: {self.profile.network_size} ({NETWORK_PRESETS[self.profile.network_size]})")
        print(f"  Workers: {self.profile.workers} instances x {self.profile.workers_per_instance} each")
        print(f"  GPU: {self.profile.gpu} (max ${self.profile.max_price}/hr)")
        print(f"  Simulations: {self.profile.simulations}")
        print(f"  Training threshold: {self.profile.threshold} games")
        print(f"  API: {self.api_url}")
        print()

        # Step 1: Check API server
        print("[1/3] Checking API server...")
        if not check_api_server(self.api_url):
            print("  ERROR: API server is not running!")
            print(f"  Please start it with: docker restart razzle-engine")
            print(f"  Or: uvicorn server.main:app --host 0.0.0.0 --port 8000")
            return 1
        print("  API server is healthy")

        # Step 2: Optionally clear training data
        if self.fresh_start:
            print("\n[2/3] Clearing training data (fresh start)...")
            try:
                result = clear_training_data(self.api_url)
                print(f"  Deleted: {result.get('games_deleted', 0)} games, "
                      f"{result.get('models_deleted', 0)} models, "
                      f"{result.get('metrics_deleted', 0)} metrics")
            except Exception as e:
                print(f"  ERROR: {e}")
                return 1
        else:
            print("\n[2/3] Keeping existing training data")
            status = get_training_status(self.api_url)
            if status:
                print(f"  Existing: {status.get('games_total', 0)} games, "
                      f"iteration {status.get('latest_model', {}).get('iteration', 0)}")

        # Step 3: Start distributed training (workers + trainer on Vast.ai)
        print("\n[3/3] Starting distributed training on Vast.ai...")
        if not self._start_workers():
            return 1

        print("\n" + "=" * 60)
        print("TRAINING LAUNCHED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nMonitoring:")
        print(f"  Dashboard: Open webapp and press 'T'")
        print(f"  API status: curl {self.api_url}/training/dashboard")
        print(f"  Latest metrics: curl {self.api_url}/training/metrics/latest")
        print(f"\nPress Ctrl+C to stop training and cleanup instances")
        print()

        # Monitor until shutdown
        return self._monitor_loop()

    def _start_workers(self) -> bool:
        """Start distributed training (workers + trainer on Vast.ai)."""
        filters, blocks = NETWORK_PRESETS[self.profile.network_size]

        cmd = [
            sys.executable, 'scripts/train_distributed.py',
            '--workers', str(self.profile.workers),
            '--workers-per-instance', str(self.profile.workers_per_instance),
            '--api-url', self.api_url,
            '--gpu', self.profile.gpu,
            '--max-price', str(self.profile.max_price),
            '--simulations', str(self.profile.simulations),
            '--filters', str(filters),
            '--blocks', str(blocks),
            '--threshold', str(self.profile.threshold),
            '--batch-size', str(self.profile.batch_size),
            '--random-opening-moves', str(self.profile.random_opening_moves),
            '--random-opening-fraction', str(self.profile.random_opening_fraction),
            # Trainer will run on Vast.ai too (no --no-trainer flag)
            '--output', str(self.output_dir),
        ]

        print(f"  Command: {' '.join(cmd[:5])}...")

        try:
            self.worker_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=Path(__file__).parent.parent,
            )
            print(f"  Distributed training started (PID {self.worker_process.pid})")
            print(f"  Creating {self.profile.workers} worker instances + 1 trainer instance on Vast.ai")
            return True
        except Exception as e:
            print(f"  ERROR starting distributed training: {e}")
            return False

    def _monitor_loop(self) -> int:
        """Monitor training progress until shutdown."""
        last_status_time = 0
        status_interval = 60  # Print status every 60 seconds

        while not self.shutdown_requested:
            try:
                # Check if process is still running
                if self.worker_process and self.worker_process.poll() is not None:
                    print("\nDistributed training process exited!")
                    self.shutdown_requested = True
                    break

                # Print status periodically
                now = time.time()
                if now - last_status_time >= status_interval:
                    self._print_status()
                    last_status_time = now

                time.sleep(5)

            except KeyboardInterrupt:
                print("\n\nShutdown requested...")
                self.shutdown_requested = True
                break

        return self._cleanup()

    def _print_status(self):
        """Print current training status."""
        status = get_training_status(self.api_url)
        metrics = get_latest_metrics(self.api_url)

        print(f"\n--- Status Update ({time.strftime('%H:%M:%S')}) ---")
        if status:
            print(f"  Games: {status.get('games_pending', 0)} pending / "
                  f"{status.get('games_total', 0)} total")
            latest = status.get('latest_model', {})
            if latest:
                print(f"  Model: iter {latest.get('iteration', 0)}, "
                      f"loss {latest.get('final_loss', 0):.4f}")

        if metrics:
            print(f"  Policy acc: {metrics.get('policy_top1_accuracy', 0)*100:.1f}%, "
                  f"EBF: {metrics.get('policy_ebf', 0):.1f}")
            print(f"  Calibration: {metrics.get('value_calibration_error', 0):.4f}")

    def _cleanup(self) -> int:
        """Clean up processes on shutdown."""
        print("\nCleaning up...")

        # Terminate distributed training (this should also cleanup Vast.ai instances)
        if self.worker_process and self.worker_process.poll() is None:
            print("  Stopping distributed training (this will destroy Vast.ai instances)...")
            self.worker_process.terminate()
            try:
                self.worker_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.worker_process.kill()

        print("  Cleanup complete")
        return 0

    def shutdown(self):
        """Signal shutdown."""
        self.shutdown_requested = True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified training launcher for Razzle Dazzle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  default    Large network (18M params), 4 workers on RTX 3060
  fast       Medium network (3.3M params), 6 workers, faster iterations
  alphazero  AlphaZero-scale (24M params), 4 workers on RTX 4090
  cheap      Budget option, 2 workers, slower but cheap

Examples:
  # Quick start with defaults
  python scripts/launch_training.py

  # Fresh start (clear old data)
  python scripts/launch_training.py --fresh

  # Use alphazero profile
  python scripts/launch_training.py --profile alphazero

  # Custom settings
  python scripts/launch_training.py --workers 8 --network-size large --gpu RTX_4090
""")

    parser.add_argument('--profile', type=str, default='default',
                        choices=list(PROFILES.keys()),
                        help='Training profile (default: default)')
    parser.add_argument('--fresh', action='store_true',
                        help='Clear all training data before starting')
    parser.add_argument('--api-url', type=str,
                        default='https://razzledazzle.lazybrains.com/api',
                        help='Training API URL')
    parser.add_argument('--output', type=Path, default=Path('output/training'),
                        help='Output directory')

    # Override profile settings
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker instances (overrides profile)')
    parser.add_argument('--network-size', type=str, default=None,
                        choices=['small', 'medium', 'large', 'alphazero'],
                        help='Network size (overrides profile)')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU type (overrides profile)')
    parser.add_argument('--max-price', type=float, default=None,
                        help='Max GPU price per hour (overrides profile)')
    parser.add_argument('--simulations', type=int, default=None,
                        help='MCTS simulations per move (overrides profile)')
    parser.add_argument('--threshold', type=int, default=None,
                        help='Games before training (overrides profile)')

    # List profiles
    parser.add_argument('--list-profiles', action='store_true',
                        help='List available profiles and exit')

    args = parser.parse_args()

    if args.list_profiles:
        print("\nAvailable Training Profiles:\n")
        for name, profile in PROFILES.items():
            filters, blocks = NETWORK_PRESETS[profile.network_size]
            print(f"  {name}:")
            print(f"    {profile.description}")
            print(f"    Network: {profile.network_size} ({filters}f/{blocks}b)")
            print(f"    Workers: {profile.workers} x {profile.workers_per_instance}")
            print(f"    GPU: {profile.gpu} (max ${profile.max_price}/hr)")
            print(f"    Simulations: {profile.simulations}")
            print()
        return 0

    # Get base profile
    profile = PROFILES[args.profile]

    # Apply overrides
    if args.workers is not None:
        profile.workers = args.workers
    if args.network_size is not None:
        profile.network_size = args.network_size
    if args.gpu is not None:
        profile.gpu = args.gpu
    if args.max_price is not None:
        profile.max_price = args.max_price
    if args.simulations is not None:
        profile.simulations = args.simulations
    if args.threshold is not None:
        profile.threshold = args.threshold

    # Create and run launcher
    launcher = TrainingLauncher(
        api_url=args.api_url,
        profile=profile,
        fresh_start=args.fresh,
        output_dir=args.output,
    )

    # Handle signals
    def signal_handler(signum, frame):
        launcher.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return launcher.run()


if __name__ == '__main__':
    sys.exit(main())
