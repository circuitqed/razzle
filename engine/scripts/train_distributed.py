#!/usr/bin/env python3
"""
Distributed training orchestrator for Razzle Dazzle.

This script creates worker instances on Vast.ai that connect to a training
API server. Workers generate games and submit them via HTTP. A separate
trainer process fetches games and trains.

Instances use a prebuilt Docker image (ghcr.io/circuitqed/razzle-worker)
with all code and dependencies pre-installed. No SSH setup, no SCP, no
pip install — Vast.ai pulls the image directly and runs an --onstart-cmd.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    VAST.AI CLOUD                            │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
    │  │Worker 0 │  │Worker 1 │  │Worker 2 │  │Worker N │       │
    │  │ GPU     │  │ GPU     │  │ GPU     │  │ GPU     │       │
    │  │selfplay │  │selfplay │  │selfplay │  │selfplay │       │
    │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
    │       │            │            │            │             │
    │       └────────────┴─────┬──────┴────────────┘             │
    │                          │ HTTP (POST games, GET models)   │
    └──────────────────────────┼──────────────────────────────────┘
                               ▼
    ┌────────────────────────────────────────────────────────────┐
    │                    API SERVER                              │
    │  (can run locally or on cloud)                            │
    │  - Receives games from workers                            │
    │  - Stores in database                                     │
    │  - Serves latest model                                    │
    └─────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
    ┌────────────────────────────────────────────────────────────┐
    │                    TRAINER                                 │
    │  (can run locally or on cloud GPU)                        │
    │  - Polls API for pending games                            │
    │  - Trains when threshold reached                          │
    │  - Uploads new models                                     │
    └────────────────────────────────────────────────────────────┘

Usage:
    # First, start the API server (in another terminal):
    cd engine && python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

    # Then, start distributed training with 4 workers:
    python scripts/train_distributed.py --workers 4 --api-url http://your-server:8000

    # The script will create workers that connect to your API server.
    # Run the trainer separately:
    python scripts/trainer.py --api-url http://your-server:8000 --device cuda
"""

import argparse
import atexit
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.vastai import VastAI, GPUOffer

DEFAULT_IMAGE = 'ghcr.io/circuitqed/razzle-worker:latest'


@dataclass
class WorkerInstance:
    """Tracks a worker instance through its lifecycle."""
    worker_id: int
    instance_id: Optional[int] = None
    offer: Optional[GPUOffer] = None
    status: str = "pending"  # pending, creating, running, failed
    error: Optional[str] = None
    role: str = "worker"  # worker or trainer
    last_activity: float = field(default_factory=time.time)
    games_submitted: int = 0


def _build_worker_onstart(
    worker_id: int,
    api_url: str,
    simulations: int,
    network_size: str,
    workers_per_instance: int,
    batch_size: int,
    random_opening_moves: int,
    random_opening_fraction: float,
    run_id: str = '',
    api_key: str = '',
) -> str:
    """Build the onstart-cmd string for a worker instance.

    Launches N worker processes via nohup so they survive the init process.
    Each worker gets an ID like "{run_id}_{sub_worker_id}" so the orchestrator
    can distinguish its workers from previous runs in the dashboard.
    """
    lines = [
        # Export API key so workers can authenticate
        f'export TRAINING_API_KEY="{api_key}"',
        # Fetch latest code from GitHub (overlay on base image)
        'cd /tmp && git clone --depth 1 https://github.com/circuitqed/razzle.git razzle_src 2>/dev/null && '
        'cp -r /tmp/razzle_src/engine/razzle_fast /workspace/razzle_fast && '
        'cp /tmp/razzle_src/engine/scripts/worker_selfplay.py /workspace/scripts/worker_selfplay.py && '
        'cp -f /tmp/razzle_src/engine/razzle/training/*.py /workspace/razzle/training/ && '
        'rm -rf /tmp/razzle_src',
        # Build C MCTS extension (install gcc if needed)
        'which gcc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc >/dev/null 2>&1)',
        'python -u /workspace/razzle_fast/_build.py',
    ]
    for i in range(workers_per_instance):
        sub_worker_id = worker_id * workers_per_instance + i
        full_id = f'{run_id}_{sub_worker_id}' if run_id else str(sub_worker_id)
        lines.append(
            f'mkdir -p /workspace/worker_{i}/model && '
            f'nohup python -u /workspace/scripts/worker_selfplay.py '
            f'--worker-id {full_id} '
            f'--api-url {api_url} '
            f'--workspace /workspace/worker_{i} '
            f'--device cuda '
            f'--simulations {simulations} '
            f'--network-size {network_size} '
            f'--batch-size {batch_size} '
            f'--random-opening-moves {random_opening_moves} '
            f'--random-opening-fraction {random_opening_fraction} '
            f'</dev/null >/workspace/worker_{i}.log 2>&1 &'
        )
    return '\n'.join(lines)


def _build_trainer_onstart(
    api_url: str,
    network_size: str,
    training_threshold: int,
    trainer_batch_size: int,
    replay_buffer_size: int,
    gamma: float,
    td_lambda: float,
    epochs: int = 5,
    api_key: str = '',
) -> str:
    """Build the onstart-cmd string for a trainer instance."""
    return (
        f'export TRAINING_API_KEY="{api_key}" && '
        f'mkdir -p /workspace/output && '
        f'nohup python -u /workspace/scripts/trainer.py '
        f'--api-url {api_url} --device cuda --threshold {training_threshold} '
        f'--network-size {network_size} --output /workspace/output '
        f'--batch-size {trainer_batch_size} --replay-buffer-size {replay_buffer_size} '
        f'--gamma {gamma} --td-lambda {td_lambda} --epochs {epochs} '
        f'</dev/null >/workspace/trainer.log 2>&1 &'
    )


# Permanently blacklisted offers/machines that consistently fail
BLACKLISTED_OFFER_IDS: set[int] = {61866, 30784976, 30784982}  # machine 54400


class DistributedOrchestrator:
    """
    Orchestrates distributed training workers and optional trainer.
    """

    def __init__(
        self,
        num_workers: int,
        api_url: str,
        output_dir: Path,
        image: str = DEFAULT_IMAGE,
        gpu_name: Optional[str] = None,
        max_price: float = 0.15,
        min_reliability: float = 0.95,
        simulations: int = 2000,
        network_size: str = 'medium',
        with_trainer: bool = True,
        training_threshold: int = 50,
        workers_per_instance: int = 1,
        batch_size: int = 32,
        random_opening_moves: int = 0,
        random_opening_fraction: float = 0.0,
        trainer_batch_size: int = 512,
        replay_buffer_size: int = 100_000,
        gamma: float = 0.99,
        td_lambda: float = 0.95,
        api_key: str = '',
    ):
        self.num_workers = num_workers
        self.api_url = api_url
        self.api_key = api_key or os.environ.get('TRAINING_API_KEY', '')
        self.output_dir = Path(output_dir)
        self.image = image
        self.gpu_name = gpu_name
        self.max_price = max_price
        self.min_reliability = min_reliability
        self.simulations = simulations
        self.network_size = network_size
        self.with_trainer = with_trainer
        self.training_threshold = training_threshold
        self.workers_per_instance = workers_per_instance
        self.batch_size = batch_size
        self.random_opening_moves = random_opening_moves
        self.random_opening_fraction = random_opening_fraction
        self.trainer_batch_size = trainer_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.td_lambda = td_lambda

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.run_id = uuid.uuid4().hex[:6]  # Short unique ID for this run
        self.vast: Optional[VastAI] = None
        self.workers: list[WorkerInstance] = []
        self.trainer: Optional[WorkerInstance] = None
        self.shutdown_requested = False
        self.failed_offer_ids: set[int] = set(BLACKLISTED_OFFER_IDS)  # Offers that produced broken instances

    def find_offers(self) -> list[GPUOffer]:
        """Find suitable GPU offers."""
        print(f"\nSearching for GPU offers under ${self.max_price}/hr...")

        offers = self.vast.search_offers(
            gpu_name=self.gpu_name,
            max_dph=self.max_price,
            min_reliability=self.min_reliability,
            order_by='dph_total'
        )

        if not offers:
            print("No suitable offers found. Try:")
            print("  --max-price 0.25  (increase price limit)")
            print("  --gpu RTX_3060    (try different GPU)")
            return []

        print(f"Found {len(offers)} offers:")
        for i, offer in enumerate(offers[:10]):
            print(f"  {i+1}. {offer.gpu_name} - ${offer.dph_total:.3f}/hr "
                  f"({offer.gpu_ram:.0f}GB RAM, {offer.reliability:.1%} reliable)")

        return offers

    def _create_worker_instance(self, worker: WorkerInstance) -> bool:
        """Create a single worker instance with onstart_cmd."""
        onstart = _build_worker_onstart(
            worker_id=worker.worker_id,
            api_url=self.api_url,
            simulations=self.simulations,
            network_size=self.network_size,
            workers_per_instance=self.workers_per_instance,
            batch_size=self.batch_size,
            random_opening_moves=self.random_opening_moves,
            random_opening_fraction=self.random_opening_fraction,
            run_id=self.run_id,
            api_key=self.api_key,
        )
        try:
            worker.status = "creating"
            instance_id = self.vast.create_instance(
                worker.offer.id,
                image=self.image,
                disk=30,
                onstart_cmd=onstart,
                label=f'razzle-worker-{worker.worker_id}',
            )
            worker.instance_id = instance_id
            worker.status = "running"
            print(f"  Worker {worker.worker_id}: Instance {instance_id} "
                  f"({worker.offer.gpu_name} @ ${worker.offer.dph_total:.3f}/hr)")
            return True
        except Exception as e:
            worker.status = "failed"
            worker.error = str(e)
            print(f"  Worker {worker.worker_id}: Failed to create - {e}")
            return False

    def _create_trainer_instance(self, trainer: WorkerInstance) -> bool:
        """Create the trainer instance with onstart_cmd."""
        onstart = _build_trainer_onstart(
            api_url=self.api_url,
            network_size=self.network_size,
            training_threshold=self.training_threshold,
            trainer_batch_size=self.trainer_batch_size,
            replay_buffer_size=self.replay_buffer_size,
            gamma=self.gamma,
            td_lambda=self.td_lambda,
            api_key=self.api_key,
        )
        try:
            trainer.status = "creating"
            instance_id = self.vast.create_instance(
                trainer.offer.id,
                image=self.image,
                disk=30,
                onstart_cmd=onstart,
                label='razzle-trainer',
            )
            trainer.instance_id = instance_id
            trainer.status = "running"
            print(f"  Trainer: Instance {instance_id} "
                  f"({trainer.offer.gpu_name} @ ${trainer.offer.dph_total:.3f}/hr)")
            return True
        except Exception as e:
            trainer.status = "failed"
            trainer.error = str(e)
            print(f"  Trainer: Failed to create - {e}")
            return False

    def create_instances(self, offers: list[GPUOffer]) -> bool:
        """Create worker and trainer instances from offers."""
        total_needed = self.num_workers + (1 if self.with_trainer else 0)
        print(f"\nCreating {self.num_workers} worker instances" +
              (" + 1 trainer instance..." if self.with_trainer else "..."))

        if len(offers) < total_needed:
            print(f"Not enough offers: need {total_needed}, found {len(offers)}")
            return False

        # Create worker instances
        for i in range(self.num_workers):
            worker = WorkerInstance(worker_id=i, offer=offers[i], role="worker")
            self.workers.append(worker)
            self._create_worker_instance(worker)

        # Create trainer instance
        if self.with_trainer:
            self.trainer = WorkerInstance(worker_id=-1, offer=offers[self.num_workers], role="trainer")
            self._create_trainer_instance(self.trainer)

        return any(w.status == "running" for w in self.workers)

    def replace_failed_instance(self, failed_worker: WorkerInstance, offers: list[GPUOffer]) -> bool:
        """Replace a failed or unresponsive worker with a new instance."""
        role_name = "Trainer" if failed_worker.role == "trainer" else f"Worker {failed_worker.worker_id}"
        print(f"\n[{role_name}] Replacing failed instance {failed_worker.instance_id}...")

        # Blacklist the offer that produced the failed instance
        if failed_worker.offer:
            self.failed_offer_ids.add(failed_worker.offer.id)

        # Destroy the old instance
        if failed_worker.instance_id:
            try:
                self.vast.destroy_instance(failed_worker.instance_id)
                print(f"[{role_name}] Destroyed old instance {failed_worker.instance_id}")
            except Exception as e:
                print(f"[{role_name}] Failed to destroy old instance: {e}")

        # Find a new offer: skip in-use and previously failed offers
        used_offer_ids = {w.offer.id for w in self.workers if w.offer}
        if self.trainer and self.trainer.offer:
            used_offer_ids.add(self.trainer.offer.id)
        skip_ids = used_offer_ids | self.failed_offer_ids

        new_offer = None
        for offer in offers:
            if offer.id not in skip_ids:
                new_offer = offer
                break

        if not new_offer:
            # Refresh offers
            offers = self.find_offers()
            for offer in offers:
                if offer.id not in skip_ids:
                    new_offer = offer
                    break

        if not new_offer:
            print(f"[{role_name}] No available offers for replacement (skipped {len(self.failed_offer_ids)} blacklisted)")
            return False

        # Create replacement and reset activity timer
        failed_worker.offer = new_offer
        failed_worker.last_activity = time.time()  # Reset so it gets a full grace period
        if failed_worker.role == "trainer":
            return self._create_trainer_instance(failed_worker)
        else:
            return self._create_worker_instance(failed_worker)

    def _ensure_initial_model(self):
        """Create and upload initial model if none exists."""
        import requests
        from razzle.ai.network import create_network

        # Check if a model already exists
        try:
            response = requests.get(f"{self.api_url}/training/models/latest", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('model') is not None:
                    print(f"\nExisting model found: {data['model']['version']}")
                    return
        except Exception as e:
            print(f"Warning: Could not check for existing model: {e}")

        # Create and upload initial model
        print("\nNo model found. Creating initial model...")
        try:
            network = create_network(preset=self.network_size, device='cpu')

            model_path = self.output_dir / "initial_model.pt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            network.save(model_path)

            with open(model_path, 'rb') as f:
                files = {'file': ('initial.pt', f, 'application/octet-stream')}
                data = {
                    'version': 'initial',
                    'iteration': '0',
                    'games_trained_on': '0',
                }
                response = requests.post(
                    f"{self.api_url}/training/models",
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()

            print(f"Uploaded initial model: initial")

        except Exception as e:
            print(f"Warning: Could not create initial model: {e}")
            print("Workers will create their own random models.")

    def _cleanup_existing_instances(self):
        """Destroy any existing instances from previous runs."""
        try:
            existing = self.vast.list_instances()
            if existing:
                print(f"\nFound {len(existing)} existing instance(s) - cleaning up...")
                for inst in existing:
                    try:
                        self.vast.destroy_instance(inst.id)
                        print(f"  Destroyed existing instance {inst.id}")
                    except Exception as e:
                        print(f"  Failed to destroy {inst.id}: {e}")
                time.sleep(5)
                print("Cleanup complete")
        except Exception as e:
            print(f"Warning: Could not check for existing instances: {e}")

    def cleanup(self):
        """Destroy all instances."""
        print("\n" + "=" * 60)
        print("Cleaning up")
        print("=" * 60)

        for worker in self.workers:
            if worker.instance_id:
                try:
                    self.vast.destroy_instance(worker.instance_id)
                    print(f"  Destroyed worker instance {worker.instance_id}")
                except Exception as e:
                    print(f"  Failed to destroy worker {worker.instance_id}: {e}")

        if self.trainer and self.trainer.instance_id:
            try:
                self.vast.destroy_instance(self.trainer.instance_id)
                print(f"  Destroyed trainer instance {self.trainer.instance_id}")
            except Exception as e:
                print(f"  Failed to destroy trainer {self.trainer.instance_id}: {e}")

    def run(self):
        """Main orchestration loop."""
        print("=" * 60)
        print("Razzle Dazzle Distributed Training")
        print("=" * 60)

        print(f"\nConfiguration:")
        print(f"  Run ID: {self.run_id}")
        print(f"  Image: {self.image}")
        print(f"  Instances: {self.num_workers} worker + {'1 trainer' if self.with_trainer else 'no trainer'}")
        print(f"  Workers per instance: {self.workers_per_instance}")
        print(f"  Total worker processes: {self.num_workers * self.workers_per_instance}")
        print(f"  API URL: {self.api_url}")
        print(f"  GPU: {self.gpu_name or 'any'}")
        print(f"  Max price: ${self.max_price}/hr")
        print(f"  Network: {self.network_size}")
        print(f"  Simulations: {self.simulations}")
        print(f"  MCTS batch size: {self.batch_size}")
        print(f"  Training threshold: {self.training_threshold} games")
        print(f"  Random openings: {self.random_opening_moves} moves in {self.random_opening_fraction:.0%} of games")
        print(f"  Output: {self.output_dir}")

        # Initialize Vast.ai
        self.vast = VastAI()

        # Clean up any existing instances from previous runs
        self._cleanup_existing_instances()

        # Only cleanup on explicit Ctrl+C, not on normal exit
        # (instances should keep running if the orchestrator disconnects)

        # Find offers
        offers = self.find_offers()
        if not offers:
            return 1

        # Estimate cost
        num_instances = self.num_workers + (1 if self.with_trainer else 0)
        total_cost = sum(o.dph_total for o in offers[:num_instances])
        print(f"\nEstimated cost: ${total_cost:.2f}/hr ({num_instances} instances)")

        # Ensure initial model exists
        self._ensure_initial_model()

        # Create instances (image pull + onstart_cmd handles everything)
        if not self.create_instances(offers):
            return 1

        # Print instructions
        print("\n" + "=" * 60)
        print("Training started successfully!")
        print("=" * 60)
        print(f"\nWorkers are submitting games to: {self.api_url}")
        if self.with_trainer:
            print(f"Trainer is polling for games and training on GPU")
        else:
            print("\nTo train on collected games (run separately):")
            print(f"  python scripts/trainer.py --api-url {self.api_url} --device cuda")
        print("\nTo view training dashboard:")
        print(f"  curl {self.api_url}/training/dashboard | jq")
        print("\nTo stop training:")
        print("  Press Ctrl+C")
        print("\n" + "=" * 60)

        # Monitor via API dashboard (no SSH needed)
        # Build map of our full worker IDs -> instance worker_id
        # Workers register as "{run_id}_{sub_id}" in the dashboard
        our_worker_map: dict[str, int] = {}
        for w in self.workers:
            for i in range(self.workers_per_instance):
                sub_id = w.worker_id * self.workers_per_instance + i
                full_id = f'{self.run_id}_{sub_id}'
                our_worker_map[full_id] = w.worker_id

        print(f"[Monitor] Run ID: {self.run_id} — tracking {len(our_worker_map)} sub-workers")

        try:
            import requests
            check_interval = 60
            # Image pull + model download can take 5-10 min on slow hosts,
            # plus first game at 2000 sims takes 5+ min of self-play
            worker_inactivity_threshold = 1800  # 30 min before replacing

            while not self.shutdown_requested:
                time.sleep(check_interval)

                try:
                    response = requests.get(f"{self.api_url}/training/dashboard", timeout=10)
                    if response.status_code != 200:
                        print(f"[Status] API error: {response.status_code}")
                        continue

                    data = response.json()
                    games_total = data.get('games_total', 0)
                    games_pending = data.get('games_pending', 0)
                    workers_data = data.get('workers', {})

                    # Count only OUR active workers from dashboard
                    current_time = time.time()
                    our_active = 0
                    for worker_id_str, worker_info in workers_data.items():
                        if worker_id_str not in our_worker_map:
                            continue

                        instance_wid = our_worker_map[worker_id_str]
                        last_seen = worker_info.get('last_seen', '')
                        games = worker_info.get('games_submitted', worker_info.get('games', 0))

                        # Parse last_seen to check recency
                        try:
                            from datetime import datetime, timezone
                            ls = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                            age_sec = (datetime.now(timezone.utc) - ls).total_seconds()
                        except Exception:
                            age_sec = float('inf')

                        if age_sec < 300:  # Active in last 5 min
                            our_active += 1
                            for w in self.workers:
                                if w.worker_id == instance_wid:
                                    w.last_activity = current_time
                                    w.games_submitted = games

                    print(f"[Status] Games: {games_total} total, {games_pending} pending, "
                          f"{our_active}/{len(self.workers)} of our workers active")

                    # Replace unresponsive workers
                    for worker in self.workers:
                        if worker.status == "running":
                            inactive_time = current_time - worker.last_activity
                            if inactive_time > worker_inactivity_threshold:
                                print(f"[Health] Worker {worker.worker_id} unresponsive for {inactive_time:.0f}s - replacing")
                                worker.status = "failed"
                                self.replace_failed_instance(worker, offers)

                except Exception as e:
                    print(f"[Status] Check failed: {e}")

        except KeyboardInterrupt:
            print("\n\nShutdown requested - destroying all instances...")
            self.shutdown_requested = True
            self.cleanup()

        print("\nOrchestrator exiting. Instances are still running.")
        print("To destroy all instances: vastai destroy instance <id>")
        return 0

    def shutdown(self):
        """Signal shutdown."""
        self.shutdown_requested = True


def main():
    parser = argparse.ArgumentParser(description='Distributed training orchestrator')
    parser.add_argument('--workers', type=int, default=3, help='Number of workers')
    parser.add_argument('--api-url', type=str, default='https://razzledazzle.lazybrains.com/api',
                        help='Training API URL (default: https://razzledazzle.lazybrains.com/api)')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE,
                        help=f'Docker image for workers (default: {DEFAULT_IMAGE})')
    parser.add_argument('--gpu', type=str, default='RTX_3060', help='GPU type')
    parser.add_argument('--max-price', type=float, default=0.10, help='Max price per hour')
    parser.add_argument('--simulations', type=int, default=2000, help='MCTS simulations (default: 2000)')
    parser.add_argument('--network-size', type=str, default='medium', choices=['small', 'medium', 'large'],
                        help='Network size preset: small (~236K), medium (~2.4M), large (~24M)')
    parser.add_argument('--output', type=Path, default=Path('output/distributed'),
                        help='Output directory')
    parser.add_argument('--no-trainer', action='store_true',
                        help='Do not create a trainer instance (run trainer separately)')
    parser.add_argument('--threshold', type=int, default=200,
                        help='Number of games before training (default: 200)')
    parser.add_argument('--workers-per-instance', type=int, default=3,
                        help='Number of worker processes per GPU instance (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='MCTS batch size for GPU parallelism (default: 32)')
    parser.add_argument('--random-opening-moves', type=int, default=8,
                        help='Number of random moves at game start (default: 8)')
    parser.add_argument('--random-opening-fraction', type=float, default=0.3,
                        help='Fraction of games with random openings (default: 0.3)')
    parser.add_argument('--trainer-batch-size', type=int, default=512,
                        help='Training batch size (default: 512)')
    parser.add_argument('--replay-buffer-size', type=int, default=100_000,
                        help='Replay buffer capacity in positions (default: 100000)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Value discount factor for TD(λ) (default: 0.99)')
    parser.add_argument('--td-lambda', type=float, default=0.95,
                        help='TD(λ) trace decay (default: 0.95, 1.0=pure MC)')

    parser.add_argument('--api-key', type=str, default=None,
                        help='Training API key (default: from TRAINING_API_KEY env var or .env)')

    args = parser.parse_args()

    # Load .env file if it exists (for TRAINING_API_KEY etc.)
    # Check engine/.env first, then repo root .env
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip())

    # Ensure unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'

    orchestrator = DistributedOrchestrator(
        num_workers=args.workers,
        api_url=args.api_url,
        output_dir=args.output,
        image=args.image,
        gpu_name=args.gpu,
        max_price=args.max_price,
        simulations=args.simulations,
        network_size=args.network_size,
        with_trainer=not args.no_trainer,
        training_threshold=args.threshold,
        workers_per_instance=args.workers_per_instance,
        batch_size=args.batch_size,
        random_opening_moves=args.random_opening_moves,
        random_opening_fraction=args.random_opening_fraction,
        trainer_batch_size=args.trainer_batch_size,
        replay_buffer_size=args.replay_buffer_size,
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        api_key=args.api_key or '',
    )

    # Handle signals
    def signal_handler(signum, frame):
        orchestrator.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return orchestrator.run()


if __name__ == '__main__':
    sys.exit(main())
