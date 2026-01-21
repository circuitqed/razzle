#!/usr/bin/env python3
"""
Distributed training orchestrator for Razzle Dazzle.

This script creates worker instances on Vast.ai that connect to a training
API server. Workers generate games and submit them via HTTP. A separate
trainer process fetches games and trains.

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
import json
import os
import signal
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.vastai import VastAI, GPUOffer


@dataclass
class WorkerInstance:
    """Tracks a worker instance through its lifecycle."""
    worker_id: int
    instance_id: Optional[int] = None
    offer: Optional[GPUOffer] = None
    host: Optional[str] = None
    port: Optional[int] = None
    status: str = "pending"  # pending, creating, starting, running, failed
    error: Optional[str] = None
    role: str = "worker"  # worker or trainer


def create_package(output_dir: Path) -> Path:
    """Create a tarball with the razzle package, worker, and trainer scripts."""
    package_path = output_dir / "razzle_package.tar.gz"
    engine_dir = Path(__file__).parent.parent

    with tarfile.open(package_path, "w:gz") as tar:
        # Add razzle package
        tar.add(engine_dir / "razzle", arcname="razzle")

        # Add worker script
        tar.add(engine_dir / "scripts" / "worker_selfplay.py", arcname="worker_selfplay.py")

        # Add trainer script
        tar.add(engine_dir / "scripts" / "trainer.py", arcname="trainer.py")

    print(f"Created package: {package_path}")
    return package_path


def setup_worker_instance(
    vast: VastAI,
    worker: WorkerInstance,
    package_path: Path,
    api_url: str,
    simulations: int,
    filters: int,
    blocks: int,
    training_threshold: int = 50,
    workers_per_instance: int = 1,
    batch_size: int = 32,
) -> bool:
    """Set up a worker instance and start the worker/trainer process."""
    role_name = "Trainer" if worker.role == "trainer" else f"Worker {worker.worker_id}"
    try:
        worker.status = "starting"
        print(f"[{role_name}] Setting up instance {worker.instance_id}")

        # Wait for SSH to be available
        for attempt in range(30):  # 5 minutes max
            try:
                result = vast.execute(worker.instance_id, "echo ready", timeout=30)
                if "ready" in result:
                    print(f"[{role_name}] SSH ready after {attempt+1} attempts")
                    break
            except Exception as e:
                if attempt % 5 == 0:  # Log every 5 attempts
                    print(f"[{role_name}] SSH attempt {attempt+1}/30: {str(e)[:50]}")
            time.sleep(10)
        else:
            raise RuntimeError("SSH not available after 5 minutes")

        # Create workspace directory
        print(f"[{role_name}] Creating workspace...")
        vast.execute(worker.instance_id, "mkdir -p /workspace/model", timeout=60)

        # Upload package
        print(f"[{role_name}] Uploading package...")
        vast.copy_to(worker.instance_id, package_path, "/workspace/razzle_package.tar.gz")
        print(f"[{role_name}] Package uploaded")

        # Extract and install
        print(f"[{role_name}] Installing dependencies...")
        setup_cmd = """
            cd /workspace && \
            tar -xzf razzle_package.tar.gz && \
            pip install torch numpy requests --quiet 2>/dev/null
        """
        vast.execute(worker.instance_id, setup_cmd, timeout=600)

        if worker.role == "trainer":
            # Start trainer process
            print(f"[{role_name}] Starting trainer process...")
            start_cmd = f"""setsid python -u /workspace/trainer.py \
                --api-url {api_url} \
                --device cuda \
                --threshold {training_threshold} \
                --filters {filters} \
                --blocks {blocks} \
                --output /workspace/output \
                </dev/null >/workspace/trainer.log 2>&1 &
                sleep 2 && echo "Trainer started"
            """
        else:
            # Start worker process(es)
            if workers_per_instance == 1:
                print(f"[{role_name}] Starting worker process...")
                start_cmd = f"""setsid python -u /workspace/worker_selfplay.py \
                    --worker-id {worker.worker_id} \
                    --api-url {api_url} \
                    --workspace /workspace \
                    --device cuda \
                    --simulations {simulations} \
                    --filters {filters} \
                    --blocks {blocks} \
                    --batch-size {batch_size} \
                    </dev/null >/workspace/worker.log 2>&1 &
                    sleep 2 && echo "Worker started"
                """
            else:
                # Launch multiple workers sharing the GPU
                print(f"[{role_name}] Starting {workers_per_instance} worker processes...")
                worker_cmds = []
                for i in range(workers_per_instance):
                    sub_worker_id = worker.worker_id * workers_per_instance + i
                    worker_cmds.append(
                        f"setsid python -u /workspace/worker_selfplay.py "
                        f"--worker-id {sub_worker_id} "
                        f"--api-url {api_url} "
                        f"--workspace /workspace/worker_{i} "
                        f"--device cuda "
                        f"--simulations {simulations} "
                        f"--filters {filters} "
                        f"--blocks {blocks} "
                        f"--batch-size {batch_size} "
                        f"</dev/null >/workspace/worker_{i}.log 2>&1 &"
                    )
                # Create workspace dirs and launch all workers
                mkdir_cmds = " && ".join([f"mkdir -p /workspace/worker_{i}/model" for i in range(workers_per_instance)])
                # Join backgrounded commands with space, not &&, since & doesn't return exit status for &&
                start_cmd = f"""{mkdir_cmds} && {" ".join(worker_cmds)}
                    sleep 2 && echo "{workers_per_instance} workers started"
                """

        result = vast.execute(worker.instance_id, start_cmd, timeout=60)
        print(f"[{role_name}] {result.strip()}")

        # Verify process started
        time.sleep(5)
        log_file = "trainer.log" if worker.role == "trainer" else "status.json"
        status_check = vast.execute(
            worker.instance_id,
            f"cat /workspace/{log_file} 2>/dev/null | tail -5 || echo 'starting...'",
            timeout=30
        )
        print(f"[{role_name}] Status: {status_check.strip()[:100]}")

        worker.status = "running"
        return True

    except Exception as e:
        worker.status = "failed"
        worker.error = str(e)
        print(f"[{role_name}] Setup failed: {e}")
        return False


class DistributedOrchestrator:
    """
    Orchestrates distributed training workers and optional trainer.
    """

    def __init__(
        self,
        num_workers: int,
        api_url: str,
        output_dir: Path,
        gpu_name: Optional[str] = None,
        max_price: float = 0.15,
        min_reliability: float = 0.95,
        simulations: int = 400,
        filters: int = 64,
        blocks: int = 6,
        with_trainer: bool = True,
        training_threshold: int = 50,
        workers_per_instance: int = 1,
        batch_size: int = 32,
    ):
        self.num_workers = num_workers
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.gpu_name = gpu_name
        self.max_price = max_price
        self.min_reliability = min_reliability
        self.simulations = simulations
        self.filters = filters
        self.blocks = blocks
        self.with_trainer = with_trainer
        self.training_threshold = training_threshold
        self.workers_per_instance = workers_per_instance
        self.batch_size = batch_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.vast: Optional[VastAI] = None
        self.workers: list[WorkerInstance] = []
        self.trainer: Optional[WorkerInstance] = None
        self.shutdown_requested = False

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
            offer = offers[i]
            worker = WorkerInstance(worker_id=i, offer=offer, role="worker")
            self.workers.append(worker)

            try:
                worker.status = "creating"
                instance_id = self.vast.create_instance(
                    offer.id,
                    image='pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
                    disk=30
                )
                worker.instance_id = instance_id
                print(f"  Worker {i}: Instance {instance_id} ({offer.gpu_name} @ ${offer.dph_total:.3f}/hr)")

            except Exception as e:
                worker.status = "failed"
                worker.error = str(e)
                print(f"  Worker {i}: Failed to create - {e}")

        # Create trainer instance
        if self.with_trainer:
            offer = offers[self.num_workers]
            self.trainer = WorkerInstance(worker_id=-1, offer=offer, role="trainer")

            try:
                self.trainer.status = "creating"
                instance_id = self.vast.create_instance(
                    offer.id,
                    image='pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
                    disk=30
                )
                self.trainer.instance_id = instance_id
                print(f"  Trainer: Instance {instance_id} ({offer.gpu_name} @ ${offer.dph_total:.3f}/hr)")

            except Exception as e:
                self.trainer.status = "failed"
                self.trainer.error = str(e)
                print(f"  Trainer: Failed to create - {e}")

        return any(w.status == "creating" for w in self.workers)

    def wait_for_instances(self, timeout: int = 600) -> bool:
        """Wait for all instances to be ready."""
        print(f"\nWaiting for instances to be ready...")

        # Combine workers and trainer for waiting
        all_instances = self.workers + ([self.trainer] if self.trainer else [])

        start = time.time()
        while time.time() - start < timeout:
            all_ready = True
            for instance in all_instances:
                if instance.status not in ["failed", "running", "ready"]:
                    try:
                        info = self.vast.get_instance(instance.instance_id)
                        if info and info.actual_status == "running" and info.ssh_host:
                            instance.host = info.ssh_host
                            instance.port = info.ssh_port
                            name = "Trainer" if instance.role == "trainer" else f"Worker {instance.worker_id}"
                            print(f"  {name}: Ready ({info.ssh_host}:{info.ssh_port})")
                            instance.status = "ready"
                        else:
                            all_ready = False
                    except Exception as e:
                        all_ready = False

            if all_ready:
                break
            time.sleep(10)

        ready_count = sum(1 for w in self.workers if w.status == "ready")
        trainer_ready = self.trainer and self.trainer.status == "ready"
        print(f"\n{ready_count}/{self.num_workers} workers ready" +
              (f", trainer {'ready' if trainer_ready else 'not ready'}" if self.with_trainer else ""))
        return ready_count > 0

    def setup_workers(self, package_path: Path) -> int:
        """Set up all worker and trainer instances in parallel."""
        print(f"\nSetting up instances...")

        ready_workers = [w for w in self.workers if w.status == "ready"]
        ready_instances = ready_workers[:]
        if self.trainer and self.trainer.status == "ready":
            ready_instances.append(self.trainer)

        with ThreadPoolExecutor(max_workers=len(ready_instances)) as executor:
            futures = {
                executor.submit(
                    setup_worker_instance,
                    self.vast,
                    instance,
                    package_path,
                    self.api_url,
                    self.simulations,
                    self.filters,
                    self.blocks,
                    self.training_threshold,
                    self.workers_per_instance,
                    self.batch_size,
                ): instance
                for instance in ready_instances
            }

            for future in as_completed(futures):
                instance = futures[future]
                name = "Trainer" if instance.role == "trainer" else f"Worker {instance.worker_id}"
                try:
                    success = future.result()
                    if success:
                        print(f"  {name}: Setup complete")
                except Exception as e:
                    print(f"  {name}: Setup failed - {e}")

        running_count = sum(1 for w in self.workers if w.status == "running")
        trainer_running = self.trainer and self.trainer.status == "running"
        print(f"\n{running_count}/{self.num_workers} workers running" +
              (f", trainer {'running' if trainer_running else 'not running'}" if self.with_trainer else ""))
        return running_count

    def _ensure_initial_model(self):
        """Create and upload initial model if none exists."""
        import requests
        from razzle.ai.network import create_network, RazzleNet

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
            # Create network with same config as workers
            network = create_network(self.filters, self.blocks, 'cpu')

            # Save locally
            model_path = self.output_dir / "initial_model.pt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            network.save(model_path)

            # Upload to API
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

    def cleanup(self):
        """Destroy all instances."""
        print("\n" + "=" * 60)
        print("Cleaning up")
        print("=" * 60)

        # Destroy worker instances
        for worker in self.workers:
            if worker.instance_id:
                try:
                    self.vast.destroy_instance(worker.instance_id)
                    print(f"  Destroyed worker instance {worker.instance_id}")
                except Exception as e:
                    print(f"  Failed to destroy worker {worker.instance_id}: {e}")

        # Destroy trainer instance
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
        print(f"  Instances: {self.num_workers} worker + {'1 trainer' if self.with_trainer else 'no trainer'}")
        print(f"  Workers per instance: {self.workers_per_instance}")
        print(f"  Total worker processes: {self.num_workers * self.workers_per_instance}")
        print(f"  API URL: {self.api_url}")
        print(f"  GPU: {self.gpu_name or 'any'}")
        print(f"  Max price: ${self.max_price}/hr")
        print(f"  Network: {self.filters} filters, {self.blocks} blocks")
        print(f"  Simulations: {self.simulations}")
        print(f"  MCTS batch size: {self.batch_size}")
        print(f"  Training threshold: {self.training_threshold} games")
        print(f"  Output: {self.output_dir}")

        # Initialize Vast.ai
        self.vast = VastAI()

        # Register cleanup
        atexit.register(self.cleanup)

        # Find offers
        offers = self.find_offers()
        if not offers:
            return 1

        # Estimate cost
        num_instances = self.num_workers + (1 if self.with_trainer else 0)
        total_cost = sum(o.dph_total for o in offers[:num_instances])
        print(f"\nEstimated cost: ${total_cost:.2f}/hr ({num_instances} instances)")

        # Create instances
        if not self.create_instances(offers):
            return 1

        # Wait for instances
        if not self.wait_for_instances():
            self.cleanup()
            return 1

        # Create package
        package_path = create_package(self.output_dir)

        # Ensure initial model exists
        self._ensure_initial_model()

        # Setup workers
        running_count = self.setup_workers(package_path)
        if running_count == 0:
            self.cleanup()
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

        # Wait for interrupt
        try:
            check_interval = 60  # Check every 60 seconds
            consecutive_failures = 0
            max_failures = 10  # Exit after 10 consecutive failures (10 minutes)

            while not self.shutdown_requested:
                time.sleep(check_interval)

                # Check worker status via API dashboard
                try:
                    import requests
                    response = requests.get(f"{self.api_url}/training/dashboard", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        games_pending = data.get('games_pending', 0)
                        games_total = data.get('games_total', 0)
                        workers_active = len(data.get('workers', {}))

                        if games_total > 0 or workers_active > 0:
                            consecutive_failures = 0
                            print(f"[Status] Games: {games_total} total, {games_pending} pending, {workers_active} workers active")
                        else:
                            consecutive_failures += 1
                            print(f"[Status] No activity detected ({consecutive_failures}/{max_failures})")
                    else:
                        consecutive_failures += 1
                        print(f"[Status] API error: {response.status_code}")
                except Exception as e:
                    consecutive_failures += 1
                    print(f"[Status] Check failed: {e}")

                if consecutive_failures >= max_failures:
                    print("\nNo activity detected for too long. Exiting.")
                    break

        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
            self.shutdown_requested = True

        self.cleanup()
        return 0

    def shutdown(self):
        """Signal shutdown."""
        self.shutdown_requested = True


def main():
    parser = argparse.ArgumentParser(description='Distributed training orchestrator')
    parser.add_argument('--workers', type=int, default=3, help='Number of workers')
    parser.add_argument('--api-url', type=str, default='https://razzledazzle.lazybrains.com/api',
                        help='Training API URL (default: https://razzledazzle.lazybrains.com/api)')
    parser.add_argument('--gpu', type=str, default='RTX_3060', help='GPU type')
    parser.add_argument('--max-price', type=float, default=0.10, help='Max price per hour')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations')
    parser.add_argument('--filters', type=int, default=None, help='Network filters (overrides --network-size)')
    parser.add_argument('--blocks', type=int, default=None, help='Network blocks (overrides --network-size)')
    parser.add_argument('--network-size', type=str, default='medium', choices=['small', 'medium', 'large'],
                        help='Network size preset: small (64f/6b), medium (128f/10b), large (256f/15b)')
    parser.add_argument('--output', type=Path, default=Path('output/distributed'),
                        help='Output directory')
    parser.add_argument('--no-trainer', action='store_true',
                        help='Do not create a trainer instance (run trainer separately)')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Number of games before training (default: 50)')
    parser.add_argument('--workers-per-instance', type=int, default=1,
                        help='Number of worker processes per GPU instance (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='MCTS batch size for GPU parallelism (default: 32)')

    args = parser.parse_args()

    # Resolve network size presets
    NETWORK_PRESETS = {
        'small': (64, 6),      # ~900K params, fast inference
        'medium': (128, 10),   # ~3.5M params, balanced
        'large': (256, 15),    # ~15M params, stronger but slower
    }

    # Use explicit args if provided, otherwise use preset
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

    orchestrator = DistributedOrchestrator(
        num_workers=args.workers,
        api_url=args.api_url,
        output_dir=args.output,
        gpu_name=args.gpu,
        max_price=args.max_price,
        simulations=args.simulations,
        filters=filters,
        blocks=blocks,
        with_trainer=not args.no_trainer,
        training_threshold=args.threshold,
        workers_per_instance=args.workers_per_instance,
        batch_size=args.batch_size,
    )

    # Handle signals
    def signal_handler(signum, frame):
        orchestrator.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return orchestrator.run()


if __name__ == '__main__':
    sys.exit(main())
