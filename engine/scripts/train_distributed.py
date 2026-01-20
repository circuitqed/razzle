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
                    break
            except:
                pass
            time.sleep(10)
        else:
            raise RuntimeError("SSH not available after 5 minutes")

        # Create workspace directory
        vast.execute(worker.instance_id, "mkdir -p /workspace/model", timeout=60)

        # Upload package
        print(f"[{role_name}] Uploading package...")
        vast.copy_to(worker.instance_id, package_path, "/workspace/razzle_package.tar.gz")

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
            # Start worker process
            print(f"[{role_name}] Starting worker process...")
            start_cmd = f"""setsid python -u /workspace/worker_selfplay.py \
                --worker-id {worker.worker_id} \
                --api-url {api_url} \
                --workspace /workspace \
                --device cuda \
                --simulations {simulations} \
                --filters {filters} \
                --blocks {blocks} \
                </dev/null >/workspace/worker.log 2>&1 &
                sleep 2 && echo "Worker started"
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

    def wait_for_instances(self, timeout: int = 300) -> bool:
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
                        if info and info.status == "running" and info.ssh_host:
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
        print(f"  Workers: {self.num_workers}")
        print(f"  Trainer: {'Yes (cloud GPU)' if self.with_trainer else 'No (run separately)'}")
        print(f"  API URL: {self.api_url}")
        print(f"  GPU: {self.gpu_name or 'any'}")
        print(f"  Max price: ${self.max_price}/hr")
        print(f"  Simulations: {self.simulations}")
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
            while not self.shutdown_requested:
                time.sleep(10)

                # Check worker status
                running = 0
                for worker in self.workers:
                    if worker.instance_id and worker.status == "running":
                        try:
                            status = self.vast.execute(
                                worker.instance_id,
                                "cat /workspace/status.json 2>/dev/null | jq -r '.games_completed'",
                                timeout=30
                            )
                            games = int(status.strip()) if status.strip().isdigit() else 0
                            running += 1
                        except:
                            pass

                if running == 0:
                    print("\nAll workers stopped. Exiting.")
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
    parser.add_argument('--api-url', type=str, default='https://razzledazzle.lazybrains.com',
                        help='Training API URL (default: https://razzledazzle.lazybrains.com)')
    parser.add_argument('--gpu', type=str, default='RTX_3060', help='GPU type')
    parser.add_argument('--max-price', type=float, default=0.10, help='Max price per hour')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations')
    parser.add_argument('--filters', type=int, default=64, help='Network filters')
    parser.add_argument('--blocks', type=int, default=6, help='Network blocks')
    parser.add_argument('--output', type=Path, default=Path('output/distributed'),
                        help='Output directory')
    parser.add_argument('--no-trainer', action='store_true',
                        help='Do not create a trainer instance (run trainer separately)')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Number of games before training (default: 50)')

    args = parser.parse_args()

    # Ensure unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'

    orchestrator = DistributedOrchestrator(
        num_workers=args.workers,
        api_url=args.api_url,
        output_dir=args.output,
        gpu_name=args.gpu,
        max_price=args.max_price,
        simulations=args.simulations,
        filters=args.filters,
        blocks=args.blocks,
        with_trainer=not args.no_trainer,
        training_threshold=args.threshold,
    )

    # Handle signals
    def signal_handler(signum, frame):
        orchestrator.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return orchestrator.run()


if __name__ == '__main__':
    sys.exit(main())
