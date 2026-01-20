#!/usr/bin/env python3
"""
Distributed training orchestrator for Razzle Dazzle.

This is the main entry point for running distributed training on Vast.ai.
It coordinates:
1. Creating worker instances on Vast.ai
2. Uploading code and initial model to workers
3. Starting worker processes on each instance
4. Running the collector locally to aggregate games and train
5. Graceful cleanup on exit

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    VAST.AI CLOUD                            │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
    │  │Worker 0 │  │Worker 1 │  │Worker 2 │  │Worker N │       │
    │  │ GPU     │  │ GPU     │  │ GPU     │  │ GPU     │       │
    │  │selfplay │  │selfplay │  │selfplay │  │selfplay │       │
    │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
    │       │            │            │            │             │
    │       └────────────┼────────────┼────────────┘             │
    │                    │  Games collected via SCP              │
    └────────────────────┼───────────────────────────────────────┘
                         ▼
    ┌────────────────────────────────────────────────────────────┐
    │                    LOCAL MACHINE                           │
    │  ┌──────────────────────────────────────────────────────┐ │
    │  │                   Collector                           │ │
    │  │  - Polls workers every 30s                           │ │
    │  │  - Downloads completed games                          │ │
    │  │  - Trains when threshold reached (100 games)         │ │
    │  │  - Uploads new model to workers                      │ │
    │  └──────────────────────────────────────────────────────┘ │
    └────────────────────────────────────────────────────────────┘

Usage:
    # Start distributed training with 4 workers
    python scripts/train_distributed.py --workers 4 --gpu RTX_3060

    # Resume with existing model
    python scripts/train_distributed.py --workers 4 --model output/model.pt

    # Custom configuration
    python scripts/train_distributed.py \\
        --workers 8 \\
        --gpu RTX_3090 \\
        --max-price 0.20 \\
        --training-threshold 200 \\
        --simulations 800
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.vastai import VastAI, GPUOffer, Instance
from scripts.collector import Collector, WorkerInfo


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


def create_package(output_dir: Path) -> Path:
    """Create a tarball with the razzle package and worker script."""
    package_path = output_dir / "razzle_package.tar.gz"
    engine_dir = Path(__file__).parent.parent

    with tarfile.open(package_path, "w:gz") as tar:
        # Add razzle package
        tar.add(engine_dir / "razzle", arcname="razzle")

        # Add worker script
        tar.add(engine_dir / "scripts" / "worker_selfplay.py", arcname="worker_selfplay.py")

        # Add requirements if exists
        requirements = engine_dir / "requirements.txt"
        if requirements.exists():
            tar.add(requirements, arcname="requirements.txt")

    print(f"Created package: {package_path}")
    return package_path


def setup_worker_instance(
    vast: VastAI,
    worker: WorkerInstance,
    package_path: Path,
    model_path: Optional[Path],
    simulations: int,
    filters: int,
    blocks: int
) -> bool:
    """Set up a worker instance and start the worker process."""
    try:
        worker.status = "starting"
        print(f"[Worker {worker.worker_id}] Setting up instance {worker.instance_id}")

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

        # Create workspace directories
        vast.execute(worker.instance_id, """
            mkdir -p /workspace/pending /workspace/model
        """, timeout=60)

        # Upload package
        print(f"[Worker {worker.worker_id}] Uploading package...")
        vast.copy_to(worker.instance_id, package_path, "/workspace/razzle_package.tar.gz")

        # Upload initial model if provided
        if model_path and model_path.exists():
            print(f"[Worker {worker.worker_id}] Uploading model...")
            vast.copy_to(worker.instance_id, model_path, f"/workspace/model/{model_path.name}")

        # Extract and install
        print(f"[Worker {worker.worker_id}] Installing dependencies...")
        setup_cmd = """
            cd /workspace && \
            tar -xzf razzle_package.tar.gz && \
            pip install torch numpy --quiet 2>/dev/null
        """
        vast.execute(worker.instance_id, setup_cmd, timeout=600)

        # Start worker process in background using setsid to fully detach
        print(f"[Worker {worker.worker_id}] Starting worker process...")
        start_cmd = f"""setsid python -u /workspace/worker_selfplay.py \
            --worker-id {worker.worker_id} \
            --workspace /workspace \
            --device cuda \
            --simulations {simulations} \
            --filters {filters} \
            --blocks {blocks} \
            </dev/null >/workspace/worker.log 2>&1 &
            sleep 1 && echo "Worker started"
        """
        result = vast.execute(worker.instance_id, start_cmd, timeout=60)
        print(f"[Worker {worker.worker_id}] {result.strip()}")

        # Verify worker started
        time.sleep(5)
        status_check = vast.execute(worker.instance_id, "cat /workspace/status.json 2>/dev/null || echo 'no status yet'", timeout=30)
        print(f"[Worker {worker.worker_id}] Status: {status_check.strip()[:100]}")

        worker.status = "running"
        return True

    except Exception as e:
        worker.status = "failed"
        worker.error = str(e)
        print(f"[Worker {worker.worker_id}] Setup failed: {e}")
        return False


def get_worker_ssh_info(vast: VastAI, instance_id: int) -> tuple[str, int]:
    """Get SSH host and port for an instance."""
    instance = vast.get_instance(instance_id)
    if instance and instance.ssh_host:
        return instance.ssh_host, instance.ssh_port
    raise RuntimeError(f"Instance {instance_id} not ready")


class DistributedTrainer:
    """
    Orchestrates distributed training.
    """

    def __init__(
        self,
        num_workers: int,
        output_dir: Path,
        gpu_name: Optional[str] = None,
        max_price: float = 0.15,
        min_reliability: float = 0.95,
        simulations: int = 400,
        filters: int = 64,
        blocks: int = 6,
        epochs: int = 10,
        training_threshold: int = 100,
        poll_interval: int = 30,
        initial_model: Optional[Path] = None,
        training_device: str = 'cpu',
    ):
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        self.gpu_name = gpu_name
        self.max_price = max_price
        self.min_reliability = min_reliability
        self.simulations = simulations
        self.filters = filters
        self.blocks = blocks
        self.epochs = epochs
        self.training_threshold = training_threshold
        self.poll_interval = poll_interval
        self.initial_model = initial_model
        self.training_device = training_device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.vast: Optional[VastAI] = None
        self.workers: list[WorkerInstance] = []
        self.collector: Optional[Collector] = None
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
        """Create worker instances from offers."""
        print(f"\nCreating {self.num_workers} worker instances...")

        for i in range(self.num_workers):
            if i >= len(offers):
                print(f"Not enough offers for {self.num_workers} workers")
                return False

            offer = offers[i]
            worker = WorkerInstance(worker_id=i, offer=offer)
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

        return any(w.status == "creating" for w in self.workers)

    def wait_for_instances(self, timeout: int = 300) -> bool:
        """Wait for all instances to be ready."""
        print(f"\nWaiting for instances to be ready...")

        start = time.time()
        while time.time() - start < timeout:
            all_ready = True

            for worker in self.workers:
                if worker.status in ["failed", "ready"]:
                    continue

                try:
                    instance = self.vast.get_instance(worker.instance_id)
                    if instance and instance.ssh_host and instance.actual_status == 'running':
                        worker.host = instance.ssh_host
                        worker.port = instance.ssh_port
                        worker.status = "ready"
                        print(f"  Worker {worker.worker_id}: Ready ({worker.host}:{worker.port})")
                    else:
                        all_ready = False
                except:
                    all_ready = False

            if all_ready or all(w.status in ["ready", "failed"] for w in self.workers):
                break

            time.sleep(10)

        ready_count = sum(1 for w in self.workers if w.status == "ready")
        print(f"\n{ready_count}/{self.num_workers} instances ready")
        return ready_count > 0

    def setup_workers(self, package_path: Path) -> bool:
        """Set up all worker instances."""
        print(f"\nSetting up workers...")

        ready_workers = [w for w in self.workers if w.status == "ready"]

        with ThreadPoolExecutor(max_workers=len(ready_workers)) as executor:
            futures = {
                executor.submit(
                    setup_worker_instance,
                    self.vast, w, package_path, self.initial_model,
                    self.simulations, self.filters, self.blocks
                ): w
                for w in ready_workers
            }

            for future in as_completed(futures):
                worker = futures[future]
                success = future.result()
                if success:
                    print(f"  Worker {worker.worker_id}: Setup complete")
                else:
                    print(f"  Worker {worker.worker_id}: Setup failed - {worker.error}")

        running_count = sum(1 for w in self.workers if w.status == "running")
        print(f"\n{running_count}/{self.num_workers} workers running")
        return running_count > 0

    def create_workers_json(self) -> Path:
        """Create workers.json file for collector."""
        workers_data = []
        for worker in self.workers:
            if worker.status == "running" and worker.host:
                workers_data.append({
                    "worker_id": worker.worker_id,
                    "host": worker.host,
                    "port": worker.port
                })

        workers_file = self.output_dir / "workers.json"
        with open(workers_file, 'w') as f:
            json.dump(workers_data, f, indent=2)

        print(f"Created {workers_file} with {len(workers_data)} workers")
        return workers_file

    def run_collector(self, workers_file: Path):
        """Run the collector to aggregate games and train."""
        print(f"\n{'='*60}")
        print("Starting collector")
        print(f"{'='*60}")

        # Load worker info from JSON
        from scripts.collector import load_workers

        workers = load_workers(workers_file)

        self.collector = Collector(
            workers=workers,
            output_dir=self.output_dir,
            device=self.training_device,
            training_threshold=self.training_threshold,
            poll_interval=self.poll_interval,
            epochs=self.epochs,
            filters=self.filters,
            blocks=self.blocks,
            initial_model=self.initial_model
        )

        # Run collector (blocks until shutdown)
        self.collector.run()

    def cleanup(self):
        """Clean up all instances."""
        if self.shutdown_requested:
            return
        self.shutdown_requested = True

        print(f"\n{'='*60}")
        print("Cleaning up")
        print(f"{'='*60}")

        # Stop collector
        if self.collector:
            self.collector.shutdown()

        # Destroy instances
        for worker in self.workers:
            if worker.instance_id:
                try:
                    self.vast.destroy_instance(worker.instance_id)
                    print(f"  Destroyed instance {worker.instance_id}")
                except Exception as e:
                    print(f"  Failed to destroy {worker.instance_id}: {e}")

    def run(self):
        """Main entry point."""
        print("="*60)
        print("Razzle Dazzle Distributed Training")
        print("="*60)

        print(f"\nConfiguration:")
        print(f"  Workers: {self.num_workers}")
        print(f"  GPU: {self.gpu_name or 'any'}")
        print(f"  Max price: ${self.max_price}/hr")
        print(f"  Simulations: {self.simulations}")
        print(f"  Training threshold: {self.training_threshold} games")
        print(f"  Training device: {self.training_device}")
        print(f"  Output: {self.output_dir}")

        # Initialize Vast.ai
        try:
            self.vast = VastAI()
        except RuntimeError as e:
            print(f"\nError: {e}")
            return 1

        # Register cleanup handlers
        atexit.register(self.cleanup)

        def signal_handler(signum, frame):
            print("\nReceived shutdown signal...")
            self.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Find offers
        offers = self.find_offers()
        if len(offers) < self.num_workers:
            print(f"\nNot enough offers ({len(offers)}) for {self.num_workers} workers")
            return 1

        # Estimate cost
        total_cost_per_hour = sum(o.dph_total for o in offers[:self.num_workers])
        print(f"\nEstimated cost: ${total_cost_per_hour:.2f}/hr")

        # Create instances
        if not self.create_instances(offers):
            print("\nFailed to create any instances")
            return 1

        # Wait for instances
        if not self.wait_for_instances():
            print("\nNo instances became ready")
            return 1

        # Create package
        package_path = create_package(self.output_dir)

        # Setup workers
        if not self.setup_workers(package_path):
            print("\nNo workers set up successfully")
            return 1

        # Create workers.json
        workers_file = self.create_workers_json()

        # Run collector (this blocks until shutdown)
        try:
            self.run_collector(workers_file)
        except KeyboardInterrupt:
            pass

        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Distributed training on Vast.ai',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic 4-worker setup
    python scripts/train_distributed.py --workers 4

    # Specific GPU type
    python scripts/train_distributed.py --workers 4 --gpu RTX_3090

    # Higher budget, more games before training
    python scripts/train_distributed.py --workers 8 --max-price 0.25 --training-threshold 200

    # Resume with existing model
    python scripts/train_distributed.py --workers 4 --model output/model_iter_005.pt
        """
    )

    # Worker configuration
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker instances')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU type (e.g., RTX_3060, RTX_3090, RTX_4090)')
    parser.add_argument('--max-price', type=float, default=0.15,
                        help='Maximum price per hour per worker')
    parser.add_argument('--min-reliability', type=float, default=0.95,
                        help='Minimum reliability score')

    # Network configuration
    parser.add_argument('--filters', type=int, default=64,
                        help='Network filter count')
    parser.add_argument('--blocks', type=int, default=6,
                        help='Network residual blocks')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs per iteration')
    parser.add_argument('--training-threshold', type=int, default=100,
                        help='Games before triggering training')
    parser.add_argument('--poll-interval', type=int, default=30,
                        help='Seconds between polling workers')
    parser.add_argument('--training-device', type=str, default='cpu',
                        help='Device for local training (cpu or cuda)')

    # Model
    parser.add_argument('--model', type=Path, default=None,
                        help='Initial model checkpoint')

    # Output
    parser.add_argument('--output', type=Path, default=Path('output/distributed'),
                        help='Output directory')

    args = parser.parse_args()

    trainer = DistributedTrainer(
        num_workers=args.workers,
        output_dir=args.output,
        gpu_name=args.gpu,
        max_price=args.max_price,
        min_reliability=args.min_reliability,
        simulations=args.simulations,
        filters=args.filters,
        blocks=args.blocks,
        epochs=args.epochs,
        training_threshold=args.training_threshold,
        poll_interval=args.poll_interval,
        initial_model=args.model,
        training_device=args.training_device,
    )

    sys.exit(trainer.run())


if __name__ == '__main__':
    main()
