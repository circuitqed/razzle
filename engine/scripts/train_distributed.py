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
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
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
    status: str = "pending"  # pending, creating, ready, starting, running, failed
    error: Optional[str] = None
    role: str = "worker"  # worker or trainer
    last_activity: float = field(default_factory=time.time)
    games_submitted: int = 0


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
    random_opening_moves: int = 0,
    random_opening_fraction: float = 0.0,
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
            start_cmd = (
                f"setsid python -u /workspace/trainer.py "
                f"--api-url {api_url} --device cuda --threshold {training_threshold} "
                f"--filters {filters} --blocks {blocks} --output /workspace/output "
                f"</dev/null >/workspace/trainer.log 2>&1 & "
                f'sleep 1 && echo "Trainer started"'
            )
        else:
            # Start worker process(es)
            if workers_per_instance == 1:
                print(f"[{role_name}] Starting worker process...")
                start_cmd = (
                    f"setsid python -u /workspace/worker_selfplay.py "
                    f"--worker-id {worker.worker_id} --api-url {api_url} "
                    f"--workspace /workspace --device cuda --simulations {simulations} "
                    f"--filters {filters} --blocks {blocks} --batch-size {batch_size} "
                    f"--random-opening-moves {random_opening_moves} "
                    f"--random-opening-fraction {random_opening_fraction} "
                    f"</dev/null >/workspace/worker.log 2>&1 & "
                    f'sleep 1 && echo "Worker started"'
                )
            else:
                # Launch multiple workers sharing the GPU
                print(f"[{role_name}] Starting {workers_per_instance} worker processes...")
                worker_cmds = []
                for i in range(workers_per_instance):
                    sub_worker_id = worker.worker_id * workers_per_instance + i
                    worker_cmds.append(
                        f"nohup python -u /workspace/worker_selfplay.py "
                        f"--worker-id {sub_worker_id} "
                        f"--api-url {api_url} "
                        f"--workspace /workspace/worker_{i} "
                        f"--device cuda "
                        f"--simulations {simulations} "
                        f"--filters {filters} "
                        f"--blocks {blocks} "
                        f"--batch-size {batch_size} "
                        f"--random-opening-moves {random_opening_moves} "
                        f"--random-opening-fraction {random_opening_fraction} "
                        f">/workspace/worker_{i}.log 2>&1 &"
                    )
                # Create workspace dirs and launch all workers
                mkdir_cmds = " && ".join([f"mkdir -p /workspace/worker_{i}/model" for i in range(workers_per_instance)])
                # Use nohup for proper detachment, then disown to release from job table
                start_cmd = f'{mkdir_cmds} && {" ".join(worker_cmds)} disown -a && sleep 1 && echo "{workers_per_instance} workers started"'

        result = vast.execute(worker.instance_id, start_cmd, timeout=120)
        print(f"[{role_name}] {result.strip()}")

        # Mark as running - workers/trainer are now started
        worker.status = "running"

        # Optional status check (non-fatal if it fails due to vastai CLI bugs)
        try:
            time.sleep(5)
            log_file = "trainer.log" if worker.role == "trainer" else "status.json"
            status_check = vast.execute(
                worker.instance_id,
                f"cat /workspace/{log_file} 2>/dev/null | tail -5 || echo 'starting...'",
                timeout=30
            )
            print(f"[{role_name}] Status: {status_check.strip()[:100]}")
        except Exception as e:
            print(f"[{role_name}] Status check skipped (CLI error)")

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
        simulations: int = 2000,
        filters: int = 64,
        blocks: int = 6,
        with_trainer: bool = True,
        training_threshold: int = 50,
        workers_per_instance: int = 1,
        batch_size: int = 32,
        random_opening_moves: int = 0,
        random_opening_fraction: float = 0.0,
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
        self.random_opening_moves = random_opening_moves
        self.random_opening_fraction = random_opening_fraction

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

    def poll_and_setup_instances(self, package_path: Path, timeout: int = 600) -> int:
        """Poll instances and setup each one as soon as it's ready."""
        print(f"\nPolling instances and setting up as ready...")

        all_instances = self.workers + ([self.trainer] if self.trainer else [])
        setup_lock = Lock()

        # Track which instances have been submitted for setup
        setup_submitted = set()

        def check_and_setup(instance: WorkerInstance):
            """Check if instance is ready and start setup."""
            try:
                info = self.vast.get_instance(instance.instance_id)
                if info and info.actual_status == "running" and info.ssh_host:
                    instance.host = info.ssh_host
                    instance.port = info.ssh_port
                    name = "Trainer" if instance.role == "trainer" else f"Worker {instance.worker_id}"
                    print(f"  {name}: Ready ({info.ssh_host}:{info.ssh_port}) - starting setup")
                    instance.status = "ready"

                    # Immediately start setup
                    success = setup_worker_instance(
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
                        self.random_opening_moves,
                        self.random_opening_fraction,
                    )
                    if success:
                        instance.last_activity = time.time()
                    return success
                return None  # Not ready yet
            except Exception as e:
                return None  # Check failed, retry later

        start = time.time()
        executor = ThreadPoolExecutor(max_workers=len(all_instances))
        pending_futures: dict[Future, WorkerInstance] = {}

        try:
            while time.time() - start < timeout:
                # Submit checks for instances not yet running or in setup
                for instance in all_instances:
                    if instance.status in ["creating", "pending"] and instance.instance_id not in setup_submitted:
                        with setup_lock:
                            if instance.instance_id not in setup_submitted:
                                setup_submitted.add(instance.instance_id)
                                future = executor.submit(check_and_setup, instance)
                                pending_futures[future] = instance

                # Check completed futures
                done_futures = []
                for future in list(pending_futures.keys()):
                    if future.done():
                        done_futures.append(future)
                        instance = pending_futures[future]
                        try:
                            result = future.result()
                            if result is None:
                                # Not ready yet, resubmit after delay
                                with setup_lock:
                                    setup_submitted.discard(instance.instance_id)
                        except Exception as e:
                            name = "Trainer" if instance.role == "trainer" else f"Worker {instance.worker_id}"
                            print(f"  {name}: Setup error - {e}")
                            with setup_lock:
                                setup_submitted.discard(instance.instance_id)

                for future in done_futures:
                    del pending_futures[future]

                # Check progress
                running_count = sum(1 for w in self.workers if w.status == "running")
                trainer_running = self.trainer and self.trainer.status == "running"

                # Exit early if all are running (and no pending setup tasks)
                all_done = all(i.status in ["running", "failed"] for i in all_instances)
                if all_done and not pending_futures:
                    break

                time.sleep(5)  # Poll interval

        finally:
            # Wait for any remaining setup tasks to complete
            if pending_futures:
                print(f"Waiting for {len(pending_futures)} pending setup tasks...")
                for future in pending_futures:
                    try:
                        future.result(timeout=300)  # 5 min max per task
                    except Exception as e:
                        pass
            executor.shutdown(wait=True)

        running_count = sum(1 for w in self.workers if w.status == "running")
        trainer_running = self.trainer and self.trainer.status == "running"
        print(f"\n{running_count}/{self.num_workers} workers running" +
              (f", trainer {'running' if trainer_running else 'not running'}" if self.with_trainer else ""))
        return running_count

    def replace_failed_instance(self, failed_worker: WorkerInstance, offers: list[GPUOffer]) -> bool:
        """Replace a failed or unresponsive worker with a new instance."""
        role_name = "Trainer" if failed_worker.role == "trainer" else f"Worker {failed_worker.worker_id}"
        print(f"\n[{role_name}] Replacing failed instance {failed_worker.instance_id}...")

        # Destroy the old instance
        if failed_worker.instance_id:
            try:
                self.vast.destroy_instance(failed_worker.instance_id)
                print(f"[{role_name}] Destroyed old instance {failed_worker.instance_id}")
            except Exception as e:
                print(f"[{role_name}] Failed to destroy old instance: {e}")

        # Find a new offer (use one not already in use)
        used_offer_ids = {w.offer.id for w in self.workers if w.offer}
        if self.trainer and self.trainer.offer:
            used_offer_ids.add(self.trainer.offer.id)

        new_offer = None
        for offer in offers:
            if offer.id not in used_offer_ids:
                new_offer = offer
                break

        if not new_offer:
            # Refresh offers
            offers = self.find_offers()
            for offer in offers:
                if offer.id not in used_offer_ids:
                    new_offer = offer
                    break

        if not new_offer:
            print(f"[{role_name}] No available offers for replacement")
            return False

        # Create new instance
        try:
            failed_worker.offer = new_offer
            failed_worker.status = "creating"
            instance_id = self.vast.create_instance(
                new_offer.id,
                image='pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
                disk=30
            )
            failed_worker.instance_id = instance_id
            print(f"[{role_name}] Created new instance {instance_id} ({new_offer.gpu_name} @ ${new_offer.dph_total:.3f}/hr)")
            return True
        except Exception as e:
            print(f"[{role_name}] Failed to create replacement: {e}")
            failed_worker.status = "failed"
            return False

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
                # Wait for cleanup
                time.sleep(5)
                print("Cleanup complete")
        except Exception as e:
            print(f"Warning: Could not check for existing instances: {e}")

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
        print(f"  Random openings: {self.random_opening_moves} moves in {self.random_opening_fraction:.0%} of games")
        print(f"  Output: {self.output_dir}")

        # Initialize Vast.ai
        self.vast = VastAI()

        # Clean up any existing instances from previous runs
        self._cleanup_existing_instances()

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

        # Create package (do this while instances are starting)
        package_path = create_package(self.output_dir)

        # Ensure initial model exists
        self._ensure_initial_model()

        # Poll instances and setup each as soon as ready
        running_count = self.poll_and_setup_instances(package_path)
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

        # Wait for interrupt with health monitoring
        try:
            import requests
            check_interval = 60  # Check every 60 seconds
            consecutive_failures = 0
            max_failures = 10  # Exit after 10 consecutive failures (10 minutes)
            worker_inactivity_threshold = 300  # 5 minutes of no games = unhealthy
            last_worker_games: dict[str, int] = {}  # Track games per worker

            while not self.shutdown_requested:
                time.sleep(check_interval)

                # Check worker status via API dashboard
                try:
                    response = requests.get(f"{self.api_url}/training/dashboard", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        games_pending = data.get('games_pending', 0)
                        games_total = data.get('games_total', 0)
                        workers_data = data.get('workers', {})
                        workers_active = len(workers_data)

                        if games_total > 0 or workers_active > 0:
                            consecutive_failures = 0
                            print(f"[Status] Games: {games_total} total, {games_pending} pending, {workers_active} workers active")

                            # Check individual worker health
                            current_time = time.time()
                            for worker_id, worker_info in workers_data.items():
                                games = worker_info.get('games_submitted', 0)
                                # Check if worker has made progress
                                prev_games = last_worker_games.get(worker_id, 0)
                                if games > prev_games:
                                    # Update activity time for matching local worker
                                    for w in self.workers:
                                        if str(w.worker_id) in worker_id or worker_id in str(w.worker_id * self.workers_per_instance):
                                            w.last_activity = current_time
                                            w.games_submitted = games
                                last_worker_games[worker_id] = games

                            # Check for unresponsive local workers
                            for worker in self.workers:
                                if worker.status == "running":
                                    inactive_time = current_time - worker.last_activity
                                    if inactive_time > worker_inactivity_threshold:
                                        print(f"[Health] Worker {worker.worker_id} unresponsive for {inactive_time:.0f}s")
                                        # Try to replace it
                                        worker.status = "failed"
                                        if self.replace_failed_instance(worker, offers):
                                            # Let the monitoring loop pick it up
                                            pass
                        else:
                            consecutive_failures += 1
                            print(f"[Status] No activity detected ({consecutive_failures}/{max_failures})")
                    else:
                        consecutive_failures += 1
                        print(f"[Status] API error: {response.status_code}")
                except Exception as e:
                    consecutive_failures += 1
                    print(f"[Status] Check failed: {e}")

                # Check for instances that need setup (replacements)
                instances_needing_setup = [w for w in self.workers if w.status == "creating"]
                if self.trainer and self.trainer.status == "creating":
                    instances_needing_setup.append(self.trainer)

                if instances_needing_setup:
                    # Poll and setup any pending replacements
                    for instance in instances_needing_setup:
                        try:
                            info = self.vast.get_instance(instance.instance_id)
                            if info and info.actual_status == "running" and info.ssh_host:
                                instance.host = info.ssh_host
                                instance.port = info.ssh_port
                                name = "Trainer" if instance.role == "trainer" else f"Worker {instance.worker_id}"
                                print(f"  {name}: Replacement ready - starting setup")
                                # Setup in background thread
                                import threading
                                threading.Thread(
                                    target=setup_worker_instance,
                                    args=(
                                        self.vast, instance, package_path, self.api_url,
                                        self.simulations, self.filters, self.blocks,
                                        self.training_threshold, self.workers_per_instance,
                                        self.batch_size, self.random_opening_moves,
                                        self.random_opening_fraction,
                                    ),
                                    daemon=True
                                ).start()
                        except Exception as e:
                            pass  # Will retry next cycle

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
    parser.add_argument('--simulations', type=int, default=2000, help='MCTS simulations (default: 2000)')
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
    parser.add_argument('--workers-per-instance', type=int, default=3,
                        help='Number of worker processes per GPU instance (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='MCTS batch size for GPU parallelism (default: 32)')
    parser.add_argument('--random-opening-moves', type=int, default=8,
                        help='Number of random moves at game start (default: 8)')
    parser.add_argument('--random-opening-fraction', type=float, default=0.3,
                        help='Fraction of games with random openings (default: 0.3)')

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
        random_opening_moves=args.random_opening_moves,
        random_opening_fraction=args.random_opening_fraction,
    )

    # Handle signals
    def signal_handler(signum, frame):
        orchestrator.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return orchestrator.run()


if __name__ == '__main__':
    sys.exit(main())
