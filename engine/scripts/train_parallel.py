#!/usr/bin/env python3
"""
Parallel cloud training script for Razzle Dazzle using Vast.ai.

Launches multiple GPU instances to generate self-play games in parallel,
aggregates the data, and trains on the combined dataset.

This is more cost-efficient than sequential training because:
1. Self-play (30+ min) is the bottleneck, not training (10 sec)
2. Multiple cheap GPUs (RTX 3060 @ $0.04/hr) can be more efficient than one expensive GPU
3. Linear speedup for game generation with N workers

Architecture:
- N worker instances generate self-play games
- Games are downloaded incrementally
- Training happens on aggregated data (can use one worker or local)

Usage:
    # 4 parallel workers with RTX 3060 (cheapest)
    python scripts/train_parallel.py --workers 4 --gpu RTX_3060 --games-per-worker 50

    # 2 workers with RTX 3090 for faster per-worker throughput
    python scripts/train_parallel.py --workers 2 --gpu RTX_3090 --games-per-worker 100

    # Mixed: use any available cheap GPUs
    python scripts/train_parallel.py --workers 4 --max-price 0.10 --games-per-worker 50
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.vastai import VastAI, GPUOffer, Instance


@dataclass
class WorkerState:
    """Track state of a worker instance."""
    worker_id: int
    instance_id: Optional[int] = None
    offer: Optional[GPUOffer] = None
    status: str = "pending"  # pending, starting, running, completed, failed
    games_completed: int = 0
    error: Optional[str] = None


def create_package(output_dir: Path) -> Path:
    """Create a tarball with the razzle package."""
    package_path = output_dir / "razzle_package.tar.gz"
    engine_dir = Path(__file__).parent.parent

    with tarfile.open(package_path, "w:gz") as tar:
        tar.add(engine_dir / "razzle", arcname="razzle")
        tar.add(engine_dir / "scripts" / "train_local.py", arcname="train_local.py")
        requirements = engine_dir / "requirements.txt"
        if requirements.exists():
            tar.add(requirements, arcname="requirements.txt")

    return package_path


def setup_worker(vast: VastAI, worker: WorkerState, package_path: Path, model_path: Optional[Path]) -> bool:
    """Set up a worker instance (upload code, install deps)."""
    try:
        worker.status = "starting"

        # Wait for SSH
        for i in range(24):  # Up to 4 minutes
            try:
                vast.execute(worker.instance_id, "echo 'ready'", timeout=60)
                break
            except:
                time.sleep(10)
        else:
            raise RuntimeError("SSH not available")

        # Create workspace
        vast.execute(worker.instance_id, "mkdir -p /workspace/output")

        # Upload package
        vast.copy_to(worker.instance_id, package_path, "/workspace/razzle_package.tar.gz")

        # Upload starting model if provided
        if model_path and model_path.exists():
            vast.copy_to(worker.instance_id, model_path, "/workspace/model_start.pt")

        # Setup environment
        setup_cmd = """cd /workspace && tar -xzf razzle_package.tar.gz && pip install torch numpy --quiet"""
        vast.execute(worker.instance_id, setup_cmd, timeout=600)

        worker.status = "running"
        return True

    except Exception as e:
        worker.error = str(e)
        worker.status = "failed"
        return False


def run_selfplay_on_worker(
    vast: VastAI,
    worker: WorkerState,
    games: int,
    simulations: int,
    filters: int,
    blocks: int,
    model_path: Optional[Path]
) -> Optional[Path]:
    """Run self-play on a worker and return path to downloaded games."""
    try:
        resume_arg = "--resume /workspace/model_start.pt" if model_path else ""

        # Run self-play only (1 iteration, skip training)
        cmd = f"""cd /workspace && python train_local.py \\
            --iterations 1 \\
            --games-per-iter {games} \\
            --simulations {simulations} \\
            --filters {filters} \\
            --blocks {blocks} \\
            --device cuda \\
            --output output/ \\
            --selfplay-only \\
            {resume_arg}"""

        output = vast.execute(worker.instance_id, cmd, timeout=7200)  # 2 hour timeout
        print(f"[Worker {worker.worker_id}] Self-play output:\n{output[-500:]}")

        worker.games_completed = games
        return True

    except Exception as e:
        worker.error = str(e)
        worker.status = "failed"
        return False


def download_games(vast: VastAI, worker: WorkerState, output_dir: Path) -> Optional[Path]:
    """Download games from a worker."""
    try:
        games_dir = output_dir / f"worker_{worker.worker_id}_games"
        games_dir.mkdir(parents=True, exist_ok=True)

        # Package games directory on remote
        vast.execute(worker.instance_id, "cd /workspace && tar -czf games.tar.gz output/games_iter_*", timeout=120)

        # Download
        vast.copy_from(worker.instance_id, "/workspace/games.tar.gz", games_dir / "games.tar.gz")

        # Extract
        with tarfile.open(games_dir / "games.tar.gz", "r:gz") as tar:
            tar.extractall(games_dir)

        return games_dir

    except Exception as e:
        print(f"[Worker {worker.worker_id}] Failed to download games: {e}")
        return None


def aggregate_games(game_dirs: list[Path], output_dir: Path) -> Path:
    """Aggregate games from multiple workers into a single directory."""
    aggregated = output_dir / "aggregated_games"
    aggregated.mkdir(parents=True, exist_ok=True)

    all_games = []

    for game_dir in game_dirs:
        # Look for pickle files (games are saved as all_games.pkl by --selfplay-only)
        for pkl_file in game_dir.rglob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    games = pickle.load(f)
                    if isinstance(games, list):
                        all_games.extend(games)
                        print(f"Loaded {len(games)} games from {pkl_file}")
            except Exception as e:
                print(f"Failed to load {pkl_file}: {e}")

    # Save aggregated games
    output_file = aggregated / "all_games.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(all_games, f)

    print(f"Aggregated {len(all_games)} total games")
    return aggregated


def train_on_aggregated(
    vast: VastAI,
    trainer_instance_id: int,
    game_tarballs: list[Path],
    model_path: Optional[Path],
    epochs: int,
    filters: int,
    blocks: int,
    output_dir: Path
) -> Optional[Path]:
    """Train on aggregated games using one of the worker instances.

    Args:
        game_tarballs: List of local paths to games.tar.gz files from each worker
    """
    try:
        # Create games directory on remote
        vast.execute(trainer_instance_id, "rm -rf /workspace/all_worker_games && mkdir -p /workspace/all_worker_games", timeout=60)

        # Upload and extract each worker's games tarball on the remote
        for i, tarball in enumerate(game_tarballs):
            if tarball.exists():
                remote_tar = f"/workspace/worker_{i}.tar.gz"
                print(f"  Uploading {tarball.name}...")
                vast.copy_to(trainer_instance_id, tarball, remote_tar)
                # Extract to all_worker_games directory
                vast.execute(trainer_instance_id,
                    f"cd /workspace/all_worker_games && tar -xzf {remote_tar} --strip-components=1",
                    timeout=120)

        # Upload model if exists
        resume_arg = ""
        if model_path and model_path.exists():
            vast.copy_to(trainer_instance_id, model_path, "/workspace/model_start.pt")
            resume_arg = "--resume /workspace/model_start.pt"

        # Run training using train_local.py with --train-only flag
        # After --strip-components=1, games are in all_worker_games/games_iter_000
        train_cmd = f"""cd /workspace && python train_local.py \\
            --train-only \\
            --games-dir all_worker_games/games_iter_000 \\
            --epochs {epochs} \\
            --filters {filters} \\
            --blocks {blocks} \\
            --device cuda \\
            --output output/ \\
            {resume_arg}"""

        output = vast.execute(trainer_instance_id, train_cmd, timeout=1800)
        print(f"Training output:\n{output}")

        # Download trained model
        model_file = output_dir / "trained_model.pt"
        vast.copy_from(trainer_instance_id, "/workspace/output/trained_model.pt", model_file)
        return model_file

    except Exception as e:
        print(f"Training failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Parallel cloud training on Vast.ai')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--gpu', type=str, default=None, help='Specific GPU type (e.g., RTX_3060, RTX_3090)')
    parser.add_argument('--max-price', type=float, default=0.15, help='Max $/hr per worker')
    parser.add_argument('--min-reliability', type=float, default=0.95, help='Min reliability score')

    # Training parameters
    parser.add_argument('--iterations', type=int, default=5, help='Training iterations')
    parser.add_argument('--games-per-worker', type=int, default=50, help='Games per worker per iteration')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs per iteration')
    parser.add_argument('--filters', type=int, default=64, help='Network filters')
    parser.add_argument('--blocks', type=int, default=6, help='Network blocks')

    # Model
    parser.add_argument('--model', type=Path, help='Starting model checkpoint')
    parser.add_argument('--output', type=Path, default=Path('output/parallel_run'), help='Output directory')

    # Options
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    parser.add_argument('--keep-instances', action='store_true', help='Keep instances running after completion')

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    total_games_per_iter = args.workers * args.games_per_worker

    print("=" * 60)
    print("Razzle Dazzle Parallel Cloud Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Workers: {args.workers}")
    print(f"  Games per worker: {args.games_per_worker}")
    print(f"  Total games per iteration: {total_games_per_iter}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Total games: {total_games_per_iter * args.iterations}")

    # Initialize Vast.ai
    try:
        vast = VastAI()
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    # Search for GPU offers
    print(f"\nSearching for GPU offers under ${args.max_price}/hr...")
    offers = vast.search_offers(
        gpu_name=args.gpu,
        max_dph=args.max_price,
        min_reliability=args.min_reliability,
        order_by='dph_total'
    )

    if len(offers) < args.workers:
        print(f"Only found {len(offers)} suitable offers, need {args.workers}")
        print("Try: --max-price 0.20 or --workers 1")
        return

    print(f"\nFound {len(offers)} offers, using top {args.workers}:")
    for i, offer in enumerate(offers[:args.workers]):
        print(f"  {i+1}. {offer.gpu_name} - ${offer.dph_total:.3f}/hr ({offer.gpu_ram:.0f}GB)")

    estimated_cost_per_iter = sum(o.dph_total for o in offers[:args.workers]) * 0.5  # ~30 min
    print(f"\nEstimated cost: ~${estimated_cost_per_iter:.2f}/iteration, ~${estimated_cost_per_iter * args.iterations:.2f} total")

    if args.dry_run:
        print("\n[DRY RUN] Would create instances and run training")
        return

    # Create package
    print("\nCreating package...")
    package_path = create_package(args.output)

    # Initialize workers
    workers = [WorkerState(worker_id=i, offer=offers[i]) for i in range(args.workers)]

    # Track all instance IDs for cleanup
    instance_ids = []

    try:
        # Create instances in parallel
        print("\nCreating instances...")
        for worker in workers:
            instance_id = vast.create_instance(
                worker.offer.id,
                image='pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
                disk=30
            )
            worker.instance_id = instance_id
            instance_ids.append(instance_id)
            print(f"  Worker {worker.worker_id}: Instance {instance_id} ({worker.offer.gpu_name})")

        # Wait for all instances to be ready
        print("\nWaiting for instances to be ready...")
        for worker in workers:
            for _ in range(60):  # 5 min timeout
                instance = vast.get_instance(worker.instance_id)
                if instance and instance.ssh_host and instance.actual_status == 'running':
                    print(f"  Worker {worker.worker_id}: Ready ({instance.ssh_host}:{instance.ssh_port})")
                    break
                time.sleep(5)
            else:
                worker.status = "failed"
                worker.error = "Instance not ready"
                print(f"  Worker {worker.worker_id}: Timeout waiting for instance")

        # Setup workers in parallel
        print("\nSetting up workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(setup_worker, vast, w, package_path, args.model): w
                for w in workers if w.status != "failed"
            }
            for future in as_completed(futures):
                worker = futures[future]
                if future.result():
                    print(f"  Worker {worker.worker_id}: Setup complete")
                else:
                    print(f"  Worker {worker.worker_id}: Setup failed - {worker.error}")

        # Check we have at least one working worker
        active_workers = [w for w in workers if w.status == "running"]
        if not active_workers:
            raise RuntimeError("No workers available")

        print(f"\n{len(active_workers)} workers ready")

        # Current model path
        current_model = args.model

        # Run training iterations
        for iteration in range(args.iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{args.iterations}")
            print(f"{'='*60}")

            # Run self-play on all workers in parallel
            print(f"\nRunning self-play ({args.games_per_worker} games per worker)...")
            game_dirs = []

            with ThreadPoolExecutor(max_workers=len(active_workers)) as executor:
                futures = {
                    executor.submit(
                        run_selfplay_on_worker,
                        vast, w, args.games_per_worker, args.simulations,
                        args.filters, args.blocks, current_model
                    ): w
                    for w in active_workers
                }
                for future in as_completed(futures):
                    worker = futures[future]
                    if future.result():
                        print(f"  Worker {worker.worker_id}: {worker.games_completed} games completed")
                    else:
                        print(f"  Worker {worker.worker_id}: Failed - {worker.error}")

            # Download games from workers (just the tarballs)
            print("\nDownloading games...")
            game_tarballs = []
            iter_output = args.output / f"iter_{iteration}"
            iter_output.mkdir(parents=True, exist_ok=True)

            for worker in active_workers:
                if worker.status == "running":
                    game_dir = download_games(vast, worker, iter_output)
                    if game_dir:
                        game_dirs.append(game_dir)
                        # Find the tarball
                        tarball = game_dir / "games.tar.gz"
                        if tarball.exists():
                            game_tarballs.append(tarball)
                        print(f"  Worker {worker.worker_id}: Downloaded")

            if not game_tarballs:
                print("No games downloaded, skipping training")
                continue

            # Train on aggregated data (aggregate on remote to avoid numpy version issues)
            print(f"\nTraining for {args.epochs} epochs (aggregating {len(game_tarballs)} workers' games on remote)...")
            trainer_worker = active_workers[0]
            new_model = train_on_aggregated(
                vast,
                trainer_worker.instance_id,
                game_tarballs,
                current_model,
                args.epochs,
                args.filters,
                args.blocks,
                iter_output
            )

            if new_model:
                # Copy to canonical location
                final_model = args.output / f"model_iter_{iteration:03d}.pt"
                shutil.copy(new_model, final_model)
                current_model = final_model
                print(f"  Saved model: {final_model}")

                # Upload new model to all workers for next iteration
                print("  Distributing model to workers...")
                for worker in active_workers[1:]:  # Skip trainer, already has it
                    try:
                        vast.copy_to(worker.instance_id, final_model, "/workspace/model_start.pt")
                    except:
                        pass

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")
        print(f"\nResults saved to {args.output}/")
        print(f"Final model: {current_model}")

    except Exception as e:
        print(f"\nError: {e}")
        raise

    finally:
        if not args.keep_instances:
            print("\nCleaning up instances...")
            for instance_id in instance_ids:
                try:
                    vast.destroy_instance(instance_id)
                    print(f"  Destroyed instance {instance_id}")
                except:
                    pass
        else:
            print("\nKeeping instances running. To destroy:")
            for instance_id in instance_ids:
                print(f"  vastai destroy instance {instance_id}")


if __name__ == '__main__':
    main()
