#!/usr/bin/env python3
"""
Cloud training script for Razzle Dazzle using Vast.ai.

This script:
1. Packages the razzle code and training data
2. Rents a GPU on Vast.ai
3. Uploads code and data
4. Runs training
5. Downloads the trained model
6. Cleans up

Requirements:
    pip install vastai
    vastai set api-key YOUR_KEY

Usage:
    # Full training run on cloud
    python scripts/train_cloud.py --gpu RTX_3090 --iterations 10

    # Use specific model as starting point
    python scripts/train_cloud.py --model output/model_iter_007.pt --iterations 5

    # Just generate self-play locally, train on cloud
    python scripts/train_cloud.py --games-dir output/games_iter_* --epochs 20
"""

import argparse
import os
import pickle
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.vastai import VastAI, GPUOffer


def create_package(output_dir: Path) -> Path:
    """
    Create a tarball with the razzle package and training scripts.
    """
    package_path = output_dir / "razzle_package.tar.gz"

    # Files to include
    engine_dir = Path(__file__).parent.parent

    with tarfile.open(package_path, "w:gz") as tar:
        # Add the razzle package
        tar.add(engine_dir / "razzle", arcname="razzle")

        # Add training script
        tar.add(engine_dir / "scripts" / "train_local.py", arcname="train_local.py")

        # Add requirements
        requirements = engine_dir / "requirements.txt"
        if requirements.exists():
            tar.add(requirements, arcname="requirements.txt")

    print(f"Created package: {package_path} ({package_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return package_path


def create_data_package(games_dirs: list[Path], output_dir: Path) -> Path:
    """
    Package game data for upload.
    """
    data_path = output_dir / "training_data.tar.gz"

    with tarfile.open(data_path, "w:gz") as tar:
        for games_dir in games_dirs:
            if games_dir.exists():
                # Add games directory preserving structure
                tar.add(games_dir, arcname=games_dir.name)

    print(f"Created data package: {data_path} ({data_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return data_path


def generate_remote_script(
    iterations: int,
    games_per_iter: int,
    simulations: int,
    epochs: int,
    filters: int,
    blocks: int,
    resume_model: str = None
) -> str:
    """
    Generate the training script to run on the remote machine.
    """
    resume_arg = f"--resume {resume_model}" if resume_model else ""

    script = f'''#!/bin/bash
set -e

echo "=== Razzle Dazzle Cloud Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(nvcc --version | grep release | awk '{{print $6}}')"

cd /workspace

# Install dependencies
pip install torch numpy --quiet

# Extract package
tar -xzf razzle_package.tar.gz

# Extract data if present
if [ -f training_data.tar.gz ]; then
    echo "Extracting training data..."
    tar -xzf training_data.tar.gz -C output/
fi

# Run training
echo "Starting training..."
python train_local.py \\
    --iterations {iterations} \\
    --games-per-iter {games_per_iter} \\
    --simulations {simulations} \\
    --epochs {epochs} \\
    --filters {filters} \\
    --blocks {blocks} \\
    --device cuda \\
    --output output/ \\
    {resume_arg}

echo "Training complete!"

# Package results
echo "Packaging results..."
tar -czf results.tar.gz output/model_iter_*.pt output/training_log.json

echo "Results ready for download"
ls -la results.tar.gz
'''
    return script


def wait_with_progress(vast: VastAI, instance_id: int, timeout: int = 300) -> 'Instance':
    """Wait for instance with progress indicator."""
    start = time.time()
    dots = 0

    while time.time() - start < timeout:
        instance = vast.get_instance(instance_id)
        if instance and instance.ssh_host and instance.actual_status == 'running':
            print()  # New line after dots
            return instance

        dots = (dots + 1) % 4
        print(f"\rWaiting for instance{'.' * (dots + 1)}{' ' * (3 - dots)}", end='', flush=True)
        time.sleep(5)

    raise TimeoutError(f"Instance not ready after {timeout}s")


def main():
    parser = argparse.ArgumentParser(description='Cloud training on Vast.ai')
    parser.add_argument('--gpu', type=str, default='RTX_3090', help='Target GPU type')
    parser.add_argument('--max-price', type=float, default=0.40, help='Max $/hr')
    parser.add_argument('--min-reliability', type=float, default=0.95, help='Min reliability score')

    # Training parameters
    parser.add_argument('--iterations', type=int, default=10, help='Training iterations')
    parser.add_argument('--games-per-iter', type=int, default=100, help='Games per iteration')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--filters', type=int, default=64, help='Network filters')
    parser.add_argument('--blocks', type=int, default=6, help='Network blocks')

    # Data/model
    parser.add_argument('--model', type=Path, help='Starting model checkpoint')
    parser.add_argument('--games-dir', type=Path, nargs='*', help='Existing game directories to upload')
    parser.add_argument('--output', type=Path, default=Path('output'), help='Output directory')

    # Options
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--keep-instance', action='store_true', help='Keep instance running after training')

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Razzle Dazzle Cloud Training")
    print("=" * 60)

    # Initialize Vast.ai
    try:
        vast = VastAI()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTo set up Vast.ai:")
        print("  pip install vastai")
        print("  vastai set api-key YOUR_KEY")
        return

    # Search for GPU offers
    print(f"\nSearching for {args.gpu} offers under ${args.max_price}/hr...")
    offers = vast.search_offers(
        gpu_name=args.gpu,
        max_dph=args.max_price,
        min_reliability=args.min_reliability,
        order_by='dph_total'
    )

    if not offers:
        print("No suitable offers found.")
        print("Try:")
        print("  --max-price 0.50  (increase budget)")
        print("  --gpu RTX_4090    (different GPU)")
        print("  --min-reliability 0.90  (lower reliability)")
        return

    print(f"\nFound {len(offers)} offers:")
    for i, offer in enumerate(offers[:5]):
        print(f"  {i+1}. {offer.gpu_name} - ${offer.dph_total:.3f}/hr "
              f"({offer.gpu_ram:.0f}GB VRAM, reliability {offer.reliability:.2f})")

    offer = offers[0]
    print(f"\nSelected: {offer.gpu_name} @ ${offer.dph_total:.3f}/hr")

    if args.dry_run:
        print("\n[DRY RUN] Would create instance and run training")
        print(f"  Iterations: {args.iterations}")
        print(f"  Games/iter: {args.games_per_iter}")
        print(f"  MCTS sims: {args.simulations}")
        print(f"  Epochs: {args.epochs}")
        return

    # Create package
    print("\nPreparing packages...")
    package_path = create_package(args.output)

    # Create data package if we have game data
    data_path = None
    if args.games_dir:
        data_path = create_data_package(args.games_dir, args.output)

    # Create remote training script
    resume_model = "model_resume.pt" if args.model else None
    train_script = generate_remote_script(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        simulations=args.simulations,
        epochs=args.epochs,
        filters=args.filters,
        blocks=args.blocks,
        resume_model=resume_model
    )

    script_path = args.output / "train_remote.sh"
    with open(script_path, 'w') as f:
        f.write(train_script)

    # Create instance
    print("\nCreating instance...")
    instance_id = vast.create_instance(
        offer.id,
        image='pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
        disk=50  # 50GB disk for games
    )
    print(f"Instance {instance_id} created")

    try:
        # Wait for instance
        print("Waiting for instance to be ready...")
        instance = wait_with_progress(vast, instance_id, timeout=300)
        print(f"Instance ready: {instance.ssh_host}:{instance.ssh_port}")

        # Create output directory on remote
        print("\nSetting up remote environment...")
        vast.execute(instance_id, "mkdir -p /workspace/output")

        # Upload files
        print("Uploading package...")
        vast.copy_to(instance_id, package_path, "/workspace/razzle_package.tar.gz")

        if data_path:
            print("Uploading training data...")
            vast.copy_to(instance_id, data_path, "/workspace/training_data.tar.gz")

        if args.model:
            print("Uploading starting model...")
            vast.copy_to(instance_id, args.model, "/workspace/output/model_resume.pt")

        # Upload and run training script
        print("Uploading training script...")
        vast.copy_to(instance_id, script_path, "/workspace/train.sh")

        print("\n" + "=" * 60)
        print("Starting training on cloud GPU...")
        print("=" * 60)

        # Run training (this will take a while)
        output = vast.execute(instance_id, "chmod +x /workspace/train.sh && /workspace/train.sh")
        print(output)

        # Download results
        print("\nDownloading results...")
        results_path = args.output / "cloud_results.tar.gz"
        vast.copy_from(instance_id, "/workspace/results.tar.gz", results_path)

        # Extract results
        print("Extracting results...")
        with tarfile.open(results_path, "r:gz") as tar:
            tar.extractall(args.output)

        print(f"\nResults saved to {args.output}/")
        print("  - Model checkpoints: model_iter_*.pt")
        print("  - Training log: training_log.json")

        # Show final status
        training_log = args.output / "output" / "training_log.json"
        if training_log.exists():
            import json
            with open(training_log) as f:
                log = json.load(f)
            print(f"\nTraining completed:")
            print(f"  Iterations: {len(log.get('iterations', []))}")
            print(f"  Total games: {log.get('total_games', 0)}")
            print(f"  Total time: {log.get('total_time_sec', 0)/60:.1f} min")

    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    finally:
        if not args.keep_instance:
            print(f"\nDestroying instance {instance_id}...")
            vast.destroy_instance(instance_id)
            print("Instance destroyed")
        else:
            print(f"\nKeeping instance {instance_id} running")
            print(f"  SSH: ssh -p {instance.ssh_port} root@{instance.ssh_host}")
            print(f"  To destroy: vastai destroy instance {instance_id}")

    print("\nCloud training complete!")


if __name__ == '__main__':
    main()
