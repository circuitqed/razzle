#!/usr/bin/env python3
"""
Cloud training script for Razzle Dazzle using Vast.ai.

Workflow:
1. Generate self-play games locally (or on cheap GPU)
2. Upload training data to cloud instance
3. Train on powerful GPU
4. Download new weights
"""

import argparse
from pathlib import Path
import pickle
import tempfile
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.vastai import VastAI, find_best_offer


def main():
    parser = argparse.ArgumentParser(description='Cloud training on Vast.ai')
    parser.add_argument('--gpu', type=str, default='RTX_3090', help='Target GPU')
    parser.add_argument('--max-price', type=float, default=0.30, help='Max $/hr')
    parser.add_argument('--data', type=Path, required=True, help='Training data directory')
    parser.add_argument('--model', type=Path, help='Starting model checkpoint')
    parser.add_argument('--output', type=Path, default=Path('output'), help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    vast = VastAI()

    # Find best offer
    print(f"Searching for {args.gpu} offers under ${args.max_price}/hr...")
    offers = vast.search_offers(
        gpu_name=args.gpu,
        max_dph=args.max_price,
        order_by='dph_total'
    )

    if not offers:
        print("No suitable offers found. Try increasing --max-price")
        return

    print(f"Found {len(offers)} offers:")
    for i, offer in enumerate(offers[:5]):
        print(f"  {i+1}. {offer.gpu_name} - ${offer.dph_total:.3f}/hr "
              f"({offer.gpu_ram:.0f}GB, reliability {offer.reliability:.2f})")

    # Select first offer
    offer = offers[0]
    print(f"\nUsing offer {offer.id}: {offer.gpu_name} @ ${offer.dph_total:.3f}/hr")

    # Create instance
    print("Creating instance...")
    instance_id = vast.create_instance(
        offer.id,
        image='pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
        disk=30
    )
    print(f"Instance {instance_id} created")

    try:
        # Wait for instance
        print("Waiting for instance to be ready...")
        instance = vast.wait_for_instance(instance_id, timeout=300)
        print(f"Instance ready: {instance.ssh_host}:{instance.ssh_port}")

        # Package and upload code + data
        print("Uploading code and data...")
        # This would be more sophisticated in production
        # For now, we'll just show the concept

        # Run training
        print(f"Starting training for {args.epochs} epochs...")
        train_cmd = f"""
cd /workspace && \
pip install torch numpy && \
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
"
"""
        output = vast.execute(instance_id, train_cmd)
        print(output)

        # In a real implementation:
        # 1. SCP the razzle package and training data
        # 2. Run the actual training script
        # 3. SCP the resulting model back

        print("\n[Demo mode - actual training would happen here]")
        print("In production, you would:")
        print("  1. vast.copy_to(instance_id, 'razzle.tar.gz', '/workspace/')")
        print("  2. vast.copy_to(instance_id, 'training_data.pkl', '/workspace/')")
        print("  3. vast.execute(instance_id, 'python train.py --epochs N')")
        print("  4. vast.copy_from(instance_id, '/workspace/model.pt', 'model.pt')")

    finally:
        # Cleanup
        print(f"\nDestroying instance {instance_id}...")
        vast.destroy_instance(instance_id)
        print("Instance destroyed")


if __name__ == '__main__':
    main()
