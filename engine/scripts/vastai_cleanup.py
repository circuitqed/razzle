#!/usr/bin/env python3
"""
Cleanup script for Vast.ai instances.

Destroys all running instances associated with this account.
Useful for cleaning up after training runs or when instances get orphaned.

Usage:
    python scripts/vastai_cleanup.py           # List instances
    python scripts/vastai_cleanup.py --destroy # Destroy all instances
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.vastai import VastAI


def main():
    parser = argparse.ArgumentParser(description='Vast.ai instance cleanup')
    parser.add_argument('--destroy', action='store_true',
                        help='Destroy all instances (without this flag, just lists them)')
    parser.add_argument('--force', action='store_true',
                        help='Skip confirmation prompt')
    args = parser.parse_args()

    vast = VastAI()

    print("Checking for running instances...")
    instances = vast.list_instances()

    if not instances:
        print("No instances found.")
        return 0

    print(f"\nFound {len(instances)} instance(s):")
    total_cost = 0
    for inst in instances:
        cost = getattr(inst, 'dph_total', 0) or 0
        total_cost += cost
        status = getattr(inst, 'actual_status', 'unknown')
        gpu = getattr(inst, 'gpu_name', 'unknown')
        print(f"  {inst.id}: {gpu} - {status} - ${cost:.3f}/hr")

    print(f"\nTotal hourly cost: ${total_cost:.2f}/hr")

    if not args.destroy:
        print("\nTo destroy all instances, run with --destroy flag")
        return 0

    if not args.force:
        response = input(f"\nDestroy all {len(instances)} instances? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    print("\nDestroying instances...")
    for inst in instances:
        try:
            vast.destroy_instance(inst.id)
            print(f"  Destroyed {inst.id}")
        except Exception as e:
            print(f"  Failed to destroy {inst.id}: {e}")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
