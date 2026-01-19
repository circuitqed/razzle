#!/usr/bin/env python3
"""
Benchmark multiple GPU types on Vast.ai to compare cost/performance.

This script:
1. Rents each GPU type briefly
2. Runs the benchmark suite
3. Destroys the instance
4. Produces a comparison table

Usage:
    python scripts/benchmark_gpus.py
    python scripts/benchmark_gpus.py --gpus RTX_3090 RTX_4090
    python scripts/benchmark_gpus.py --dry-run
"""

import argparse
import json
import re
import sys
import tarfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.vastai import VastAI, GPUOffer


@dataclass
class BenchmarkResult:
    """Results from a GPU benchmark run."""
    gpu_name: str
    price_per_hour: float

    # Network inference (evals/sec at optimal batch size)
    inference_evals_per_sec: float
    optimal_batch_size: int

    # MCTS performance (sims/sec at 400 sims)
    mcts_sims_per_sec: float

    # Cost efficiency
    evals_per_dollar: float  # inference_evals_per_sec / price_per_hour * 3600

    # Raw output for debugging
    raw_output: str = ""


def create_benchmark_package(output_dir: Path) -> Path:
    """Create a minimal package for benchmarking."""
    package_path = output_dir / "benchmark_package.tar.gz"
    engine_dir = Path(__file__).parent.parent

    with tarfile.open(package_path, "w:gz") as tar:
        tar.add(engine_dir / "razzle", arcname="razzle")
        tar.add(engine_dir / "scripts" / "benchmark.py", arcname="benchmark.py")

        # Add a minimal requirements for benchmark
        requirements = engine_dir / "requirements.txt"
        if requirements.exists():
            tar.add(requirements, arcname="requirements.txt")

    return package_path


def parse_benchmark_output(output: str) -> dict:
    """Parse benchmark output to extract key metrics."""
    result = {
        'inference_evals_per_sec': 0,
        'optimal_batch_size': 1,
        'mcts_sims_per_sec': 0,
        'gpu_name': 'Unknown',
    }

    # Parse GPU name
    gpu_match = re.search(r'GPU: (.+?)$', output, re.MULTILINE)
    if gpu_match:
        result['gpu_name'] = gpu_match.group(1).strip()

    # Parse the comparison table for best GPU performance
    # Looking for lines like: 32       1234.5         5678.9         4.60x      0.176
    batch_pattern = r'^(\d+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+x\s+[\d.]+'
    best_evals = 0
    best_batch = 1

    for line in output.split('\n'):
        match = re.match(batch_pattern, line.strip())
        if match:
            batch_size = int(match.group(1))
            gpu_evals = float(match.group(2))
            if gpu_evals > best_evals:
                best_evals = gpu_evals
                best_batch = batch_size

    if best_evals > 0:
        result['inference_evals_per_sec'] = best_evals
        result['optimal_batch_size'] = best_batch

    # Parse MCTS comparison for 400 sims
    # Looking for: GPU: 123.4ms (3245 sims/sec)
    mcts_pattern = r'400 simulations.*?GPU:\s*[\d.]+ms\s*\((\d+)\s*sims/sec\)'
    mcts_match = re.search(mcts_pattern, output, re.DOTALL)
    if mcts_match:
        result['mcts_sims_per_sec'] = int(mcts_match.group(1))

    return result


def run_benchmark_on_gpu(vast: VastAI, offer: GPUOffer, package_path: Path) -> Optional[BenchmarkResult]:
    """Run benchmark on a specific GPU and return results."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {offer.gpu_name}")
    print(f"Price: ${offer.dph_total:.3f}/hr")
    print(f"{'='*60}")

    # Create instance
    print("Creating instance...")
    try:
        instance_id = vast.create_instance(
            offer.id,
            image='pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime',
            disk=10
        )
    except RuntimeError as e:
        print(f"Failed to create instance: {e}")
        return None

    print(f"Instance {instance_id} created")

    try:
        # Wait for instance
        print("Waiting for instance to be ready...")
        start_wait = time.time()
        while time.time() - start_wait < 300:
            instance = vast.get_instance(instance_id)
            if instance and instance.ssh_host and instance.actual_status == 'running':
                print(f"Instance ready: {instance.ssh_host}:{instance.ssh_port}")
                break
            print(".", end="", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("Instance not ready after 5 minutes")
        print()

        # Wait for SSH
        print("Waiting for SSH...")
        for i in range(12):
            try:
                vast.execute(instance_id, "echo 'ready'")
                print("SSH ready!")
                break
            except RuntimeError:
                print(".", end="", flush=True)
                time.sleep(10)
        else:
            raise RuntimeError("SSH not available")
        print()

        # Upload benchmark package
        print("Uploading benchmark package...")
        vast.copy_to(instance_id, package_path, "/workspace/benchmark_package.tar.gz")

        # Run benchmark
        print("Running benchmarks...")
        run_cmd = """
cd /workspace && \
tar -xzf benchmark_package.tar.gz && \
pip install torch numpy --quiet && \
python benchmark.py --compare
"""
        output = vast.execute(instance_id, run_cmd, timeout=600)
        print(output)

        # Parse results
        metrics = parse_benchmark_output(output)

        return BenchmarkResult(
            gpu_name=metrics['gpu_name'],
            price_per_hour=offer.dph_total,
            inference_evals_per_sec=metrics['inference_evals_per_sec'],
            optimal_batch_size=metrics['optimal_batch_size'],
            mcts_sims_per_sec=metrics['mcts_sims_per_sec'],
            evals_per_dollar=metrics['inference_evals_per_sec'] / offer.dph_total * 3600 if offer.dph_total > 0 else 0,
            raw_output=output
        )

    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None

    finally:
        print(f"Destroying instance {instance_id}...")
        try:
            vast.destroy_instance(instance_id)
            print("Instance destroyed")
        except Exception as e:
            print(f"Warning: Failed to destroy instance: {e}")


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print a comparison table of all benchmark results."""
    print("\n" + "=" * 90)
    print("GPU BENCHMARK COMPARISON")
    print("=" * 90)

    # Sort by inference performance
    results = sorted(results, key=lambda r: r.inference_evals_per_sec, reverse=True)

    print(f"\n{'GPU':<25} {'$/hr':<8} {'Infer/s':<12} {'MCTS/s':<10} {'Batch':<8} {'Evals/$':<12}")
    print("-" * 90)

    for r in results:
        print(f"{r.gpu_name:<25} ${r.price_per_hour:<7.3f} {r.inference_evals_per_sec:<12.0f} "
              f"{r.mcts_sims_per_sec:<10.0f} {r.optimal_batch_size:<8} {r.evals_per_dollar:<12,.0f}")

    print("-" * 90)

    if results:
        # Find best performers
        fastest = max(results, key=lambda r: r.inference_evals_per_sec)
        most_efficient = max(results, key=lambda r: r.evals_per_dollar)
        cheapest = min(results, key=lambda r: r.price_per_hour)

        print(f"\nFastest:        {fastest.gpu_name} ({fastest.inference_evals_per_sec:.0f} evals/sec)")
        print(f"Best value:     {most_efficient.gpu_name} ({most_efficient.evals_per_dollar:,.0f} evals per dollar)")
        print(f"Cheapest:       {cheapest.gpu_name} (${cheapest.price_per_hour:.3f}/hr)")

        # Speed comparison
        if len(results) > 1:
            slowest = min(results, key=lambda r: r.inference_evals_per_sec)
            speedup = fastest.inference_evals_per_sec / slowest.inference_evals_per_sec
            print(f"\nSpeedup range:  {speedup:.1f}x ({slowest.gpu_name} -> {fastest.gpu_name})")


def main():
    parser = argparse.ArgumentParser(description='Benchmark multiple GPU types on Vast.ai')
    parser.add_argument('--gpus', nargs='+', default=['RTX_3090', 'RTX_4090', 'A100_SXM4'],
                        help='GPU types to benchmark')
    parser.add_argument('--max-price', type=float, default=1.0, help='Max $/hr for any GPU')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--output', type=Path, default=Path('output'), help='Output directory')

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Razzle Dazzle Multi-GPU Benchmark")
    print("=" * 60)

    # Initialize Vast.ai
    try:
        vast = VastAI()
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    # Find offers for each GPU
    print("\nFinding GPU offers...")
    gpu_offers = {}
    for gpu in args.gpus:
        offers = vast.search_offers(gpu_name=gpu, max_dph=args.max_price, min_reliability=0.92)
        if offers:
            gpu_offers[gpu] = offers[0]
            print(f"  {gpu}: ${offers[0].dph_total:.3f}/hr ({len(offers)} offers)")
        else:
            print(f"  {gpu}: No offers available under ${args.max_price}/hr")

    if not gpu_offers:
        print("\nNo GPU offers found. Try increasing --max-price")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would benchmark these GPUs:")
        for gpu, offer in gpu_offers.items():
            print(f"  {gpu}: ${offer.dph_total:.3f}/hr")

        # Estimate costs (assume ~5 min per benchmark)
        total_cost = sum(o.dph_total * (5/60) for o in gpu_offers.values())
        print(f"\nEstimated total cost: ~${total_cost:.2f}")
        return

    # Create benchmark package
    print("\nCreating benchmark package...")
    package_path = create_benchmark_package(args.output)
    print(f"Package: {package_path}")

    # Run benchmarks
    results = []
    for gpu, offer in gpu_offers.items():
        result = run_benchmark_on_gpu(vast, offer, package_path)
        if result:
            results.append(result)

    # Print comparison
    if results:
        print_comparison(results)

        # Save results to JSON
        results_file = args.output / "gpu_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump([{
                'gpu_name': r.gpu_name,
                'price_per_hour': r.price_per_hour,
                'inference_evals_per_sec': r.inference_evals_per_sec,
                'optimal_batch_size': r.optimal_batch_size,
                'mcts_sims_per_sec': r.mcts_sims_per_sec,
                'evals_per_dollar': r.evals_per_dollar,
            } for r in results], f, indent=2)
        print(f"\nResults saved to: {results_file}")
    else:
        print("\nNo benchmark results collected.")

    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
