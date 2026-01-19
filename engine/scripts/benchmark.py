#!/usr/bin/env python3
"""
Performance benchmarks for Razzle Dazzle engine.

Measures:
- Move generation speed
- Neural network inference speed
- MCTS simulation throughput
- End-to-end move selection time
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves, MoveGenerator
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import DummyEvaluator, BatchedEvaluator
from razzle.ai.network import RazzleNet


def benchmark_move_generation(iterations: int = 10000) -> dict:
    """Benchmark move generation speed."""
    state = GameState.new_game()

    start = time.perf_counter()
    for _ in range(iterations):
        moves = get_legal_moves(state)
    elapsed = time.perf_counter() - start

    return {
        "name": "Move Generation",
        "iterations": iterations,
        "total_ms": elapsed * 1000,
        "per_call_us": (elapsed / iterations) * 1_000_000,
        "calls_per_sec": iterations / elapsed,
    }


def benchmark_state_copy(iterations: int = 10000) -> dict:
    """Benchmark state copying speed."""
    state = GameState.new_game()

    start = time.perf_counter()
    for _ in range(iterations):
        copy = state.copy()
    elapsed = time.perf_counter() - start

    return {
        "name": "State Copy",
        "iterations": iterations,
        "total_ms": elapsed * 1000,
        "per_call_us": (elapsed / iterations) * 1_000_000,
        "calls_per_sec": iterations / elapsed,
    }


def benchmark_state_tensor(iterations: int = 1000) -> dict:
    """Benchmark state to tensor conversion."""
    state = GameState.new_game()

    start = time.perf_counter()
    for _ in range(iterations):
        tensor = state.to_tensor()
    elapsed = time.perf_counter() - start

    return {
        "name": "State to Tensor",
        "iterations": iterations,
        "total_ms": elapsed * 1000,
        "per_call_us": (elapsed / iterations) * 1_000_000,
        "calls_per_sec": iterations / elapsed,
    }


def benchmark_network_inference(iterations: int = 100, device: str = 'cpu') -> dict:
    """Benchmark neural network forward pass."""
    import torch

    net = RazzleNet()
    net.to(device)
    net.eval()

    state = GameState.new_game()
    tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            net(tensor)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            policy, value = net(tensor)
    elapsed = time.perf_counter() - start

    return {
        "name": f"Network Inference ({device})",
        "iterations": iterations,
        "total_ms": elapsed * 1000,
        "per_call_ms": (elapsed / iterations) * 1000,
        "calls_per_sec": iterations / elapsed,
    }


def benchmark_network_batch(batch_sizes: list[int] = [1, 8, 16, 32], device: str = 'cpu') -> list[dict]:
    """Benchmark batched neural network inference."""
    import torch

    net = RazzleNet()
    net.to(device)
    net.eval()

    state = GameState.new_game()
    single_tensor = torch.from_numpy(state.to_tensor()).to(device)

    results = []
    for batch_size in batch_sizes:
        batch = single_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        iterations = max(10, 100 // batch_size)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                net(batch)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                policy, value = net(batch)
        elapsed = time.perf_counter() - start

        total_evals = iterations * batch_size
        results.append({
            "name": f"Batch size {batch_size}",
            "iterations": iterations,
            "batch_size": batch_size,
            "total_evals": total_evals,
            "total_ms": elapsed * 1000,
            "per_eval_ms": (elapsed / total_evals) * 1000,
            "evals_per_sec": total_evals / elapsed,
        })

    return results


def benchmark_mcts(simulations: list[int] = [100, 400, 800], use_network: bool = False, device: str = 'cpu') -> list[dict]:
    """Benchmark MCTS search."""
    state = GameState.new_game()

    if use_network:
        net = RazzleNet()
        evaluator = BatchedEvaluator(net, device=device)
        eval_name = f"Neural Net ({device})"
    else:
        evaluator = DummyEvaluator()
        eval_name = "Random Policy"

    results = []
    for sims in simulations:
        config = MCTSConfig(num_simulations=sims)
        mcts = MCTS(evaluator, config)

        # Run multiple times for stability
        times = []
        for _ in range(3):
            start = time.perf_counter()
            root = mcts.search(state)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)

        results.append({
            "name": f"MCTS {sims} sims ({eval_name})",
            "simulations": sims,
            "evaluator": eval_name,
            "avg_time_ms": avg_time * 1000,
            "sims_per_sec": sims / avg_time,
        })

    return results


def benchmark_mcts_batched(
    simulations: int = 400,
    batch_sizes: list[int] = [1, 4, 8, 16, 32],
    device: str = 'cpu'
) -> list[dict]:
    """Benchmark batched MCTS search with different batch sizes."""
    state = GameState.new_game()
    net = RazzleNet()
    evaluator = BatchedEvaluator(net, device=device)

    results = []
    for batch_size in batch_sizes:
        config = MCTSConfig(
            num_simulations=simulations,
            batch_size=batch_size,
            virtual_loss=3
        )
        mcts = MCTS(evaluator, config)

        # Run multiple times for stability
        times = []
        for _ in range(3):
            start = time.perf_counter()
            root = mcts.search_batched(state)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)

        results.append({
            "name": f"Batched MCTS (batch={batch_size})",
            "simulations": simulations,
            "batch_size": batch_size,
            "avg_time_ms": avg_time * 1000,
            "sims_per_sec": simulations / avg_time,
        })

    return results


def benchmark_full_game(max_moves: int = 50, simulations: int = 100, use_network: bool = False) -> dict:
    """Benchmark a full game with AI moves."""
    state = GameState.new_game()

    if use_network:
        net = RazzleNet()
        evaluator = BatchedEvaluator(net, device='cpu')
        eval_name = "Neural Net"
    else:
        evaluator = DummyEvaluator()
        eval_name = "Random"

    config = MCTSConfig(num_simulations=simulations, temperature=0.1)

    move_times = []
    moves_played = 0

    start_total = time.perf_counter()
    while not state.is_terminal() and moves_played < max_moves:
        mcts = MCTS(evaluator, config)

        start_move = time.perf_counter()
        root = mcts.search(state)
        move = mcts.select_move(root)
        move_time = time.perf_counter() - start_move

        move_times.append(move_time)
        state.apply_move(move)
        moves_played += 1

    total_time = time.perf_counter() - start_total

    return {
        "name": f"Full Game ({eval_name}, {simulations} sims)",
        "moves_played": moves_played,
        "total_time_sec": total_time,
        "avg_move_time_ms": np.mean(move_times) * 1000,
        "min_move_time_ms": np.min(move_times) * 1000,
        "max_move_time_ms": np.max(move_times) * 1000,
    }


def print_result(result: dict) -> None:
    """Pretty print a benchmark result."""
    name = result.pop("name")
    print(f"\n{name}:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def benchmark_cpu_vs_gpu(batch_sizes: list[int] = [1, 8, 16, 32, 64]) -> dict:
    """
    Compare CPU vs GPU performance across different batch sizes.

    Returns dict with comparison data.
    """
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available - cannot compare CPU vs GPU")
        return None

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Create networks for each device
    net_cpu = RazzleNet()
    net_cpu.eval()

    net_gpu = RazzleNet()
    net_gpu.to('cuda')
    net_gpu.eval()

    state = GameState.new_game()
    tensor_np = state.to_tensor()

    results = []

    for batch_size in batch_sizes:
        iterations = max(20, 200 // batch_size)

        # Prepare batches
        batch_cpu = torch.from_numpy(tensor_np).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        batch_gpu = batch_cpu.to('cuda')

        # Warmup CPU
        with torch.no_grad():
            for _ in range(5):
                net_cpu(batch_cpu)

        # Benchmark CPU
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                net_cpu(batch_cpu)
        cpu_time = time.perf_counter() - start
        cpu_evals_per_sec = (iterations * batch_size) / cpu_time

        # Warmup GPU (important for accurate timing)
        with torch.no_grad():
            for _ in range(10):
                net_gpu(batch_gpu)
        torch.cuda.synchronize()

        # Benchmark GPU
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                net_gpu(batch_gpu)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.perf_counter() - start
        gpu_evals_per_sec = (iterations * batch_size) / gpu_time

        speedup = gpu_evals_per_sec / cpu_evals_per_sec

        results.append({
            'batch_size': batch_size,
            'cpu_evals_per_sec': cpu_evals_per_sec,
            'gpu_evals_per_sec': gpu_evals_per_sec,
            'speedup': speedup,
            'cpu_ms_per_eval': 1000 / cpu_evals_per_sec,
            'gpu_ms_per_eval': 1000 / gpu_evals_per_sec,
        })

    return results


def benchmark_mcts_cpu_vs_gpu(simulations: int = 400, batch_size: int = 16) -> dict:
    """Compare MCTS performance between CPU and GPU."""
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available - cannot compare CPU vs GPU")
        return None

    state = GameState.new_game()

    # CPU MCTS
    net_cpu = RazzleNet()
    eval_cpu = BatchedEvaluator(net_cpu, device='cpu')
    config = MCTSConfig(num_simulations=simulations, batch_size=batch_size)
    mcts_cpu = MCTS(eval_cpu, config)

    # Warmup
    mcts_cpu.search_batched(state)

    cpu_times = []
    for _ in range(3):
        start = time.perf_counter()
        mcts_cpu.search_batched(state)
        cpu_times.append(time.perf_counter() - start)
    cpu_avg = np.mean(cpu_times)

    # GPU MCTS
    net_gpu = RazzleNet()
    net_gpu.to('cuda')
    eval_gpu = BatchedEvaluator(net_gpu, device='cuda')
    mcts_gpu = MCTS(eval_gpu, config)

    # Warmup
    import torch
    mcts_gpu.search_batched(state)
    torch.cuda.synchronize()

    gpu_times = []
    for _ in range(3):
        start = time.perf_counter()
        mcts_gpu.search_batched(state)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start)
    gpu_avg = np.mean(gpu_times)

    return {
        'simulations': simulations,
        'batch_size': batch_size,
        'cpu_time_ms': cpu_avg * 1000,
        'gpu_time_ms': gpu_avg * 1000,
        'cpu_sims_per_sec': simulations / cpu_avg,
        'gpu_sims_per_sec': simulations / gpu_avg,
        'speedup': cpu_avg / gpu_avg,
    }


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 75)
    print("CPU vs GPU Inference Comparison")
    print("=" * 75)
    print(f"{'Batch':<8} {'CPU (eval/s)':<14} {'GPU (eval/s)':<14} {'Speedup':<10} {'GPU ms/eval':<12}")
    print("-" * 75)

    for r in results:
        print(f"{r['batch_size']:<8} {r['cpu_evals_per_sec']:<14.1f} {r['gpu_evals_per_sec']:<14.1f} "
              f"{r['speedup']:<10.2f}x {r['gpu_ms_per_eval']:<12.3f}")

    print("-" * 75)

    # Find optimal batch size for GPU
    best = max(results, key=lambda x: x['gpu_evals_per_sec'])
    print(f"\nOptimal GPU batch size: {best['batch_size']} ({best['gpu_evals_per_sec']:.0f} evals/sec)")
    print(f"Peak GPU speedup: {best['speedup']:.1f}x over CPU")


def main():
    parser = argparse.ArgumentParser(description='Razzle Dazzle Engine Benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--core', action='store_true', help='Run core engine benchmarks')
    parser.add_argument('--network', action='store_true', help='Run neural network benchmarks')
    parser.add_argument('--mcts', action='store_true', help='Run MCTS benchmarks')
    parser.add_argument('--game', action='store_true', help='Run full game benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare CPU vs GPU performance')
    parser.add_argument('--device', type=str, default='cpu', help='Device for neural network (cpu/cuda)')

    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.all, args.core, args.network, args.mcts, args.game, args.compare]):
        args.all = True

    print("=" * 50)
    print("Razzle Dazzle Engine Benchmarks")
    print("=" * 50)

    # CPU vs GPU comparison mode
    if args.compare:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            # Network inference comparison
            results = benchmark_cpu_vs_gpu(batch_sizes=[1, 4, 8, 16, 32, 64, 128])
            if results:
                print_comparison_table(results)

            # MCTS comparison
            print("\n" + "=" * 75)
            print("CPU vs GPU MCTS Comparison")
            print("=" * 75)

            for sims in [200, 400, 800]:
                mcts_result = benchmark_mcts_cpu_vs_gpu(simulations=sims, batch_size=16)
                if mcts_result:
                    print(f"\n{sims} simulations (batch_size=16):")
                    print(f"  CPU: {mcts_result['cpu_time_ms']:.1f}ms ({mcts_result['cpu_sims_per_sec']:.0f} sims/sec)")
                    print(f"  GPU: {mcts_result['gpu_time_ms']:.1f}ms ({mcts_result['gpu_sims_per_sec']:.0f} sims/sec)")
                    print(f"  Speedup: {mcts_result['speedup']:.2f}x")
        else:
            print("\nCUDA not available. Install PyTorch with CUDA support to compare.")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

        print("\n" + "=" * 50)
        print("Comparison complete")
        return

    if args.all or args.core:
        print("\n### Core Engine ###")
        print_result(benchmark_move_generation())
        print_result(benchmark_state_copy())
        print_result(benchmark_state_tensor())

    if args.all or args.network:
        print("\n### Neural Network ###")
        print_result(benchmark_network_inference(device=args.device))

        print("\n### Batched Inference ###")
        for result in benchmark_network_batch(device=args.device):
            print_result(result)

    if args.all or args.mcts:
        print("\n### MCTS (Random Policy) ###")
        for result in benchmark_mcts(use_network=False):
            print_result(result)

        print("\n### MCTS (Neural Network - Sequential) ###")
        for result in benchmark_mcts(simulations=[50, 100, 200], use_network=True, device=args.device):
            print_result(result)

        print("\n### MCTS (Neural Network - Batched) ###")
        for result in benchmark_mcts_batched(simulations=400, device=args.device):
            print_result(result)

    if args.all or args.game:
        print("\n### Full Game ###")
        print_result(benchmark_full_game(use_network=False, simulations=100))
        print_result(benchmark_full_game(use_network=True, simulations=50))

    print("\n" + "=" * 50)
    print("Benchmarks complete")


if __name__ == '__main__':
    main()
