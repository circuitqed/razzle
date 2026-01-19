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


def main():
    parser = argparse.ArgumentParser(description='Razzle Dazzle Engine Benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--core', action='store_true', help='Run core engine benchmarks')
    parser.add_argument('--network', action='store_true', help='Run neural network benchmarks')
    parser.add_argument('--mcts', action='store_true', help='Run MCTS benchmarks')
    parser.add_argument('--game', action='store_true', help='Run full game benchmark')
    parser.add_argument('--device', type=str, default='cpu', help='Device for neural network (cpu/cuda)')

    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.all, args.core, args.network, args.mcts, args.game]):
        args.all = True

    print("=" * 50)
    print("Razzle Dazzle Engine Benchmarks")
    print("=" * 50)

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
