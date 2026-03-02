"""Benchmark C engine vs Python engine."""

import time
import random
import numpy as np

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.network import NUM_ACTIONS
from razzle_fast.wrapper import FastState, FastMCTS


class DummyEvaluator:
    def evaluate(self, state):
        return np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS, 0.0

    def evaluate_batch(self, states):
        return [self.evaluate(s) for s in states]


def bench_state_ops():
    """Benchmark state operations: init, apply_move, get_legal_moves, to_tensor."""
    N = 10000
    rng = random.Random(42)

    # Python engine
    t0 = time.perf_counter()
    for _ in range(N):
        gs = GameState.new_game()
        for step in range(20):
            if gs.is_terminal():
                break
            moves = get_legal_moves(gs)
            if not moves:
                break
            gs.apply_move(rng.choice(moves))
            _ = gs.to_tensor()
    py_time = time.perf_counter() - t0

    # C engine
    rng = random.Random(42)
    t0 = time.perf_counter()
    for _ in range(N):
        fs = FastState()
        for step in range(20):
            if fs.is_terminal():
                break
            moves = fs.get_legal_moves()
            if not moves:
                break
            fs.apply_move(rng.choice(moves))
            _ = fs.to_tensor()
    c_time = time.perf_counter() - t0

    print(f"State ops ({N} games x 20 steps):")
    print(f"  Python: {py_time:.3f}s")
    print(f"  C:      {c_time:.3f}s")
    print(f"  Speedup: {py_time / c_time:.1f}x")
    print()


def bench_mcts():
    """Benchmark MCTS search (with dummy evaluator, measures tree overhead)."""
    evaluator = DummyEvaluator()
    config = MCTSConfig(
        num_simulations=200,
        batch_size=16,
        early_termination=False,
        pass_quiescence=False,
    )
    N = 20

    gs = GameState.new_game()

    # Python MCTS
    t0 = time.perf_counter()
    for _ in range(N):
        mcts = MCTS(evaluator, config)
        mcts.search_batched(gs, add_noise=False)
    py_time = time.perf_counter() - t0

    # C MCTS
    t0 = time.perf_counter()
    for _ in range(N):
        mcts = FastMCTS(evaluator, config)
        mcts.search_batched(gs, add_noise=False)
    c_time = time.perf_counter() - t0

    print(f"MCTS ({N} searches x {config.num_simulations} sims, batch={config.batch_size}):")
    print(f"  Python: {py_time:.3f}s ({N * config.num_simulations / py_time:.0f} sims/s)")
    print(f"  C:      {c_time:.3f}s ({N * config.num_simulations / c_time:.0f} sims/s)")
    print(f"  Speedup: {py_time / c_time:.1f}x")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Razzle Fast C Engine Benchmark")
    print("=" * 60)
    print()
    bench_state_ops()
    bench_mcts()
