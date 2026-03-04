#!/usr/bin/env python3
"""Generate MCTS reference fixtures for cross-validating TypeScript MCTS.

Plays 10 seeded random games, runs Python MCTS at each position with a
deterministic evaluator (uniform policy, ply-based value), and records
root children stats and policy output.

Outputs JSON to stdout.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.network import NUM_ACTIONS, END_TURN_ACTION


class FixedValueEvaluator:
    """Deterministic evaluator: uniform policy over legal moves, ply-based value.

    Must be replicated exactly in the TypeScript test.

    Policy is computed in float32 (matching TS Float32Array), then cast to
    float64 so Python's expand() normalizes in float64 — matching how TS
    expandNode() accumulates policySum in float64 (JS number).
    """

    def evaluate(self, state):
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        moves = get_legal_moves(state)
        for m in moves:
            idx = END_TURN_ACTION if m == -1 else m
            policy[idx] = 1.0
        s = policy.sum()
        if s > 0:
            policy /= s
        # Cast to float64: values keep float32 rounding but expand() will
        # normalize in float64, matching TS behavior exactly.
        policy = policy.astype(np.float64)
        value = float(np.float32(0.1 * (state.ply % 5 - 2)))
        return policy, value

    def evaluate_batch(self, states):
        return [self.evaluate(s) for s in states]


def state_to_dict(state: GameState) -> dict:
    """Serialize game state to JSON-compatible dict (bigints as strings)."""
    return {
        "pieces": [str(state.pieces[0]), str(state.pieces[1])],
        "balls": [str(state.balls[0]), str(state.balls[1])],
        "currentPlayer": state.current_player,
        "touchedMask": str(state.touched_mask),
        "hasPassed": state.has_passed,
        "lastKnightDst": state.last_knight_dst,
        "ply": state.ply,
    }


def run_mcts_at_position(gs: GameState, evaluator, config: MCTSConfig) -> dict:
    """Run MCTS at a position and capture results."""
    mcts = MCTS(evaluator, config)
    root = mcts.search(gs, add_noise=False)
    policy = mcts.get_policy(root)

    # Collect per-child data
    children = []
    for action, child in sorted(root.children.items()):
        children.append({
            "action": action,
            "prior": float(child.prior),
            "visitCount": child.visit_count,
            "valueSum": float(child.value_sum),
        })

    return {
        "state": state_to_dict(gs),
        "children": children,
        "policy": policy.tolist(),
    }


def generate_game_fixtures(game_seed: int, evaluator, config: MCTSConfig,
                           max_positions: int = 30) -> dict:
    """Play a random game, run MCTS at each position."""
    rng = random.Random(game_seed)
    gs = GameState.new_game()
    positions = []

    pos_idx = 0
    while not gs.is_terminal() and pos_idx < max_positions:
        result = run_mcts_at_position(gs, evaluator, config)
        positions.append(result)

        moves = get_legal_moves(gs)
        gs.apply_move(rng.choice(moves))
        pos_idx += 1

    return {
        "gameSeed": game_seed,
        "numPositions": len(positions),
        "positions": positions,
    }


def main():
    evaluator = FixedValueEvaluator()
    config = MCTSConfig(
        num_simulations=100,
        batch_size=1,
        temperature=1.0,
        early_termination=False,
        pass_quiescence=False,
        dirichlet_epsilon=0,
    )

    games = []
    for seed in range(10):
        print(f"Generating game {seed}...", file=sys.stderr)
        game = generate_game_fixtures(seed, evaluator, config)
        games.append(game)

    output = {
        "description": "MCTS cross-validation fixtures (Python reference)",
        "config": {
            "numSimulations": config.num_simulations,
            "batchSize": config.batch_size,
            "cPuct": config.c_puct,
            "temperature": config.temperature,
            "earlyTermination": config.early_termination,
            "passQuiescence": config.pass_quiescence,
            "dirichletEpsilon": config.dirichlet_epsilon,
        },
        "games": games,
    }

    json.dump(output, sys.stdout, indent=2)
    print(file=sys.stderr)
    total_pos = sum(g["numPositions"] for g in games)
    print(f"Generated {len(games)} games, {total_pos} positions total",
          file=sys.stderr)


if __name__ == "__main__":
    main()
