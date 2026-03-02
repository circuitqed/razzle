"""
Compare Python MCTS vs C FastMCTS with the same evaluator and seed.

Plays 50 random games and runs MCTS at every position, verifying that
both engines produce identical or near-identical visit distributions.
"""

import numpy as np
import random
import pytest

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.network import NUM_ACTIONS, END_TURN_ACTION
from razzle_fast.wrapper import FastMCTS


class FixedValueEvaluator:
    """Evaluator with uniform policy and fixed value based on state ply.

    Returns float32 values to match real NN output precision.
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
        value = float(np.float32(0.1 * (state.ply % 5 - 2)))
        return policy, value

    def evaluate_batch(self, states):
        return [self.evaluate(s) for s in states]


def _compare_at_position(gs, evaluator, num_sims=100, batch_size=8):
    """Run both engines on a position and return policies."""
    config = MCTSConfig(
        num_simulations=num_sims,
        batch_size=batch_size,
        temperature=1.0,
        early_termination=False,
        pass_quiescence=False,
        dirichlet_epsilon=0,
    )

    py_mcts = MCTS(evaluator, config)
    py_root = py_mcts.search_batched(gs, add_noise=False)
    py_policy = py_mcts.get_policy(py_root)

    c_mcts = FastMCTS(evaluator, config)
    c_result = c_mcts.search_batched(gs, add_noise=False)
    c_policy = c_result.policy

    return py_policy, c_policy


class TestCompareFullGames:
    """Play 50 random games and compare MCTS at every position.

    With batch_size=1, both engines produce identical results. With
    batch_size=8, virtual loss during within-batch leaf selection can
    cause ~0.1% of positions to differ by 1 visit on 2 actions. This
    is an inherent property of batched MCTS tie-breaking and is
    acceptable.
    """

    @pytest.mark.parametrize("game_seed", range(50))
    def test_every_position_in_game(self, game_seed):
        """Play a random game, run MCTS at each position."""
        rng = random.Random(game_seed)
        gs = GameState.new_game()
        evaluator = FixedValueEvaluator()
        position = 0
        mismatches = []

        while not gs.is_terminal():
            py_pol, c_pol = _compare_at_position(gs, evaluator)

            max_diff = np.max(np.abs(py_pol - c_pol))
            if max_diff > 0:
                mismatches.append((position, gs.ply, max_diff))

            moves = get_legal_moves(gs)
            gs.apply_move(rng.choice(moves))
            position += 1

            if position >= 60:
                break

        # Allow at most 2 positions per game with tiny 1-visit diffs
        bad = [(pos, ply, d) for pos, ply, d in mismatches if d > 0.02]
        assert not bad, (
            f"Game {game_seed}: {len(bad)} positions with large diffs: "
            + ", ".join(f"pos={p} ply={pl} diff={d:.4f}" for p, pl, d in bad)
        )
        assert len(mismatches) <= 2, (
            f"Game {game_seed}: too many mismatched positions ({len(mismatches)}): "
            + ", ".join(f"pos={p} ply={pl} diff={d:.4f}" for p, pl, d in mismatches)
        )

    @pytest.mark.parametrize("game_seed", range(10))
    def test_sequential_exact_match(self, game_seed):
        """With batch_size=1, both engines should be exactly identical."""
        rng = random.Random(game_seed)
        gs = GameState.new_game()
        evaluator = FixedValueEvaluator()
        position = 0

        while not gs.is_terminal():
            config = MCTSConfig(
                num_simulations=100,
                batch_size=1,
                temperature=1.0,
                early_termination=False,
                pass_quiescence=False,
                dirichlet_epsilon=0,
            )

            py_mcts = MCTS(evaluator, config)
            py_root = py_mcts.search_batched(gs, add_noise=False)
            py_pol = py_mcts.get_policy(py_root)

            c_mcts = FastMCTS(evaluator, config)
            c_result = c_mcts.search_batched(gs, add_noise=False)
            c_pol = c_result.policy

            np.testing.assert_array_equal(
                py_pol, c_pol,
                err_msg=(
                    f"Game {game_seed}, position {position} "
                    f"(ply={gs.ply}, player={gs.current_player}): "
                    f"sequential search should match exactly"
                ),
            )

            moves = get_legal_moves(gs)
            gs.apply_move(rng.choice(moves))
            position += 1

            if position >= 30:
                break
