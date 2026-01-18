"""Tests for neural network evaluators."""

import pytest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.evaluator import DummyEvaluator, BatchedEvaluator
from razzle.ai.network import RazzleNet, NUM_ACTIONS


class TestDummyEvaluator:
    def test_returns_uniform_policy(self):
        evaluator = DummyEvaluator()
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        legal_moves = get_legal_moves(state)
        expected_prob = 1.0 / len(legal_moves)

        for move in legal_moves:
            assert policy[move] == pytest.approx(expected_prob)

    def test_returns_zero_value(self):
        evaluator = DummyEvaluator()
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        assert value == 0.0

    def test_policy_sums_to_one(self):
        evaluator = DummyEvaluator()
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        assert policy.sum() == pytest.approx(1.0)

    def test_illegal_moves_have_zero_prob(self):
        evaluator = DummyEvaluator()
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        legal_moves = set(get_legal_moves(state))
        for i in range(NUM_ACTIONS):
            if i not in legal_moves:
                assert policy[i] == 0.0

    def test_tracks_eval_count(self):
        evaluator = DummyEvaluator()
        assert evaluator.total_evals == 0

        state = GameState.new_game()
        evaluator.evaluate(state)
        assert evaluator.total_evals == 1

        evaluator.evaluate(state)
        evaluator.evaluate(state)
        assert evaluator.total_evals == 3

    def test_batch_evaluation(self):
        evaluator = DummyEvaluator()
        state = GameState.new_game()

        results = evaluator.evaluate_batch([state, state, state])

        assert len(results) == 3
        for policy, value in results:
            assert policy.shape == (NUM_ACTIONS,)
            assert value == 0.0


class TestBatchedEvaluator:
    def test_single_evaluation(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        assert policy.shape == (NUM_ACTIONS,)
        assert isinstance(value, float)

    def test_policy_is_probability(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        # All values should be non-negative
        assert (policy >= 0).all()
        # Should sum to approximately 1
        assert policy.sum() == pytest.approx(1.0, rel=0.01)

    def test_value_in_range(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        # Value should be in [-1, 1] (tanh output)
        assert -1.0 <= value <= 1.0

    def test_batch_evaluation(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        state = GameState.new_game()

        results = evaluator.evaluate_batch([state, state, state])

        assert len(results) == 3
        for policy, value in results:
            assert policy.shape == (NUM_ACTIONS,)
            assert -1.0 <= value <= 1.0

    def test_batch_vs_single_consistency(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        state = GameState.new_game()

        # Single evaluation
        single_policy, single_value = evaluator.evaluate(state)

        # Batch evaluation
        batch_results = evaluator.evaluate_batch([state])
        batch_policy, batch_value = batch_results[0]

        # Should be very close (may differ slightly due to floating point)
        np.testing.assert_allclose(single_policy, batch_policy, rtol=1e-5)
        assert single_value == pytest.approx(batch_value, rel=1e-5)

    def test_empty_batch(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)

        results = evaluator.evaluate_batch([])

        assert results == []

    def test_stats_tracking(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        state = GameState.new_game()

        evaluator.evaluate(state)
        evaluator.evaluate(state)
        evaluator.evaluate_batch([state, state, state])

        stats = evaluator.stats()
        assert stats['total_evals'] == 5
        assert stats['total_batches'] == 3

    def test_deterministic_evaluation(self):
        """Same state should give same output (network in eval mode)."""
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        state = GameState.new_game()

        policy1, value1 = evaluator.evaluate(state)
        policy2, value2 = evaluator.evaluate(state)

        np.testing.assert_array_equal(policy1, policy2)
        assert value1 == value2


class TestBatchedEvaluatorDevice:
    def test_cpu_device(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net, device='cpu')
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        assert policy.shape == (NUM_ACTIONS,)

    @pytest.mark.skipif(
        not __import__('torch').cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_device(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net, device='cuda')
        state = GameState.new_game()

        policy, value = evaluator.evaluate(state)

        assert policy.shape == (NUM_ACTIONS,)
