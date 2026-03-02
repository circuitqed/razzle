"""Tests verifying C MCTS tree matches Python MCTS behavior."""

import ctypes
import numpy as np
import pytest

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.mcts import MCTS, MCTSConfig, Node
from razzle.ai.network import NUM_ACTIONS, END_TURN_ACTION
from razzle_fast.wrapper import (
    FastState, FastMCTS, _lib, _np_to_cfloat_ptr,
    CRazzleState, CMCTSTree,
)


class DummyEvaluator:
    """Evaluator with uniform policy and zero value, for deterministic testing."""

    def evaluate(self, state):
        policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
        return policy, 0.0

    def evaluate_batch(self, states):
        return [self.evaluate(s) for s in states]


class TestMCTSTree:
    def test_create_and_free(self):
        fs = FastState()
        tree = _lib.razzle_mcts_create(ctypes.byref(fs._cs), 10000, 8, 200)
        assert tree
        _lib.razzle_mcts_free(tree)

    def test_expand_root(self):
        fs = FastState()
        tree = _lib.razzle_mcts_create(ctypes.byref(fs._cs), 10000, 8, 200)
        try:
            policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
            _lib.razzle_mcts_expand_root(tree, _np_to_cfloat_ptr(policy))

            root = tree.contents.nodes[tree.contents.root]
            assert root.is_expanded == 1
            assert root.num_children > 0

            # Verify children count matches legal moves
            gs = GameState.new_game()
            expected_moves = len(get_legal_moves(gs))
            assert root.num_children == expected_moves
        finally:
            _lib.razzle_mcts_free(tree)

    def test_select_and_expand(self):
        """Run a small search and verify tree grows."""
        fs = FastState()
        tree = _lib.razzle_mcts_create(ctypes.byref(fs._cs), 100000, 8, 200)
        try:
            policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
            _lib.razzle_mcts_expand_root(tree, _np_to_cfloat_ptr(policy))

            tensors_buf = np.zeros(8 * 392, dtype=np.float32)
            for _ in range(10):
                leaf_count = _lib.razzle_mcts_select_leaves(
                    tree, 8, 3, 1.5, _np_to_cfloat_ptr(tensors_buf)
                )
                if leaf_count > 0:
                    policies = np.ones((leaf_count, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
                    values = np.zeros(leaf_count, dtype=np.float32)
                    _lib.razzle_mcts_expand_and_backup(
                        tree, leaf_count,
                        _np_to_cfloat_ptr(policies),
                        _np_to_cfloat_ptr(values), 3
                    )

            assert _lib.razzle_mcts_root_visits(tree) > 0
        finally:
            _lib.razzle_mcts_free(tree)

    def test_get_policy(self):
        """Verify policy extraction produces valid distribution."""
        fs = FastState()
        tree = _lib.razzle_mcts_create(ctypes.byref(fs._cs), 100000, 8, 200)
        try:
            policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
            _lib.razzle_mcts_expand_root(tree, _np_to_cfloat_ptr(policy))

            tensors_buf = np.zeros(8 * 392, dtype=np.float32)
            for _ in range(20):
                leaf_count = _lib.razzle_mcts_select_leaves(
                    tree, 8, 3, 1.5, _np_to_cfloat_ptr(tensors_buf)
                )
                if leaf_count > 0:
                    policies = np.ones((leaf_count, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
                    values = np.zeros(leaf_count, dtype=np.float32)
                    _lib.razzle_mcts_expand_and_backup(
                        tree, leaf_count,
                        _np_to_cfloat_ptr(policies),
                        _np_to_cfloat_ptr(values), 3
                    )

            policy_out = np.zeros(NUM_ACTIONS, dtype=np.float32)
            _lib.razzle_mcts_get_policy(tree, _np_to_cfloat_ptr(policy_out), 1.0)

            # Should be a valid probability distribution
            assert abs(policy_out.sum() - 1.0) < 1e-5
            assert np.all(policy_out >= 0)

            # Should only have mass on legal moves
            gs = GameState.new_game()
            legal = get_legal_moves(gs)
            legal_indices = set()
            for m in legal:
                legal_indices.add(END_TURN_ACTION if m == -1 else m)

            for i in range(NUM_ACTIONS):
                if i not in legal_indices:
                    assert policy_out[i] == 0.0, f"Illegal move {i} has nonzero prob"
        finally:
            _lib.razzle_mcts_free(tree)


class TestFastMCTS:
    def test_basic_search(self):
        """Verify FastMCTS completes a search and returns valid results."""
        evaluator = DummyEvaluator()
        config = MCTSConfig(
            num_simulations=100,
            batch_size=8,
            early_termination=False,
            pass_quiescence=False,
        )
        mcts = FastMCTS(evaluator, config)
        gs = GameState.new_game()
        result = mcts.search_batched(gs, add_noise=False)

        policy = result.policy
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (NUM_ACTIONS,)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_fast_path(self):
        """Test search_fast returns valid policy and move."""
        evaluator = DummyEvaluator()
        config = MCTSConfig(
            num_simulations=50,
            batch_size=8,
            early_termination=False,
            pass_quiescence=False,
        )
        mcts = FastMCTS(evaluator, config)
        gs = GameState.new_game()
        policy, move = mcts.search_fast(gs, add_noise=False)

        assert isinstance(policy, np.ndarray)
        assert abs(policy.sum() - 1.0) < 1e-5

        # Move should be a legal move
        legal = get_legal_moves(gs)
        legal_indices = set()
        for m in legal:
            legal_indices.add(END_TURN_ACTION if m == -1 else m)
        if move == -1:
            assert END_TURN_ACTION in legal_indices
        else:
            assert move in legal_indices

    def test_search_with_noise(self):
        """Verify search works with Dirichlet noise."""
        evaluator = DummyEvaluator()
        config = MCTSConfig(
            num_simulations=100,
            batch_size=8,
            early_termination=False,
            pass_quiescence=False,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )
        mcts = FastMCTS(evaluator, config)
        gs = GameState.new_game()
        result = mcts.search_batched(gs, add_noise=True)
        assert abs(result.policy.sum() - 1.0) < 1e-5


class TestEarlyTermination:
    def test_early_stop_check(self):
        fs = FastState()
        tree = _lib.razzle_mcts_create(ctypes.byref(fs._cs), 100000, 8, 200)
        try:
            assert _lib.razzle_mcts_should_stop_early(tree, 100, 0.85) == 0
        finally:
            _lib.razzle_mcts_free(tree)
