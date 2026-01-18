"""Tests for Monte Carlo Tree Search."""

import pytest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves, move_to_algebraic
from razzle.ai.mcts import MCTS, MCTSConfig, Node
from razzle.ai.evaluator import DummyEvaluator, BatchedEvaluator
from razzle.ai.network import RazzleNet, NUM_ACTIONS


class TestMCTSConfig:
    def test_default_config(self):
        config = MCTSConfig()
        assert config.num_simulations == 800
        assert config.c_puct == 1.5
        assert config.temperature == 1.0

    def test_custom_config(self):
        config = MCTSConfig(num_simulations=100, temperature=0.5)
        assert config.num_simulations == 100
        assert config.temperature == 0.5


class TestNode:
    def test_initial_node(self):
        state = GameState.new_game()
        node = Node(state=state)
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.value == 0.0
        assert not node.is_expanded
        assert len(node.children) == 0

    def test_value_calculation(self):
        state = GameState.new_game()
        node = Node(state=state)
        node.visit_count = 10
        node.value_sum = 5.0
        assert node.value == 0.5

    def test_ucb_score_unexplored(self):
        state = GameState.new_game()
        node = Node(state=state, prior=0.5)
        # UCB = Q + c * P * sqrt(parent) / (1 + child)
        # With 0 visits: UCB = 0 + 1.5 * 0.5 * sqrt(100) / 1 = 7.5
        score = node.ucb_score(parent_visits=100, c_puct=1.5)
        assert score == pytest.approx(7.5)

    def test_ucb_score_explored(self):
        state = GameState.new_game()
        node = Node(state=state, prior=0.5)
        node.visit_count = 10
        node.value_sum = 3.0  # Q = 0.3
        # UCB = 0.3 + 1.5 * 0.5 * sqrt(100) / 11
        expected = 0.3 + 1.5 * 0.5 * 10 / 11
        score = node.ucb_score(parent_visits=100, c_puct=1.5)
        assert score == pytest.approx(expected)


class TestNodeExpansion:
    def test_expand_creates_children(self):
        state = GameState.new_game()
        node = Node(state=state)

        # Uniform policy
        policy = np.ones(NUM_ACTIONS, dtype=np.float32)
        policy /= policy.sum()

        node.expand(policy)

        assert node.is_expanded
        legal_moves = get_legal_moves(state)
        assert len(node.children) == len(legal_moves)

    def test_expand_normalizes_priors(self):
        state = GameState.new_game()
        node = Node(state=state)

        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal_moves = get_legal_moves(state)
        for m in legal_moves:
            policy[m] = 1.0

        node.expand(policy)

        prior_sum = sum(c.prior for c in node.children.values())
        assert prior_sum == pytest.approx(1.0)

    def test_expand_twice_no_change(self):
        state = GameState.new_game()
        node = Node(state=state)
        policy = np.ones(NUM_ACTIONS, dtype=np.float32)

        node.expand(policy)
        num_children = len(node.children)

        node.expand(policy)  # Second expand should be no-op
        assert len(node.children) == num_children


class TestMCTSWithDummyEvaluator:
    def test_search_returns_root(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=10))

        state = GameState.new_game()
        root = mcts.search(state)

        assert root is not None
        assert root.is_expanded
        assert root.visit_count > 0

    def test_search_visits_children(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=50))

        state = GameState.new_game()
        root = mcts.search(state)

        total_child_visits = sum(c.visit_count for c in root.children.values())
        # Root visits = 1 (initial) + simulations
        # Child visits should be close to simulations
        assert total_child_visits >= 40  # Some margin

    def test_select_move_returns_legal_move(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=20, temperature=0))

        state = GameState.new_game()
        root = mcts.search(state)
        move = mcts.select_move(root)

        legal_moves = get_legal_moves(state)
        assert move in legal_moves

    def test_get_best_move(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=20, temperature=0))

        state = GameState.new_game()
        move = mcts.get_best_move(state)

        legal_moves = get_legal_moves(state)
        assert move in legal_moves


class TestMCTSTemperature:
    def test_temperature_zero_greedy(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=50, temperature=0))

        state = GameState.new_game()
        root = mcts.search(state)

        # With temp=0, should always select most visited
        move = mcts.select_move(root)
        best_visits = max(c.visit_count for c in root.children.values())
        assert root.children[move].visit_count == best_visits

    def test_temperature_nonzero_samples(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=100, temperature=1.0))

        state = GameState.new_game()
        root = mcts.search(state)

        # With temp=1, should sample proportionally
        # Run multiple times and check we get different moves
        moves = set()
        for _ in range(20):
            move = mcts.select_move(root)
            moves.add(move)

        # Should get at least a couple different moves with stochastic sampling
        assert len(moves) >= 2


class TestMCTSPolicy:
    def test_get_policy_shape(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=20))

        state = GameState.new_game()
        root = mcts.search(state)
        policy = mcts.get_policy(root)

        assert policy.shape == (NUM_ACTIONS,)

    def test_get_policy_sums_to_one(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=50, temperature=1.0))

        state = GameState.new_game()
        root = mcts.search(state)
        policy = mcts.get_policy(root)

        assert policy.sum() == pytest.approx(1.0)

    def test_get_policy_only_legal_moves(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=30))

        state = GameState.new_game()
        root = mcts.search(state)
        policy = mcts.get_policy(root)

        legal_moves = set(get_legal_moves(state))
        for i, prob in enumerate(policy):
            if i in legal_moves:
                # Legal moves should have probability
                pass  # May or may not be > 0 depending on visits
            else:
                # Illegal moves must have zero probability
                assert prob == 0.0


class TestMCTSAnalysis:
    def test_analyze_returns_top_moves(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=50))

        state = GameState.new_game()
        root = mcts.search(state)
        analysis = mcts.analyze(root, top_k=3)

        assert len(analysis) == 3
        for m in analysis:
            assert 'move' in m
            assert 'algebraic' in m
            assert 'visits' in m
            assert 'value' in m
            assert 'prior' in m

    def test_analyze_sorted_by_visits(self):
        evaluator = DummyEvaluator()
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=100))

        state = GameState.new_game()
        root = mcts.search(state)
        analysis = mcts.analyze(root, top_k=5)

        visits = [m['visits'] for m in analysis]
        assert visits == sorted(visits, reverse=True)


class TestMCTSDirichletNoise:
    def test_noise_changes_priors(self):
        state = GameState.new_game()
        node = Node(state=state)

        policy = np.ones(NUM_ACTIONS, dtype=np.float32)
        legal_moves = get_legal_moves(state)
        for m in legal_moves:
            policy[m] = 1.0 / len(legal_moves)

        node.expand(policy)

        # Record original priors
        original_priors = {a: c.prior for a, c in node.children.items()}

        # Add noise
        node.add_dirichlet_noise(alpha=0.3, epsilon=0.25)

        # Priors should have changed
        changed = False
        for a, c in node.children.items():
            if abs(c.prior - original_priors[a]) > 0.001:
                changed = True
                break
        assert changed

    def test_noise_priors_still_valid(self):
        state = GameState.new_game()
        node = Node(state=state)

        policy = np.ones(NUM_ACTIONS, dtype=np.float32)
        node.expand(policy)
        node.add_dirichlet_noise(alpha=0.3, epsilon=0.25)

        # Priors should still sum to ~1
        prior_sum = sum(c.prior for c in node.children.values())
        assert prior_sum == pytest.approx(1.0, rel=0.01)


class TestMCTSWithNetwork:
    def test_search_with_real_network(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=10))

        state = GameState.new_game()
        root = mcts.search(state)

        assert root.is_expanded
        assert len(root.children) > 0

    def test_move_selection_with_network(self):
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        mcts = MCTS(evaluator, MCTSConfig(num_simulations=20, temperature=0))

        state = GameState.new_game()
        move = mcts.get_best_move(state)

        legal_moves = get_legal_moves(state)
        assert move in legal_moves
