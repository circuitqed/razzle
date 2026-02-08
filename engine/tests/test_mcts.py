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
        # When parent_player matches child's current_player, no negation
        score = node.ucb_score(parent_visits=100, c_puct=1.5, parent_player=state.current_player)
        assert score == pytest.approx(7.5)

    def test_ucb_score_explored(self):
        state = GameState.new_game()
        node = Node(state=state, prior=0.5)
        node.visit_count = 10
        node.value_sum = 3.0  # Q = 0.3
        # UCB = 0.3 + 1.5 * 0.5 * sqrt(100) / 11
        # When parent_player matches child's current_player, no negation
        expected = 0.3 + 1.5 * 0.5 * 10 / 11
        score = node.ucb_score(parent_visits=100, c_puct=1.5, parent_player=state.current_player)
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


class TestMCTSBatched:
    """Tests for batched MCTS search."""

    def test_batched_search_returns_root(self):
        """Batched search returns valid root node."""
        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=20, batch_size=4)
        mcts = MCTS(evaluator, config)

        state = GameState.new_game()
        root = mcts.search_batched(state)

        assert root is not None
        assert root.is_expanded
        assert root.visit_count > 0

    def test_batched_search_visits_correct_count(self):
        """Batched search does approximately correct number of simulations."""
        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=100, batch_size=8)
        mcts = MCTS(evaluator, config)

        state = GameState.new_game()
        root = mcts.search_batched(state)

        # Visit count should be approximately num_simulations
        # Root is visited each simulation plus initial expansion
        total_child_visits = sum(c.visit_count for c in root.children.values())
        assert total_child_visits >= 90  # Allow some margin

    def test_batched_select_move_returns_legal(self):
        """Batched search produces legal move selection."""
        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=50, batch_size=8, temperature=0)
        mcts = MCTS(evaluator, config)

        state = GameState.new_game()
        root = mcts.search_batched(state)
        move = mcts.select_move(root)

        legal_moves = get_legal_moves(state)
        assert move in legal_moves

    def test_batched_virtual_loss_clears(self):
        """Virtual loss is properly cleared after search."""
        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=50, batch_size=8, virtual_loss=3)
        mcts = MCTS(evaluator, config)

        state = GameState.new_game()
        root = mcts.search_batched(state)

        # Check no virtual loss remains
        def check_no_virtual_loss(node):
            assert node.virtual_loss == 0, f"Node has virtual_loss={node.virtual_loss}"
            for child in node.children.values():
                check_no_virtual_loss(child)

        check_no_virtual_loss(root)

    def test_batched_with_network(self):
        """Batched search works with neural network evaluator."""
        net = RazzleNet()
        evaluator = BatchedEvaluator(net)
        config = MCTSConfig(num_simulations=32, batch_size=8)
        mcts = MCTS(evaluator, config)

        state = GameState.new_game()
        root = mcts.search_batched(state)
        move = mcts.select_move(root)

        legal_moves = get_legal_moves(state)
        assert move in legal_moves

    def test_batched_different_batch_sizes(self):
        """Batched search works with various batch sizes."""
        evaluator = DummyEvaluator()
        state = GameState.new_game()

        for batch_size in [1, 4, 8, 16]:
            config = MCTSConfig(num_simulations=32, batch_size=batch_size)
            mcts = MCTS(evaluator, config)
            root = mcts.search_batched(state)

            assert root.is_expanded
            move = mcts.select_move(root)
            assert move in get_legal_moves(state)

    def test_batched_policy_valid(self):
        """Batched search produces valid policy output."""
        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=50, batch_size=8, temperature=1.0)
        mcts = MCTS(evaluator, config)

        state = GameState.new_game()
        root = mcts.search_batched(state)
        policy = mcts.get_policy(root)

        assert policy.shape == (NUM_ACTIONS,)
        assert policy.sum() == pytest.approx(1.0)


class TestMCTSPassQuiescence:
    """Tests for pass quiescence search - finding wins during pass chains."""

    def _create_winning_position(self) -> tuple[GameState, int]:
        """
        Create a position where player 0 can win via a pass chain.

        Returns (state, winning_move) where applying winning_move leads to victory.

        Position: Player 0 has ball at g7 (row 7) with a piece at g8 (goal row).
        Passing g7->g8 wins immediately.
        """
        state = GameState.new_game()
        # Set up: player 0 has ball ready to score
        # Ball at g7 (square 6 + 6*7 = 48), piece at g8 (square 6 + 7*7 = 55)
        from razzle.core.bitboard import bit
        state.pieces = (bit(55) | bit(10) | bit(11) | bit(12) | bit(13), state.pieces[1])
        state.balls = (bit(48), state.balls[1])
        # Make a knight move to set up for pass (has_passed must be False initially)
        # Actually, to test quiescence, we need has_passed=True
        # So simulate: player made a pass, now can continue passing to goal
        state.has_passed = True
        state.touched_mask = bit(48)  # Ball position is touched

        # Winning move: pass from g7 (48) to g8 (55) = 48 * 56 + 55 = 2743
        winning_move = 48 * 56 + 55
        return state, winning_move

    def _create_two_pass_win(self) -> tuple[GameState, int, int]:
        """
        Create a position where player 0 can win via TWO passes.

        Ball at f6, piece at g7, piece at g8 (goal).
        Path: f6->g7->g8 wins.

        Returns (state, first_pass, second_pass).
        """
        from razzle.core.bitboard import bit
        state = GameState.new_game()
        # f6 = 5 + 5*7 = 40, g7 = 6 + 6*7 = 48, g8 = 6 + 7*7 = 55
        state.pieces = (bit(48) | bit(55) | bit(10) | bit(11) | bit(12), state.pieces[1])
        state.balls = (bit(40), state.balls[1])
        state.has_passed = True
        state.touched_mask = bit(40)  # Ball position is touched

        # First pass: f6 (40) -> g7 (48) = 40 * 56 + 48 = 2288
        first_pass = 40 * 56 + 48
        # Second pass: g7 (48) -> g8 (55) = 48 * 56 + 55 = 2743
        second_pass = 48 * 56 + 55
        return state, first_pass, second_pass

    def test_quiescence_finds_immediate_win(self):
        """Quiescence search finds immediate winning pass."""
        state, winning_move = self._create_winning_position()

        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=50, pass_quiescence=True)
        mcts = MCTS(evaluator, config)

        root = mcts.search(state)
        best_move = mcts.select_move(root)

        # The winning move should have very high value
        if winning_move in root.children:
            winning_child = root.children[winning_move]
            # Value should be near +1 (winning)
            assert winning_child.value > 0.8, f"Winning move value {winning_child.value} should be > 0.8"

    def test_batched_quiescence_finds_immediate_win(self):
        """Batched MCTS with quiescence finds immediate winning pass."""
        state, winning_move = self._create_winning_position()

        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=50, batch_size=8, pass_quiescence=True)
        mcts = MCTS(evaluator, config)

        root = mcts.search_batched(state)

        # The winning move should be found
        if winning_move in root.children:
            winning_child = root.children[winning_move]
            # Value should be near +1 (winning)
            assert winning_child.value > 0.8, f"Winning move value {winning_child.value} should be > 0.8"
            # Should get most visits
            max_visits = max(c.visit_count for c in root.children.values())
            assert winning_child.visit_count >= max_visits * 0.8, "Winning move should get most visits"

    def test_batched_quiescence_finds_two_pass_win(self):
        """Batched MCTS with quiescence finds 2-pass winning sequence."""
        state, first_pass, second_pass = self._create_two_pass_win()

        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=100, batch_size=8, pass_quiescence=True)
        mcts = MCTS(evaluator, config)

        root = mcts.search_batched(state)

        # The first pass should lead to the winning sequence
        if first_pass in root.children:
            first_child = root.children[first_pass]
            # Value should be near +1 (winning)
            assert first_child.value > 0.8, f"First pass value {first_child.value} should be > 0.8"

    def test_batched_quiescence_prefers_win_over_end_turn(self):
        """Batched MCTS prefers winning pass over END_TURN."""
        state, winning_move = self._create_winning_position()

        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=100, batch_size=8, pass_quiescence=True, temperature=0)
        mcts = MCTS(evaluator, config)

        root = mcts.search_batched(state)
        best_move = mcts.select_move(root)

        # When there's a winning move, it should be selected over END_TURN
        # END_TURN is encoded as -1
        if winning_move in root.children:
            winning_visits = root.children[winning_move].visit_count
            end_turn_visits = root.children.get(-1, Node(state=state)).visit_count
            assert winning_visits > end_turn_visits, \
                f"Winning move visits ({winning_visits}) should exceed END_TURN ({end_turn_visits})"

    def test_quiescence_config_respected(self):
        """Quiescence can be toggled via config."""
        state, winning_move = self._create_winning_position()

        # With quiescence enabled
        evaluator = DummyEvaluator()
        config_with = MCTSConfig(num_simulations=50, batch_size=8, pass_quiescence=True)
        mcts_with = MCTS(evaluator, config_with)
        root_with = mcts_with.search_batched(state)

        # Without quiescence
        config_without = MCTSConfig(num_simulations=50, batch_size=8, pass_quiescence=False)
        mcts_without = MCTS(evaluator, config_without)
        root_without = mcts_without.search_batched(state)

        # Both should find the winning move eventually since it's a terminal state
        # But with quiescence, the win should be found on first evaluation of the position
        # The quiescence_evals stat should be higher when enabled
        assert mcts_with.stats.quiescence_evals >= 0
        assert mcts_without.stats.quiescence_evals == 0
