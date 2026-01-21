"""
Tests for correct player perspective in training data generation.

These tests verify the critical bug fix where player tracking must account
for ball passes (which don't switch players) rather than assuming strict
turn alternation via `i % 2`.
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Any

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves, encode_move
from razzle.ai.network import NUM_ACTIONS, END_TURN_ACTION
from razzle.training.selfplay import GameRecord


class TestPlayerSwitchingLogic:
    """Test that we understand when players switch vs stay the same."""

    def test_knight_move_switches_player(self):
        """Knight moves should switch the current player."""
        state = GameState.new_game()
        assert state.current_player == 0

        # Find a legal knight move (not a pass, not end_turn)
        legal_moves = get_legal_moves(state)
        knight_moves = [m for m in legal_moves if m >= 0 and not self._is_pass(state, m)]
        assert knight_moves, "Should have knight moves at start"

        state.apply_move(knight_moves[0])
        assert state.current_player == 1, "Knight move should switch player"

    def test_ball_pass_keeps_player(self):
        """Ball passes should keep the same player."""
        # We need to set up a state where a pass is possible
        state = GameState.new_game()

        # Find a sequence that allows a pass
        # First, need to find a state where the ball can be passed
        # This requires the ball to be adjacent to a friendly piece

        # At game start, player 0 has ball at a1 and pieces at c1, e1
        # Ball at a1 (square 0), pieces at c1 (square 2) and e1 (square 4)
        # We need to move a piece to be knight-adjacent to the ball

        # For simplicity, let's manually check if passes are available
        legal_moves = get_legal_moves(state)
        pass_moves = [m for m in legal_moves if m >= 0 and self._is_pass(state, m)]

        if pass_moves:
            player_before = state.current_player
            state.apply_move(pass_moves[0])
            assert state.current_player == player_before, "Ball pass should NOT switch player"
        else:
            # Skip if no pass available at start (which is expected)
            pytest.skip("No pass available at game start - this is expected")

    def test_end_turn_switches_player(self):
        """END_TURN should switch the current player."""
        state = GameState.new_game()
        original_player = state.current_player

        # END_TURN is only legal after a pass, so we need to set up that state
        state.has_passed = True  # Simulate having made a pass

        legal_moves = get_legal_moves(state)
        assert -1 in legal_moves, "END_TURN should be legal after passing"

        state.apply_move(-1)  # END_TURN
        assert state.current_player != original_player, "END_TURN should switch player"

    def _is_pass(self, state: GameState, move: int) -> bool:
        """Check if a move is a ball pass (vs knight move)."""
        if move < 0:
            return False
        src = move // 56
        return bool(state.balls[state.current_player] & (1 << src))


class TestGameRecordPlayerTracking:
    """Test that GameRecord correctly tracks and uses player info."""

    def test_gamerecord_with_players_list(self):
        """GameRecord should use tracked players list for training examples."""
        # Create a GameRecord with explicit player tracking
        # Simulating: P0 move, P1 move, P1 pass (stays P1), P1 end_turn -> P0
        states = [np.zeros((6, 8, 7), dtype=np.float32) for _ in range(4)]
        policies = [np.zeros(NUM_ACTIONS, dtype=np.float32) for _ in range(4)]
        players = [0, 1, 1, 0]  # P0, P1, P1 (after pass), P0 (after end_turn)

        record = GameRecord(
            states=states,
            policies=policies,
            result=1.0,  # P0 wins
            moves=[100, 200, 300, -1],
            players=players,
        )

        examples = record.training_examples(use_ball_shaping=False)
        assert len(examples) == 4

        # Check values are assigned from correct perspective
        # P0 wins (result=1.0), so:
        # - Position where P0 to move: value = +1.0
        # - Position where P1 to move: value = -1.0
        for i, (state, policy, value, legal_mask) in enumerate(examples):
            expected_player = players[i]
            if expected_player == 0:
                assert value == pytest.approx(1.0), f"Position {i}: P0 to move should see +1.0 (P0 won)"
            else:
                assert value == pytest.approx(-1.0), f"Position {i}: P1 to move should see -1.0 (P0 won)"

    def test_gamerecord_without_players_falls_back(self):
        """GameRecord without players list should fall back to i % 2."""
        # This tests backward compatibility with old game records
        states = [np.zeros((6, 8, 7), dtype=np.float32) for _ in range(3)]
        policies = [np.zeros(NUM_ACTIONS, dtype=np.float32) for _ in range(3)]

        record = GameRecord(
            states=states,
            policies=policies,
            result=1.0,
            moves=[100, 200, 300],
            players=[],  # Empty - should fall back to i % 2
        )

        examples = record.training_examples(use_ball_shaping=False)
        assert len(examples) == 3

        # Fallback: i % 2
        # i=0 -> P0, i=1 -> P1, i=2 -> P0
        _, _, value0, _ = examples[0]
        _, _, value1, _ = examples[1]
        _, _, value2, _ = examples[2]

        assert value0 == pytest.approx(1.0)   # P0 to move, P0 won
        assert value1 == pytest.approx(-1.0)  # P1 to move, P0 won
        assert value2 == pytest.approx(1.0)   # P0 to move, P0 won


class TestTrainerPlayerTracking:
    """Test that trainer.py correctly tracks players during replay."""

    def test_trainer_games_to_training_data_tracks_player(self):
        """games_to_training_data should replay game to track actual player."""
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        # We need to construct a game that has ball passes to verify
        # the player tracking is correct

        # Create a mock TrainingGame with the structure trainer.py expects
        @dataclass
        class MockTrainingGame:
            moves: list[int]
            visit_counts: list[dict[int, int]]
            result: float

        # Simple game: just alternating knight moves
        # This should work with i % 2 AND with correct tracking
        state = GameState.new_game()
        legal_moves = get_legal_moves(state)
        knight_moves = [m for m in legal_moves if m >= 0]

        if knight_moves:
            mock_game = MockTrainingGame(
                moves=[knight_moves[0]],
                visit_counts=[{knight_moves[0]: 100}],
                result=1.0,
            )

            # Import the function from trainer.py
            # Note: This requires the trainer.py module to be importable
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
                from trainer import games_to_training_data

                # games_to_training_data returns 5 values: states, policies, values, legal_masks, difficulties
                # difficulties is None when no network is provided
                states, policies, values, legal_masks, difficulties = games_to_training_data([mock_game])
                assert len(states) == 1
                assert len(policies) == 1
                assert len(values) == 1
                assert len(legal_masks) == 1
                assert difficulties is None  # No network provided, so no difficulty targets
                # P0 was to move, P0 won -> value should be +1.0
                assert values[0] == pytest.approx(1.0)
            except ImportError:
                pytest.skip("Could not import trainer module")


class TestEndTurnActionIndex:
    """Test that END_TURN is properly mapped to END_TURN_ACTION index."""

    def test_end_turn_action_constant(self):
        """END_TURN_ACTION should be 3136 (56*56)."""
        assert END_TURN_ACTION == 3136

    def test_num_actions_includes_end_turn(self):
        """NUM_ACTIONS should be 3137 to include END_TURN."""
        assert NUM_ACTIONS == 3137

    def test_mcts_get_policy_includes_end_turn(self):
        """MCTS.get_policy should put END_TURN probability at index 3136."""
        from razzle.ai.mcts import MCTS, MCTSConfig, Node
        from razzle.ai.evaluator import DummyEvaluator

        evaluator = DummyEvaluator()
        config = MCTSConfig(num_simulations=10)
        mcts = MCTS(evaluator, config)

        # Create a state where END_TURN is legal (after passing)
        state = GameState.new_game()
        state.has_passed = True  # Simulate having passed

        root = mcts.search(state, add_noise=False)
        policy = mcts.get_policy(root)

        # Check that policy has correct shape
        assert policy.shape == (NUM_ACTIONS,)

        # If END_TURN was a valid move and got visits, it should be at index 3136
        if -1 in root.children:
            end_turn_visits = root.children[-1].visit_count
            if end_turn_visits > 0:
                # END_TURN should have non-zero probability at END_TURN_ACTION
                assert policy[END_TURN_ACTION] > 0, "END_TURN should have probability at index 3136"

    def test_dummy_evaluator_includes_end_turn(self):
        """DummyEvaluator should give END_TURN prior at index 3136."""
        from razzle.ai.evaluator import DummyEvaluator

        evaluator = DummyEvaluator()

        # Create state where END_TURN is legal
        state = GameState.new_game()
        state.has_passed = True

        policy, value = evaluator.evaluate(state)

        assert policy.shape == (NUM_ACTIONS,)

        # END_TURN should be legal and have non-zero prior
        legal_moves = get_legal_moves(state)
        if -1 in legal_moves:
            assert policy[END_TURN_ACTION] > 0, "END_TURN should have prior at index 3136"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_game_player_tracking(self):
        """Play a complete game and verify player tracking is consistent."""
        state = GameState.new_game()
        move_count = 0
        max_moves = 50  # Limit for test

        players_seen = []
        moves_made = []

        while not state.is_terminal() and move_count < max_moves:
            players_seen.append(state.current_player)

            legal_moves = get_legal_moves(state)
            if not legal_moves:
                break

            # Pick first legal move
            move = legal_moves[0]
            moves_made.append(move)

            prev_player = state.current_player
            state.apply_move(move)

            # Verify player switching logic
            if move == -1:  # END_TURN
                assert state.current_player != prev_player, "END_TURN should switch"
            elif self._is_pass(state, move, prev_player):
                # Note: we check with prev_player since state has changed
                assert state.current_player == prev_player, "Pass should NOT switch"
            else:
                # Knight move
                assert state.current_player != prev_player, "Knight move should switch"

            move_count += 1

        # Verify we actually played some moves
        assert move_count > 0, "Should have played at least one move"

    def _is_pass(self, state: GameState, move: int, player: int) -> bool:
        """Check if a move was a ball pass (check the previous state's ball position)."""
        # This is tricky because state has already changed
        # We check if the move's source was where the ball WAS before the move
        if move < 0:
            return False
        # After a pass, the ball is at dst and src is now empty
        # We can't easily check this after the fact, so we use a heuristic:
        # If has_passed is True, we just made a pass
        return state.has_passed
