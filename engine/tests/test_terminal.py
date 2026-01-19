"""Tests for terminal conditions and win detection."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.bitboard import (
    bit, algebraic_to_sq, ROW_1_MASK, ROW_8_MASK,
    P1_START_PIECES, P2_START_PIECES
)


class TestP1WinCondition:
    """Player 1 wins when their ball reaches row 8."""

    def test_p1_wins_ball_on_a8(self):
        state = GameState.new_game()
        # Move P1's ball to a8
        state.balls = (bit(algebraic_to_sq('a8')), state.balls[1])
        assert state.is_terminal()
        assert state.get_winner() == 0
        assert state.get_result(0) == 1.0
        assert state.get_result(1) == 0.0

    def test_p1_wins_ball_on_d8(self):
        state = GameState.new_game()
        state.balls = (bit(algebraic_to_sq('d8')), state.balls[1])
        assert state.is_terminal()
        assert state.get_winner() == 0

    def test_p1_wins_ball_on_g8(self):
        state = GameState.new_game()
        state.balls = (bit(algebraic_to_sq('g8')), state.balls[1])
        assert state.is_terminal()
        assert state.get_winner() == 0

    def test_p1_ball_on_row7_not_terminal(self):
        state = GameState.new_game()
        state.balls = (bit(algebraic_to_sq('d7')), state.balls[1])
        assert not state.is_terminal()
        assert state.get_winner() is None


class TestP2WinCondition:
    """Player 2 wins when their ball reaches row 1."""

    def test_p2_wins_ball_on_a1(self):
        state = GameState.new_game()
        state.balls = (state.balls[0], bit(algebraic_to_sq('a1')))
        assert state.is_terminal()
        assert state.get_winner() == 1
        assert state.get_result(1) == 1.0
        assert state.get_result(0) == 0.0

    def test_p2_wins_ball_on_d1(self):
        state = GameState.new_game()
        state.balls = (state.balls[0], bit(algebraic_to_sq('d1')))
        assert state.is_terminal()
        assert state.get_winner() == 1

    def test_p2_wins_ball_on_g1(self):
        state = GameState.new_game()
        state.balls = (state.balls[0], bit(algebraic_to_sq('g1')))
        assert state.is_terminal()
        assert state.get_winner() == 1

    def test_p2_ball_on_row2_not_terminal(self):
        state = GameState.new_game()
        state.balls = (state.balls[0], bit(algebraic_to_sq('d2')))
        assert not state.is_terminal()
        assert state.get_winner() is None


class TestMoveLimitCondition:
    """Move limit: current player loses if ply > 200."""

    def test_current_player_loses_at_ply_201(self):
        state = GameState.new_game()
        state.ply = 201
        # Player 0 (current player) loses, player 1 wins
        assert state.is_terminal()
        assert state.get_winner() == 1  # Opponent of current player wins
        assert state.get_result(0) == 0.0  # Player 0 loses
        assert state.get_result(1) == 1.0  # Player 1 wins

    def test_not_draw_at_ply_200(self):
        state = GameState.new_game()
        state.ply = 200
        assert not state.is_terminal()

    def test_not_draw_at_ply_0(self):
        state = GameState.new_game()
        assert state.ply == 0
        assert not state.is_terminal()


class TestGetResult:
    """Test get_result returns correct values from each player's perspective."""

    def test_result_for_ongoing_game(self):
        state = GameState.new_game()
        # Ongoing game returns 0.5 (draw/undetermined)
        assert state.get_result(0) == 0.5
        assert state.get_result(1) == 0.5

    def test_result_p1_win(self):
        state = GameState.new_game()
        state.balls = (bit(algebraic_to_sq('d8')), state.balls[1])
        assert state.get_result(0) == 1.0  # P1 perspective: win
        assert state.get_result(1) == 0.0  # P2 perspective: loss

    def test_result_p2_win(self):
        state = GameState.new_game()
        state.balls = (state.balls[0], bit(algebraic_to_sq('d1')))
        assert state.get_result(0) == 0.0  # P1 perspective: loss
        assert state.get_result(1) == 1.0  # P2 perspective: win


class TestRowMasks:
    """Verify row masks are correct."""

    def test_row_1_mask_covers_a1_to_g1(self):
        for col in 'abcdefg':
            sq = algebraic_to_sq(f'{col}1')
            assert ROW_1_MASK & bit(sq), f"{col}1 should be in ROW_1_MASK"

    def test_row_8_mask_covers_a8_to_g8(self):
        for col in 'abcdefg':
            sq = algebraic_to_sq(f'{col}8')
            assert ROW_8_MASK & bit(sq), f"{col}8 should be in ROW_8_MASK"

    def test_row_1_mask_excludes_row_2(self):
        for col in 'abcdefg':
            sq = algebraic_to_sq(f'{col}2')
            assert not (ROW_1_MASK & bit(sq)), f"{col}2 should not be in ROW_1_MASK"

    def test_row_8_mask_excludes_row_7(self):
        for col in 'abcdefg':
            sq = algebraic_to_sq(f'{col}7')
            assert not (ROW_8_MASK & bit(sq)), f"{col}7 should not be in ROW_8_MASK"
