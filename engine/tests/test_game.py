"""Tests for game state and moves."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import (
    get_legal_moves, encode_move, decode_move,
    move_to_algebraic, algebraic_to_move, MoveGenerator
)
from razzle.core.bitboard import algebraic_to_sq, bit


class TestGameState:
    def test_new_game(self):
        state = GameState.new_game()
        assert state.current_player == 0
        assert state.ply == 0
        assert not state.is_terminal()

    def test_initial_pieces(self):
        state = GameState.new_game()
        # Player 1 has pieces and ball
        assert state.pieces[0] != 0
        assert state.balls[0] != 0
        # Player 2 has pieces and ball
        assert state.pieces[1] != 0
        assert state.balls[1] != 0

    def test_copy(self):
        state = GameState.new_game()
        copy = state.copy()
        assert copy.pieces == state.pieces
        assert copy.balls == state.balls
        assert copy.current_player == state.current_player

    def test_to_tensor(self):
        state = GameState.new_game()
        tensor = state.to_tensor()
        assert tensor.shape == (7, 8, 7)
        # Should have non-zero values in piece planes
        assert tensor[0].sum() > 0  # Current player pieces
        assert tensor[1].sum() > 0  # Current player ball


class TestMoves:
    def test_encode_decode(self):
        src, dst = 3, 16  # d1-c3
        move = encode_move(src, dst)
        decoded_src, decoded_dst = decode_move(move)
        assert decoded_src == src
        assert decoded_dst == dst

    def test_algebraic_conversion(self):
        move = algebraic_to_move('d1-c3')
        assert move_to_algebraic(move) == 'd1-c3'

    def test_initial_moves(self):
        state = GameState.new_game()
        moves = get_legal_moves(state)
        assert len(moves) > 0

        # Should have knight moves (pieces can move)
        # Ball piece cannot move, but can pass
        # Other 4 pieces can move

    def test_knight_moves_from_start(self):
        state = GameState.new_game()
        knight_moves = list(MoveGenerator.get_knight_moves(state))

        # Ball is on d1 (sq 3), so that piece can't move
        # Other pieces on b1, c1, e1, f1 can move
        assert len(knight_moves) > 0

        # Convert to algebraic for easier checking
        algebraic = [move_to_algebraic(m) for m in knight_moves]
        # b1 piece should be able to reach a3, c3, d2
        assert 'b1-a3' in algebraic or 'b1-c3' in algebraic

    def test_pass_moves_from_start(self):
        state = GameState.new_game()
        pass_moves = list(MoveGenerator.get_pass_moves(state))

        # Ball on d1 can pass to c1, e1 (horizontal)
        # Cannot pass to b1 or f1 (blocked by c1, e1)
        algebraic = [move_to_algebraic(m) for m in pass_moves]
        assert 'd1-c1' in algebraic
        assert 'd1-e1' in algebraic


class TestApplyMove:
    def test_knight_move(self):
        state = GameState.new_game()
        # Move piece from b1 to c3
        move = algebraic_to_move('b1-c3')

        # Verify it's legal
        assert move in get_legal_moves(state)

        state.apply_move(move)

        # Player should change
        assert state.current_player == 1
        # Ply should increment
        assert state.ply == 1

    def test_ball_pass(self):
        state = GameState.new_game()
        # Pass ball from d1 to e1
        move = algebraic_to_move('d1-e1')

        state.apply_move(move)

        # Ball should have moved
        e1_sq = algebraic_to_sq('e1')
        assert state.balls[0] & bit(e1_sq)

    def test_terminal_detection(self):
        state = GameState.new_game()
        assert not state.is_terminal()
        assert state.get_winner() is None


class TestMustPass:
    def test_no_must_pass_initially(self):
        state = GameState.new_game()
        assert not MoveGenerator.must_pass(state)
