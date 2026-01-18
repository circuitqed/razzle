"""Tests for bitboard utilities."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.bitboard import (
    ROWS, COLS, NUM_SQUARES,
    sq_to_rowcol, rowcol_to_sq, sq_to_algebraic, algebraic_to_sq,
    bit, popcount, iter_bits, KNIGHT_ATTACKS,
    P1_START_PIECES, P1_START_BALL, P2_START_PIECES, P2_START_BALL
)


class TestSquareConversion:
    def test_sq_to_rowcol(self):
        assert sq_to_rowcol(0) == (0, 0)  # a1
        assert sq_to_rowcol(6) == (0, 6)  # g1
        assert sq_to_rowcol(7) == (1, 0)  # a2
        assert sq_to_rowcol(55) == (7, 6)  # g8

    def test_rowcol_to_sq(self):
        assert rowcol_to_sq(0, 0) == 0
        assert rowcol_to_sq(0, 6) == 6
        assert rowcol_to_sq(7, 6) == 55

    def test_algebraic_conversion(self):
        assert sq_to_algebraic(0) == 'a1'
        assert sq_to_algebraic(3) == 'd1'
        assert sq_to_algebraic(55) == 'g8'

        assert algebraic_to_sq('a1') == 0
        assert algebraic_to_sq('d1') == 3
        assert algebraic_to_sq('g8') == 55

    def test_roundtrip(self):
        for sq in range(NUM_SQUARES):
            row, col = sq_to_rowcol(sq)
            assert rowcol_to_sq(row, col) == sq

            alg = sq_to_algebraic(sq)
            assert algebraic_to_sq(alg) == sq


class TestBitOperations:
    def test_bit(self):
        assert bit(0) == 1
        assert bit(1) == 2
        assert bit(3) == 8
        assert bit(55) == 1 << 55

    def test_popcount(self):
        assert popcount(0) == 0
        assert popcount(1) == 1
        assert popcount(0b1111) == 4
        assert popcount(P1_START_PIECES) == 5  # 5 pieces

    def test_iter_bits(self):
        bb = 0b1010101
        bits = list(iter_bits(bb))
        assert bits == [0, 2, 4, 6]

    def test_iter_bits_numpy_int64(self):
        """Test that iter_bits handles numpy.int64 (from bitboard operations)."""
        import numpy as np
        bb = np.int64(0b1010101)
        bits = list(iter_bits(bb))
        assert bits == [0, 2, 4, 6]

    def test_lsb_numpy_int64(self):
        """Test that lsb handles numpy.int64."""
        import numpy as np
        from razzle.core.bitboard import lsb
        assert lsb(np.int64(8)) == 3
        assert lsb(np.int64(0)) == -1


class TestKnightAttacks:
    def test_corner_knight(self):
        # Knight on a1 (sq 0) can only go to b3 and c2
        attacks = KNIGHT_ATTACKS[0]
        squares = list(iter_bits(attacks))
        algebraic = [sq_to_algebraic(sq) for sq in squares]
        assert set(algebraic) == {'b3', 'c2'}

    def test_center_knight(self):
        # Knight on d4 (sq 24) should have 8 moves
        attacks = KNIGHT_ATTACKS[24]
        assert popcount(attacks) == 8

    def test_edge_knight(self):
        # Knight on d1 (sq 3) has limited moves
        attacks = KNIGHT_ATTACKS[3]
        squares = list(iter_bits(attacks))
        algebraic = [sq_to_algebraic(sq) for sq in squares]
        # Should reach b2, c3, e3, f2
        assert set(algebraic) == {'b2', 'c3', 'e3', 'f2'}


class TestStartingPosition:
    def test_p1_pieces(self):
        # Player 1 pieces on b1-f1 (squares 1-5)
        squares = list(iter_bits(P1_START_PIECES))
        assert squares == [1, 2, 3, 4, 5]

    def test_p1_ball(self):
        # Player 1 ball on d1 (square 3)
        squares = list(iter_bits(P1_START_BALL))
        assert squares == [3]

    def test_p2_pieces(self):
        # Player 2 pieces on b8-f8
        squares = list(iter_bits(P2_START_PIECES))
        algebraic = [sq_to_algebraic(sq) for sq in squares]
        assert set(algebraic) == {'b8', 'c8', 'd8', 'e8', 'f8'}

    def test_p2_ball(self):
        # Player 2 ball on d8
        squares = list(iter_bits(P2_START_BALL))
        assert sq_to_algebraic(squares[0]) == 'd8'
