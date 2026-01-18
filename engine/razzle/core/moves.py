"""
Move generation for Razzle Dazzle.

Handles knight moves and ball passes with proper rule enforcement.
"""

from __future__ import annotations
from typing import Iterator

from .bitboard import (
    ROWS, COLS, NUM_SQUARES,
    KNIGHT_ATTACKS, RAY_MASKS, BETWEEN,
    bit, iter_bits, sq_to_rowcol, rowcol_to_sq, sq_to_algebraic,
    is_valid_sq
)
from .state import GameState


# Move encoding: src * 56 + dst
def encode_move(src: int, dst: int) -> int:
    """Encode a move as a single integer."""
    return src * NUM_SQUARES + dst


def decode_move(move: int) -> tuple[int, int]:
    """Decode a move into (src, dst)."""
    return move // NUM_SQUARES, move % NUM_SQUARES


def move_to_algebraic(move: int) -> str:
    """Convert move to algebraic notation."""
    src, dst = decode_move(move)
    return f"{sq_to_algebraic(src)}-{sq_to_algebraic(dst)}"


def algebraic_to_move(s: str) -> int:
    """Parse algebraic notation to move."""
    from .bitboard import algebraic_to_sq
    parts = s.strip().split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid move format: {s}")
    src = algebraic_to_sq(parts[0])
    dst = algebraic_to_sq(parts[1])
    return encode_move(src, dst)


class MoveGenerator:
    """Generates legal moves for a game state."""

    @staticmethod
    def get_knight_moves(state: GameState) -> Iterator[int]:
        """
        Generate all legal knight moves for current player.

        A piece can only move if it does NOT have the ball.
        """
        p = state.current_player
        pieces = state.pieces[p]
        ball_sq = None

        # Find which piece has the ball (if any)
        for sq in iter_bits(state.balls[p]):
            ball_sq = sq

        # A piece at ball position cannot move (it's holding the ball)
        movable_pieces = pieces & ~state.balls[p]

        for src in iter_bits(movable_pieces):
            # Knight attacks from this square, filtered to empty squares
            targets = KNIGHT_ATTACKS[src] & state.empty
            for dst in iter_bits(targets):
                yield encode_move(src, dst)

    @staticmethod
    def get_pass_moves(state: GameState) -> Iterator[int]:
        """
        Generate all legal ball passes for current player.

        The ball can be passed in a straight line (horizontal, vertical, diagonal)
        to any friendly piece that hasn't touched the ball this turn.
        """
        p = state.current_player
        ball_pos = state.balls[p]

        if not ball_pos:
            return

        ball_sq = next(iter_bits(ball_pos))
        my_pieces = state.pieces[p]
        occupied = state.occupied

        # Can't pass to pieces that already touched ball this turn
        valid_receivers = my_pieces & ~state.touched_mask

        # Check each direction
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # N, S, E, W
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonals
        ]

        ball_row, ball_col = sq_to_rowcol(ball_sq)

        for dr, dc in directions:
            r, c = ball_row + dr, ball_col + dc
            while is_valid_sq(r, c):
                sq = rowcol_to_sq(r, c)
                sq_bit = bit(sq)

                if occupied & sq_bit:
                    # Hit something
                    if valid_receivers & sq_bit:
                        # It's our piece and can receive
                        yield encode_move(ball_sq, sq)
                    break  # Can't pass through pieces

                r += dr
                c += dc

    @staticmethod
    def must_pass(state: GameState) -> bool:
        """
        Check if current player MUST pass (opponent adjacent to ball).

        "If your opponent's previous move results in one of her pieces
        being adjacent to your ball, you must pass if you can."
        """
        p = state.current_player
        ball_pos = state.balls[p]
        if not ball_pos:
            return False

        ball_sq = next(iter_bits(ball_pos))
        ball_row, ball_col = sq_to_rowcol(ball_sq)

        opp_pieces = state.pieces[1 - p]

        # Check all 8 adjacent squares
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = ball_row + dr, ball_col + dc
                if is_valid_sq(r, c):
                    adj_sq = rowcol_to_sq(r, c)
                    if opp_pieces & bit(adj_sq):
                        return True

        return False

    @staticmethod
    def get_legal_moves(state: GameState) -> list[int]:
        """
        Get all legal moves for current player.

        Returns list of encoded moves. Move type is determined by context:
        - If source is ball position: it's a pass
        - Otherwise: it's a knight move

        Rules:
        1. If opponent is adjacent to your ball, you MUST pass if possible
        2. You can chain passes (each is a separate move in atomic encoding)
        3. After passing, you can move a piece OR continue passing
        4. A knight move ends your turn
        """
        moves = []

        # Check if we must pass
        forced_pass = MoveGenerator.must_pass(state)
        pass_moves = list(MoveGenerator.get_pass_moves(state))

        if forced_pass and pass_moves:
            # Must pass - only return pass moves
            return pass_moves

        # Can do either passes or knight moves
        moves.extend(pass_moves)
        moves.extend(MoveGenerator.get_knight_moves(state))

        return moves

    @staticmethod
    def get_move_mask(state: GameState) -> list[bool]:
        """
        Get a mask indicating which moves are legal.

        Returns a list of NUM_SQUARES * NUM_SQUARES booleans.
        Index i is True if move i is legal.

        Useful for neural network output masking.
        """
        mask = [False] * (NUM_SQUARES * NUM_SQUARES)
        for move in MoveGenerator.get_legal_moves(state):
            mask[move] = True
        return mask


# Convenience functions
def get_legal_moves(state: GameState) -> list[int]:
    """Get all legal moves for the current player."""
    return MoveGenerator.get_legal_moves(state)


def is_legal_move(state: GameState, move: int) -> bool:
    """Check if a move is legal."""
    return move in MoveGenerator.get_legal_moves(state)


def get_move_count(state: GameState) -> int:
    """Get number of legal moves."""
    return len(MoveGenerator.get_legal_moves(state))
