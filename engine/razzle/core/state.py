"""
Game state representation for Razzle Dazzle.

Uses bitboards for efficient move generation and state manipulation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .bitboard import (
    ROWS, COLS, NUM_SQUARES, VALID_MASK,
    P1_START_PIECES, P1_START_BALL, P2_START_PIECES, P2_START_BALL,
    ROW_1_MASK, ROW_8_MASK,
    bit, popcount, iter_bits, sq_to_algebraic, print_bitboard
)


@dataclass(frozen=False)
class GameState:
    """
    Represents the complete state of a Razzle Dazzle game.

    Attributes:
        pieces: Tuple of (p1_pieces, p2_pieces) bitboards
        balls: Tuple of (p1_ball, p2_ball) bitboards (single bit each)
        current_player: 0 for player 1, 1 for player 2
        touched_mask: Bitboard of pieces that have touched ball this turn
        ply: Number of half-moves played
        history: Stack of (move, captured_state) for undo
    """
    pieces: tuple[int, int] = (P1_START_PIECES, P2_START_PIECES)
    balls: tuple[int, int] = (P1_START_BALL, P2_START_BALL)
    current_player: int = 0
    touched_mask: int = 0
    ply: int = 0
    history: list = field(default_factory=list)

    @classmethod
    def new_game(cls) -> GameState:
        """Create a new game in the starting position."""
        return cls()

    @property
    def my_pieces(self) -> int:
        """Bitboard of current player's pieces (excluding ball)."""
        return self.pieces[self.current_player]

    @property
    def my_ball(self) -> int:
        """Bitboard of current player's ball position."""
        return self.balls[self.current_player]

    @property
    def opp_pieces(self) -> int:
        """Bitboard of opponent's pieces (excluding ball)."""
        return self.pieces[1 - self.current_player]

    @property
    def opp_ball(self) -> int:
        """Bitboard of opponent's ball position."""
        return self.balls[1 - self.current_player]

    @property
    def occupied(self) -> int:
        """Bitboard of all occupied squares."""
        return self.pieces[0] | self.pieces[1] | self.balls[0] | self.balls[1]

    @property
    def empty(self) -> int:
        """Bitboard of all empty squares."""
        return (~self.occupied) & VALID_MASK

    def is_terminal(self) -> bool:
        """Check if game is over."""
        # Player 1 wins by getting ball to row 8
        if self.balls[0] & ROW_8_MASK:
            return True
        # Player 2 wins by getting ball to row 1
        if self.balls[1] & ROW_1_MASK:
            return True
        # Draw by excessive length (optional)
        if self.ply > 200:
            return True
        return False

    def get_winner(self) -> Optional[int]:
        """Return winner (0 or 1) or None if no winner yet."""
        if self.balls[0] & ROW_8_MASK:
            return 0
        if self.balls[1] & ROW_1_MASK:
            return 1
        return None

    def get_result(self, player: int) -> float:
        """Get game result from player's perspective: 1.0=win, 0.0=loss, 0.5=draw."""
        winner = self.get_winner()
        if winner is None:
            return 0.5  # Draw or ongoing
        return 1.0 if winner == player else 0.0

    def copy(self) -> GameState:
        """Create a shallow copy (history is shared, but that's fine for MCTS)."""
        return GameState(
            pieces=self.pieces,
            balls=self.balls,
            current_player=self.current_player,
            touched_mask=self.touched_mask,
            ply=self.ply,
            history=[]  # Fresh history for copy
        )

    def apply_move(self, move: int) -> None:
        """
        Apply a move to the state. Modifies state in-place.

        Move encoding:
        - Knight move: src * 56 + dst (where piece moves from src to dst)
        - Pass: src * 56 + dst (where ball moves from src to dst)
        - End turn: -1 (explicitly end turn after passing)

        Move type is determined by whether src has the ball or a piece.
        """
        # Handle end turn
        if move == -1:
            # Save state for undo
            self.history.append((
                move,
                self.pieces,
                self.balls,
                self.current_player,
                self.touched_mask
            ))
            # Switch player
            self.current_player = 1 - self.current_player
            self.touched_mask = 0
            self.ply += 1
            return

        src = move // NUM_SQUARES
        dst = move % NUM_SQUARES
        src_bit = bit(src)
        dst_bit = bit(dst)

        # Save state for undo
        self.history.append((
            move,
            self.pieces,
            self.balls,
            self.current_player,
            self.touched_mask
        ))

        p = self.current_player

        # Determine move type based on what's at source
        if self.balls[p] & src_bit:
            # Ball pass: move ball from src to dst (dst has our piece)
            new_balls = list(self.balls)
            new_balls[p] = dst_bit
            self.balls = tuple(new_balls)

            # Mark both passer (src) and receiver (dst) as having touched ball
            self.touched_mask |= src_bit | dst_bit

            # Don't switch player - can continue passing or move a piece
            # Actually, in atomic moves, we DO switch if this completes the turn
            # For now, let's say pass doesn't end turn (handled by move generator)

        else:
            # Knight move: piece moves from src to dst
            new_pieces = list(self.pieces)
            new_pieces[p] = (new_pieces[p] & ~src_bit) | dst_bit
            self.pieces = tuple(new_pieces)

            # If this piece had the ball, ball moves with it?
            # No - in Razzle Dazzle, the piece with ball CANNOT move
            # So this is a non-ball piece moving

            # Knight move ends the turn
            self.current_player = 1 - p
            self.touched_mask = 0
            self.ply += 1

    def undo_move(self) -> None:
        """Undo the last move."""
        if not self.history:
            raise ValueError("No moves to undo")

        _, self.pieces, self.balls, self.current_player, self.touched_mask = self.history.pop()
        if self.current_player != self.current_player:  # Turn changed
            self.ply -= 1

    def to_tensor(self) -> np.ndarray:
        """
        Convert state to neural network input tensor.

        Returns (6, 8, 7) float32 array:
          - Plane 0: Current player's pieces
          - Plane 1: Current player's ball
          - Plane 2: Opponent's pieces
          - Plane 3: Opponent's ball
          - Plane 4: Touched mask (pieces that can't receive passes)
          - Plane 5: Current player indicator (all 1s if player 1, all 0s if player 2)
        """
        planes = np.zeros((6, ROWS, COLS), dtype=np.float32)

        p = self.current_player
        opp = 1 - p

        for sq in iter_bits(self.pieces[p]):
            row, col = sq // COLS, sq % COLS
            planes[0, row, col] = 1.0

        for sq in iter_bits(self.balls[p]):
            row, col = sq // COLS, sq % COLS
            planes[1, row, col] = 1.0

        for sq in iter_bits(self.pieces[opp]):
            row, col = sq // COLS, sq % COLS
            planes[2, row, col] = 1.0

        for sq in iter_bits(self.balls[opp]):
            row, col = sq // COLS, sq % COLS
            planes[3, row, col] = 1.0

        for sq in iter_bits(self.touched_mask):
            row, col = sq // COLS, sq % COLS
            planes[4, row, col] = 1.0

        if p == 0:
            planes[5, :, :] = 1.0

        return planes

    def __hash__(self) -> int:
        """Hash for transposition table."""
        return hash((self.pieces, self.balls, self.current_player, self.touched_mask))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return (
            self.pieces == other.pieces and
            self.balls == other.balls and
            self.current_player == other.current_player and
            self.touched_mask == other.touched_mask
        )

    def __repr__(self) -> str:
        """Pretty print the board."""
        symbols = {}

        # Place pieces
        for sq in iter_bits(self.pieces[0]):
            symbols[sq] = 'x'
        for sq in iter_bits(self.pieces[1]):
            symbols[sq] = 'o'

        # Place balls (uppercase)
        for sq in iter_bits(self.balls[0]):
            symbols[sq] = 'X'
        for sq in iter_bits(self.balls[1]):
            symbols[sq] = 'O'

        lines = []
        for row in range(ROWS - 1, -1, -1):
            rank = f"{row + 1} |"
            for col in range(COLS):
                sq = row * COLS + col
                rank += " " + symbols.get(sq, ".")
            lines.append(rank)

        lines.append("   +" + "-" * (COLS * 2))
        lines.append("    " + " ".join("abcdefg"))
        lines.append(f"\nPlayer {self.current_player + 1} to move (ply {self.ply})")

        return "\n".join(lines)
