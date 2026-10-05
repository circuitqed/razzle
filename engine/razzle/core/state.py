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
from .symmetry import rotate_tensor_180

# Rule variant: threefold repetition of a turn-start position is a draw. Off by default
# (set REPETITION_DRAW = True before creating games). A turn with its pass chain is one
# move, so only positions at the start of a turn are counted; the position is the full
# state (pieces, balls, ineligibility, side to move, last knight destination).
REPETITION_DRAW = False


@dataclass(frozen=False)
class GameState:
    """
    Represents the complete state of a Razzle Dazzle game.

    Attributes:
        pieces: Tuple of (p1_pieces, p2_pieces) bitboards
        balls: Tuple of (p1_ball, p2_ball) bitboards (single bit each)
        current_player: 0 for player 1, 1 for player 2
        touched_mask: Bitboard of pieces that are INELIGIBLE to receive passes.
                      A piece becomes ineligible when it passes or receives the ball.
                      Ineligibility persists across turns until the piece moves.
        has_passed: Whether a pass has been made this turn. If True, only more
                    passes or end_turn are allowed (no knight moves).
        last_knight_dst: Destination square of opponent's last knight move, or -1.
                         Used to check forced pass rule (must pass if opponent
                         just moved adjacent to your ball).
        ply: Number of half-moves played
        history: Stack of (move, captured_state) for undo
    """
    pieces: tuple[int, int] = (P1_START_PIECES, P2_START_PIECES)
    balls: tuple[int, int] = (P1_START_BALL, P2_START_BALL)
    current_player: int = 0
    touched_mask: int = field(default_factory=lambda: P1_START_PIECES | P2_START_PIECES)
    has_passed: bool = False
    last_knight_dst: int = -1
    ply: int = 0
    history: list = field(default_factory=list)
    # Turn-start position -> occurrences, when the repetition rule is on (else None)
    position_counts: Optional[dict] = None

    @classmethod
    def new_game(cls) -> GameState:
        """Create a new game in the starting position."""
        s = cls()
        if REPETITION_DRAW:
            s.position_counts = {s.position_key(): 1}
        return s

    def position_key(self) -> tuple:
        """The repetition rule's notion of a position (see REPETITION_DRAW)."""
        return (self.pieces, self.balls, self.current_player, self.touched_mask, self.last_knight_dst)

    def is_repetition_draw(self) -> bool:
        return (self.position_counts is not None and not self.has_passed
                and self.position_counts.get(self.position_key(), 0) >= 3)

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
        # Move limit: current player loses if exceeded
        if self.ply > 200:
            return True
        if self.is_repetition_draw():
            return True
        return False

    def get_winner(self) -> Optional[int]:
        """Return winner (0 or 1) or None if no winner yet."""
        if self.balls[0] & ROW_8_MASK:
            return 0
        if self.balls[1] & ROW_1_MASK:
            return 1
        # Move limit exceeded: current player loses (couldn't win in time)
        if self.ply > 200:
            return 1 - self.current_player  # Opponent wins
        return None

    def get_result(self, player: int) -> float:
        """Get game result from player's perspective: 1.0=win, 0.0=loss."""
        winner = self.get_winner()
        if winner is None:
            return 0.5  # Game still ongoing
        return 1.0 if winner == player else 0.0

    def copy(self) -> GameState:
        """Create a shallow copy (history is shared, but that's fine for MCTS)."""
        return GameState(
            pieces=self.pieces,
            balls=self.balls,
            current_player=self.current_player,
            touched_mask=self.touched_mask,
            has_passed=self.has_passed,
            last_knight_dst=self.last_knight_dst,
            ply=self.ply,
            history=[],  # Fresh history for copy
            position_counts=dict(self.position_counts) if self.position_counts is not None else None,
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
                self.touched_mask,
                self.has_passed,
                self.last_knight_dst,
                self.ply
            ))
            # Switch player - DO NOT reset touched_mask (ineligibility persists!)
            self.current_player = 1 - self.current_player
            self.has_passed = False
            self.last_knight_dst = -1  # No knight move this turn (was a pass)
            self.ply += 1
            self._count_position(+1)
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
            self.touched_mask,
            self.has_passed,
            self.last_knight_dst,
            self.ply
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

            # Mark that we've passed this turn (no knight moves allowed after)
            self.has_passed = True

        else:
            # Knight move: piece moves from src to dst
            new_pieces = list(self.pieces)
            new_pieces[p] = (new_pieces[p] & ~src_bit) | dst_bit
            self.pieces = tuple(new_pieces)

            # Clear ineligibility for the piece that moved (it can receive again)
            self.touched_mask &= ~src_bit

            # Knight move ends the turn - record destination for forced pass check
            self.last_knight_dst = dst
            self.current_player = 1 - p
            self.has_passed = False
            self.ply += 1

        self._count_position(+1)      # no-op mid-pass (has_passed) or with the rule off

    def _count_position(self, delta: int) -> None:
        if self.position_counts is None or self.has_passed:
            return
        k = self.position_key()
        n = self.position_counts.get(k, 0) + delta
        if n > 0:
            self.position_counts[k] = n
        else:
            self.position_counts.pop(k, None)

    def undo_move(self) -> None:
        """Undo the last move."""
        if not self.history:
            raise ValueError("No moves to undo")

        self._count_position(-1)
        entry = self.history.pop()
        (_, self.pieces, self.balls, self.current_player, self.touched_mask,
         self.has_passed, self.last_knight_dst, self.ply) = entry

    def to_tensor(self, num_planes: int = 7) -> np.ndarray:
        """
        Convert state to neural network input tensor.

        The board is always presented from the current player's perspective:
        - Current player's pieces appear at the "bottom" (low row indices)
        - Goal is at the "top" (high row indices)

        For player 1, the board is rotated 180° so both players see the same
        spatial pattern - "advance toward the far rank".

        Returns (7, 8, 7) float32 array:
          - Plane 0: Current player's pieces
          - Plane 1: Current player's ball
          - Plane 2: Opponent's pieces
          - Plane 3: Opponent's ball
          - Plane 4: Touched mask (pieces that can't receive passes)
          - Plane 5: Always 1 (reserved for compatibility, player perspective is normalized)
          - Plane 6: Has passed indicator (all 1s if has_passed=True, all 0s otherwise)

        With num_planes=9 (v2 networks), two more planes expose the forced-pass
        rule, which depends on history the board alone doesn't show:
          - Plane 7: Opponent's last knight move destination (one-hot; empty if none)
          - Plane 8: Forced pass (all 1s if the side to move must pass now)
        """
        if num_planes not in (7, 9):
            raise ValueError(f"num_planes must be 7 or 9, got {num_planes}")
        planes = np.zeros((num_planes, ROWS, COLS), dtype=np.float32)

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

        # Plane 5: Always 1 since perspective is now normalized
        planes[5, :, :] = 1.0

        if self.has_passed:
            planes[6, :, :] = 1.0

        if num_planes == 9:
            if self.last_knight_dst >= 0:
                planes[7, self.last_knight_dst // COLS, self.last_knight_dst % COLS] = 1.0
            if self.is_forced_pass():
                planes[8, :, :] = 1.0

        # Rotate 180° for player 1 so both players see "advance toward far rank"
        if p == 1:
            planes = rotate_tensor_180(planes)

        return planes

    def is_forced_pass(self) -> bool:
        """True if the side to move must pass now (same rule as move generation)."""
        from .moves import MoveGenerator
        if self.has_passed or not MoveGenerator.must_pass(self):
            return False
        return next(iter(MoveGenerator.get_pass_moves(self)), None) is not None

    def __hash__(self) -> int:
        """Hash for transposition/eval caches.

        Includes last_knight_dst: it decides the forced-pass rule (so the legal
        moves) and is an input plane of v2 networks, so positions that differ
        only in it must not share a cached evaluation.
        """
        return hash((tuple(self.pieces), tuple(self.balls), self.current_player, self.touched_mask,
                     self.has_passed, self.last_knight_dst))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return (
            self.pieces == other.pieces and
            self.balls == other.balls and
            self.current_player == other.current_player and
            self.touched_mask == other.touched_mask and
            self.has_passed == other.has_passed and
            self.last_knight_dst == other.last_knight_dst
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
