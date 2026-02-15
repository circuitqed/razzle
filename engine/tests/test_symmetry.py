"""
Tests for board symmetry transformations.
"""

import numpy as np
import pytest

from razzle.core.symmetry import (
    ROWS, COLS, TOTAL_SQUARES, NUM_ACTIONS, END_TURN_INDEX,
    rotate_square_180, flip_square_horizontal,
    rotate_move_180, flip_move_horizontal,
    rotate_bitboard_180, flip_bitboard_horizontal,
    rotate_policy_180, flip_policy_horizontal,
    rotate_tensor_180, flip_tensor_horizontal,
    MOVE_ROTATION_MAP, MOVE_FLIP_MAP,
)
from razzle.core.state import GameState
from razzle.core.bitboard import P1_START_PIECES, P2_START_PIECES, P1_START_BALL, P2_START_BALL


class TestSquareTransformations:
    """Tests for square index transformations."""

    def test_rotate_square_180_is_self_inverse(self):
        """180° rotation applied twice returns original."""
        for sq in range(TOTAL_SQUARES):
            rotated = rotate_square_180(sq)
            back = rotate_square_180(rotated)
            assert back == sq, f"Square {sq}: rotated={rotated}, back={back}"

    def test_flip_square_horizontal_is_self_inverse(self):
        """Horizontal flip applied twice returns original."""
        for sq in range(TOTAL_SQUARES):
            flipped = flip_square_horizontal(sq)
            back = flip_square_horizontal(flipped)
            assert back == sq, f"Square {sq}: flipped={flipped}, back={back}"

    def test_rotate_square_180_corners(self):
        """Test 180° rotation on corner squares."""
        # a1 (0) <-> g8 (55)
        assert rotate_square_180(0) == 55
        assert rotate_square_180(55) == 0
        # g1 (6) <-> a8 (49)
        assert rotate_square_180(6) == 49
        assert rotate_square_180(49) == 6

    def test_flip_square_horizontal_corners(self):
        """Test horizontal flip on corner squares."""
        # a1 (0) <-> g1 (6)
        assert flip_square_horizontal(0) == 6
        assert flip_square_horizontal(6) == 0
        # a8 (49) <-> g8 (55)
        assert flip_square_horizontal(49) == 55
        assert flip_square_horizontal(55) == 49

    def test_rotate_square_180_center(self):
        """Center column should stay in center after rotation."""
        # d4 is at (row=3, col=3) -> sq = 3*7+3 = 24
        # After 180°: (7-3, 6-3) = (4, 3) -> sq = 4*7+3 = 31
        assert rotate_square_180(24) == 31
        # d5 is at (row=4, col=3) -> sq = 31
        # After 180°: (7-4, 6-3) = (3, 3) -> sq = 24
        assert rotate_square_180(31) == 24


class TestMoveTransformations:
    """Tests for move index transformations."""

    def test_rotate_move_180_is_self_inverse(self):
        """180° rotation applied twice returns original."""
        for move in range(NUM_ACTIONS):
            rotated = rotate_move_180(move)
            back = rotate_move_180(rotated)
            assert back == move, f"Move {move}: rotated={rotated}, back={back}"

    def test_flip_move_horizontal_is_self_inverse(self):
        """Horizontal flip applied twice returns original."""
        for move in range(NUM_ACTIONS):
            flipped = flip_move_horizontal(move)
            back = flip_move_horizontal(flipped)
            assert back == move, f"Move {move}: flipped={flipped}, back={back}"

    def test_end_turn_unchanged_by_rotation(self):
        """END_TURN action should not change with rotation."""
        assert rotate_move_180(END_TURN_INDEX) == END_TURN_INDEX

    def test_end_turn_unchanged_by_flip(self):
        """END_TURN action should not change with flip."""
        assert flip_move_horizontal(END_TURN_INDEX) == END_TURN_INDEX

    def test_move_rotation_map_is_self_inverse(self):
        """The precomputed rotation map should be its own inverse."""
        for move in range(NUM_ACTIONS):
            rotated = MOVE_ROTATION_MAP[move]
            back = MOVE_ROTATION_MAP[rotated]
            assert back == move


class TestBitboardTransformations:
    """Tests for bitboard transformations."""

    def test_rotate_bitboard_180_is_self_inverse(self):
        """180° rotation applied twice returns original."""
        test_bbs = [0, 1, P1_START_PIECES, P2_START_PIECES, 0x1234567890ABCD]
        for bb in test_bbs:
            rotated = rotate_bitboard_180(bb)
            back = rotate_bitboard_180(rotated)
            assert back == bb, f"Bitboard {hex(bb)}: rotated={hex(rotated)}, back={hex(back)}"

    def test_rotate_p1_to_p2_position(self):
        """Player 1 starting pieces should rotate to player 2 position."""
        # P1 starts on row 1 (b1-f1)
        # After 180° rotation, should be on row 8 (b8-f8)
        rotated = rotate_bitboard_180(P1_START_PIECES)
        assert rotated == P2_START_PIECES

    def test_rotate_p1_ball_to_p2_ball(self):
        """Player 1 ball should rotate to player 2 ball position."""
        rotated = rotate_bitboard_180(P1_START_BALL)
        assert rotated == P2_START_BALL


class TestPolicyTransformations:
    """Tests for policy array transformations."""

    def test_rotate_policy_180_is_self_inverse(self):
        """180° rotation applied twice returns original."""
        policy = np.random.rand(NUM_ACTIONS).astype(np.float32)
        policy /= policy.sum()  # Normalize
        rotated = rotate_policy_180(policy)
        back = rotate_policy_180(rotated)
        np.testing.assert_array_almost_equal(back, policy)

    def test_flip_policy_horizontal_is_self_inverse(self):
        """Horizontal flip applied twice returns original."""
        policy = np.random.rand(NUM_ACTIONS).astype(np.float32)
        policy /= policy.sum()
        flipped = flip_policy_horizontal(policy)
        back = flip_policy_horizontal(flipped)
        np.testing.assert_array_almost_equal(back, policy)

    def test_policy_rotation_preserves_end_turn(self):
        """END_TURN probability should be unchanged by rotation."""
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[END_TURN_INDEX] = 1.0
        rotated = rotate_policy_180(policy)
        assert rotated[END_TURN_INDEX] == 1.0
        assert rotated.sum() == 1.0

    def test_policy_rotation_batch(self):
        """Policy rotation should work with batched inputs."""
        batch_size = 4
        policies = np.random.rand(batch_size, NUM_ACTIONS).astype(np.float32)
        policies /= policies.sum(axis=1, keepdims=True)

        rotated = rotate_policy_180(policies)
        back = rotate_policy_180(rotated)
        np.testing.assert_array_almost_equal(back, policies)


class TestTensorTransformations:
    """Tests for board tensor transformations."""

    def test_rotate_tensor_180_is_self_inverse(self):
        """180° rotation applied twice returns original."""
        tensor = np.random.rand(7, ROWS, COLS).astype(np.float32)
        rotated = rotate_tensor_180(tensor)
        back = rotate_tensor_180(rotated)
        np.testing.assert_array_almost_equal(back, tensor)

    def test_rotate_tensor_180_batch(self):
        """Tensor rotation should work with batched inputs."""
        batch_size = 4
        tensors = np.random.rand(batch_size, 7, ROWS, COLS).astype(np.float32)
        rotated = rotate_tensor_180(tensors)
        back = rotate_tensor_180(rotated)
        np.testing.assert_array_almost_equal(back, tensors)

    def test_rotate_tensor_shape(self):
        """Rotation should preserve tensor shape."""
        tensor = np.random.rand(7, ROWS, COLS).astype(np.float32)
        rotated = rotate_tensor_180(tensor)
        assert rotated.shape == tensor.shape


class TestGameStateIntegration:
    """Integration tests with GameState."""

    def test_to_tensor_shape(self):
        """Test that to_tensor returns correct shape."""
        state = GameState.new_game()
        tensor = state.to_tensor()
        assert tensor.shape == (7, ROWS, COLS)

    def test_starting_position_symmetry(self):
        """
        Both players should see their pieces at similar positions in the tensor.
        P0 sees their pieces at the bottom (low row indices).
        P1 (after rotation) should also see their pieces at the bottom.
        """
        state = GameState.new_game()
        tensor = state.to_tensor()

        # P0's pieces should be in low rows (plane 0)
        p0_pieces_rows = np.where(tensor[0].sum(axis=1) > 0)[0]
        assert p0_pieces_rows.max() <= 1, "P0 pieces should be in rows 0-1"

        # P0's goal (opponent's start) should be in high rows
        p0_opp_rows = np.where(tensor[2].sum(axis=1) > 0)[0]
        assert p0_opp_rows.min() >= 6, "P0 opponent should be in rows 6-7"

    def test_p1_sees_normalized_view(self):
        """
        When current_player=1 (P2), they should see their pieces at the bottom
        (after 180° rotation), same as current_player=0 (P1).
        """
        from razzle.core.moves import get_legal_moves
        from razzle.core.bitboard import P1_START_BALL

        state = GameState.new_game()
        # current_player=0 means P1, current_player=1 means P2 (0-indexed)
        assert state.current_player == 0

        # Find the ball square (where the ball is)
        ball_sq = None
        for sq in range(TOTAL_SQUARES):
            if P1_START_BALL & (1 << sq):
                ball_sq = sq
                break

        # Make a KNIGHT move (not ball pass) to switch to P2
        # Knight moves have src != ball_sq
        moves = get_legal_moves(state)
        knight_moves = [m for m in moves if m >= 0 and (m // TOTAL_SQUARES) != ball_sq]
        assert len(knight_moves) > 0, "Should have knight moves"
        state.apply_move(knight_moves[0])

        # Now it should be P2's turn (current_player=1)
        assert state.current_player == 1, f"Expected P2's turn, got current_player={state.current_player}"
        tensor = state.to_tensor()

        # After rotation, P2's "my pieces" (plane 0) should be toward the bottom
        # P2 pieces started at rows 6-7, after 180° rotation should be at rows 0-1
        my_pieces_rows = np.where(tensor[0].sum(axis=1) > 0)[0]
        assert len(my_pieces_rows) > 0, "P2 should have pieces on board"
        # After rotation, P2's pieces (originally at top) should now be at bottom
        assert my_pieces_rows.max() <= 2, f"P2 pieces should be rotated to bottom, got rows {my_pieces_rows}"

    def test_plane_5_always_one(self):
        """Plane 5 should always be 1.0 after normalization."""
        from razzle.core.moves import get_legal_moves

        # Test for P1 (current_player=0)
        state = GameState.new_game()
        tensor = state.to_tensor()
        assert tensor[5].sum() == ROWS * COLS, "Plane 5 should be all 1s for P1"

        # Test for P2 (current_player=1)
        moves = get_legal_moves(state)
        knight_moves = [m for m in moves if m >= 0]
        state.apply_move(knight_moves[0])
        tensor = state.to_tensor()
        assert tensor[5].sum() == ROWS * COLS, "Plane 5 should be all 1s for P2 too"
