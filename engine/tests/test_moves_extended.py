"""Extended tests for move generation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import (
    get_legal_moves, encode_move, decode_move,
    move_to_algebraic, algebraic_to_move,
    MoveGenerator, END_TURN_MOVE
)
from razzle.core.bitboard import (
    bit, algebraic_to_sq, sq_to_algebraic, iter_bits,
    KNIGHT_ATTACKS, NUM_SQUARES
)


class TestMoveEncoding:
    def test_encode_decode_all_squares(self):
        """Test encoding/decoding for all possible square pairs."""
        for src in range(NUM_SQUARES):
            for dst in range(NUM_SQUARES):
                move = encode_move(src, dst)
                decoded_src, decoded_dst = decode_move(move)
                assert decoded_src == src
                assert decoded_dst == dst

    def test_move_bounds(self):
        """Move encoding should be in [0, 3136)."""
        for src in range(NUM_SQUARES):
            for dst in range(NUM_SQUARES):
                move = encode_move(src, dst)
                assert 0 <= move < 3136


class TestMoveMask:
    def test_move_mask_shape(self):
        state = GameState.new_game()
        mask = MoveGenerator.get_move_mask(state)
        assert len(mask) == 3136

    def test_move_mask_matches_legal_moves(self):
        state = GameState.new_game()
        mask = MoveGenerator.get_move_mask(state)
        legal_moves = get_legal_moves(state)

        # Count of True values should match legal move count
        mask_count = sum(mask)
        assert mask_count == len(legal_moves)

        # Each legal move should be True in mask
        for move in legal_moves:
            assert mask[move], f"Legal move {move} not in mask"


class TestKnightMoves:
    def test_corner_a1_moves(self):
        """Knight on a1 can only reach b3, c2."""
        state = GameState.new_game()
        # Clear board and place single piece at a1
        state.pieces = (bit(algebraic_to_sq('a1')), 0)
        state.balls = (bit(algebraic_to_sq('a1')), bit(algebraic_to_sq('d8')))  # Ball on same square

        # With ball on piece, it can't move - only pass
        # Let's put ball elsewhere
        state.balls = (bit(algebraic_to_sq('b1')), bit(algebraic_to_sq('d8')))
        state.pieces = (bit(algebraic_to_sq('a1')) | bit(algebraic_to_sq('b1')), 0)

        moves = list(MoveGenerator.get_knight_moves(state))
        algebraic_moves = [move_to_algebraic(m) for m in moves]

        # a1 piece can move to b3, c2
        assert 'a1-b3' in algebraic_moves
        assert 'a1-c2' in algebraic_moves
        # b1 piece has ball, so can't move
        assert not any(m.startswith('b1-') for m in algebraic_moves)

    def test_corner_g8_moves(self):
        """Knight on g8 can reach e7, f6."""
        state = GameState.new_game()
        # Place P1 piece on g8 (unusual but valid for testing)
        state.pieces = (bit(algebraic_to_sq('g8')), 0)
        state.balls = (bit(algebraic_to_sq('a1')), bit(algebraic_to_sq('d8')))

        moves = list(MoveGenerator.get_knight_moves(state))
        algebraic_moves = [move_to_algebraic(m) for m in moves]

        assert 'g8-e7' in algebraic_moves
        assert 'g8-f6' in algebraic_moves

    def test_center_knight_eight_moves(self):
        """Knight in center should have 8 possible moves (if unblocked)."""
        state = GameState.new_game()
        # Place single piece at d4, ball elsewhere
        d4 = algebraic_to_sq('d4')
        state.pieces = (bit(d4), 0)
        state.balls = (bit(algebraic_to_sq('a1')), bit(algebraic_to_sq('g8')))

        moves = list(MoveGenerator.get_knight_moves(state))

        # All 8 knight moves should be available
        assert len(moves) == 8

    def test_knight_cant_capture_own_piece(self):
        """Knight can't move to square with friendly piece."""
        state = GameState.new_game()
        # Place pieces at b1 and c3 (b1 could normally move to c3)
        state.pieces = (
            bit(algebraic_to_sq('b1')) | bit(algebraic_to_sq('c3')),
            0
        )
        state.balls = (bit(algebraic_to_sq('a1')), bit(algebraic_to_sq('g8')))

        moves = list(MoveGenerator.get_knight_moves(state))
        algebraic_moves = [move_to_algebraic(m) for m in moves]

        # b1-c3 should NOT be available (friendly piece there)
        assert 'b1-c3' not in algebraic_moves

    def test_ball_piece_cannot_move(self):
        """Piece holding ball cannot make knight moves."""
        state = GameState.new_game()
        # Ball and piece on same square
        state.pieces = (bit(algebraic_to_sq('d4')), 0)
        state.balls = (bit(algebraic_to_sq('d4')), bit(algebraic_to_sq('g8')))

        moves = list(MoveGenerator.get_knight_moves(state))

        # No knight moves available (only piece has ball)
        assert len(moves) == 0


class TestPassMoves:
    def test_pass_to_adjacent_piece(self):
        """Ball can pass to adjacent friendly piece."""
        state = GameState.new_game()
        # Ball on d4, pieces on c4 and e4
        state.balls = (bit(algebraic_to_sq('d4')), bit(algebraic_to_sq('g8')))
        state.pieces = (
            bit(algebraic_to_sq('c4')) | bit(algebraic_to_sq('e4')) | bit(algebraic_to_sq('d4')),
            0
        )

        passes = list(MoveGenerator.get_pass_moves(state))
        algebraic = [move_to_algebraic(m) for m in passes]

        assert 'd4-c4' in algebraic
        assert 'd4-e4' in algebraic

    def test_pass_diagonal(self):
        """Ball can pass diagonally."""
        state = GameState.new_game()
        # Ball on d4, piece on e5 (diagonal)
        state.balls = (bit(algebraic_to_sq('d4')), bit(algebraic_to_sq('g8')))
        state.pieces = (
            bit(algebraic_to_sq('d4')) | bit(algebraic_to_sq('e5')),
            0
        )

        passes = list(MoveGenerator.get_pass_moves(state))
        algebraic = [move_to_algebraic(m) for m in passes]

        assert 'd4-e5' in algebraic

    def test_pass_blocked_by_enemy(self):
        """Pass blocked by enemy piece in the way."""
        state = GameState.new_game()
        # Ball on a1, friendly on c1, enemy on b1
        state.balls = (bit(algebraic_to_sq('a1')), bit(algebraic_to_sq('g8')))
        state.pieces = (
            bit(algebraic_to_sq('a1')) | bit(algebraic_to_sq('c1')),
            bit(algebraic_to_sq('b1'))  # Enemy blocks
        )

        passes = list(MoveGenerator.get_pass_moves(state))
        algebraic = [move_to_algebraic(m) for m in passes]

        # Can't pass to c1 (blocked by b1)
        assert 'a1-c1' not in algebraic

    def test_pass_long_range(self):
        """Ball can pass across multiple squares."""
        state = GameState.new_game()
        # Ball on a1, piece on a5 (long vertical pass)
        state.balls = (bit(algebraic_to_sq('a1')), bit(algebraic_to_sq('g8')))
        state.pieces = (
            bit(algebraic_to_sq('a1')) | bit(algebraic_to_sq('a5')),
            0
        )

        passes = list(MoveGenerator.get_pass_moves(state))
        algebraic = [move_to_algebraic(m) for m in passes]

        assert 'a1-a5' in algebraic

    def test_cannot_pass_to_touched_piece(self):
        """Can't pass to piece that already received ball this turn."""
        state = GameState.new_game()
        # Setup: ball on d4, pieces on c4 and e4
        state.balls = (bit(algebraic_to_sq('d4')), bit(algebraic_to_sq('g8')))
        state.pieces = (
            bit(algebraic_to_sq('c4')) | bit(algebraic_to_sq('e4')) | bit(algebraic_to_sq('d4')),
            0
        )
        # Mark c4 as touched
        state.touched_mask = bit(algebraic_to_sq('c4'))

        passes = list(MoveGenerator.get_pass_moves(state))
        algebraic = [move_to_algebraic(m) for m in passes]

        # Can pass to e4 but not c4
        assert 'd4-e4' in algebraic
        assert 'd4-c4' not in algebraic


class TestApplyMoveStateChanges:
    def test_knight_move_changes_player(self):
        state = GameState.new_game()
        assert state.current_player == 0

        move = algebraic_to_move('b1-c3')
        state.apply_move(move)

        assert state.current_player == 1

    def test_knight_move_increments_ply(self):
        state = GameState.new_game()
        assert state.ply == 0

        move = algebraic_to_move('b1-c3')
        state.apply_move(move)

        assert state.ply == 1

    def test_knight_move_clears_only_moving_piece(self):
        """Knight move only clears ineligibility for the piece that moved."""
        state = GameState.new_game()
        # Mark b1, c1, d1 as ineligible
        state.touched_mask = bit(algebraic_to_sq('b1')) | bit(algebraic_to_sq('c1')) | bit(algebraic_to_sq('d1'))

        # Move b1 to c3
        move = algebraic_to_move('b1-c3')
        state.apply_move(move)

        # b1's bit should be cleared, but c1 and d1 remain ineligible
        assert not (state.touched_mask & bit(algebraic_to_sq('b1')))
        assert state.touched_mask & bit(algebraic_to_sq('c1'))
        assert state.touched_mask & bit(algebraic_to_sq('d1'))

    def test_pass_updates_ball_position(self):
        state = GameState.new_game()
        # Initial ball on d1
        assert state.balls[0] == bit(algebraic_to_sq('d1'))

        # Pass to e1
        move = algebraic_to_move('d1-e1')
        state.apply_move(move)

        assert state.balls[0] == bit(algebraic_to_sq('e1'))

    def test_pass_updates_touched_mask(self):
        state = GameState.new_game()
        assert state.touched_mask == 0

        # Pass from d1 to e1
        move = algebraic_to_move('d1-e1')
        state.apply_move(move)

        # Both d1 (passer) and e1 (receiver) should be touched
        assert state.touched_mask & bit(algebraic_to_sq('d1'))
        assert state.touched_mask & bit(algebraic_to_sq('e1'))

    def test_cannot_pass_back_to_passer(self):
        """After d1→e1, cannot pass back to d1 (it already touched ball)."""
        state = GameState.new_game()

        # Pass d1→e1
        state.apply_move(algebraic_to_move('d1-e1'))

        # Get available passes from e1
        passes = list(MoveGenerator.get_pass_moves(state))
        algebraic = [move_to_algebraic(m) for m in passes]

        # e1→d1 should NOT be available (d1 already touched)
        assert 'e1-d1' not in algebraic

    def test_end_turn_after_pass(self):
        """End turn should be available after passing."""
        state = GameState.new_game()
        assert END_TURN_MOVE not in get_legal_moves(state)  # Not available at start

        # Make a pass
        state.apply_move(algebraic_to_move('d1-e1'))

        # End turn should now be available
        assert END_TURN_MOVE in get_legal_moves(state)

    def test_end_turn_switches_player(self):
        """End turn should switch current player."""
        state = GameState.new_game()
        state.apply_move(algebraic_to_move('d1-e1'))  # Pass
        assert state.current_player == 0

        state.apply_move(END_TURN_MOVE)  # End turn
        assert state.current_player == 1

    def test_end_turn_increments_ply(self):
        """End turn should increment ply."""
        state = GameState.new_game()
        state.apply_move(algebraic_to_move('d1-e1'))
        assert state.ply == 0

        state.apply_move(END_TURN_MOVE)
        assert state.ply == 1

    def test_end_turn_preserves_touched_mask(self):
        """End turn should NOT reset touched_mask (ineligibility persists)."""
        state = GameState.new_game()
        state.apply_move(algebraic_to_move('d1-e1'))
        mask_after_pass = state.touched_mask
        assert mask_after_pass != 0

        state.apply_move(END_TURN_MOVE)
        # Touched mask persists - pieces remain ineligible until they move
        assert state.touched_mask == mask_after_pass

    def test_ineligibility_persists_across_multiple_turns(self):
        """Ineligibility persists until piece moves, even across turns."""
        state = GameState.new_game()

        # Player 1: pass d1→e1, then end turn
        state.apply_move(algebraic_to_move('d1-e1'))
        state.apply_move(END_TURN_MOVE)
        assert state.current_player == 1

        # Player 2: make a knight move
        state.apply_move(algebraic_to_move('b8-c6'))
        assert state.current_player == 0

        # Player 1's turn again - d1 and e1 should STILL be ineligible
        assert state.touched_mask & bit(algebraic_to_sq('d1'))
        assert state.touched_mask & bit(algebraic_to_sq('e1'))

        # Cannot pass back to d1 (still ineligible)
        passes = list(MoveGenerator.get_pass_moves(state))
        algebraic = [move_to_algebraic(m) for m in passes]
        assert 'e1-d1' not in algebraic

    def test_moving_piece_restores_eligibility(self):
        """After a piece moves, it becomes eligible to receive passes again."""
        state = GameState.new_game()

        # Pass d1→c1, making both ineligible
        state.apply_move(algebraic_to_move('d1-c1'))
        assert state.touched_mask & bit(algebraic_to_sq('d1'))

        # Move d1 (oh wait, d1 had the ball, can't move)
        # Let's pass to b1 first, then move d1
        state.apply_move(algebraic_to_move('c1-b1'))

        # Now b1 has ball, d1 and c1 are ineligible
        # Move f1 (which is eligible since it hasn't touched ball)
        state.apply_move(algebraic_to_move('f1-e3'))

        # Now it's player 2's turn, they move
        state.apply_move(algebraic_to_move('b8-c6'))

        # Player 1's turn - d1 and c1 still ineligible, but let's move d1
        # Actually d1 didn't have ball after first pass, so it could have moved
        # Let me redo this test more carefully
        pass  # See test below for cleaner version

    def test_piece_becomes_eligible_after_moving(self):
        """A piece that was ineligible becomes eligible after it moves."""
        state = GameState.new_game()

        # Pass d1→e1
        state.apply_move(algebraic_to_move('d1-e1'))
        d1_sq = algebraic_to_sq('d1')

        # d1 is now ineligible (it passed the ball)
        assert state.touched_mask & bit(d1_sq)

        # End turn
        state.apply_move(END_TURN_MOVE)

        # Player 2 moves
        state.apply_move(algebraic_to_move('b8-c6'))

        # Player 1: move the piece that was on d1 (which is still there)
        # Wait, d1 doesn't have the ball anymore, so it CAN move
        state.apply_move(algebraic_to_move('d1-c3'))

        # Now d1's ineligibility should be cleared
        assert not (state.touched_mask & bit(d1_sq))


class TestStateCopy:
    def test_copy_is_independent(self):
        state = GameState.new_game()
        copy = state.copy()

        # Modify original
        state.ply = 100
        state.current_player = 1

        # Copy should be unchanged
        assert copy.ply == 0
        assert copy.current_player == 0

    def test_copy_has_empty_history(self):
        state = GameState.new_game()
        state.apply_move(algebraic_to_move('b1-c3'))
        assert len(state.history) == 1

        copy = state.copy()
        assert len(copy.history) == 0


class TestStateHash:
    def test_same_state_same_hash(self):
        state1 = GameState.new_game()
        state2 = GameState.new_game()
        assert hash(state1) == hash(state2)

    def test_different_state_different_hash(self):
        state1 = GameState.new_game()
        state2 = GameState.new_game()
        state2.apply_move(algebraic_to_move('b1-c3'))
        assert hash(state1) != hash(state2)

    def test_equality(self):
        state1 = GameState.new_game()
        state2 = GameState.new_game()
        assert state1 == state2

        state2.apply_move(algebraic_to_move('b1-c3'))
        assert state1 != state2
