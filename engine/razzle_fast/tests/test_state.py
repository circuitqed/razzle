"""Tests verifying C state matches Python engine."""

import random
import numpy as np
import pytest

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle_fast.wrapper import FastState


def _random_game(max_moves=100, seed=None):
    """Play a random game, yielding (state, legal_moves) at each step."""
    rng = random.Random(seed)
    gs = GameState.new_game()
    for _ in range(max_moves):
        if gs.is_terminal():
            break
        moves = get_legal_moves(gs)
        if not moves:
            break
        yield gs.copy(), moves
        move = rng.choice(moves)
        gs.apply_move(move)
    yield gs.copy(), get_legal_moves(gs)


class TestStateInit:
    def test_initial_state_matches(self):
        gs = GameState.new_game()
        fs = FastState()
        assert fs.pieces == gs.pieces
        assert fs.balls == gs.balls
        assert fs.current_player == gs.current_player
        assert fs.has_passed == gs.has_passed
        assert fs.touched_mask == gs.touched_mask
        assert fs.last_knight_dst == gs.last_knight_dst
        assert fs.ply == gs.ply


class TestFromGameState:
    def test_roundtrip(self):
        gs = GameState.new_game()
        fs = FastState.from_game_state(gs)
        gs2 = fs.to_game_state()
        assert gs == gs2

    def test_roundtrip_mid_game(self):
        gs = GameState.new_game()
        moves = get_legal_moves(gs)
        gs.apply_move(moves[0])
        fs = FastState.from_game_state(gs)
        gs2 = fs.to_game_state()
        assert gs.pieces == gs2.pieces
        assert gs.balls == gs2.balls
        assert gs.current_player == gs2.current_player
        assert gs.touched_mask == gs2.touched_mask
        assert gs.has_passed == gs2.has_passed


class TestApplyMove:
    @pytest.mark.parametrize("seed", range(50))
    def test_apply_move_matches(self, seed):
        """Apply each legal move in a random game position and verify states match."""
        rng = random.Random(seed)
        gs = GameState.new_game()
        # Play some random moves to get to an interesting position
        for _ in range(rng.randint(0, 30)):
            if gs.is_terminal():
                break
            moves = get_legal_moves(gs)
            if not moves:
                break
            gs.apply_move(rng.choice(moves))

        if gs.is_terminal():
            return

        moves = get_legal_moves(gs)
        for move in moves:
            gs_copy = gs.copy()
            gs_copy.apply_move(move)

            fs = FastState.from_game_state(gs)
            fs.apply_move(move)

            assert fs.pieces == gs_copy.pieces, f"pieces mismatch on move {move}"
            assert fs.balls == gs_copy.balls, f"balls mismatch on move {move}"
            assert fs.current_player == gs_copy.current_player, f"player mismatch on move {move}"
            assert fs.touched_mask == gs_copy.touched_mask, f"touched_mask mismatch on move {move}"
            assert fs.has_passed == gs_copy.has_passed, f"has_passed mismatch on move {move}"
            assert fs.last_knight_dst == gs_copy.last_knight_dst, f"last_knight_dst mismatch on move {move}"
            assert fs.ply == gs_copy.ply, f"ply mismatch on move {move}"


class TestLegalMoves:
    @pytest.mark.parametrize("seed", range(100))
    def test_legal_moves_match(self, seed):
        """Verify legal moves match between Python and C at random positions."""
        rng = random.Random(seed)
        gs = GameState.new_game()
        for _ in range(rng.randint(0, 40)):
            if gs.is_terminal():
                break
            moves = get_legal_moves(gs)
            if not moves:
                break
            gs.apply_move(rng.choice(moves))

        if gs.is_terminal():
            return

        py_moves = sorted(get_legal_moves(gs))
        fs = FastState.from_game_state(gs)
        c_moves = sorted(fs.get_legal_moves())

        assert py_moves == c_moves, (
            f"Legal moves mismatch at seed={seed}, ply={gs.ply}, "
            f"player={gs.current_player}, passed={gs.has_passed}\n"
            f"Python: {py_moves}\nC: {c_moves}"
        )


class TestTerminal:
    def test_initial_not_terminal(self):
        fs = FastState()
        assert not fs.is_terminal()

    def test_winner_none_initially(self):
        fs = FastState()
        assert fs.get_winner() is None

    @pytest.mark.parametrize("seed", range(20))
    def test_terminal_matches(self, seed):
        """Play random games to completion and verify terminal detection."""
        rng = random.Random(seed)
        gs = GameState.new_game()
        fs = FastState()
        for _ in range(250):
            assert gs.is_terminal() == fs.is_terminal()
            if gs.is_terminal():
                assert gs.get_winner() == fs.get_winner()
                break
            moves = get_legal_moves(gs)
            if not moves:
                break
            move = rng.choice(moves)
            gs.apply_move(move)
            fs.apply_move(move)


class TestTensor:
    def test_initial_tensor_matches(self):
        gs = GameState.new_game()
        fs = FastState()
        py_tensor = gs.to_tensor()
        c_tensor = fs.to_tensor()
        np.testing.assert_array_equal(py_tensor, c_tensor)

    @pytest.mark.parametrize("seed", range(50))
    def test_tensor_matches_random(self, seed):
        """Verify tensor conversion matches at random positions."""
        rng = random.Random(seed)
        gs = GameState.new_game()
        for _ in range(rng.randint(0, 30)):
            if gs.is_terminal():
                break
            moves = get_legal_moves(gs)
            if not moves:
                break
            gs.apply_move(rng.choice(moves))

        if gs.is_terminal():
            return

        fs = FastState.from_game_state(gs)
        py_tensor = gs.to_tensor()
        c_tensor = fs.to_tensor()
        np.testing.assert_array_equal(
            py_tensor, c_tensor,
            err_msg=f"Tensor mismatch at seed={seed}, player={gs.current_player}"
        )


class TestCopy:
    def test_copy_independent(self):
        fs = FastState()
        fs2 = fs.copy()
        # Pick a knight move (changes ply/player), not a pass
        # Knight moves have src that is NOT the ball square
        ball_sq = None
        bb = fs.balls[fs.current_player]
        while bb:
            ball_sq = (bb & -bb).bit_length() - 1
            break
        moves = fs.get_legal_moves()
        knight_moves = [m for m in moves if m >= 0 and m // 56 != ball_sq]
        assert len(knight_moves) > 0, "No knight moves available"
        fs.apply_move(knight_moves[0])
        # Original changed, copy should not
        assert fs.ply != fs2.ply or fs.current_player != fs2.current_player


class TestEquals:
    def test_same_state_equal(self):
        fs1 = FastState()
        fs2 = FastState()
        assert fs1 == fs2

    def test_different_state_not_equal(self):
        fs1 = FastState()
        fs2 = FastState()
        moves = fs1.get_legal_moves()
        fs1.apply_move(moves[0])
        assert fs1 != fs2
