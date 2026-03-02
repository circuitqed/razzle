"""Full game replay tests - play random games move by move in both engines."""

import random
import pytest

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle_fast.wrapper import FastState


class TestGameReplay:
    @pytest.mark.parametrize("seed", range(100))
    def test_full_game_replay(self, seed):
        """Play a random game in both engines simultaneously, verify states match at every step."""
        rng = random.Random(seed)
        gs = GameState.new_game()
        fs = FastState()

        for step in range(200):
            # Check state equality at every step
            assert fs.pieces == gs.pieces, f"pieces mismatch at step {step}"
            assert fs.balls == gs.balls, f"balls mismatch at step {step}"
            assert fs.current_player == gs.current_player, f"player mismatch at step {step}"
            assert fs.touched_mask == gs.touched_mask, f"touched_mask mismatch at step {step}"
            assert fs.has_passed == gs.has_passed, f"has_passed mismatch at step {step}"
            assert fs.last_knight_dst == gs.last_knight_dst, f"last_knight_dst mismatch at step {step}"
            assert fs.ply == gs.ply, f"ply mismatch at step {step}"

            # Check terminal
            assert gs.is_terminal() == fs.is_terminal(), f"terminal mismatch at step {step}"
            if gs.is_terminal():
                assert gs.get_winner() == fs.get_winner(), f"winner mismatch at step {step}"
                break

            # Check legal moves
            py_moves = sorted(get_legal_moves(gs))
            c_moves = sorted(fs.get_legal_moves())
            assert py_moves == c_moves, (
                f"legal moves mismatch at step {step}, "
                f"player={gs.current_player}, passed={gs.has_passed}"
            )

            if not py_moves:
                break

            # Apply same random move to both
            move = rng.choice(py_moves)
            gs.apply_move(move)
            fs.apply_move(move)


class TestStressTest:
    def test_many_random_positions(self):
        """Generate 1000 random positions, verify legal moves at each."""
        rng = random.Random(42)
        mismatches = 0
        total = 0

        for game_seed in range(100):
            game_rng = random.Random(game_seed + 1000)
            gs = GameState.new_game()
            fs = FastState()

            for _ in range(50):
                if gs.is_terminal():
                    break

                total += 1
                py_moves = sorted(get_legal_moves(gs))
                c_moves = sorted(fs.get_legal_moves())
                if py_moves != c_moves:
                    mismatches += 1

                move = game_rng.choice(py_moves)
                gs.apply_move(move)
                fs.apply_move(move)

        assert mismatches == 0, f"{mismatches}/{total} positions had legal move mismatches"
