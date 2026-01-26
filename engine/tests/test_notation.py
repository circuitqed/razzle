"""Tests for game notation (PGN-like format)."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import algebraic_to_move, END_TURN_MOVE
from razzle.core.notation import (
    GameRecord, game_to_pgn, pgn_to_game
)


class TestGameRecord:
    def test_empty_game(self):
        """Empty game produces valid PGN."""
        state = GameState.new_game()
        pgn = game_to_pgn(state)

        assert '[Event' in pgn
        assert '[White' in pgn
        assert '[Result "*"]' in pgn

    def test_simple_moves(self):
        """Simple game with moves."""
        state = GameState.new_game()
        state.apply_move(algebraic_to_move('b1-c3'))  # P1 knight
        state.apply_move(algebraic_to_move('b8-c6'))  # P2 knight

        pgn = game_to_pgn(state)

        assert 'b1-c3' in pgn
        assert 'b8-c6' in pgn

    def test_pass_and_end(self):
        """Game with passes and end turns - end is implicit in new format."""
        state = GameState.new_game()
        state.apply_move(algebraic_to_move('d1-c1'))  # P1 pass
        state.apply_move(END_TURN_MOVE)  # P1 end

        pgn = game_to_pgn(state)

        assert 'd1-c1' in pgn
        # 'end' is no longer shown in new format - it's implicit
        assert 'end' not in pgn

    def test_round_trip(self):
        """Parse PGN and replay produces same state."""
        state = GameState.new_game()
        state.apply_move(algebraic_to_move('d1-c1'))
        state.apply_move(END_TURN_MOVE)
        state.apply_move(algebraic_to_move('d8-e8'))
        state.apply_move(END_TURN_MOVE)
        state.apply_move(algebraic_to_move('b1-c3'))

        pgn = game_to_pgn(state)
        state2 = pgn_to_game(pgn)

        assert state2.ply == state.ply
        assert state2.current_player == state.current_player
        assert state2.pieces == state.pieces
        assert state2.balls == state.balls

    def test_pass_chain_notation(self):
        """Pass chains are shown as continuous notation like c1-d1-e1."""
        state = GameState.new_game()
        # P1: pass d1->c1, pass c1->b1, end turn
        state.apply_move(algebraic_to_move('d1-c1'))
        state.apply_move(algebraic_to_move('c1-b1'))
        state.apply_move(END_TURN_MOVE)

        pgn = game_to_pgn(state)

        # Should show as chain notation, not separate moves
        assert 'd1-c1-b1' in pgn
        assert 'end' not in pgn

    def test_pass_chain_round_trip(self):
        """Pass chain notation parses correctly."""
        state = GameState.new_game()
        # P1: pass d1->c1, pass c1->b1, end turn
        state.apply_move(algebraic_to_move('d1-c1'))
        state.apply_move(algebraic_to_move('c1-b1'))
        state.apply_move(END_TURN_MOVE)
        # P2: knight move
        state.apply_move(algebraic_to_move('b8-c6'))

        pgn = game_to_pgn(state)
        state2 = pgn_to_game(pgn)

        assert state2.ply == state.ply
        assert state2.current_player == state.current_player
        assert state2.pieces == state.pieces
        assert state2.balls == state.balls

    def test_metadata_preserved(self):
        """Metadata is preserved in round-trip."""
        state = GameState.new_game()
        pgn = game_to_pgn(
            state,
            event="Test Tournament",
            white="Alice",
            black="Bob"
        )

        record = GameRecord.from_pgn(pgn)

        assert record.event == "Test Tournament"
        assert record.white == "Alice"
        assert record.black == "Bob"

    def test_result_p1_wins(self):
        """Result shows P1 win correctly."""
        record = GameRecord()
        record.result = "1-0"

        pgn = record.to_pgn()

        assert '[Result "1-0"]' in pgn
        assert pgn.strip().endswith("1-0")

    def test_result_p2_wins(self):
        """Result shows P2 win correctly."""
        record = GameRecord()
        record.result = "0-1"

        pgn = record.to_pgn()

        assert '[Result "0-1"]' in pgn

    def test_parse_pgn_with_extra_whitespace(self):
        """Parser handles extra whitespace."""
        pgn = """
        [Event "Test"]
        [White "A"]
        [Black "B"]
        [Result "*"]

        1.  b1-c3    b8-c6
        2.  c1-d3
        """

        record = GameRecord.from_pgn(pgn)

        assert len(record.moves) == 3
        assert record.event == "Test"


class TestNotationConvenience:
    def test_game_to_pgn_function(self):
        """game_to_pgn convenience function works."""
        state = GameState.new_game()
        pgn = game_to_pgn(state, event="Quick Game")

        assert '[Event "Quick Game"]' in pgn

    def test_pgn_to_game_function(self):
        """pgn_to_game convenience function works."""
        pgn = """
        [Result "*"]

        1. b1-c3 b8-c6
        """

        state = pgn_to_game(pgn)

        assert state.ply == 2
        assert state.current_player == 0
