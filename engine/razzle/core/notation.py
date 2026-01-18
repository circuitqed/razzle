"""
Game notation for Razzle Dazzle (PGN-like format).

Format example:
```
[Event "Casual Game"]
[Date "2026.01.18"]
[White "Player 1"]
[Black "Player 2"]
[Result "1-0"]

1. d1-c1 d8-e8 2. c1-b1 end 3. b1-c3 e8-f8 4. end c6-e5 ...
```

Moves:
- Knight moves: "b1-c3" (piece moves from b1 to c3)
- Ball passes: "d1-e1" (ball passes from d1 to e1)
- End turn: "end" (end turn after passing)

Move numbers increment after both players have moved (like chess).
Multiple actions in a turn are space-separated.
"""

from __future__ import annotations
import re
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

from .state import GameState
from .moves import (
    get_legal_moves, move_to_algebraic, algebraic_to_move,
    END_TURN_MOVE, decode_move
)


@dataclass
class GameRecord:
    """Record of a complete or in-progress game."""

    # Metadata (PGN-style tags)
    event: str = "Razzle Dazzle Game"
    site: str = "?"
    date: str = field(default_factory=lambda: date.today().strftime("%Y.%m.%d"))
    round: str = "?"
    white: str = "Player 1"  # Player 1 (bottom)
    black: str = "Player 2"  # Player 2 (top)
    result: str = "*"  # "*" = ongoing, "1-0" = P1 wins, "0-1" = P2 wins, "1/2-1/2" = draw

    # Move history
    moves: list[int] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: GameState, **metadata) -> GameRecord:
        """Create a record from a game state with history."""
        record = cls(**metadata)

        # Extract moves from history
        for entry in state.history:
            move = entry[0]
            record.moves.append(move)

        # Set result if game is over
        if state.is_terminal():
            winner = state.get_winner()
            if winner == 0:
                record.result = "1-0"
            elif winner == 1:
                record.result = "0-1"
            else:
                record.result = "1/2-1/2"

        return record

    def to_pgn(self) -> str:
        """Export to PGN-like format."""
        lines = []

        # Tags
        lines.append(f'[Event "{self.event}"]')
        lines.append(f'[Site "{self.site}"]')
        lines.append(f'[Date "{self.date}"]')
        lines.append(f'[Round "{self.round}"]')
        lines.append(f'[White "{self.white}"]')
        lines.append(f'[Black "{self.black}"]')
        lines.append(f'[Result "{self.result}"]')
        lines.append('')

        # Moves
        move_text = self._format_moves()

        # Word wrap at 80 chars
        wrapped = []
        current_line = ""
        for word in move_text.split():
            if len(current_line) + len(word) + 1 > 80:
                wrapped.append(current_line)
                current_line = word
            else:
                current_line = f"{current_line} {word}".strip()
        if current_line:
            wrapped.append(current_line)

        lines.extend(wrapped)

        # Result at end
        if self.result != "*":
            lines.append(self.result)

        return '\n'.join(lines)

    def _format_moves(self) -> str:
        """Format moves into notation string."""
        parts = []
        move_num = 1
        ply = 0

        # Group moves by turn
        # Player 1 moves first, then Player 2, then move number increments
        current_turn_moves = []
        current_player = 0

        for move in self.moves:
            alg = move_to_algebraic(move)

            # Check if this move ends the turn
            is_turn_end = (move == END_TURN_MOVE or
                          (move >= 0 and not self._is_pass_move(move, ply)))

            current_turn_moves.append(alg)

            if is_turn_end:
                # Format this player's turn
                turn_str = ' '.join(current_turn_moves)

                if current_player == 0:
                    parts.append(f"{move_num}. {turn_str}")
                else:
                    parts.append(turn_str)
                    move_num += 1

                current_player = 1 - current_player
                current_turn_moves = []

            ply += 1

        # Handle incomplete turn at end
        if current_turn_moves:
            turn_str = ' '.join(current_turn_moves)
            if current_player == 0:
                parts.append(f"{move_num}. {turn_str}")
            else:
                parts.append(turn_str)

        return ' '.join(parts)

    def _is_pass_move(self, move: int, ply: int) -> bool:
        """Check if a move is a pass (vs knight move) by replaying."""
        # Replay game to this point to check
        state = GameState.new_game()
        for i, m in enumerate(self.moves[:ply]):
            state.apply_move(m)

        # Check if source is ball position
        src, _ = decode_move(move)
        return bool(state.balls[state.current_player] & (1 << src))

    @classmethod
    def from_pgn(cls, pgn_text: str) -> GameRecord:
        """Parse PGN-like format."""
        record = cls()

        # Parse tags
        tag_pattern = r'\[(\w+)\s+"([^"]+)"\]'
        for match in re.finditer(tag_pattern, pgn_text):
            tag, value = match.groups()
            tag_lower = tag.lower()
            if tag_lower == 'event':
                record.event = value
            elif tag_lower == 'site':
                record.site = value
            elif tag_lower == 'date':
                record.date = value
            elif tag_lower == 'round':
                record.round = value
            elif tag_lower == 'white':
                record.white = value
            elif tag_lower == 'black':
                record.black = value
            elif tag_lower == 'result':
                record.result = value

        # Parse moves - remove tags and result markers
        move_text = re.sub(tag_pattern, '', pgn_text)
        move_text = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', move_text)

        # Parse move tokens
        tokens = move_text.split()
        for token in tokens:
            # Skip move numbers like "1." or "12."
            if re.match(r'^\d+\.$', token):
                continue

            # Parse move
            if token.lower() == 'end':
                record.moves.append(END_TURN_MOVE)
            else:
                try:
                    move = algebraic_to_move(token)
                    record.moves.append(move)
                except (ValueError, IndexError):
                    # Skip unparseable tokens
                    pass

        return record

    def replay(self) -> GameState:
        """Replay all moves and return final state."""
        state = GameState.new_game()
        for move in self.moves:
            state.apply_move(move)
        return state


def game_to_pgn(state: GameState, **metadata) -> str:
    """Convert a game state to PGN notation."""
    record = GameRecord.from_state(state, **metadata)
    return record.to_pgn()


def pgn_to_game(pgn_text: str) -> GameState:
    """Parse PGN and return the resulting game state."""
    record = GameRecord.from_pgn(pgn_text)
    return record.replay()
