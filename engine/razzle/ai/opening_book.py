"""
Opening book for fast, knowledge-based move selection in known positions.

Loads a JSON book generated from training games and provides instant
move lookup without needing MCTS search.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _state_key(state) -> str:
    """Compute position key from a GameState.

    Format matches generate_opening_book.py:
      "{pieces[0]}:{balls[0]}:{pieces[1]}:{balls[1]}:{current_player}:{touched_mask}:{has_passed}"
    """
    return (
        f"{state.pieces[0]}:{state.balls[0]}:"
        f"{state.pieces[1]}:{state.balls[1]}:"
        f"{state.current_player}:{state.touched_mask}:{int(state.has_passed)}"
    )


class OpeningBook:
    """Opening book for instant move lookup in known positions."""

    def __init__(self, path: str | Path):
        """Load opening book from JSON file.

        Args:
            path: Path to the opening book JSON file.
        """
        with open(path) as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self._positions: dict[str, dict] = data.get("positions", {})
        logger.info(
            "Loaded opening book: %d positions, %s",
            len(self._positions),
            self.metadata.get("model_range", "unknown"),
        )

    def __len__(self) -> int:
        return len(self._positions)

    def lookup(self, state) -> Optional[int]:
        """Look up the most popular move for a position.

        Args:
            state: A GameState instance.

        Returns:
            Action index of the most popular move, or None if position not in book.
        """
        key = _state_key(state)
        entry = self._positions.get(key)
        if entry is None:
            return None
        # moves are sorted by count descending; first entry is most popular
        return entry["moves"][0][0]

    def sample(self, state, temperature: float = 1.0) -> Optional[int]:
        """Sample a move from the book distribution with temperature.

        Args:
            state: A GameState instance.
            temperature: Controls randomness. 0 = always pick most popular,
                1 = sample proportional to visit counts, >1 = more uniform.

        Returns:
            Sampled action index, or None if position not in book.
        """
        key = _state_key(state)
        entry = self._positions.get(key)
        if entry is None:
            return None

        moves = entry["moves"]  # [[action, count, win_rate], ...]

        if temperature == 0.0 or len(moves) == 1:
            return moves[0][0]

        # Apply temperature to visit counts
        counts = [m[1] for m in moves]
        if temperature != 1.0:
            max_count = max(counts)
            # Work in log space for numerical stability
            log_counts = [math.log(c / max_count) / temperature for c in counts]
            max_log = max(log_counts)
            weights = [math.exp(lc - max_log) for lc in log_counts]
        else:
            weights = counts

        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return moves[i][0]
        return moves[-1][0]

    def lookup_distribution(self, state) -> Optional[list[tuple[int, float]]]:
        """Look up the move distribution for a position.

        Args:
            state: A GameState instance.

        Returns:
            List of (action, probability) tuples, or None if not in book.
            Probabilities are proportional to visit counts.
        """
        key = _state_key(state)
        entry = self._positions.get(key)
        if entry is None:
            return None

        total = entry["total"]
        return [(m[0], m[1] / total) for m in entry["moves"]]
