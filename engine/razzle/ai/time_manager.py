"""
Time management for MCTS in timed games.

Dynamically allocates simulations based on position difficulty:
- Easy positions (high network confidence) get fewer simulations
- Hard positions (high search difficulty) get more simulations

This enables better time allocation in timed games, spending thinking
time where it matters most.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class TimeConfig:
    """Configuration for time-managed search."""

    # Total game time in seconds (None = unlimited)
    total_time: Optional[float] = None

    # Simulation limits
    min_simulations: int = 100      # Always do at least this many
    max_simulations: int = 2000     # Never exceed this
    default_simulations: int = 800  # Used when no time limit

    # Difficulty scaling
    # difficulty_scale controls how much difficulty affects simulation count
    # At difficulty=0.5 (neutral), multiplier is 0.2 + 0.5*1.6 = 1.0
    # At difficulty=0.2 (easy),    multiplier is 0.2 + 0.2*1.6 = 0.52
    # At difficulty=0.8 (hard),    multiplier is 0.2 + 0.8*1.6 = 1.48
    difficulty_scale: float = 1.6
    difficulty_base: float = 0.2    # Minimum multiplier

    # Time management
    time_buffer: float = 2.0        # Reserve this much time at end of game
    move_overhead: float = 0.1      # Fixed overhead per move (network eval, etc.)

    # Estimated game length for initial time allocation
    estimated_moves: int = 40


@dataclass
class TimeManager:
    """
    Manages time allocation for MCTS in timed games.

    Tracks remaining time and converts difficulty predictions to
    target simulation counts.
    """

    config: TimeConfig = field(default_factory=TimeConfig)

    # Time tracking
    remaining_time: float = field(init=False)
    moves_played: int = 0
    estimated_moves_remaining: int = field(init=False)

    # Statistics for analysis
    total_simulations: int = 0
    total_time_used: float = 0.0
    move_count: int = 0

    def __post_init__(self):
        """Initialize time tracking."""
        if self.config.total_time is not None:
            self.remaining_time = self.config.total_time - self.config.time_buffer
        else:
            self.remaining_time = float('inf')
        self.estimated_moves_remaining = self.config.estimated_moves

    def get_target_simulations(self, difficulty: float) -> int:
        """
        Convert difficulty score to target simulation count.

        Args:
            difficulty: Predicted difficulty in [0, 1] where:
                       0 = easy (network is confident)
                       1 = hard (network is uncertain)

        Returns:
            Target number of simulations to run.
        """
        # Clamp difficulty to valid range
        difficulty = max(0.0, min(1.0, difficulty))

        # If no time limit, use fixed simulations scaled by difficulty
        if self.config.total_time is None:
            base = self.config.default_simulations
            multiplier = self.config.difficulty_base + difficulty * self.config.difficulty_scale
            target = int(base * multiplier)
            return max(self.config.min_simulations,
                       min(self.config.max_simulations, target))

        # Calculate time-based allocation
        time_per_move = self._get_time_per_move()
        base_sims = self._time_to_sims(time_per_move)

        # Scale by difficulty
        multiplier = self.config.difficulty_base + difficulty * self.config.difficulty_scale
        target = int(base_sims * multiplier)

        return max(self.config.min_simulations,
                   min(self.config.max_simulations, target))

    def _get_time_per_move(self) -> float:
        """Calculate base time allocation per move."""
        if self.estimated_moves_remaining <= 0:
            return self.config.move_overhead

        available = max(0, self.remaining_time - self.config.move_overhead)
        return available / self.estimated_moves_remaining

    def _time_to_sims(self, time_budget: float) -> int:
        """
        Estimate how many simulations can run in given time.

        This is approximate - actual speed depends on hardware and batch size.
        Assumes roughly 500-1000 sims/second for neural network MCTS.
        """
        # Conservative estimate: 500 sims/second
        # This accounts for network evaluation latency
        sims_per_second = 500

        return int(time_budget * sims_per_second)

    def update(self, elapsed_time: float, simulations_run: int) -> None:
        """
        Update time manager after a move.

        Args:
            elapsed_time: Time spent on this move (seconds)
            simulations_run: Number of simulations actually run
        """
        self.remaining_time = max(0, self.remaining_time - elapsed_time)
        self.moves_played += 1
        self.estimated_moves_remaining = max(1, self.estimated_moves_remaining - 1)

        # Update statistics
        self.total_simulations += simulations_run
        self.total_time_used += elapsed_time
        self.move_count += 1

    def update_moves_estimate(self, new_estimate: int) -> None:
        """
        Update estimated moves remaining based on game progress.

        Can be called when we have better information about likely game length.
        """
        self.estimated_moves_remaining = max(1, new_estimate)

    @property
    def avg_sims_per_move(self) -> float:
        """Average simulations per move so far."""
        if self.move_count == 0:
            return 0.0
        return self.total_simulations / self.move_count

    @property
    def avg_time_per_move(self) -> float:
        """Average time per move so far (seconds)."""
        if self.move_count == 0:
            return 0.0
        return self.total_time_used / self.move_count

    def stats(self) -> dict:
        """Return statistics about time management."""
        return {
            'remaining_time': self.remaining_time,
            'moves_played': self.moves_played,
            'estimated_moves_remaining': self.estimated_moves_remaining,
            'total_simulations': self.total_simulations,
            'total_time_used': self.total_time_used,
            'avg_sims_per_move': self.avg_sims_per_move,
            'avg_time_per_move': self.avg_time_per_move,
        }


def create_time_manager(
    total_time: Optional[float] = None,
    min_sims: int = 100,
    max_sims: int = 2000,
    default_sims: int = 800,
) -> TimeManager:
    """
    Create a time manager with given settings.

    Args:
        total_time: Total game time in seconds, or None for unlimited
        min_sims: Minimum simulations per move
        max_sims: Maximum simulations per move
        default_sims: Default simulations when no time limit

    Returns:
        Configured TimeManager instance.
    """
    config = TimeConfig(
        total_time=total_time,
        min_simulations=min_sims,
        max_simulations=max_sims,
        default_simulations=default_sims,
    )
    return TimeManager(config=config)
