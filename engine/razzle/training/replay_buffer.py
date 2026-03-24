"""
Replay buffer for storing training positions across iterations.

Prevents catastrophic forgetting by mixing old and new positions during training.

Buffer size is 10% of total positions seen, with a minimum floor. This mirrors
AlphaZero's approach (which kept the last ~1M games out of ~25M total) but
scaled for our smaller training runs.
"""

import numpy as np
import random


class ReplayBuffer:
    """
    Stores training positions across iterations for replay.

    Buffer capacity grows as training progresses: max(min_positions,
    total_positions_seen * fraction). Oldest positions are evicted
    when capacity is exceeded.
    """

    def __init__(
        self,
        max_positions: int = 100_000,
        fraction: float = 0.10,
        min_positions: int = 5_000,
    ):
        """
        Initialize replay buffer.

        Args:
            max_positions: Hard upper cap on buffer size.
            fraction: Keep this fraction of total positions seen.
            min_positions: Minimum buffer size regardless of fraction.
        """
        self.hard_cap = max_positions
        self.fraction = fraction
        self.min_positions = min_positions
        self.total_positions_seen = 0

        self.states: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.legal_masks: list[np.ndarray | None] = []

    @property
    def capacity(self) -> int:
        """Current buffer capacity based on total positions seen."""
        dynamic = int(self.total_positions_seen * self.fraction)
        return min(self.hard_cap, max(self.min_positions, dynamic))

    def add(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        legal_masks: np.ndarray | None
    ) -> None:
        """
        Add batch of positions to buffer.

        Args:
            states: Board state tensors (N, C, H, W)
            policies: Policy targets (N, NUM_ACTIONS)
            values: Value targets (N,)
            legal_masks: Legal move masks (N, NUM_ACTIONS) or None
        """
        n = len(states)
        self.total_positions_seen += n

        for i in range(n):
            self.states.append(states[i])
            self.policies.append(policies[i])
            self.values.append(values[i])
            self.legal_masks.append(
                legal_masks[i] if legal_masks is not None else None
            )

        self._evict()

    def _evict(self) -> None:
        """Remove oldest positions if over capacity."""
        cap = self.capacity
        excess = len(self.states) - cap
        if excess > 0:
            self.states = self.states[excess:]
            self.policies = self.policies[excess:]
            self.values = self.values[excess:]
            self.legal_masks = self.legal_masks[excess:]

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Sample n random positions from buffer.

        Args:
            n: Number of positions to sample

        Returns:
            Tuple of (states, policies, values, legal_masks).
            legal_masks is None if no masks were stored.
        """
        n = min(n, len(self.states))
        if n == 0:
            return (
                np.empty((0, 6, 8, 7), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                None,
            )

        indices = random.sample(range(len(self.states)), n)

        sampled_states = np.stack([self.states[i] for i in indices])
        sampled_policies = np.stack([self.policies[i] for i in indices])
        sampled_values = np.array([self.values[i] for i in indices], dtype=np.float32)

        # Handle legal masks - check if first one exists
        if self.legal_masks[indices[0]] is not None:
            sampled_masks = np.stack([self.legal_masks[i] for i in indices])
        else:
            sampled_masks = None

        return sampled_states, sampled_policies, sampled_values, sampled_masks

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        """Clear all positions from buffer."""
        self.states.clear()
        self.policies.clear()
        self.values.clear()
        self.legal_masks.clear()
