"""
Replay buffer for storing training positions across iterations.

Prevents catastrophic forgetting by mixing old and new positions during training.
"""

from collections import deque
import numpy as np
import random


class ReplayBuffer:
    """
    Stores training positions across iterations for replay.

    Uses deques with a max size to automatically evict oldest positions
    when capacity is reached.
    """

    def __init__(self, max_positions: int = 100_000):
        """
        Initialize replay buffer.

        Args:
            max_positions: Maximum number of positions to store.
                          Oldest positions are evicted when full.
        """
        self.max_positions = max_positions
        self.states = deque(maxlen=max_positions)
        self.policies = deque(maxlen=max_positions)
        self.values = deque(maxlen=max_positions)
        self.legal_masks = deque(maxlen=max_positions)

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
            states: Board state tensors (N, 6, 8, 7)
            policies: Policy targets (N, NUM_ACTIONS)
            values: Value targets (N,)
            legal_masks: Legal move masks (N, NUM_ACTIONS) or None
        """
        for i in range(len(states)):
            self.states.append(states[i])
            self.policies.append(policies[i])
            self.values.append(values[i])
            self.legal_masks.append(
                legal_masks[i] if legal_masks is not None else None
            )

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
