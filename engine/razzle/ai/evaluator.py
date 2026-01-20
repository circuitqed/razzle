"""
Batched neural network evaluator for MCTS.

Collects evaluation requests from multiple MCTS workers and processes
them in batches for efficient GPU utilization.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
from threading import Lock, Condition
import numpy as np
import torch

from ..core.state import GameState
from .network import RazzleNet, NUM_ACTIONS, END_TURN_ACTION


@dataclass
class EvalRequest:
    """A pending evaluation request."""
    state: GameState
    callback: Callable[[np.ndarray, float], None]


class BatchedEvaluator:
    """
    Batched neural network evaluator.

    Collects evaluation requests and processes them in batches.
    Can be used synchronously (flush manually) or asynchronously (auto-flush).
    """

    def __init__(
        self,
        network: RazzleNet,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        self.network = network.to(device)
        self.network.eval()
        self.batch_size = batch_size
        self.device = device

        self.pending: list[EvalRequest] = []
        self.lock = Lock()
        self.condition = Condition(self.lock)

        # Statistics
        self.total_evals = 0
        self.total_batches = 0

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """
        Synchronous evaluation of a single state.

        Returns (policy, value) where:
          - policy: numpy array of shape (3136,) with move probabilities
          - value: float in [-1, 1]
        """
        tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_policy, value = self.network(tensor)

        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
        value = value.item()

        self.total_evals += 1
        self.total_batches += 1

        return policy, value

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[np.ndarray, float]]:
        """
        Evaluate a batch of states.

        Returns list of (policy, value) tuples.
        """
        if not states:
            return []

        # Stack state tensors
        tensors = np.stack([s.to_tensor() for s in states])
        batch = torch.from_numpy(tensors).to(self.device)

        with torch.no_grad():
            log_policies, values = self.network(batch)

        policies = torch.exp(log_policies).cpu().numpy()
        values = values.squeeze(-1).cpu().numpy()

        self.total_evals += len(states)
        self.total_batches += 1

        return [(policies[i], values[i]) for i in range(len(states))]

    def request_eval(
        self,
        state: GameState,
        callback: Callable[[np.ndarray, float], None]
    ) -> None:
        """
        Submit an evaluation request (for async batching).

        The callback will be called with (policy, value) when evaluation completes.
        """
        with self.lock:
            self.pending.append(EvalRequest(state, callback))
            if len(self.pending) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> None:
        """Process all pending requests."""
        with self.lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Process pending requests (must hold lock)."""
        if not self.pending:
            return

        requests = self.pending
        self.pending = []

        # Evaluate batch
        states = [r.state for r in requests]
        results = self.evaluate_batch(states)

        # Call callbacks
        for request, (policy, value) in zip(requests, results):
            request.callback(policy, value)

    def stats(self) -> dict:
        """Get evaluation statistics."""
        return {
            'total_evals': self.total_evals,
            'total_batches': self.total_batches,
            'avg_batch_size': self.total_evals / max(1, self.total_batches)
        }


class DummyEvaluator:
    """
    Dummy evaluator that returns uniform policy and zero value.

    Useful for testing MCTS without a trained network.
    """

    def __init__(self):
        self.total_evals = 0

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """Return uniform policy and zero value."""
        from ..core.moves import get_legal_moves

        self.total_evals += 1

        # Uniform over legal moves
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal_moves = get_legal_moves(state)
        if legal_moves:
            prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                # END_TURN (-1) maps to END_TURN_ACTION index
                if move == -1:
                    policy[END_TURN_ACTION] = prob
                else:
                    policy[move] = prob

        return policy, 0.0

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[np.ndarray, float]]:
        """Evaluate batch of states."""
        return [self.evaluate(s) for s in states]
