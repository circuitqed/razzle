"""
Batched neural network evaluator for MCTS.

Collects evaluation requests from multiple MCTS workers and processes
them in batches for efficient GPU utilization.
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional
from threading import Lock, Condition
import numpy as np
import torch

from ..core.state import GameState
from ..core.symmetry import rotate_policy_180
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
        device: str = 'cpu',
        cache_size: int = 10000,
    ):
        self.network = network.to(device)
        self.network.eval()
        self.batch_size = batch_size
        self.device = device

        self.pending: list[EvalRequest] = []
        self.lock = Lock()
        self.condition = Condition(self.lock)

        # LRU eval cache: GameState -> (policy, value, difficulty)
        self._cache: OrderedDict[GameState, tuple[np.ndarray, float, float]] = OrderedDict()
        self._cache_size = cache_size

        # Statistics
        self.total_evals = 0
        self.total_batches = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_put(self, state: GameState, policy: np.ndarray, value: float, difficulty: float) -> None:
        """Add an entry to the LRU cache, evicting oldest if full."""
        if self._cache_size <= 0:
            return
        self._cache[state] = (policy, value, difficulty)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _cache_get(self, state: GameState) -> tuple[np.ndarray, float, float] | None:
        """Look up state in cache. Returns (policy, value, difficulty) or None."""
        entry = self._cache.get(state)
        if entry is not None:
            self._cache.move_to_end(state)
            self.cache_hits += 1
            return entry
        self.cache_misses += 1
        return None

    def clear_cache(self) -> None:
        """Clear the eval cache (e.g. after loading a new model)."""
        self._cache.clear()

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """
        Synchronous evaluation of a single state.

        Returns (policy, value) where:
          - policy: numpy array of shape (3137,) with move probabilities
          - value: float in [-1, 1]
        """
        cached = self._cache_get(state)
        if cached is not None:
            return cached[0], cached[1]

        tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_policy, value, difficulty = self.network(tensor)

        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
        value = value.item()
        difficulty = difficulty.item()

        # Rotate policy back for player 1 (board was rotated in to_tensor)
        if state.current_player == 1:
            policy = rotate_policy_180(policy)

        self._cache_put(state, policy, value, difficulty)
        self.total_evals += 1
        self.total_batches += 1

        return policy, value

    def evaluate_with_difficulty(self, state: GameState) -> tuple[np.ndarray, float, float]:
        """
        Synchronous evaluation including difficulty prediction.

        Returns (policy, value, difficulty) where:
          - policy: numpy array of shape (3137,) with move probabilities
          - value: float in [-1, 1]
          - difficulty: float in [0, 1] predicting search difficulty
        """
        cached = self._cache_get(state)
        if cached is not None:
            return cached

        tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_policy, value, difficulty = self.network(tensor)

        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
        value = value.item()
        difficulty = difficulty.item()

        # Rotate policy back for player 1 (board was rotated in to_tensor)
        if state.current_player == 1:
            policy = rotate_policy_180(policy)

        self._cache_put(state, policy, value, difficulty)
        self.total_evals += 1
        self.total_batches += 1

        return policy, value, difficulty

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[np.ndarray, float]]:
        """
        Evaluate a batch of states.

        Returns list of (policy, value) tuples.
        """
        if not states:
            return []

        # Check cache for each state
        results: list[tuple[np.ndarray, float] | None] = []
        miss_indices: list[int] = []
        miss_states: list[GameState] = []
        for i, state in enumerate(states):
            cached = self._cache_get(state)
            if cached is not None:
                results.append((cached[0], cached[1]))
            else:
                results.append(None)
                miss_indices.append(i)
                miss_states.append(state)

        # Evaluate cache misses
        if miss_states:
            tensors = np.stack([s.to_tensor() for s in miss_states])
            batch = torch.from_numpy(tensors).to(self.device)

            with torch.no_grad():
                log_policies, values, difficulties = self.network(batch)

            policies = torch.exp(log_policies).cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()
            difficulties_np = difficulties.squeeze(-1).cpu().numpy()

            # Rotate policies back for player 1 states
            for j, state in enumerate(miss_states):
                if state.current_player == 1:
                    policies[j] = rotate_policy_180(policies[j])
                self._cache_put(state, policies[j], float(values_np[j]), float(difficulties_np[j]))
                results[miss_indices[j]] = (policies[j], float(values_np[j]))

            self.total_evals += len(miss_states)
            self.total_batches += 1

        return results

    def evaluate_batch_with_difficulty(
        self, states: list[GameState]
    ) -> list[tuple[np.ndarray, float, float]]:
        """
        Evaluate a batch of states including difficulty.

        Returns list of (policy, value, difficulty) tuples.
        """
        if not states:
            return []

        # Check cache for each state
        results: list[tuple[np.ndarray, float, float] | None] = []
        miss_indices: list[int] = []
        miss_states: list[GameState] = []
        for i, state in enumerate(states):
            cached = self._cache_get(state)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                miss_indices.append(i)
                miss_states.append(state)

        # Evaluate cache misses
        if miss_states:
            tensors = np.stack([s.to_tensor() for s in miss_states])
            batch = torch.from_numpy(tensors).to(self.device)

            with torch.no_grad():
                log_policies, values, difficulties = self.network(batch)

            policies = torch.exp(log_policies).cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()
            difficulties_np = difficulties.squeeze(-1).cpu().numpy()

            # Rotate policies back for player 1 states
            for j, state in enumerate(miss_states):
                if state.current_player == 1:
                    policies[j] = rotate_policy_180(policies[j])
                entry = (policies[j], float(values_np[j]), float(difficulties_np[j]))
                self._cache_put(state, *entry)
                results[miss_indices[j]] = entry

            self.total_evals += len(miss_states)
            self.total_batches += 1

        return results

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
        total_lookups = self.cache_hits + self.cache_misses
        return {
            'total_evals': self.total_evals,
            'total_batches': self.total_batches,
            'avg_batch_size': self.total_evals / max(1, self.total_batches),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, total_lookups),
            'cache_size': len(self._cache),
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

    def evaluate_with_difficulty(self, state: GameState) -> tuple[np.ndarray, float, float]:
        """Return uniform policy, zero value, and neutral difficulty."""
        policy, value = self.evaluate(state)
        return policy, value, 0.5  # Neutral difficulty for dummy evaluator

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[np.ndarray, float]]:
        """Evaluate batch of states."""
        return [self.evaluate(s) for s in states]

    def evaluate_batch_with_difficulty(
        self, states: list[GameState]
    ) -> list[tuple[np.ndarray, float, float]]:
        """Evaluate batch of states with difficulty."""
        return [self.evaluate_with_difficulty(s) for s in states]
