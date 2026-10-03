"""
Compact replay buffer for self-play training.

Positions are stored the way the distillation shards store them: board planes
bit-packed (binary planes), policy targets and legal moves as sparse lists.
About 200 bytes per position instead of ~26 KB for dense float arrays, so the
window can hold millions of positions (tens of thousands of games).

Each position carries a policy weight: 0 for positions whose policy target
carries no information (random-opening moves recorded with uniform visit
counts, or quick searches recorded without visit counts), so they train the
value head only.

The window is the most recent `capacity` positions:
    capacity = clamp(fraction * total_positions_seen, min_positions, max_positions)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_ACTIONS = 3137


def _sparse(dense: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    idx, val = [], []
    for row in dense:
        nz = np.nonzero(row)[0]
        idx.append(nz.astype(np.int16))
        val.append(row[nz].astype(np.float16))
    return idx, val


class CompactReplayBuffer:
    def __init__(self, max_positions: int = 3_000_000, min_positions: int = 250_000,
                 fraction: float = 0.25):
        self.max_positions = max_positions
        self.min_positions = min_positions
        self.fraction = fraction
        self.total_positions_seen = 0
        self.planes: int | None = None
        # Chunks appended per add(); evicted oldest-first.
        self._chunks: list[dict] = []
        self._size = 0

    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        dynamic = int(self.total_positions_seen * self.fraction)
        return min(self.max_positions, max(self.min_positions, dynamic))

    def __len__(self) -> int:
        return self._size

    def add(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray,
            legal_masks: np.ndarray, policy_weights: np.ndarray | None = None) -> None:
        """Add positions. states: (N, C, 8, 7) binary floats; policies/legal_masks: (N, 3137)."""
        n = len(states)
        if n == 0:
            return
        c = states.shape[1]
        if self.planes is None:
            self.planes = c
        elif self.planes != c:
            raise ValueError(f'buffer holds {self.planes}-plane states, got {c}')
        pol_idx, pol_val = _sparse(policies)
        leg_idx, _ = _sparse(legal_masks)
        if policy_weights is None:
            policy_weights = (policies.sum(axis=1) > 0).astype(np.float32)
        chunk = dict(
            packed=np.packbits(states.reshape(n, -1).astype(bool), axis=1),
            pol_ptr=np.concatenate([[0], np.cumsum([len(a) for a in pol_idx])]).astype(np.int64),
            pol_idx=np.concatenate(pol_idx) if pol_idx else np.zeros(0, np.int16),
            pol_val=np.concatenate(pol_val) if pol_val else np.zeros(0, np.float16),
            leg_ptr=np.concatenate([[0], np.cumsum([len(a) for a in leg_idx])]).astype(np.int64),
            leg_idx=np.concatenate(leg_idx) if leg_idx else np.zeros(0, np.int16),
            value=values.astype(np.float32),
            pweight=policy_weights.astype(np.float32),
        )
        self._chunks.append(chunk)
        self._size += n
        self.total_positions_seen += n
        self._evict()

    def _evict(self) -> None:
        cap = self.capacity
        while self._size > cap and len(self._chunks) > 1:
            first = self._chunks[0]
            n0 = len(first['value'])
            if self._size - n0 >= cap:
                self._chunks.pop(0)
                self._size -= n0
            else:
                # Trim the oldest chunk partially
                drop = self._size - cap
                self._chunks[0] = _slice_chunk(first, drop, n0)
                self._size -= drop
                break

    # ------------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator | None = None):
        """Sample n positions uniformly from the window. Returns dense numpy arrays:
        states (n, C, 8, 7) float32, policies (n, 3137), values (n,), legal (n, 3137), pweights (n,)."""
        rng = rng or np.random.default_rng()
        sizes = np.array([len(ch['value']) for ch in self._chunks])
        starts = np.concatenate([[0], np.cumsum(sizes)])
        picks = np.sort(rng.integers(0, self._size, n))
        chunk_of = np.searchsorted(starts, picks, side='right') - 1

        c = self.planes
        states = np.zeros((n, c * 56), np.float32)
        policies = np.zeros((n, NUM_ACTIONS), np.float32)
        legal = np.zeros((n, NUM_ACTIONS), np.float32)
        values = np.zeros(n, np.float32)
        pweights = np.zeros(n, np.float32)
        out = 0
        for ci in np.unique(chunk_of):
            ch = self._chunks[ci]
            local = picks[chunk_of == ci] - starts[ci]
            k = len(local)
            sl = slice(out, out + k)
            states[sl] = np.unpackbits(ch['packed'][local], axis=1)[:, : c * 56]
            values[sl] = ch['value'][local]
            pweights[sl] = ch['pweight'][local]
            for j, i in enumerate(local):
                a, b = ch['pol_ptr'][i], ch['pol_ptr'][i + 1]
                policies[out + j, ch['pol_idx'][a:b]] = ch['pol_val'][a:b]
                a, b = ch['leg_ptr'][i], ch['leg_ptr'][i + 1]
                legal[out + j, ch['leg_idx'][a:b]] = 1.0
            out += k
        perm = rng.permutation(n)        # undo the sort so batches are mixed
        return (states.reshape(n, c, 8, 7)[perm], policies[perm], values[perm],
                legal[perm], pweights[perm])

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if not self._chunks:
            return
        merged = _merge(self._chunks)
        np.savez(path, planes=self.planes, total_positions_seen=self.total_positions_seen, **merged)

    def load(self, path: str | Path) -> None:
        d = np.load(path)
        self.planes = int(d['planes'])
        chunk = {k: d[k] for k in ('packed', 'pol_ptr', 'pol_idx', 'pol_val', 'leg_ptr',
                                   'leg_idx', 'value', 'pweight')}
        self._chunks = [chunk]
        self._size = len(chunk['value'])
        self.total_positions_seen = int(d['total_positions_seen'])
        self._evict()


def _slice_chunk(ch: dict, lo: int, hi: int) -> dict:
    pa, pb = ch['pol_ptr'][lo], ch['pol_ptr'][hi]
    la, lb = ch['leg_ptr'][lo], ch['leg_ptr'][hi]
    return dict(
        packed=ch['packed'][lo:hi],
        pol_ptr=ch['pol_ptr'][lo:hi + 1] - pa, pol_idx=ch['pol_idx'][pa:pb], pol_val=ch['pol_val'][pa:pb],
        leg_ptr=ch['leg_ptr'][lo:hi + 1] - la, leg_idx=ch['leg_idx'][la:lb],
        value=ch['value'][lo:hi], pweight=ch['pweight'][lo:hi],
    )


def _merge(chunks: list[dict]) -> dict:
    pol_ptr, leg_ptr, po, lo = [np.zeros(1, np.int64)], [np.zeros(1, np.int64)], 0, 0
    for ch in chunks:
        pol_ptr.append(ch['pol_ptr'][1:] + po)
        leg_ptr.append(ch['leg_ptr'][1:] + lo)
        po += ch['pol_ptr'][-1]
        lo += ch['leg_ptr'][-1]
    cat = lambda k: np.concatenate([ch[k] for ch in chunks])
    return dict(packed=cat('packed'), pol_ptr=np.concatenate(pol_ptr), pol_idx=cat('pol_idx'),
                pol_val=cat('pol_val'), leg_ptr=np.concatenate(leg_ptr), leg_idx=cat('leg_idx'),
                value=cat('value'), pweight=cat('pweight'))
