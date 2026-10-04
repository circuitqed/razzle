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


def _csr(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense (N, A) -> CSR (ptr int64 (N+1,), idx int16, val float16)."""
    rows, cols = np.nonzero(dense)                     # row-major order = CSR order
    ptr = np.zeros(len(dense) + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=len(dense)), out=ptr[1:])
    return ptr, cols.astype(np.int16), dense[rows, cols].astype(np.float16)


def pack_chunk(states: np.ndarray, policies: np.ndarray, values: np.ndarray,
               legal_masks: np.ndarray, policy_weights: np.ndarray | None = None) -> dict:
    """Pack dense arrays into one buffer chunk (see CompactReplayBuffer.add_chunk).
    Module-level so trainer worker processes can pack their share of a batch."""
    n = len(states)
    if policy_weights is None:
        policy_weights = (policies.sum(axis=1) > 0).astype(np.float32)
    pol_ptr, pol_idx, pol_val = _csr(policies)
    leg_ptr, leg_idx, _ = _csr(legal_masks)
    return dict(
        packed=np.packbits(states.reshape(n, -1).astype(bool), axis=1),
        pol_ptr=pol_ptr, pol_idx=pol_idx, pol_val=pol_val,
        leg_ptr=leg_ptr, leg_idx=leg_idx,
        value=values.astype(np.float32),
        pweight=policy_weights.astype(np.float32),
        planes=np.int64(states.shape[1]),
    )


def _gather_rows(ptr: np.ndarray, local: np.ndarray):
    """For CSR rows `local`: (flat element positions, output row number per element)."""
    a, b = ptr[local], ptr[local + 1]
    lens = b - a
    rows = np.repeat(np.arange(len(local)), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(a, lens) + offs, rows


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
        if len(states) == 0:
            return
        self.add_chunk(pack_chunk(states, policies, values, legal_masks, policy_weights))

    def add_chunk(self, chunk: dict) -> None:
        """Add positions already packed by pack_chunk()."""
        chunk = dict(chunk)
        c = int(chunk.pop('planes'))
        n = len(chunk['value'])
        if n == 0:
            return
        if self.planes is None:
            self.planes = c
        elif self.planes != c:
            raise ValueError(f'buffer holds {self.planes}-plane states, got {c}')
        self._chunks.append(chunk)
        self._size += n
        self.total_positions_seen += n
        self._evict()

    def _evict(self) -> None:
        cap = self.capacity
        while self._size > cap:
            first = self._chunks[0]
            n0 = len(first['value'])
            if self._size - n0 >= cap and len(self._chunks) > 1:
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
        picks = rng.integers(0, self._size, n)             # random order = batch order
        chunk_of = np.searchsorted(starts, picks, side='right') - 1

        c = self.planes
        states = np.zeros((n, c * 56), np.float32)
        policies = np.zeros((n, NUM_ACTIONS), np.float32)
        legal = np.zeros((n, NUM_ACTIONS), np.float32)
        values = np.zeros(n, np.float32)
        pweights = np.zeros(n, np.float32)
        for ci in np.unique(chunk_of):
            ch = self._chunks[ci]
            out_rows = np.nonzero(chunk_of == ci)[0]
            local = picks[out_rows] - starts[ci]
            states[out_rows] = np.unpackbits(ch['packed'][local], axis=1)[:, : c * 56]
            values[out_rows] = ch['value'][local]
            pweights[out_rows] = ch['pweight'][local]
            el, r = _gather_rows(ch['pol_ptr'], local)
            policies[out_rows[r], ch['pol_idx'][el]] = ch['pol_val'][el]
            el, r = _gather_rows(ch['leg_ptr'], local)
            legal[out_rows[r], ch['leg_idx'][el]] = 1.0
        return states.reshape(n, c, 8, 7), policies, values, legal, pweights

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
