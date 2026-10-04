#!/usr/bin/env python3
"""
Fast head-to-head arena between two RazzleNet models (or one model at two
simulation counts), for picking a distillation teacher, evaluating students
and calibrating difficulty levels.

Plays many games concurrently in one process: each game has its own C search
tree (razzle_fast), and every tick the pending leaves of all trees are batched
into one forward pass per model. Works with GPUs in exclusive-process mode.

Openings: --opening-moves random legal moves, then each opening is played twice
with colors swapped (paired games), so first-move advantage cancels.
Search: fixed simulations, no Dirichlet noise, no early stop, move = most visits.

Example:
  python3 arena.py --a teacher.pt --b student.pt --sims 256 --games 1000
  python3 arena.py --a m.pt --sims-a 64 --b m.pt --sims-b 256 --games 400
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / 'razzle_fast'))
from razzle.ai.network import RazzleNet, NUM_ACTIONS  # noqa: E402
from razzle.core.symmetry import MOVE_ROTATION_MAP  # noqa: E402
from razzle_fast.wrapper import _lib, CRazzleState, _np_to_cfloat_ptr  # noqa: E402

_lib.razzle_state_extra_planes.argtypes = [ctypes.POINTER(CRazzleState), ctypes.POINTER(ctypes.c_float)]
_lib.razzle_state_extra_planes.restype = None
EXTRA = 2 * 8 * 7   # v2 planes 7-8 (last knight destination, forced pass)
_HAS_LEAF_INFO = hasattr(_lib, 'razzle_mcts_leaf_info')   # absent in stale prebuilt .so files
if _HAS_LEAF_INFO:
    _lib.razzle_mcts_leaf_info.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)]
    _lib.razzle_mcts_leaf_info.restype = None


def leaf_info(tree, n: int) -> tuple[list[int], np.ndarray]:
    """Side to move and v2 extra planes of the n leaves from the last select_leaves."""
    extras = np.zeros((n, EXTRA), dtype=np.float32)
    if _HAS_LEAF_INFO:
        players = np.zeros(n, dtype=np.int32)
        _lib.razzle_mcts_leaf_info(ctypes.cast(tree, ctypes.c_void_p), n,
                                   players.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                   extras.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return players.tolist(), extras
    tc = tree.contents
    players = []
    for k in range(n):
        b = tc.leaf_indices[k]
        node = tc.nodes[tc.path_buf[b * tc.max_depth + tc.path_lens[b] - 1]]
        players.append(node.state.current_player)
        _lib.razzle_state_extra_planes(ctypes.byref(node.state), _np_to_cfloat_ptr(extras[k]))
    return players, extras


def extra_planes(cs) -> np.ndarray:
    e = np.zeros(EXTRA, dtype=np.float32)
    _lib.razzle_state_extra_planes(ctypes.byref(cs), _np_to_cfloat_ptr(e))
    return e

TENSOR = 7 * 8 * 7
END_TURN_ACTION = 3136
C_PUCT = 1.5
VLOSS = 3


def legal_moves(cs: CRazzleState) -> list[int]:
    buf = (ctypes.c_int * 256)()
    n = _lib.razzle_state_get_legal_moves(ctypes.byref(cs), buf)
    return [buf[i] for i in range(n)]


class Search:
    """One in-progress MCTS for the side to move in one game."""

    def __init__(self, cs: CRazzleState, sims: int, leaf_batch: int):
        self.sims = sims
        self.leaf_batch = max(1, min(leaf_batch, sims))
        self.tree = _lib.razzle_mcts_create(ctypes.byref(cs), max(4096, sims * 48), self.leaf_batch, 256)
        if not self.tree:
            raise MemoryError('tree alloc failed')
        self.root_cs = cs
        self.expanded = False
        self.done_sims = 0
        self.buf = np.zeros(self.leaf_batch * TENSOR, dtype=np.float32)
        self.pending = 0
        self.result_move: int | None = None

    def free(self):
        if self.tree:
            _lib.razzle_mcts_free(self.tree)
            self.tree = None

    def request(self) -> tuple[np.ndarray, list[int], np.ndarray]:
        """Tensors needing evaluation now, the side to move for each, and v2 extra planes."""
        if not self.expanded:
            t = np.zeros(TENSOR, dtype=np.float32)
            _lib.razzle_state_to_tensor(ctypes.byref(self.root_cs), _np_to_cfloat_ptr(t))
            self.pending = -1
            return t[None], [self.root_cs.current_player], extra_planes(self.root_cs)[None]
        want = min(self.leaf_batch, self.sims - self.done_sims)
        n = _lib.razzle_mcts_select_leaves(self.tree, want, VLOSS, C_PUCT, _np_to_cfloat_ptr(self.buf))
        self.done_sims += want          # terminal leaves were backed up inside select_leaves
        self.pending = n
        if n == 0:
            return np.zeros((0, TENSOR), np.float32), [], np.zeros((0, EXTRA), np.float32)
        players, extras = leaf_info(self.tree, n)
        return self.buf[: n * TENSOR].reshape(n, TENSOR), players, extras

    def deliver(self, policies: np.ndarray, values: np.ndarray):
        if self.pending == -1:
            _lib.razzle_mcts_expand_root(self.tree, _np_to_cfloat_ptr(np.ascontiguousarray(policies[0])))
            self.expanded = True
            win = _lib.razzle_mcts_check_immediate_win(self.tree)
            if win != -2:
                self.result_move = win
        elif self.pending > 0:
            _lib.razzle_mcts_expand_and_backup(
                self.tree, self.pending,
                _np_to_cfloat_ptr(np.ascontiguousarray(policies)),
                _np_to_cfloat_ptr(np.ascontiguousarray(values)), VLOSS)
        self.pending = 0
        if self.result_move is None and self.expanded and self.done_sims >= self.sims:
            self.result_move = self._best()

    def finished(self) -> bool:
        return self.result_move is not None

    def _best(self) -> int:
        actions = (ctypes.c_int * 256)()
        visits = (ctypes.c_int * 256)()
        priors = (ctypes.c_float * 256)()
        vals = (ctypes.c_float * 256)()
        n = _lib.razzle_mcts_get_root_children(self.tree, actions, visits, priors, vals)
        best = max(range(n), key=lambda i: (visits[i], priors[i]))
        return actions[best]


class Game:
    def __init__(self, opening: list[int], a_is_p0: bool):
        self.cs = CRazzleState()
        _lib.razzle_state_init(ctypes.byref(self.cs))
        for m in opening:
            _lib.razzle_state_apply_move(ctypes.byref(self.cs), m)
        self.a_is_p0 = a_is_p0
        self.moves = 0
        self.search: Search | None = None

    def over(self) -> bool:
        return bool(_lib.razzle_state_is_terminal(ctypes.byref(self.cs))) or self.moves >= 300

    def score_for_a(self) -> float:
        w = _lib.razzle_state_get_winner(ctypes.byref(self.cs))
        if w < 0:
            return 0.5
        return 1.0 if (w == 0) == self.a_is_p0 else 0.0

    def mover_is_a(self) -> bool:
        return (self.cs.current_player == 0) == self.a_is_p0


def random_openings(n: int, plies: int, rng: random.Random) -> list[list[int]]:
    out = []
    while len(out) < n:
        cs = CRazzleState()
        _lib.razzle_state_init(ctypes.byref(cs))
        seq = []
        for _ in range(plies):
            if _lib.razzle_state_is_terminal(ctypes.byref(cs)):
                break
            m = rng.choice(legal_moves(cs))
            seq.append(m)
            _lib.razzle_state_apply_move(ctypes.byref(cs), m)
        if not _lib.razzle_state_is_terminal(ctypes.byref(cs)):
            out.append(seq)
    return out


class Model:
    def __init__(self, path: str, device: torch.device, value_scale: float = 1.0, half: bool = False):
        self.value_scale = value_scale   # diagnostic: stretch value outputs (clamped to [-1, 1])
        self.net = RazzleNet.load(path, device=str(device)).to(device).eval()
        # fp16 inference: ~2.4x faster forward on RTX 3060 tensor cores; vs fp32 on real
        # positions max |dp| 0.004, mean |dv| 0.0007, top move agrees 99.96%.
        self.half = half and torch.device(device).type == 'cuda'
        if self.half:
            self.net = self.net.half()
        self.device = device
        self.rot = torch.from_numpy(MOVE_ROTATION_MAP.astype(np.int64)).to(device)
        self.evals = 0

    @torch.no_grad()
    def __call__(self, x: np.ndarray, players: list[int], extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = torch.from_numpy(x).to(self.device).view(-1, 7, 8, 7)
        if self.net.config.num_input_planes == 9:
            t = torch.cat([t, torch.from_numpy(extra).to(self.device).view(-1, 2, 8, 7)], dim=1)
        if self.half:
            t = t.half()
        logp, v, _ = self.net(t)
        p = logp.float().exp()
        p1 = torch.tensor(players, device=self.device) == 1
        if p1.any():
            p[p1] = p[p1][:, self.rot]      # back to absolute orientation for player 1
        self.evals += len(players)
        v = v.float().squeeze(1)
        if self.value_scale != 1.0:
            v = (v * self.value_scale).clamp(-1.0, 1.0)
        return p.cpu().numpy(), v.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--sims', type=int, default=256)
    ap.add_argument('--sims-a', type=int, default=0)
    ap.add_argument('--sims-b', type=int, default=0)
    ap.add_argument('--games', type=int, default=400, help='rounded up to an even number (paired)')
    ap.add_argument('--concurrency', type=int, default=128)
    ap.add_argument('--leaf-batch', type=int, default=8)
    ap.add_argument('--opening-moves', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--json', default='', help='append a result line to this file')
    ap.add_argument('--value-scale-a', type=float, default=1.0, help='diagnostic: multiply A value outputs')
    ap.add_argument('--value-scale-b', type=float, default=1.0)
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sims_a = args.sims_a or args.sims
    sims_b = args.sims_b or args.sims
    model_a = Model(args.a, dev, args.value_scale_a)
    same = args.b == args.a and args.value_scale_a == args.value_scale_b
    model_b = model_a if same else Model(args.b, dev, args.value_scale_b)

    rng = random.Random(args.seed)
    pairs = (args.games + 1) // 2
    queue = []
    for op in random_openings(pairs, args.opening_moves, rng):
        queue += [(op, True), (op, False)]

    active: list[Game] = []
    scores, plies = [], []
    t0 = time.time()
    while queue or active:
        while queue and len(active) < args.concurrency:
            active.append(Game(*queue.pop()))

        # Start searches where needed, finish games that ended
        still = []
        for g in active:
            if g.over():
                scores.append(g.score_for_a())
                plies.append(int(g.cs.ply))
                continue
            if g.search is None:
                sims = sims_a if g.mover_is_a() else sims_b
                g.search = Search(g.cs, sims, args.leaf_batch)
            still.append(g)
        active = still

        # Gather requests per model
        req = {id(model_a): ([], [], [], [], model_a), id(model_b): ([], [], [], [], model_b)}
        for g in active:
            x, players, ex = g.search.request()
            if len(players) == 0:
                g.search.deliver(np.zeros((0, NUM_ACTIONS), np.float32), np.zeros(0, np.float32))
                continue
            xs, ps, exs, owners, _ = req[id(model_a if g.mover_is_a() else model_b)]
            xs.append(x)
            ps += players
            exs.append(ex)
            owners.append((g, len(players)))

        for xs, ps, exs, owners, model in req.values():
            if not xs:
                continue
            pol, val = model(np.concatenate(xs), ps, np.concatenate(exs))
            i = 0
            for g, n in owners:
                g.search.deliver(pol[i:i + n], val[i:i + n])
                i += n

        # Apply finished searches
        for g in active:
            if g.search is not None and g.search.finished():
                _lib.razzle_state_apply_move(ctypes.byref(g.cs), g.search.result_move)
                g.search.free()
                g.search = None
                g.moves += 1

    n = len(scores)
    s = sum(scores) / n
    se = math.sqrt(max(s * (1 - s), 1e-9) / n)
    elo = lambda p: -400 * math.log10(1 / min(max(p, 1e-3), 1 - 1e-3) - 1)
    res = dict(a=args.a, b=args.b, sims_a=sims_a, sims_b=sims_b, games=n,
               value_scale_a=args.value_scale_a, value_scale_b=args.value_scale_b,
               score_a=round(s, 4), ci95=round(1.96 * se, 4),
               elo_a_minus_b=round(elo(s)), elo_ci95=[round(elo(s - 1.96 * se)), round(elo(s + 1.96 * se))],
               median_ply=int(np.median(plies)), secs=round(time.time() - t0),
               evals_a=model_a.evals, evals_b=model_b.evals if model_b is not model_a else None)
    print(json.dumps(res), flush=True)
    if args.json:
        with open(args.json, 'a') as f:
            f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
