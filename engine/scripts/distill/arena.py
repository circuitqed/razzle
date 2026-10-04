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
    def __init__(self, opening: list[int], a_is_p0: bool, pair: int = -1):
        self.pair = pair
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


GRAPH_BUCKETS = (64, 128, 256, 384, 512, 640, 768, 1024, 1536, 2048)


class Model:
    def __init__(self, path: str, device: torch.device, value_scale: float = 1.0, half: bool = False,
                 cuda_graphs: bool = False):
        self.value_scale = value_scale   # diagnostic: stretch value outputs (clamped to [-1, 1])
        self.net = RazzleNet.load(path, device=str(device)).to(device).eval()
        # fp16 inference: ~2.4x faster forward on RTX 3060 tensor cores; vs fp32 on real
        # positions max |dp| 0.004, mean |dv| 0.0007, top move agrees 99.96%.
        on_cuda = torch.device(device).type == 'cuda'
        self.half = half and on_cuda
        if self.half:
            self.net = self.net.half()
        self.device = device
        self.planes = self.net.config.num_input_planes
        self.rot = torch.from_numpy(MOVE_ROTATION_MAP.astype(np.int64)).to(device)
        self.evals = 0
        # CUDA graphs: replay the whole forward pass as one launch instead of ~30
        # Python-dispatched kernels. One graph per padded batch size (bucket).
        self.cuda_graphs = cuda_graphs and on_cuda
        self._graphs: dict[int, tuple] = {}

    def _forward(self, t: torch.Tensor):
        logp, v, _ = self.net(t)
        return logp.float().exp(), v.float().squeeze(1)

    def _graph_for(self, n: int):
        size = next((b for b in GRAPH_BUCKETS if b >= n), None)
        if size is None:
            return None
        if size not in self._graphs:
            dtype = torch.float16 if self.half else torch.float32
            static_in = torch.zeros(size, self.planes, 8, 7, device=self.device, dtype=dtype)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):                       # warm up (cuDNN autotune) off the graph
                    self._forward(static_in)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_p, static_v = self._forward(static_in)
            self._graphs[size] = (g, static_in, static_p, static_v)
        return self._graphs[size]

    @torch.no_grad()
    def __call__(self, x: np.ndarray, players: list[int], extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(players)
        t = torch.from_numpy(x).to(self.device).view(-1, 7, 8, 7)
        if self.planes == 9:
            t = torch.cat([t, torch.from_numpy(extra).to(self.device).view(-1, 2, 8, 7)], dim=1)
        if self.half:
            t = t.half()
        gr = self._graph_for(n) if self.cuda_graphs else None
        if gr is not None:
            g, static_in, static_p, static_v = gr
            static_in[:n].copy_(t)
            g.replay()
            p, v = static_p[:n].clone(), static_v[:n]
        else:
            p, v = self._forward(t)
        p1 = torch.from_numpy(np.asarray(players, dtype=np.int64)).to(self.device) == 1
        if p1.any():
            p[p1] = p[p1][:, self.rot]      # back to absolute orientation for player 1
        self.evals += n
        if self.value_scale != 1.0:
            v = (v * self.value_scale).clamp(-1.0, 1.0)
        return p.cpu().numpy(), v.cpu().numpy()


def build_trt_engine(path: str, engine_path: str, max_batch: int = 2048) -> None:
    """ONNX-export a checkpoint and build a TensorRT fp16 engine (dynamic batch 1..max_batch)."""
    import os
    import tensorrt as trt
    net = RazzleNet.load(path, device='cpu').eval()
    planes = net.config.num_input_planes
    log = trt.Logger(trt.Logger.ERROR)
    onnx_path = f'{path}.{os.getpid()}.onnx'
    torch.onnx.export(net, torch.zeros(1, planes, 8, 7), onnx_path, opset_version=17,
                      input_names=['x'], output_names=['logp', 'v', 'd'],
                      dynamic_axes={k: {0: 'b'} for k in ('x', 'logp', 'v', 'd')})
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    ok = parser.parse(open(onnx_path, 'rb').read())
    os.remove(onnx_path)
    if not ok:
        raise RuntimeError(f'ONNX parse failed: {parser.get_error(0)}')
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = 1     # 15 s build, same speed as level 3 here
    prof = builder.create_optimization_profile()
    prof.set_shape('x', (1, planes, 8, 7), (768, planes, 8, 7), (max_batch, planes, 8, 7))
    cfg.add_optimization_profile(prof)
    data = builder.build_serialized_network(network, cfg)
    if data is None:
        raise RuntimeError('TensorRT engine build failed')
    tmp = f'{engine_path}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, engine_path)


class TRTModel:
    """Same interface as Model, running a TensorRT fp16 engine (~3x faster than PyTorch
    fp16 on an RTX 3060; vs fp32 top move agrees 99.9%, mean |dv| 0.0013).

    The engine is built from an ONNX export of the checkpoint (~15 s) and cached next to
    it as <model>.fp16.engine; processes sharing the directory build it once (file lock).
    Needs: pip install tensorrt-cu12 onnx "numpy<2".
    """
    MAX_BATCH = 2048

    def __init__(self, path: str, device):
        import fcntl
        import os
        import tensorrt as trt
        self.trt = trt
        self.device = torch.device(device)
        self.value_scale = 1.0
        self.evals = 0
        self.planes = RazzleNet.load(path, device='cpu').config.num_input_planes
        self.rot = torch.from_numpy(MOVE_ROTATION_MAP.astype(np.int64)).to(self.device)
        log = trt.Logger(trt.Logger.ERROR)
        engine_path = f'{path}.fp16.engine'
        with open(f'{path}.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(engine_path):
                # Build in a child process: the builder's ~3 GB of host memory is never
                # returned to the OS, and with processes taking turns to build, every
                # self-play process would otherwise grow to ~5-6 GB.
                import subprocess
                subprocess.run([sys.executable, '-c',
                                'import sys; sys.path.insert(0, sys.argv[1]); import arena; '
                                'arena.build_trt_engine(sys.argv[2], sys.argv[3])',
                                str(Path(__file__).resolve().parent), path, engine_path], check=True)
            with open(engine_path, 'rb') as f:
                self.engine = trt.Runtime(log).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)
        n = self.MAX_BATCH
        self.out = {'logp': torch.empty(n, NUM_ACTIONS, device=self.device),
                    'v': torch.empty(n, 1, device=self.device), 'd': torch.empty(n, 1, device=self.device)}

    @torch.no_grad()
    def __call__(self, x: np.ndarray, players: list[int], extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(players)
        if n > self.MAX_BATCH:
            parts = [self(x[i:i + self.MAX_BATCH], players[i:i + self.MAX_BATCH], extra[i:i + self.MAX_BATCH])
                     for i in range(0, n, self.MAX_BATCH)]
            return np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])
        with torch.cuda.stream(self.stream):
            t = torch.from_numpy(x).to(self.device).view(-1, 7, 8, 7)
            if self.planes == 9:
                t = torch.cat([t, torch.from_numpy(extra).to(self.device).view(-1, 2, 8, 7)], dim=1)
            t = t.contiguous()
            self.ctx.set_input_shape('x', tuple(t.shape))
            self.ctx.set_tensor_address('x', t.data_ptr())
            for k, buf in self.out.items():
                self.ctx.set_tensor_address(k, buf.data_ptr())
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            p = self.out['logp'][:n].exp()
            v = self.out['v'][:n, 0].clone()
            p1 = torch.from_numpy(np.asarray(players, dtype=np.int64)).to(self.device) == 1
            if p1.any():
                p[p1] = p[p1][:, self.rot]
            self.evals += n
            p, v = p.cpu().numpy(), v.cpu().numpy()      # .cpu() synchronizes the stream
        return p, v


def make_model(path: str, device, backend: str = 'torch', half: bool = False, cuda_graphs: bool = False):
    if backend == 'trt':
        return TRTModel(path, device)
    return Model(path, device, half=half, cuda_graphs=cuda_graphs)


def _color_stats(outcomes: dict) -> dict:
    """How much colour decides games: first-player (P0) win rate, and for each opening
    played twice with colours swapped, whether the same colour won both games
    (colour-decided) or the same model won both (skill-decided)."""
    games = [w for v in outcomes.values() for _, w in v]
    decided = [w for w in games if w >= 0]
    same_colour = same_model = 0
    full = [v for v in outcomes.values() if len(v) == 2 and all(w >= 0 for _, w in v)]
    for (a0, w0), (a1, w1) in full:
        if w0 == w1:
            same_colour += 1
        else:
            same_model += 1
    def score(as_p0):   # model A's score in the games where it played first (as_p0) / second
        r = [(0.5 if w < 0 else float((w == 0) == a0)) for v in outcomes.values() for a0, w in v if a0 == as_p0]
        return round(sum(r) / len(r), 4) if r else None
    sa0, sa1 = score(True), score(False)
    if sa0 is None or sa1 is None:      # fixed colours: no first-move / skill split
        return dict(p0_win_rate=round(sum(w == 0 for w in decided) / max(1, len(decided)), 4),
                    draws=len(games) - len(decided), score_a_first=sa0, score_a_second=sa1)
    lo = lambda p: math.log10(min(max(p, 1e-3), 1 - 1e-3) / (1 - min(max(p, 1e-3), 1 - 1e-3)))
    # Bradley-Terry with a first-move term: logit(A first) = d + f, logit(A second) = d - f
    return dict(p0_win_rate=round(sum(w == 0 for w in decided) / max(1, len(decided)), 4),
                draws=len(games) - len(decided),
                pairs_colour_decided=same_colour, pairs_skill_decided=same_model,
                score_a_first=sa0, score_a_second=sa1,
                elo_first_move=round(200 * (lo(sa0) - lo(sa1))),
                elo_skill=round(200 * (lo(sa0) + lo(sa1))))


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
    ap.add_argument('--colours', choices=['both', 'a-first', 'a-second'], default='both',
                    help='both: each opening twice with colours swapped; a-first / a-second: A always '
                         'moves first / second (handicap tests: how much stronger must the second player be?)')
    ap.add_argument('--leaf-batch-a', type=int, default=0, help='per-side override (search-quality tests)')
    ap.add_argument('--leaf-batch-b', type=int, default=0)
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
    if args.colours == 'both':
        for k, op in enumerate(random_openings(pairs, args.opening_moves, rng)):
            queue += [(op, True, k), (op, False, k)]
    else:
        # A plays P0 (the side that moves first from the initial position) or P1. Openings
        # use an even number of plies so the side to move after the opening is still P0.
        plies = args.opening_moves - (args.opening_moves % 2)
        for k, op in enumerate(random_openings(2 * pairs, plies, rng)):
            queue.append((op, args.colours == 'a-first', k))

    active: list[Game] = []
    scores, plies, outcomes = [], [], {}   # outcomes[pair] = [(a_is_p0, winner), ...]
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
                outcomes.setdefault(g.pair, []).append(
                    (g.a_is_p0, int(_lib.razzle_state_get_winner(ctypes.byref(g.cs)))))
                continue
            if g.search is None:
                sims = sims_a if g.mover_is_a() else sims_b
                lb = (args.leaf_batch_a if g.mover_is_a() else args.leaf_batch_b) or args.leaf_batch
                g.search = Search(g.cs, sims, lb)
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
    res = dict(a=args.a, b=args.b, sims_a=sims_a, sims_b=sims_b, games=n, colours=args.colours,
               leaf_batch_a=args.leaf_batch_a or args.leaf_batch, leaf_batch_b=args.leaf_batch_b or args.leaf_batch,
               value_scale_a=args.value_scale_a, value_scale_b=args.value_scale_b,
               score_a=round(s, 4), ci95=round(1.96 * se, 4),
               elo_a_minus_b=round(elo(s)), elo_ci95=[round(elo(s - 1.96 * se)), round(elo(s + 1.96 * se))],
               median_ply=int(np.median(plies)), secs=round(time.time() - t0),
               **_color_stats(outcomes),
               evals_a=model_a.evals, evals_b=model_b.evals if model_b is not model_a else None)
    print(json.dumps(res), flush=True)
    if args.json:
        with open(args.json, 'a') as f:
            f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
