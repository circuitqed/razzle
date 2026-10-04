#!/usr/bin/env python3
"""
Self-play worker v2: many games per process, one batched GPU forward per tick.

Compared with worker_selfplay.py:
  * Many concurrent games share each network call (GPUs in exclusive-process
    mode can't be filled by one-game-per-process workers), and leaf tensors go
    straight from the C tree to torch (no per-leaf Python GameState).
  * Tree reuse: after a move the chosen child's subtree becomes the new root,
    so pass-chain continuations (and the next turn) start with most of their
    search already done.
  * Playout-cap randomization (KataGo): each turn is a full search with
    probability --full-prob (Dirichlet noise, no early stop, visit counts
    recorded as policy targets) or a quick search (--fast-sims, early stop,
    recorded with empty visit counts -> value-only training targets).
  * Supports v1 (7-plane) and v2 (9-plane) networks.

Output: games are POSTed to the training API (same format as the old worker)
and/or appended to a local JSONL file.

Example:
  python3 scripts/selfplay_v2.py --api-url https://.../api --games 0 \
      --sims 800 --fast-sims 160 --concurrency 64
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / 'razzle_fast'))
sys.path.insert(0, str(ENGINE / 'scripts' / 'distill'))
from arena import Model, make_model, extra_planes, leaf_info, legal_moves, TENSOR, EXTRA  # noqa: E402
from razzle.ai.network import RazzleNet  # noqa: E402
from razzle_fast.wrapper import _lib, CRazzleState, CMCTSTree, _np_to_cfloat_ptr  # noqa: E402

_lib.razzle_mcts_reroot.argtypes = [ctypes.POINTER(CMCTSTree), ctypes.c_int]
_lib.razzle_mcts_reroot.restype = ctypes.c_int

NUM_ACTIONS = 3137
C_PUCT = 1.5
VLOSS = 3
STATS = {'reuse_hit': 0, 'reuse_miss': 0}


class SelfPlayGame:
    def __init__(self, args, rng: random.Random):
        self.args = args
        self.rng = rng
        self.cs = CRazzleState()
        _lib.razzle_state_init(ctypes.byref(self.cs))
        self.moves: list[int] = []
        self.visit_counts: list[dict[int, int]] = []
        self.tree = None
        self.capacity = 0
        self.random_moves = args.random_opening_moves if rng.random() < args.random_opening_fraction else 0
        self.turn_full = True          # playout cap decided at each turn start
        self.search_active = False
        self.pending = 0
        self.buf = np.zeros(args.leaf_batch * TENSOR, dtype=np.float32)
        self.target = 0
        self.start_visits = 0
        self.noise_done = False
        self.finished_move: int | None = None

    # -- game state ---------------------------------------------------------
    def over(self) -> bool:
        return bool(_lib.razzle_state_is_terminal(ctypes.byref(self.cs))) or len(self.moves) >= 300

    def result(self) -> float:
        w = _lib.razzle_state_get_winner(ctypes.byref(self.cs))
        return 0.0 if w < 0 else (1.0 if w == 0 else -1.0)

    def free(self):
        if self.tree:
            _lib.razzle_mcts_free(self.tree)
            self.tree = None

    # -- search -------------------------------------------------------------
    def begin_move(self):
        """Prepare a search for the side to move (reusing the tree if we have one)."""
        a = self.args
        if not self.cs.has_passed:                       # new turn: decide full vs quick
            self.turn_full = self.rng.random() < a.full_prob
        self.sims = a.sims if self.turn_full else a.fast_sims
        if self.tree is None:
            self.capacity = max(200_000, a.sims * 64)
            self.tree = _lib.razzle_mcts_create(ctypes.byref(self.cs), self.capacity, a.leaf_batch, 256)
            if not self.tree:
                raise MemoryError('tree alloc failed')
        root_visits = _lib.razzle_mcts_root_visits(self.tree)
        self.start_visits = root_visits
        # top up to the budget, but always add some fresh simulations
        self.target = max(self.sims, root_visits + max(8, self.sims // 8))
        self.noise_done = False
        self.finished_move = None
        self.search_active = True

    def _root_expanded(self) -> bool:
        tc = self.tree.contents
        return bool(tc.nodes[tc.root].is_expanded)

    def request(self):
        """Positions needing evaluation now: (tensors, players, extras)."""
        if not self._root_expanded():
            t = np.zeros(TENSOR, dtype=np.float32)
            _lib.razzle_state_to_tensor(ctypes.byref(self.cs), _np_to_cfloat_ptr(t))
            self.pending = -1
            return t[None], [self.cs.current_player], extra_planes(self.cs)[None]
        self._after_expansion()
        if self.finished_move is not None:
            return None
        visits = _lib.razzle_mcts_root_visits(self.tree)
        want = min(self.args.leaf_batch, self.target - visits)
        if want <= 0:
            self._finish()
            return None
        n = _lib.razzle_mcts_select_leaves(self.tree, want, VLOSS, C_PUCT, _np_to_cfloat_ptr(self.buf))
        self.pending = n
        if n == 0:
            return np.zeros((0, TENSOR), np.float32), [], np.zeros((0, EXTRA), np.float32)
        players, extras = leaf_info(self.tree, n)
        return self.buf[: n * TENSOR].reshape(n, TENSOR), players, extras

    def _after_expansion(self):
        """Once the root is expanded: immediate-win shortcut and root noise (full searches)."""
        if self.noise_done:
            return
        self.noise_done = True
        win = _lib.razzle_mcts_check_immediate_win(self.tree)
        if win != -2:
            self.finished_move = win
            self._record({win: 1} if self.turn_full else {})
            return
        if self.turn_full and self.args.dirichlet_eps > 0:
            tc = self.tree.contents
            nc = tc.nodes[tc.root].num_children
            if nc > 0:
                noise = np.random.dirichlet([self.args.dirichlet_alpha] * nc).astype(np.float32)
                _lib.razzle_mcts_add_dirichlet_noise(self.tree, self.args.dirichlet_eps,
                                                     _np_to_cfloat_ptr(noise), nc)

    def deliver(self, policies, values):
        if self.pending == -1:
            _lib.razzle_mcts_expand_root(self.tree, _np_to_cfloat_ptr(np.ascontiguousarray(policies[0])))
        elif self.pending > 0:
            _lib.razzle_mcts_expand_and_backup(self.tree, self.pending,
                                               _np_to_cfloat_ptr(np.ascontiguousarray(policies)),
                                               _np_to_cfloat_ptr(np.ascontiguousarray(values)), VLOSS)
        self.pending = 0
        if self.finished_move is None and self._root_expanded():
            visits = _lib.razzle_mcts_root_visits(self.tree)
            done = visits >= self.target
            if not done and not self.turn_full and visits - self.start_visits >= 64:
                done = bool(_lib.razzle_mcts_should_stop_early(self.tree, 64, 0.85))
            if done:
                self._finish()

    def _root_visits_dict(self) -> dict[int, int]:
        actions = (ctypes.c_int * 256)()
        visits = (ctypes.c_int * 256)()
        a2 = (ctypes.c_float * 256)()
        a3 = (ctypes.c_float * 256)()
        n = _lib.razzle_mcts_get_root_children(self.tree, actions, visits, a2, a3)
        return {actions[i]: visits[i] for i in range(n) if visits[i] > 0}

    def _finish(self):
        vc = self._root_visits_dict()
        if not vc:
            self.finished_move = self.rng.choice(legal_moves(self.cs))
            self._record({})
            return
        moves = list(vc)
        counts = np.array([vc[m] for m in moves], dtype=np.float64)
        if self.turn_full and len(self.moves) < self.args.temperature_moves:
            self.finished_move = moves[int(np.random.choice(len(moves), p=counts / counts.sum()))]
        else:
            self.finished_move = moves[int(np.argmax(counts))]
        self._record(vc if self.turn_full else {})

    def _record(self, vc):
        self._pending_record = vc

    def play_random_move(self):
        lm = legal_moves(self.cs)
        m = self.rng.choice(lm)
        self.visit_counts.append({x: 1 for x in lm})     # uniform -> value-only target
        self._apply(m, reuse=False)

    def commit_move(self):
        self.visit_counts.append(self._pending_record)
        self._apply(self.finished_move, reuse=True)

    def _apply(self, move: int, reuse: bool):
        self.moves.append(move)
        _lib.razzle_state_apply_move(ctypes.byref(self.cs), move)
        self.search_active = False
        if reuse and self.tree is not None and self.tree.contents.count < 0.6 * self.capacity \
                and _lib.razzle_mcts_reroot(self.tree, move):
            STATS['reuse_hit'] += 1
            return
        if reuse:
            STATS['reuse_miss'] += 1
        self.free()


# ---------------------------------------------------------------------------

class Submitter:
    """Submits finished games from a background thread, retrying with backoff.

    A failed or slow submission no longer stalls the self-play loop or loses the
    game. Games still unsent at shutdown are saved to `spool` and resent on the
    next start.
    """
    MAX_QUEUE = 50_000

    def __init__(self, client, worker_id: str, spool: Path):
        import queue
        import threading
        self.client, self.worker_id, self.spool = client, worker_id, spool
        self.q: queue.Queue = queue.Queue()
        self.failed = self.sent = self.dropped = 0
        self._stop = threading.Event()
        if spool.exists():
            n = 0
            for line in spool.read_text().splitlines():
                if line.strip():
                    self.q.put(json.loads(line))
                    n += 1
            spool.unlink()
            print(f'[selfplay] resending {n} games from {spool}', flush=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, record: dict) -> None:
        if self.q.qsize() >= self.MAX_QUEUE:
            self.q.get_nowait()
            self.dropped += 1
        self.q.put(record)

    def _run(self) -> None:
        import queue
        delay = 1.0
        while not self._stop.is_set():
            try:
                rec = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            while not self._stop.is_set():
                try:
                    self.client.submit_game(worker_id=self.worker_id, moves=rec['moves'], result=rec['result'],
                                            visit_counts=[{int(k): v for k, v in d.items()}
                                                          for d in rec['visit_counts']],
                                            model_version=rec['model'])
                    self.sent += 1
                    delay = 1.0
                    break
                except Exception as e:
                    self.failed += 1
                    if self.failed <= 5 or self.failed % 100 == 0:
                        print(f'[selfplay] submit failed ({self.failed} so far, {self.q.qsize()} queued): {e}',
                              flush=True)
                    self._stop.wait(delay)
                    delay = min(delay * 2, 60.0)
            else:
                self.q.put(rec)          # stopping: keep it for the spool

    def close(self, timeout: float = 20.0) -> None:
        """Try to drain the queue, then spool whatever is left to disk."""
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.2)
        self._stop.set()
        self.thread.join(timeout=35)
        left = []
        while not self.q.empty():
            left.append(self.q.get_nowait())
        if left:
            with open(self.spool, 'a') as f:
                for rec in left:
                    f.write(json.dumps(rec) + '\n')
            print(f'[selfplay] saved {len(left)} unsent games to {self.spool}', flush=True)


class ModelSource:
    """Latest model from the training API (or a fixed local file)."""

    def __init__(self, args, device):
        self.args, self.device = args, device
        self.version = None
        self.model: Model | None = None
        self.client = None
        if args.api_url:
            from razzle.training.api_client import TrainingAPIClient
            self.client = TrainingAPIClient(base_url=args.api_url)
        self.dir = Path(args.model_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._pending = None
        self._executor = None

    def _make(self, path: str):
        return make_model(path, self.device, backend=self.args.backend,
                          half=self.args.fp16, cuda_graphs=self.args.cuda_graphs)

    def refresh(self) -> bool:
        # TensorRT engines take ~15 s to build: after the first model, load new ones in a
        # background thread and keep playing with the current model until it is ready.
        if self.args.backend == 'trt' and self.model is not None and not self.args.model:
            if self._pending is not None:
                if not self._pending.done():
                    return False
                fut, self._pending = self._pending, None
                try:
                    return fut.result()
                except Exception as e:
                    print(f'[selfplay] background model load failed: {e}', flush=True)
                    return False
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor
                self._executor = ThreadPoolExecutor(1)
            self._pending = self._executor.submit(self._refresh_now)
            return False
        return self._refresh_now()

    def _refresh_now(self) -> bool:
        if self.args.model:
            if self.model is None:
                self.model = self._make(self.args.model)
                self.version = Path(self.args.model).stem
                return True
            return False
        info = self.client.get_latest_model()
        if info is None or info.version == self.version:
            return False
        path = self.dir / f'{info.version}.pt'
        for attempt in range(6):
            try:
                if not path.exists():
                    self.client.download_model(info.version, path)
                model = self._make(str(path))
                break
            except Exception as e:      # network error or a bad file: refetch and retry
                print(f'[selfplay] loading {info.version} failed ({e}); retrying', flush=True)
                path.unlink(missing_ok=True)
                if self.model is not None and attempt >= 1:
                    return False        # keep playing with the current model
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f'could not load {info.version}')
        self.model = model
        self.version = info.version
        print(f'[selfplay] loaded model {self.version}', flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api-url', default='')
    ap.add_argument('--model', default='', help='fixed local model (no API polling)')
    ap.add_argument('--model-dir', default='output/selfplay_models')
    ap.add_argument('--out-jsonl', default='', help='also append games here')
    ap.add_argument('--worker-id', default=f'sp2-{os.uname().nodename}-{os.getpid()}')
    ap.add_argument('--games', type=int, default=0, help='stop after N games (0 = run forever)')
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument('--leaf-batch', type=int, default=8)
    ap.add_argument('--sims', type=int, default=800, help='full-search simulations')
    ap.add_argument('--fast-sims', type=int, default=160, help='quick-search simulations')
    ap.add_argument('--full-prob', type=float, default=0.25)
    ap.add_argument('--temperature-moves', type=int, default=15)
    ap.add_argument('--dirichlet-alpha', type=float, default=0.3)
    ap.add_argument('--dirichlet-eps', type=float, default=0.25)
    ap.add_argument('--random-opening-moves', type=int, default=8)
    ap.add_argument('--random-opening-fraction', type=float, default=0.3)
    ap.add_argument('--refresh-every', type=int, default=200, help='check for a new model every N games')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True,
                    help='half-precision inference on CUDA (default on; --no-fp16 for fp32)')
    ap.add_argument('--backend', choices=['torch', 'trt'], default='torch',
                    help='trt: TensorRT fp16 engine (~3x faster forward; needs tensorrt-cu12 + onnx)')
    ap.add_argument('--cuda-graphs', action=argparse.BooleanOptionalAction, default=False,
                    help='replay the forward pass as captured CUDA graphs (padded batch buckets)')
    args = ap.parse_args()
    if not args.api_url and not args.model:
        raise SystemExit('need --api-url or --model')

    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    rng = random.Random(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)

    source = ModelSource(args, dev)
    while not source.refresh() and source.model is None:
        print('[selfplay] waiting for a model...', flush=True)
        time.sleep(30)

    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__('flag', True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__('flag', True))
    out = open(args.out_jsonl, 'a') if args.out_jsonl else None
    submitter = (Submitter(source.client, args.worker_id, Path(args.model_dir).parent / f'unsent_{args.worker_id}.jsonl')
                 if source.client else None)

    games: list[SelfPlayGame] = []
    done = 0
    since_refresh = 0
    evals = 0
    last_report = 0
    t0 = time.time()
    while not stop['flag'] and (args.games == 0 or done < args.games):
        target_active = args.concurrency if args.games == 0 else min(args.concurrency, args.games - done)
        while len(games) < target_active:
            games.append(SelfPlayGame(args, rng))

        # Finish games, play random opening moves, start searches
        still = []
        for g in games:
            if g.over():
                g.free()
                record = dict(worker_id=args.worker_id, model=source.version, moves=g.moves,
                              result=g.result(), visit_counts=[{str(k): v for k, v in d.items()} for d in g.visit_counts])
                if submitter:
                    submitter.put(record)
                if out:
                    out.write(json.dumps(record) + '\n')
                    out.flush()
                done += 1
                since_refresh += 1
                continue
            while len(g.moves) < g.random_moves and not g.over():
                g.play_random_move()
            if g.over():
                still.append(g)
                continue
            if not g.search_active:
                g.begin_move()
            still.append(g)
        games = still

        # Gather evaluation requests
        xs, ps, exs, owners = [], [], [], []
        for g in games:
            if g.over() or not g.search_active or g.finished_move is not None:
                continue
            req = g.request()
            if req is None:
                continue
            x, players, ex = req
            if not players:
                g.deliver(np.zeros((0, NUM_ACTIONS), np.float32), np.zeros(0, np.float32))
                continue
            xs.append(x); ps += players; exs.append(ex); owners.append((g, len(players)))
        if xs:
            pol, val = source.model(np.concatenate(xs), ps, np.concatenate(exs))
            evals += len(ps)
            i = 0
            for g, n in owners:
                g.deliver(pol[i:i + n], val[i:i + n])
                i += n

        # Commit finished moves
        for g in games:
            if g.search_active and g.finished_move is not None:
                g.commit_move()

        if since_refresh >= args.refresh_every and not args.model:
            since_refresh = 0
            try:
                source.refresh()
            except Exception as e:
                print(f'[selfplay] model refresh failed: {e}', flush=True)

        if done >= last_report + 50 and evals:
            last_report = done
            el = time.time() - t0
            print(f'[selfplay] {done} games, {done / el * 3600:.0f} games/h, {evals / el:,.0f} evals/s', flush=True)

    for g in games:
        g.free()
    el = time.time() - t0
    if submitter:
        submitter.close()
        print(f'[selfplay] submitted {submitter.sent}, retried failures {submitter.failed}, '
              f'dropped {submitter.dropped}', flush=True)
    print(f'[selfplay] finished {done} games in {el:.0f}s ({done / max(el, 1e-9) * 3600:.0f} games/h, '
          f'{evals / max(el, 1e-9):,.0f} evals/s); tree reuse {STATS}', flush=True)


if __name__ == '__main__':
    main()
