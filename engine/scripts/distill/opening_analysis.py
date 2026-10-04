#!/usr/bin/env python3
"""
How big is the first-move advantage, and where does it come from?

For every legal first move (or every position reached by a list of opening lines):
  1. deep search value: one long search from the position (--value-sims), value for
     the FIRST player (player 0) as the visit-weighted Q over the root's children;
  2. empirical result: --games self-play games from the position, same network on
     both sides, sampling moves from the search visits for the first --sample-plies
     plies (variety) and playing the most-visited move after that.

A first-move advantage that comes from one or two dominant openings shows up as a
few moves with a much higher P0 score than the rest; a small inherent edge shows up
as a similar modest edge after every reasonable first move.

Example:
  python3 opening_analysis.py --model phoenix2_iter_000.pt --value-sims 4096 \\
      --games 200 --play-sims 256 --json openings.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena import (Model, Search, CRazzleState, _lib, legal_moves, TENSOR, EXTRA,  # noqa: E402
                   make_model)


def root_stats(search: Search) -> tuple[list[int], list[int], list[float]]:
    """(actions, visits, q from the root player's perspective) of the root's children."""
    actions = (ctypes.c_int * 256)()
    visits = (ctypes.c_int * 256)()
    values = (ctypes.c_float * 256)()
    priors = (ctypes.c_float * 256)()
    n = _lib.razzle_mcts_get_root_children(search.tree, actions, visits, values, priors)
    return list(actions[:n]), list(visits[:n]), list(values[:n])


def state_after(moves: list[int]) -> CRazzleState:
    cs = CRazzleState()
    _lib.razzle_state_init(ctypes.byref(cs))
    for m in moves:
        _lib.razzle_state_apply_move(ctypes.byref(cs), m)
    return cs


def run_searches(model, searches: list[Search], max_batch: int = 2048) -> None:
    """Drive many searches to completion with batched evaluations."""
    active = list(searches)
    while active:
        reqs = []
        for s in active:
            r = s.request()
            if r is not None and len(r[1]):
                reqs.append((s, r))
            elif r is not None:
                s.deliver(np.zeros((0, 3137), np.float32), np.zeros(0, np.float32))
        if reqs:
            for i in range(0, len(reqs), max(1, max_batch // 8)):
                part = reqs[i:i + max(1, max_batch // 8)]
                x = np.concatenate([r[0] for _, r in part])
                pl = [p for _, r in part for p in r[1]]
                ex = np.concatenate([r[2] for _, r in part])
                pol, val = model(x, pl, ex)
                k = 0
                for s, r in part:
                    n = len(r[1])
                    s.deliver(pol[k:k + n], val[k:k + n])
                    k += n
        active = [s for s in active if not s.finished()]


def search_value_p0(search: Search, cs: CRazzleState) -> float:
    acts, vis, q = root_stats(search)
    tot = sum(vis)
    v = sum(n * x for n, x in zip(vis, q)) / max(tot, 1)      # root player's perspective
    return v if cs.current_player == 0 else -v


def play_games(model, starts: list[list[int]], games_per: int, sims: int, sample_plies: int,
               leaf_batch: int, seed: int, max_plies: int = 300, concurrency: int = 512) -> list[list[int]]:
    """Self-play from each start; returns winners (0/1/-1) per start."""
    rng = random.Random(seed)
    queue = [(si, list(st)) for si, st in enumerate(starts) for _ in range(games_per)]
    rng.shuffle(queue)
    results: list[list[int]] = [[] for _ in starts]
    active = []   # [start_idx, cs, plies_played, search]
    while queue or active:
        while queue and len(active) < concurrency:
            si, st = queue.pop()
            active.append([si, state_after(st), 0, None])
        still = []
        for g in active:
            si, cs, ply, s = g
            if _lib.razzle_state_is_terminal(ctypes.byref(cs)) or ply >= max_plies:
                results[si].append(int(_lib.razzle_state_get_winner(ctypes.byref(cs))))
                continue
            if s is None:
                g[3] = Search(cs, sims, leaf_batch)
            still.append(g)
        active = still
        if not active:
            continue
        run_one_round(model, [g[3] for g in active])
        for g in active:
            s = g[3]
            if s.finished():
                if g[2] < sample_plies:
                    acts, vis, _ = root_stats(s)
                    move = rng.choices(acts, weights=vis)[0] if sum(vis) else s.result_move
                else:
                    move = s.result_move
                s.free()
                _lib.razzle_state_apply_move(ctypes.byref(g[1]), move)
                g[2] += 1
                g[3] = None
    return results


def run_one_round(model, searches: list[Search]) -> None:
    """One request/deliver round for every unfinished search (batched)."""
    reqs = []
    for s in searches:
        if s.finished():
            continue
        r = s.request()
        if r is None:
            continue
        if len(r[1]) == 0:
            s.deliver(np.zeros((0, 3137), np.float32), np.zeros(0, np.float32))
            continue
        reqs.append((s, r))
    if not reqs:
        return
    x = np.concatenate([r[0] for _, r in reqs])
    pl = [p for _, r in reqs for p in r[1]]
    ex = np.concatenate([r[2] for _, r in reqs])
    pol, val = model(x, pl, ex)
    k = 0
    for s, r in reqs:
        n = len(r[1])
        s.deliver(pol[k:k + n], val[k:k + n])
        k += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--value-sims', type=int, default=4096)
    ap.add_argument('--games', type=int, default=200, help='self-play games per start position')
    ap.add_argument('--play-sims', type=int, default=256)
    ap.add_argument('--sample-plies', type=int, default=6,
                    help='plies after the start position chosen by sampling visit counts')
    ap.add_argument('--lines', default='', help='JSON list of move lists to analyse instead of all first moves')
    ap.add_argument('--leaf-batch', type=int, default=8)
    ap.add_argument('--backend', default='torch', choices=['torch', 'trt'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = make_model(args.model, dev, backend=args.backend, half=dev.type == 'cuda')
    t0 = time.time()
    if args.lines:
        starts = json.loads(Path(args.lines).read_text())
    else:
        starts = [[m] for m in legal_moves(state_after([]))]

    # 1. deep search value of the start position itself and of each line
    root = state_after([])
    root_search = Search(root, args.value_sims, args.leaf_batch)
    states = [state_after(st) for st in starts]
    searches = [Search(cs, args.value_sims, args.leaf_batch) for cs in states]
    run_searches(model, [root_search] + searches)
    root_v = search_value_p0(root_search, root)
    racts, rvis, _ = root_stats(root_search)
    root_share = {a: v / max(1, sum(rvis)) for a, v in zip(racts, rvis)}
    values = [search_value_p0(s, cs) for s, cs in zip(searches, states)]
    for s in [root_search] + searches:
        s.free()
    print(f'[openings] root value for player 0: {root_v:+.3f} ({len(starts)} lines, '
          f'{time.time() - t0:.0f}s)', flush=True)

    # 2. empirical self-play results from each line
    results = play_games(model, starts, args.games, args.play_sims, args.sample_plies,
                         args.leaf_batch, args.seed) if args.games > 0 else [[] for _ in starts]

    rows = []
    for st, v, res in zip(starts, values, results):
        dec = [w for w in res if w >= 0]
        p0 = sum(w == 0 for w in dec) / max(1, len(dec))
        se = math.sqrt(max(p0 * (1 - p0), 1e-9) / max(1, len(dec)))
        rows.append(dict(line=st, value_p0=round(v, 4), p0_score=round(p0, 4), ci95=round(1.96 * se, 4),
                         games=len(res), draws=len(res) - len(dec),
                         root_visit_share=round(root_share.get(st[0], 0.0), 4) if len(st) == 1 else None))
    rows.sort(key=lambda r: -r['value_p0'])
    all_dec = [w for res in results for w in res if w >= 0]
    out = dict(model=args.model, value_sims=args.value_sims, play_sims=args.play_sims, games_per_line=args.games,
               sample_plies=args.sample_plies, root_value_p0=round(root_v, 4),
               overall_p0_score=round(sum(w == 0 for w in all_dec) / max(1, len(all_dec)), 4),
               lines=rows, secs=round(time.time() - t0))
    print(f"{'line':>14} {'value P0':>9} {'P0 score':>9} {'±':>6} {'root visits':>11}")
    for r in rows:
        share = f"{100 * r['root_visit_share']:.1f}%" if r['root_visit_share'] is not None else ''
        print(f"{str(r['line']):>14} {r['value_p0']:+9.3f} {r['p0_score']:9.3f} {r['ci95']:6.3f} {share:>11}")
    print(f"overall P0 score {out['overall_p0_score']:.3f} over {len(all_dec)} decided games; "
          f"root value {root_v:+.3f}; {out['secs']}s", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
