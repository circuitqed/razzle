#!/usr/bin/env python3
"""
Difficulty-ladder calibration: put (model, simulations) configurations on one Elo scale.

  gen  — write arena match lines pairing each config with its nearest neighbours in a
         guessed strength order (lopsided pairs carry little information).
  fit  — Bradley–Terry fit (MM iterations, weak prior) over arena_results.jsonl lines
         whose label starts with "cal_"; prints Elo relative to an anchor config.

Config names: "<model>@<sims>", models resolved via --models (name -> path).
"""
import argparse
import json
import math
import sys
from collections import defaultdict

MODELS = {
    'p050': 'models/pegasus_iter_050.pt', 'p100': 'models/pegasus_iter_100.pt',
    'p150': 'models/pegasus_iter_150.pt', 'p200': 'models/pegasus_iter_200.pt',
    'p250': 'models/pegasus_iter_250.pt', 'p600': 'teacher.pt',
    's32x4': 'runs/s32x4/student_final.pt', 's48x6': 'runs/s48x6/student_final.pt',
    's64x6': 'runs/s64x6/student_final.pt', 's64x8': 'runs/s64x8/student_final.pt',
    's96x12': 'runs/s96x12/student_final.pt',
}

# Guessed weak -> strong. Current app levels (p050..p250) plus student candidates.
ORDER = [
    'p050@1', 's32x4@1', 'p050@8', 'p050@16', 's32x4@4', 's32x4@8', 'p050@64', 'p100@64',
    's32x4@32', 'p100@128', 's48x6@16', 'p150@128', 's48x6@32', 'p150@256', 'p200@256',
    's32x4@128', 'p250@256', 's48x6@128', 'p600@256', 'p250@512', 's64x8@128', 'p250@1024',
    's96x12@256', 'p250@2048', 's48x6@512', 's64x8@512', 's96x12@1024',
]


def gen(root: str, neighbours: int, games: int):
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:i + 1 + neighbours]:
            ma, sa = a.split('@')
            mb, sb = b.split('@')
            print(f'cal_{a}__{b} --a {root}/{MODELS[ma]} --sims-a {sa} '
                  f'--b {root}/{MODELS[mb]} --sims-b {sb} --games {games}')


def fit(results_path: str, anchor: str, iters: int = 2000):
    wins = defaultdict(float)     # wins[(a, b)] = points a scored vs b
    games = defaultdict(float)
    for line in open(results_path):
        line = line.strip()
        if not line.startswith('{'):
            continue
        r = json.loads(line)
        if not r.get('label', '').startswith('cal_'):
            continue
        a, b = r['label'][4:].split('__')
        n = r['games']
        wins[(a, b)] += r['score_a'] * n
        wins[(b, a)] += (1 - r['score_a']) * n
        games[frozenset((a, b))] += n
    players = sorted({p for k in games for p in k})
    s = {p: 1.0 for p in players}
    prior = 1.0   # one virtual draw vs a reference of equal strength, keeps 0/N pairs finite
    for _ in range(iters):
        new = {}
        for p in players:
            w = prior * 0.5 + sum(wins[(p, q)] for q in players if q != p)
            denom = prior / (s[p] + 1.0) * 1.0
            for q in players:
                if q != p and frozenset((p, q)) in games:
                    denom += games[frozenset((p, q))] / (s[p] + s[q])
            new[p] = w / denom
        s = new
    elo = {p: 400 * math.log10(s[p]) for p in players}
    base = elo.get(anchor, 0.0)
    rows = sorted(((elo[p] - base, p) for p in players))
    for e, p in rows:
        n = sum(v for k, v in games.items() if p in k)
        print(f'{p:14s} {e:+7.0f}   ({int(n)} games)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--root', required=True)
    g.add_argument('--neighbours', type=int, default=3)
    g.add_argument('--games', type=int, default=200)
    f = sub.add_parser('fit')
    f.add_argument('results')
    f.add_argument('--anchor', default='p050@1')
    a = ap.parse_args()
    if a.cmd == 'gen':
        gen(a.root, a.neighbours, a.games)
    else:
        fit(a.results, a.anchor)
