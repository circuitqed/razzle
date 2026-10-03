#!/usr/bin/env python3
"""
Export self-play games from the training DB to gzip JSONL for distillation.

Read-only (opens the SQLite file with mode=ro). One line per game:
  {"id", "model", "moves", "result", "visit_counts"}

Usage (inside the engine container, where the DB lives):
  python3 export_games.py /app/server/data/games.db /tmp/games.jsonl.gz [--prefix gryphon]
"""
import argparse, gzip, json, sqlite3, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('db')
ap.add_argument('out')
ap.add_argument('--prefix', default='', help='only games whose model_version starts with this')
args = ap.parse_args()

con = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
q = 'SELECT id, model_version, moves, result, visit_counts FROM training_games'
params = ()
if args.prefix:
    q += ' WHERE model_version LIKE ?'
    params = (args.prefix + '%',)
q += ' ORDER BY id'

t0 = time.time()
n = 0
with gzip.open(args.out, 'wt', compresslevel=6) as f:
    for gid, model, moves, result, vc in con.execute(q, params):
        # moves / visit_counts are stored as JSON text already; splice without re-parsing
        f.write(f'{{"id":{gid},"model":{json.dumps(model)},"moves":{moves},"result":{result},"visit_counts":{vc}}}\n')
        n += 1
        if n % 50000 == 0:
            print(f'{n} games ({time.time() - t0:.0f}s)', file=sys.stderr, flush=True)
print(f'exported {n} games to {args.out} in {time.time() - t0:.0f}s', file=sys.stderr)
