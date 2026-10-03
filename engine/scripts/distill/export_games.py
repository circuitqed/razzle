#!/usr/bin/env python3
"""
Export self-play games from the training DB to gzip JSONL for distillation.

Read-only (opens the SQLite file with mode=ro). One line per game:
  {"id", "model", "moves", "result", "visit_counts"}

Reads either the legacy training_games table (games.db, JSON text columns)
or selfplay_games (training.db, zlib-compressed JSON), whichever the DB has.

Usage (inside the engine container, where the DBs live):
  python3 export_games.py /app/server/data/training.db /tmp/games.jsonl.gz [--prefix v2run]
  python3 export_games.py /app/server/data/games.db /tmp/legacy.jsonl.gz
"""
import argparse, gzip, json, sqlite3, sys, time, zlib

ap = argparse.ArgumentParser()
ap.add_argument('db')
ap.add_argument('out')
ap.add_argument('--prefix', default='', help='only games whose model_version starts with this')
args = ap.parse_args()

con = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
compressed = 'selfplay_games' in tables
if compressed:
    q = 'SELECT id, model_version, data, result FROM selfplay_games'
else:
    q = 'SELECT id, model_version, moves, result, visit_counts FROM training_games'
params = ()
if args.prefix:
    q += ' WHERE model_version LIKE ?'
    params = (args.prefix + '%',)
q += ' ORDER BY id'

t0 = time.time()
n = 0
with gzip.open(args.out, 'wt', compresslevel=6) as f:
    for row in con.execute(q, params):
        if compressed:
            gid, model, blob, result = row
            d = json.loads(zlib.decompress(blob))
            f.write(json.dumps({"id": gid, "model": model, "moves": d["moves"], "result": result,
                                "visit_counts": d["visit_counts"]}) + '\n')
        else:
            gid, model, moves, result, vc = row
            # legacy columns are JSON text already; splice without re-parsing
            f.write(f'{{"id":{gid},"model":{json.dumps(model)},"moves":{moves},"result":{result},"visit_counts":{vc}}}\n')
        n += 1
        if n % 50000 == 0:
            print(f'{n} games ({time.time() - t0:.0f}s)', file=sys.stderr, flush=True)
print(f'exported {n} games to {args.out} in {time.time() - t0:.0f}s', file=sys.stderr)
