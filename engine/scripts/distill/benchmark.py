#!/usr/bin/env python3
"""
Fixed-position benchmark for checkpoints: how well does each network predict a set of
held-out games' search moves and outcomes?

  python3 benchmark.py --games deep3200.jsonl.gz --skip 45000 --limit 600 --label deep a.pt b.pt

Prints, per model: agreement of the network's top move with the recorded search's top move
(positions with a policy target), policy cross-entropy against the search visits, EBF on
those positions, value MSE against the game outcome, and mean |value| (confidence).

Two sets are useful: games from much deeper searches than the candidates were trained on
(does the network move toward stronger play?) and broad games from another run (does it
forget general knowledge?).
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / 'scripts'))
import trainer as T  # noqa: E402
from razzle.ai.network import RazzleNet  # noqa: E402
from razzle.training.api_client import TrainingGame  # noqa: E402


def load(path, skip, limit):
    out = []
    for i, line in enumerate(gzip.open(path, 'rt')):
        if i < skip:
            continue
        if len(out) >= limit:
            break
        g = json.loads(line)
        out.append(TrainingGame(id=i, worker_id='x', moves=g['moves'], result=g['result'],
                                visit_counts=[{int(k): v for k, v in d.items()} for d in g['visit_counts']],
                                model_version=g.get('model'), created_at=''))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--games', required=True)
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--limit', type=int, default=600)
    ap.add_argument('--label', default='')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    games = load(args.games, args.skip, args.limit)
    st, po, va, le, _ = T.games_to_training_data(games, num_planes=9)
    pw = T.DistributedTrainer._policy_weights(po, le) > 0
    for m in args.models:
        net = RazzleNet.load(m, device=args.device).to(args.device).eval()
        lps, vs = [], []
        with torch.no_grad():
            for i in range(0, len(st), 8192):
                lp, v, _ = net(torch.from_numpy(st[i:i + 8192]).to(args.device))
                lps.append(lp.float().cpu().numpy()); vs.append(v.squeeze(1).float().cpu().numpy())
        lp = np.concatenate(lps); v = np.concatenate(vs)
        L, P = lp[pw], po[pw]
        ent = -(np.exp(L) * L * le[pw]).sum(1).mean()
        print(json.dumps(dict(
            label=args.label, model=Path(m).name, positions=int(len(st)), policy_positions=int(pw.sum()),
            move_match=round(float((L.argmax(1) == P.argmax(1)).mean()), 4),
            policy_ce=round(float(-(P * L).sum(1).mean()), 4), ebf=round(float(np.exp(ent)), 3),
            value_mse=round(float(np.mean((v - va) ** 2)), 4), mean_abs_v=round(float(np.abs(v).mean()), 4))),
            flush=True)


if __name__ == '__main__':
    main()
