#!/usr/bin/env python3
"""
Offline replay of a self-play run's trainer: start from a model, feed stored
games in order through the same conversion / compact window / train_steps code
the distributed trainer uses, and save checkpoints. Lets you test trainer
settings (learning rate, reuse, window seeding) on games you already have,
without new self-play.

Example:
  python3 replay_train.py --games phoenix_canary.jsonl.gz --model phoenix_iter_000.pt \
      --lr 2e-4 --out runs/replay_lr2e-4 --batch-games 512 --reuse 2
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / 'scripts'))
from razzle.ai.network import RazzleNet  # noqa: E402
from razzle.training.api_client import TrainingGame  # noqa: E402
from razzle.training.compact_buffer import CompactReplayBuffer  # noqa: E402
from razzle.training.trainer import Trainer as NetworkTrainer, TrainingConfig  # noqa: E402
import trainer as T  # noqa: E402  (scripts/trainer.py: games_to_training_data, _policy_weights)


def load_games(path):
    out = []
    for i, line in enumerate(gzip.open(path, 'rt')):
        g = json.loads(line)
        out.append(TrainingGame(id=g.get('id', i), worker_id='replay', moves=g['moves'], result=g['result'],
                                visit_counts=[{int(k): v for k, v in d.items()} for d in g['visit_counts']],
                                model_version=g.get('model'), created_at=''))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-games', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--reuse', type=float, default=2.0)
    ap.add_argument('--value-weight', type=float, default=1.5)
    ap.add_argument('--window-min', type=int, default=250_000)
    ap.add_argument('--checkpoints', type=int, default=4, help='save this many evenly spaced checkpoints')
    ap.add_argument('--ema', type=float, default=0.0,
                    help='also keep an exponential moving average of the weights (e.g. 0.999 per step); '
                         'checkpoints then save the averaged weights')
    ap.add_argument('--max-games', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8, help='processes converting games (trainer code)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    net = RazzleNet.load(args.model, device=args.device).to(args.device)
    planes = net.config.num_input_planes
    trainer = NetworkTrainer(net, TrainingConfig(batch_size=args.batch_size, learning_rate=args.lr,
                                                 device=args.device, value_weight=args.value_weight,
                                                 value_weight_quartic=0.0))
    buf = CompactReplayBuffer(min_positions=args.window_min)
    rng = np.random.default_rng(0)

    games = load_games(args.games)
    if args.max_games:
        games = games[:args.max_games]
    ema = None
    if args.ema > 0:
        import copy
        ema = copy.deepcopy(net).eval()
        for q in ema.parameters():
            q.requires_grad_(False)

    def ema_update():
        with torch.no_grad():
            for e, q in zip(ema.parameters(), net.parameters()):
                e.mul_(args.ema).add_(q.detach(), alpha=1 - args.ema)
            for e, q in zip(ema.buffers(), net.buffers()):
                e.copy_(q)
    chunks = [games[i:i + args.batch_games] for i in range(0, len(games), args.batch_games)]
    save_at = {round(len(chunks) * (k + 1) / args.checkpoints) - 1 for k in range(args.checkpoints)}
    t0 = time.time()
    log = open(out / 'log.jsonl', 'w')
    import multiprocessing as mp
    pool = mp.get_context('spawn').Pool(args.workers) if args.workers > 1 else None
    for it, chunk in enumerate(chunks):
        k = max(1, args.workers)
        jobs = [(chunk[i::k], planes, 1, i) for i in range(k)]      # same conversion as the trainer
        results = pool.map(T._convert_games_compact, jobs) if pool else [T._convert_games_compact(jobs[0])]
        n_new = 0
        for r in results:
            buf.add_chunk(r[0])
            n_new += r[2]
        steps = max(1, int(np.ceil(n_new * args.reuse / args.batch_size)))
        if ema is None:
            m = trainer.train_steps(lambda: buf.sample(args.batch_size, rng), steps, verbose=False)
        else:   # one step at a time so the average updates after every step
            ms = []
            for _ in range(steps):
                ms.append(trainer.train_steps(lambda: buf.sample(args.batch_size, rng), 1, verbose=False))
                ema_update()
            m = {k: float(np.mean([x[k] for x in ms])) for k in ('loss', 'policy_loss', 'value_loss')}
        m.update(iteration=it + 1, games=len(chunk), positions=n_new, window=len(buf))
        log.write(json.dumps(m) + '\n')
        log.flush()
        print(f"iter {it + 1}/{len(chunks)}: steps {steps} loss {m['loss']:.4f} pol {m['policy_loss']:.4f} "
              f"val {m['value_loss']:.4f} window {len(buf):,} ({time.time() - t0:.0f}s)", flush=True)
        if it in save_at:
            (ema or net).save(str(out / f'replay_iter_{it + 1:03d}.pt'))
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
