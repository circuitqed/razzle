#!/usr/bin/env python3
"""
Replay exported self-play games into compact training shards for distillation.

Input: gzip JSONL from export_games.py.
Output: shard_XXXX.npz files, each with (N positions):
  packed      uint8  [N, 49]   to_tensor() planes (7x8x7 = 392 bits), np.packbits
  pol_ptr     int64  [N+1]     CSR offsets into pol_idx / pol_p
  pol_idx     int16  [nnz]     MCTS visit targets, side-to-move orientation
  pol_p       float16[nnz]
  leg_ptr     int64  [N+1]     CSR offsets into leg_idx
  leg_idx     int16  [nnz]     legal moves, side-to-move orientation
  z           int8   [N]       game outcome from the side to move (+1/-1, 0 = no result)
  flags       uint8  [N]       bit0 = uniform target (random-opening move),
                               bit1 = mid-pass position,
                               bit2 = forced pass (side to move must pass)
  lkd         int8   [N]       opponent's last knight destination square in
                               side-to-move orientation (-1 = none); with bit2
                               this rebuilds the v2 input planes 7 and 8
  visits      int16  [N]       total root visits of the recorded search
  game        int32  [N]       game id (for a by-game validation split)

Orientation matches the trainer (scripts/trainer.py games_to_training_data):
to_tensor() rotates the board 180 degrees for player 1, so player-1 policy and
legal indices are rotated with MOVE_ROTATION_MAP too.

Usage:
  python3 build_dataset.py games.jsonl.gz OUT_DIR [--workers N] [--games-per-shard 20000]
"""
import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from razzle.core.state import GameState  # noqa: E402
from razzle.core.moves import get_legal_moves  # noqa: E402
from razzle.core.symmetry import MOVE_ROTATION_MAP, rotate_square_180  # noqa: E402

END_TURN_ACTION = 3136


def idx_of(move: int) -> int:
    return END_TURN_ACTION if move == -1 else move


def replay_games(lines: list[str]) -> dict:
    packed, z, flags, visits, game, lkd = [], [], [], [], [], []
    pol_idx, pol_p, pol_len = [], [], []
    leg_idx, leg_len = [], []

    for line in lines:
        g = json.loads(line)
        moves, vcs, result = g['moves'], g['visit_counts'], g['result']
        state = GameState.new_game()
        for move, vc in zip(moves, vcs):
            if vc:
                p = state.current_player
                rot = (lambda i: int(MOVE_ROTATION_MAP[i])) if p == 1 else (lambda i: i)
                vals = list(vc.values())
                total = sum(vals)
                if total > 0:
                    packed.append(np.packbits(state.to_tensor().astype(bool).reshape(-1)))
                    keys = [rot(idx_of(int(m))) for m in vc.keys()]
                    pol_idx.extend(keys)
                    pol_p.extend(v / total for v in vals)
                    pol_len.append(len(keys))
                    legal = [rot(idx_of(m)) for m in get_legal_moves(state)]
                    leg_idx.extend(legal)
                    leg_len.append(len(legal))
                    z.append(0 if result == 0 else (result if p == 0 else -result))
                    uniform = len(vals) > 3 and len(set(vals)) == 1
                    flags.append((1 if uniform else 0) | (2 if state.has_passed else 0)
                                 | (4 if state.is_forced_pass() else 0))
                    sq = state.last_knight_dst
                    lkd.append(-1 if sq < 0 else (rotate_square_180(sq) if p == 1 else sq))
                    visits.append(min(total, 32767))
                    game.append(g['id'])
            state.apply_move(move)

    def ptr(lengths):
        out = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=out[1:])
        return out

    return dict(
        packed=np.stack(packed) if packed else np.zeros((0, 49), np.uint8),
        pol_ptr=ptr(pol_len), pol_idx=np.array(pol_idx, np.int16), pol_p=np.array(pol_p, np.float16),
        leg_ptr=ptr(leg_len), leg_idx=np.array(leg_idx, np.int16),
        z=np.array(z, np.int8), flags=np.array(flags, np.uint8),
        visits=np.array(visits, np.int16), game=np.array(game, np.int32),
        lkd=np.array(lkd, np.int8),
    )


def work(job):
    shard_no, lines, out_dir = job
    data = replay_games(lines)
    path = Path(out_dir) / f'shard_{shard_no:04d}.npz'
    np.savez(path, **data)
    return shard_no, len(lines), len(data['z'])


def chunks(path, size, out_dir):
    buf, n = [], 0
    with gzip.open(path, 'rt') as f:
        for line in f:
            buf.append(line)
            if len(buf) == size:
                yield (n, buf, out_dir)
                buf, n = [], n + 1
    if buf:
        yield (n, buf, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('games')
    ap.add_argument('out_dir')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)))
    ap.add_argument('--games-per-shard', type=int, default=20000)
    ap.add_argument('--limit-shards', type=int, default=0, help='stop after N shards (testing)')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    games = positions = 0
    jobs = chunks(args.games, args.games_per_shard, args.out_dir)
    if args.limit_shards:
        jobs = (j for j in jobs if j[0] < args.limit_shards)
    with Pool(args.workers) as pool:
        for shard_no, ng, npos in pool.imap_unordered(work, jobs):
            games += ng
            positions += npos
            print(f'shard {shard_no}: {ng} games, {npos} positions '
                  f'(total {games} games / {positions} positions, {time.time() - t0:.0f}s)', flush=True)
    print(f'done: {games} games, {positions} positions in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
