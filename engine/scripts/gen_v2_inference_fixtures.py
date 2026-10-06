#!/usr/bin/env python3
"""Generate v2 (9-plane input, spatial policy head) inference fixtures for the webapp.

Replays real self-play games (jsonl or jsonl.gz, one {"moves": [...]} per line),
picks a mix of positions (player 0/1, mid pass-chain, forced-pass, last knight
destination set but no forced pass, ...) and records, for each:

  - the move prefix that reaches it from the start (so TS can replay it),
  - the state in the TS fixture format (bitboards as decimal strings),
  - Python's to_tensor(num_planes=9),
  - PyTorch log-policy (3137) and value.

Output feeds webapp/src/engine/__tests__/inference-v2.test.ts. The test looks
for <model>.onnx next to the other ONNX models (engine/output/models,
/tmp/models, ...); export it with scripts/export_onnx.py.

Usage (from engine/):
    python scripts/gen_v2_inference_fixtures.py \
        --checkpoint output/models/phoenix3_iter_235.pt \
        --games selfplay.jsonl.gz \
        --output ../webapp/src/engine/__tests__/inference-fixtures-v2.json
"""

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from razzle.ai.network import RazzleNet, spatial_action_index  # noqa: E402
from razzle.core.moves import MoveGenerator  # noqa: E402
from razzle.core.state import GameState  # noqa: E402


def open_games(path: str):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)['moves']


def ts_state(s: GameState) -> dict:
    return {
        'pieces': [str(s.pieces[0]), str(s.pieces[1])],
        'balls': [str(s.balls[0]), str(s.balls[1])],
        'currentPlayer': s.current_player,
        'touchedMask': str(s.touched_mask),
        'hasPassed': bool(s.has_passed),
        'lastKnightDst': int(s.last_knight_dst),
        'ply': int(s.ply),
    }


def categories(s: GameState) -> list[str]:
    cats = []
    if s.is_forced_pass():
        cats.append('forced_pass')
    elif not s.has_passed and MoveGenerator.must_pass(s):
        cats.append('must_pass_no_pass_available')   # plane 8 must stay zero
    elif s.last_knight_dst >= 0:
        cats.append('last_knight_dst')
    else:
        cats.append('no_last_knight_dst')             # plane 7 must stay zero
    if s.has_passed:
        cats.append('pass_chain')
    cats.append(f'player{s.current_player}')
    if s.ply <= 6:
        cats.append('opening')
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--games', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-games', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model = RazzleNet.load(args.checkpoint)
    model.eval()
    if model.config.num_input_planes != 9 or model.config.policy_head != 'spatial':
        sys.exit(f'not a v2 model: {model.config}')

    # Bucket every position of the first --max-games games by category.
    buckets: dict[str, list] = {}
    for gi, moves in enumerate(open_games(args.games)):
        if gi >= args.max_games:
            break
        s = GameState.new_game()
        for k, mv in enumerate(moves):
            if s.is_terminal():
                break
            for c in categories(s):
                buckets.setdefault(c, []).append((gi, moves[:k]))
            s.apply_move(mv)

    # Quotas: deliberately over-weight the v2-specific situations.
    quotas = [
        ('forced_pass', 5), ('must_pass_no_pass_available', 2), ('last_knight_dst', 3), ('no_last_knight_dst', 2),
        ('pass_chain', 4), ('player1', 2), ('player0', 1), ('opening', 1),
    ]
    picked: list[tuple[int, list]] = []
    seen = set()
    for cat, n in quotas:
        pool = buckets.get(cat, [])
        rng.shuffle(pool)
        got = 0
        for gi, prefix in pool:
            key = (gi, len(prefix))
            if key in seen:
                continue
            seen.add(key)
            picked.append((gi, prefix))
            got += 1
            if got == n:
                break
        print(f'{cat:30s} {got}/{n} (pool {len(pool)})')

    positions = []
    for gi, prefix in picked:
        s = GameState.new_game()
        for mv in prefix:
            s.apply_move(mv)
        t = s.to_tensor(num_planes=9)
        with torch.no_grad():
            logp, v, _ = model(torch.from_numpy(t).unsqueeze(0))
        positions.append({
            'game': gi,
            'moves': list(prefix),
            'state': ts_state(s),
            'categories': categories(s),
            'forcedPass': bool(s.is_forced_pass()),
            'tensor': t.tolist(),
            'policy': logp.squeeze(0).tolist(),
            'value': round(v.item(), 8),
        })

    out = {
        'model': Path(args.checkpoint).name,
        'num_input_planes': 9,
        'policy_head': 'spatial',
        'source_games': Path(args.games).name,
        'spatial_action_index': spatial_action_index().tolist(),
        'num_positions': len(positions),
        'positions': positions,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f'wrote {len(positions)} positions to {args.output}')


if __name__ == '__main__':
    main()
