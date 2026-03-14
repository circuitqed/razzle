#!/usr/bin/env python3
"""
Sims Scaling Tournament — measure how Elo scales with simulation count.

Plays the same model against itself at different sim counts to establish
a sims→Elo mapping. Results are posted to the arena API with a special
simulations value encoding (e.g. model "iter_780_s64" vs "iter_780_s256").

Usage:
    python sims_tournament.py \
        --api-url https://razzledazzle.lazybrains.com/api \
        --model iter_780 \
        --sim-levels 32 64 128 256 512 1024 2000 \
        --games 20 --device cuda
"""

import argparse
import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


def _api_headers() -> dict:
    headers = {}
    api_key = os.environ.get("TRAINING_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def download_model(api_url: str, version: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Model {version} already cached at {dest}")
        return dest

    url = f"{api_url}/training/models/{version}/download"
    print(f"Downloading {version} from {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    tmp.rename(dest)
    print(f"Saved {version} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


def post_result(api_url: str, model1: str, model2: str,
                m1_wins: int, m2_wins: int, draws: int, sims: int) -> bool:
    try:
        resp = requests.post(
            f"{api_url}/arena/matches",
            json={
                "model1_version": model1,
                "model2_version": model2,
                "model1_wins": m1_wins,
                "model2_wins": m2_wins,
                "draws": draws,
                "simulations": sims,
            },
            headers=_api_headers(),
            timeout=15,
        )
        return resp.ok
    except Exception as e:
        print(f"  POST failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Sims scaling tournament")
    parser.add_argument("--api-url", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        help="Model version to test (e.g. iter_780)")
    parser.add_argument("--sim-levels", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512, 1024, 2000],
                        help="Simulation counts to test")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per pair")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-dir", type=str, default="/workspace/models")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--opening-moves", type=int, default=3)
    args = parser.parse_args()

    os.environ["PYTHONUNBUFFERED"] = "1"

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Download model
    model_dir = Path(args.model_dir)
    model_path = model_dir / f"{args.model}.pt"
    download_model(args.api_url, args.model, model_path)

    from razzle.ai.network import RazzleNet
    from razzle.ai.mcts import MCTS, MCTSConfig
    from razzle.ai.evaluator import BatchedEvaluator
    from scripts.model_arena import play_game

    # Load model once
    print(f"Loading {args.model}...")
    net = RazzleNet.load(str(model_path), device=args.device)
    net.eval()
    evaluator = BatchedEvaluator(net, device=args.device)

    sim_levels = sorted(args.sim_levels)
    pairs = list(combinations(sim_levels, 2))
    random.shuffle(pairs)

    n_pairs = len(pairs)
    total_games = n_pairs * args.games
    print(f"\nModel: {args.model}")
    print(f"Sim levels: {sim_levels}")
    print(f"Pairs: {n_pairs}")
    print(f"Games per pair: {args.games}")
    print(f"Total games: {total_games}")
    print()

    # Use a unique simulations value so these don't mix with regular arena data
    # Convention: 9999 for sims-scaling tournament
    SIMS_TAG = 9999

    games_played = 0
    for idx, (s1, s2) in enumerate(pairs, 1):
        label1 = f"{args.model}_s{s1}"
        label2 = f"{args.model}_s{s2}"
        print(f"[{idx}/{n_pairs}] {label1} vs {label2}")

        config1 = MCTSConfig(num_simulations=s1, batch_size=min(args.batch_size, max(8, s1 // 8)))
        config2 = MCTSConfig(num_simulations=s2, batch_size=min(args.batch_size, max(8, s2 // 8)))

        wins1, wins2, draws = 0, 0, 0
        for g in range(args.games):
            # Alternate sides
            if g % 2 == 0:
                mcts_p0 = MCTS(evaluator, config1)
                mcts_p1 = MCTS(evaluator, config2)
                p0_is_s1 = True
            else:
                mcts_p0 = MCTS(evaluator, config2)
                mcts_p1 = MCTS(evaluator, config1)
                p0_is_s1 = False

            t0 = time.time()
            outcome = play_game(mcts_p0, mcts_p1,
                                opening_moves=args.opening_moves,
                                opening_temperature=1.0)
            elapsed = time.time() - t0

            # Convert to s1's perspective
            if not p0_is_s1:
                outcome = -outcome

            if outcome == 1:
                wins1 += 1
                winner = f"s{s1}"
            elif outcome == -1:
                wins2 += 1
                winner = f"s{s2}"
            else:
                draws += 1
                winner = "draw"

            games_played += 1
            print(f"  Game {g+1}/{args.games}: {winner} [{elapsed:.0f}s]")

        print(f"  Result: s{s1} {wins1}-{wins2}-{draws} s{s2}")

        # Post result
        post_result(args.api_url, label1, label2, wins1, wins2, draws, SIMS_TAG)

    # Trigger Elo compute
    print("\nTriggering Elo recompute...")
    try:
        resp = requests.post(f"{args.api_url}/arena/compute",
                             headers=_api_headers(), timeout=30)
        if resp.ok:
            data = resp.json()
            ratings = data.get("ratings", {}).get(str(SIMS_TAG), {})
            if ratings:
                print(f"\nSims scaling Elo ({args.model}):")
                sorted_ratings = sorted(ratings.items(),
                                        key=lambda x: x[1]["elo"], reverse=True)
                for name, info in sorted_ratings:
                    print(f"  {name}: {info['elo']:.0f} ({info['games']} games)")
    except Exception as e:
        print(f"Elo compute failed: {e}")

    print(f"\nDone: {games_played} games played across {n_pairs} pairs")


if __name__ == "__main__":
    main()
