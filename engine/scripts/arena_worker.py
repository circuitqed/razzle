#!/usr/bin/env python3
"""
Arena Worker for Vast.ai GPU instances.

Streams individual game results as they finish rather than waiting for a
full match to complete.  Multiple workers can run the same tournament
simultaneously — each game result is POSTed immediately, and workers
check the accumulated game count before starting each new game so they
naturally stop once enough games exist for a pair.

Usage:
    python arena_worker.py \
        --api-url https://razzledazzle.lazybrains.com/api \
        --models initial iter_020 iter_040 iter_060 iter_080 iter_100 \
        --games 20 --simulations 1000
"""

import argparse
import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def wait_for_api(api_url: str, timeout: int = 120):
    """Wait for API server to be reachable."""
    print(f"Waiting for API at {api_url} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{api_url}/health", timeout=5)
            if resp.ok:
                print("API is reachable.")
                return
        except Exception:
            pass
        time.sleep(5)
    print(f"WARNING: API not reachable after {timeout}s, proceeding anyway.")


def download_model(api_url: str, version: str, dest: Path) -> Path:
    """Download a model file from the API."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Model {version} already cached at {dest}")
        return dest

    url = f"{api_url}/training/models/{version}/download"
    print(f"  Downloading {version} from {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    tmp.rename(dest)
    print(f"  Saved {version} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


def get_match_game_count(
    api_url: str, model1: str, model2: str, simulations: int
) -> int:
    """Get total games played so far for a specific pair."""
    try:
        resp = requests.get(
            f"{api_url}/arena/matches",
            params={"simulations": simulations},
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            matches = data.get("matches", []) if isinstance(data, dict) else data
            for m in matches:
                if m.get("simulations") != simulations:
                    continue
                m1 = m.get("model1_version", "")
                m2 = m.get("model2_version", "")
                if (m1 == model1 and m2 == model2) or (m1 == model2 and m2 == model1):
                    return (
                        m.get("model1_wins", 0)
                        + m.get("model2_wins", 0)
                        + m.get("draws", 0)
                    )
    except Exception as e:
        print(f"    Warning: could not fetch match count: {e}")
    return 0


def post_game_result(api_url: str, match_data: dict) -> bool:
    """POST a single game result to the arena API."""
    try:
        resp = requests.post(
            f"{api_url}/arena/matches", json=match_data, timeout=15
        )
        if resp.ok:
            return True
        print(f"    Warning: POST failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        print(f"    Warning: POST failed: {e}")
    return False


def trigger_elo_compute(api_url: str):
    """Ask the server to recompute ELO ratings."""
    try:
        resp = requests.post(f"{api_url}/arena/compute", timeout=30)
        if resp.ok:
            data = resp.json()
            print(f"ELO recompute done: {data.get('status', '?')}")
        else:
            print(f"ELO recompute failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        print(f"ELO recompute failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Arena worker for Vast.ai GPU instances"
    )
    parser.add_argument(
        "--api-url", type=str, required=True,
        help="Base URL of the training/arena API",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Model versions to compare (e.g. initial iter_020 iter_040)",
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Target number of games per match (default: 20)",
    )
    parser.add_argument(
        "--simulations", type=int, default=1000,
        help="MCTS simulations per move (default: 1000)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--model-dir", type=str, default="/workspace/models",
        help="Directory to cache downloaded models",
    )
    args = parser.parse_args()

    # Ensure unbuffered output for log tailing
    os.environ["PYTHONUNBUFFERED"] = "1"

    import torch
    import numpy as np

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_dir = Path(args.model_dir)
    models = args.models
    pairs = list(combinations(models, 2))
    # Shuffle so multiple workers don't march through pairs in the same order
    random.shuffle(pairs)

    print(f"Models:      {models}")
    print(f"Pairs:       {len(pairs)}")
    print(f"Games/match: {args.games}")
    print(f"Sims/move:   {args.simulations}")
    print(f"Device:      {args.device}")
    print()

    # Wait for API
    wait_for_api(args.api_url)

    # Download all models up front
    print("Downloading models ...")
    model_paths: dict[str, Path] = {}
    for version in models:
        dest = model_dir / f"{version}.pt"
        download_model(args.api_url, version, dest)
        model_paths[version] = dest
    print()

    # Import game engine (after torch is confirmed working)
    from razzle.ai.network import RazzleNet
    from razzle.ai.mcts import MCTS, MCTSConfig
    from razzle.ai.evaluator import BatchedEvaluator
    from scripts.model_arena import play_game

    # Pre-load all models into memory
    print("Loading models into memory ...")
    loaded_nets: dict[str, RazzleNet] = {}
    for version, path in model_paths.items():
        net = RazzleNet.load(str(path), device=args.device)
        net.eval()
        loaded_nets[version] = net
        print(f"  Loaded {version}")
    print()

    config = MCTSConfig(num_simulations=args.simulations, batch_size=32)

    # Play all pairs, one game at a time, streaming results
    games_played = 0
    games_skipped = 0
    pairs_done = 0
    elo_trigger_counter = 0

    for idx, (m1, m2) in enumerate(pairs, 1):
        # Check how many games already exist for this pair
        existing_count = get_match_game_count(
            args.api_url, m1, m2, args.simulations
        )
        if existing_count >= args.games:
            print(f"[{idx}/{len(pairs)}] {m1} vs {m2} -- {existing_count}/{args.games} games, skipping")
            pairs_done += 1
            continue

        print(f"[{idx}/{len(pairs)}] {m1} vs {m2} ({existing_count}/{args.games} games so far)")

        # Create evaluators for this pair
        eval1 = BatchedEvaluator(loaded_nets[m1], device=args.device)
        eval2 = BatchedEvaluator(loaded_nets[m2], device=args.device)

        game_num = existing_count
        while game_num < args.games:
            # Alternate who plays as player 0
            m1_is_p0 = (game_num % 2 == 0)

            mcts_p0 = MCTS(eval1 if m1_is_p0 else eval2, config)
            mcts_p1 = MCTS(eval2 if m1_is_p0 else eval1, config)

            t0 = time.time()
            outcome = play_game(mcts_p0, mcts_p1, opening_moves=3, opening_temperature=1.0)
            elapsed = time.time() - t0

            # outcome is from p0's perspective: +1 p0 wins, -1 p1 wins, 0 draw
            # Convert to m1's perspective
            if not m1_is_p0:
                outcome = -outcome

            m1_wins = 1 if outcome == 1 else 0
            m2_wins = 1 if outcome == -1 else 0
            draws = 1 if outcome == 0 else 0

            winner_str = m1 if outcome == 1 else (m2 if outcome == -1 else "draw")
            game_num += 1
            print(f"    Game {game_num}/{args.games}: {winner_str} [{elapsed:.0f}s]")

            # POST immediately
            post_game_result(args.api_url, {
                "model1_version": m1,
                "model2_version": m2,
                "model1_wins": m1_wins,
                "model2_wins": m2_wins,
                "draws": draws,
                "simulations": args.simulations,
            })
            games_played += 1
            elo_trigger_counter += 1

            # Trigger ELO recompute every 10 games
            if elo_trigger_counter >= 10:
                trigger_elo_compute(args.api_url)
                elo_trigger_counter = 0

            # Re-check total count periodically (other workers may have contributed)
            if game_num % 5 == 0:
                actual_count = get_match_game_count(
                    args.api_url, m1, m2, args.simulations
                )
                if actual_count >= args.games:
                    print(f"    Pair complete ({actual_count} games total from all workers)")
                    break
                game_num = actual_count  # Sync with reality

        pairs_done += 1

    print()
    print(f"Done: {games_played} games played, {pairs_done}/{len(pairs)} pairs done")

    # Final ELO recompute
    if games_played > 0:
        print("Final ELO recompute ...")
        trigger_elo_compute(args.api_url)

    print("Arena worker finished.")


if __name__ == "__main__":
    main()
