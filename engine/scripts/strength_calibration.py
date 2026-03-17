#!/usr/bin/env python3
"""
Strength Calibration Tournament — find 5 evenly-spaced AI difficulty levels.

Tests candidate models at multiple simulation counts to build Elo curves.
Multiple workers can run simultaneously — each game result is POSTed
immediately and workers check accumulated counts to avoid duplicate work.

Usage:
    # Run full tournament on GPU (single worker)
    python scripts/strength_calibration.py \
        --api-url https://razzledazzle.lazybrains.com/api \
        --games 10 --device cuda

    # Run with multiple workers (each gets different matchup order)
    python scripts/strength_calibration.py \
        --api-url ... --worker-id 0 --num-workers 3
    python scripts/strength_calibration.py \
        --api-url ... --worker-id 1 --num-workers 3

    # Show current results only
    python scripts/strength_calibration.py \
        --api-url https://razzledazzle.lazybrains.com/api --results-only
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Configuration ---

CANDIDATE_MODELS = [
    "initial",
    "iter_010",
    "iter_035",
    "iter_075",
    "iter_150",
    "iter_300",
    "iter_500",
    "iter_858",
]

SIM_LEVELS = [1, 64, 256, 1024, 2048, 4096]

# Arena tag — keeps results separate from regular arena
SIMS_TAG = 9998

# ---


def _api_headers() -> dict:
    headers = {}
    api_key = os.environ.get("TRAINING_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def wait_for_api(api_url: str, timeout: int = 120):
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  {version} cached ({dest.stat().st_size / 1024:.0f} KB)")
        return dest

    url = f"{api_url}/training/models/{version}/download"
    print(f"  Downloading {version} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    tmp.rename(dest)
    print(f"  Saved {version} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


def get_match_game_count(api_url: str, m1: str, m2: str) -> int:
    """Check how many games have been played for this matchup."""
    try:
        resp = requests.get(
            f"{api_url}/arena/matches",
            params={"simulations": SIMS_TAG, "limit": 10000},
            headers=_api_headers(),
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            matches = data.get("matches", []) if isinstance(data, dict) else data
            for m in matches:
                if m.get("simulations") != SIMS_TAG:
                    continue
                a = m.get("model1_version", "")
                b = m.get("model2_version", "")
                if (a == m1 and b == m2) or (a == m2 and b == m1):
                    return (m.get("model1_wins", 0) +
                            m.get("model2_wins", 0) +
                            m.get("draws", 0))
    except Exception as e:
        print(f"    Warning: could not fetch match count: {e}")
    return 0


def post_game_result(api_url: str, m1: str, m2: str,
                     m1_wins: int, m2_wins: int, draws: int) -> bool:
    try:
        headers = _api_headers()
        resp = requests.post(
            f"{api_url}/arena/matches",
            json={
                "model1_version": m1,
                "model2_version": m2,
                "model1_wins": m1_wins,
                "model2_wins": m2_wins,
                "draws": draws,
                "simulations": SIMS_TAG,
            },
            headers=headers,
            timeout=15,
        )
        if not resp.ok:
            print(f"    POST failed ({resp.status_code}): {resp.text[:200]}")
        return resp.ok
    except Exception as e:
        print(f"    POST failed: {e}")
        return False


def trigger_elo_compute(api_url: str) -> dict:
    try:
        resp = requests.post(f"{api_url}/arena/compute",
                             headers=_api_headers(), timeout=60)
        if resp.ok:
            data = resp.json()
            return data.get("ratings", {}).get(str(SIMS_TAG), {})
    except Exception as e:
        print(f"Elo compute failed: {e}")
    return {}


def player_label(model: str, sims: int) -> str:
    return f"{model}_s{sims}"


def build_matchup_list(models: list[str], sim_levels: list[int]) -> list[tuple]:
    """Build the list of all matchups to play.

    Returns list of (model1, sims1, model2, sims2) tuples.
    Strategy:
      1) Within each model: adjacent sims levels
      2) Across adjacent models at each sims level
      3) Cross-calibration bridges (model[i] high sims vs model[i+1] low sims)
    """
    matchups_set = set()

    def add(m1, s1, m2, s2):
        key = tuple(sorted([(m1, s1), (m2, s2)]))
        matchups_set.add(key)

    # 1) Within-model sims scaling
    for model in models:
        for i in range(len(sim_levels) - 1):
            add(model, sim_levels[i], model, sim_levels[i + 1])

    # 2) Cross-model at each sim level (adjacent models)
    for sims in sim_levels:
        for i in range(len(models) - 1):
            add(models[i], sims, models[i + 1], sims)

    # 3) Cross-calibration bridges
    for i in range(len(models) - 1):
        # High sims of weaker model vs low sims of stronger model
        add(models[i], sim_levels[-1], models[i + 1], sim_levels[0])
        if len(sim_levels) >= 4:
            add(models[i], sim_levels[3], models[i + 1], sim_levels[1])
        # Also: model[i] at 2nd-highest vs model[i+1] at 256
        if len(sim_levels) >= 3:
            add(models[i], sim_levels[-2], models[i + 1], sim_levels[2])

    # 4) Round-robin at 256 sims (primary ranking sim level)
    from itertools import combinations
    ref_sims = 256
    if ref_sims in sim_levels:
        for m1, m2 in combinations(models, 2):
            add(m1, ref_sims, m2, ref_sims)

    # Convert to list
    matchups = []
    for (m1, s1), (m2, s2) in matchups_set:
        matchups.append((m1, s1, m2, s2))

    return matchups


def show_results(api_url: str, models: list[str]):
    """Display current Elo ratings in a model × sims table."""
    print("\n" + "=" * 70)
    print(f"STRENGTH CALIBRATION RESULTS (arena tag={SIMS_TAG})")
    print("=" * 70)

    ratings = trigger_elo_compute(api_url)
    if not ratings:
        print("No ratings available yet.")
        return

    # Group by model
    model_ratings: dict[str, dict[int, tuple]] = {}  # model -> {sims: (elo, games)}
    for label, info in ratings.items():
        parts = label.rsplit("_s", 1)
        if len(parts) == 2:
            model = parts[0]
            try:
                sims = int(parts[1])
            except ValueError:
                continue
            if model not in model_ratings:
                model_ratings[model] = {}
            model_ratings[model][sims] = (info["elo"], info.get("games", 0))

    sim_levels = sorted(set(s for r in model_ratings.values() for s in r))
    if not sim_levels:
        print("No sim levels found in results.")
        return

    # Print table
    header = f"{'Model':<15}" + "".join(f"{'s'+str(s):>9}" for s in sim_levels)
    print(header)
    print("-" * len(header))

    for model in models:
        if model in model_ratings:
            row = f"{model:<15}"
            for s in sim_levels:
                data = model_ratings[model].get(s)
                if data:
                    elo, games = data
                    row += f"{elo:>8.0f}g" if games < 10 else f"{elo:>9.0f}"
                else:
                    row += f"{'—':>9}"
            print(row)

    # Collect all players with enough games
    all_players = []
    for model, sims_dict in model_ratings.items():
        for sims, (elo, games) in sims_dict.items():
            all_players.append((model, sims, elo, games))

    all_players.sort(key=lambda x: x[2])

    if len(all_players) < 5:
        print(f"\nOnly {len(all_players)} players rated so far. Need more games.")
        return

    min_elo = all_players[0][2]
    max_elo = all_players[-1][2]
    spread = max_elo - min_elo

    print(f"\nElo range: {min_elo:.0f} — {max_elo:.0f} (spread: {spread:.0f})")
    print(f"Total rated players: {len(all_players)}")

    # Find 5 evenly spaced target Elos
    if spread > 0:
        targets = [min_elo + i * spread / 4 for i in range(5)]
        print(f"\nSuggested 5 difficulty levels (evenly spaced):")
        print(f"{'Level':<8} {'Model':<15} {'Sims':>6} {'Elo':>7} {'Target':>7} {'Games':>6}")
        print("-" * 55)
        used = set()
        for i, target in enumerate(targets, 1):
            # Find closest unused player
            best = None
            best_dist = float('inf')
            for p in all_players:
                key = (p[0], p[1])
                if key not in used:
                    dist = abs(p[2] - target)
                    if dist < best_dist:
                        best_dist = dist
                        best = p
            if best:
                used.add((best[0], best[1]))
                print(f"  {i:<6} {best[0]:<15} {best[1]:>6} {best[2]:>7.0f} {target:>7.0f} {best[3]:>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Strength calibration tournament for AI difficulty levels")
    parser.add_argument("--api-url", type=str, required=True)
    parser.add_argument("--games", type=int, default=10,
                        help="Target games per matchup (default: 10)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-dir", type=str, default="/workspace/models",
                        help="Directory to cache downloaded models")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Override candidate models")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker ID for multi-worker partitioning")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of workers")
    parser.add_argument("--results-only", action="store_true",
                        help="Just show current results, don't play games")
    parser.add_argument("--opening-moves", type=int, default=3)
    args = parser.parse_args()

    os.environ["PYTHONUNBUFFERED"] = "1"

    models = args.models or CANDIDATE_MODELS

    if args.results_only:
        show_results(args.api_url, models)
        return

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build matchup list
    matchups = build_matchup_list(models, SIM_LEVELS)

    # Partition matchups across workers (deterministic)
    matchups.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    my_matchups = [m for i, m in enumerate(matchups)
                   if i % args.num_workers == args.worker_id]

    # Shuffle for better load distribution (seed with worker_id for determinism)
    rng = random.Random(42 + args.worker_id)
    rng.shuffle(my_matchups)

    print(f"Worker {args.worker_id}/{args.num_workers}")
    print(f"Models: {models}")
    print(f"Sim levels: {SIM_LEVELS}")
    print(f"Total matchups: {len(matchups)}, mine: {len(my_matchups)}")
    print(f"Games per matchup: {args.games}")
    print(f"Device: {args.device}")
    print(f"Arena tag: {SIMS_TAG}")
    print()

    wait_for_api(args.api_url)

    # Download all models
    model_dir = Path(args.model_dir)
    print("Downloading models ...")
    for model in models:
        download_model(args.api_url, model, model_dir / f"{model}.pt")
    print()

    # Load all models
    from razzle.ai.network import RazzleNet
    from razzle.ai.mcts import MCTS, MCTSConfig
    from razzle.ai.evaluator import BatchedEvaluator
    from scripts.model_arena import play_game

    print("Loading models ...")
    evaluators = {}
    for model in models:
        model_path = model_dir / f"{model}.pt"
        net = RazzleNet.load(str(model_path), device=args.device)
        net.eval()
        evaluators[model] = BatchedEvaluator(net, device=args.device)
        print(f"  Loaded {model}")
    print()

    # Play matchups
    total_games = 0
    elo_counter = 0

    for idx, (m1, s1, m2, s2) in enumerate(my_matchups, 1):
        label1 = player_label(m1, s1)
        label2 = player_label(m2, s2)

        # Check how many games already exist
        existing = get_match_game_count(args.api_url, label1, label2)
        if existing >= args.games:
            print(f"[{idx}/{len(my_matchups)}] {label1} vs {label2} — "
                  f"{existing}/{args.games} games, skipping")
            continue

        print(f"\n[{idx}/{len(my_matchups)}] {label1} vs {label2} "
              f"({existing}/{args.games} games so far)")

        config1 = MCTSConfig(
            num_simulations=s1,
            batch_size=min(32, max(1, s1 // 4)) if s1 > 1 else 1,
        )
        config2 = MCTSConfig(
            num_simulations=s2,
            batch_size=min(32, max(1, s2 // 4)) if s2 > 1 else 1,
        )

        game_num = existing
        while game_num < args.games:
            # Alternate sides
            m1_is_p0 = (game_num % 2 == 0)

            if m1_is_p0:
                mcts_p0 = MCTS(evaluators[m1], config1)
                mcts_p1 = MCTS(evaluators[m2], config2)
            else:
                mcts_p0 = MCTS(evaluators[m2], config2)
                mcts_p1 = MCTS(evaluators[m1], config1)

            t0 = time.time()
            outcome = play_game(mcts_p0, mcts_p1,
                                opening_moves=args.opening_moves,
                                opening_temperature=1.0)
            elapsed = time.time() - t0

            if not m1_is_p0:
                outcome = -outcome

            m1w = 1 if outcome == 1 else 0
            m2w = 1 if outcome == -1 else 0
            d = 1 if outcome == 0 else 0
            tag = "W" if outcome == 1 else ("L" if outcome == -1 else "D")

            game_num += 1
            total_games += 1
            elo_counter += 1
            print(f"    Game {game_num}/{args.games}: {tag} [{elapsed:.1f}s]")

            post_game_result(args.api_url, label1, label2, m1w, m2w, d)

            # Trigger Elo recompute periodically
            if elo_counter >= 20:
                print("    (recomputing Elo...)")
                trigger_elo_compute(args.api_url)
                elo_counter = 0

            # Re-check count every 5 games (other workers may contribute)
            if game_num % 5 == 0 and args.num_workers > 1:
                actual = get_match_game_count(args.api_url, label1, label2)
                if actual >= args.games:
                    print(f"    Pair complete ({actual} games from all workers)")
                    break
                game_num = max(game_num, actual)

    print(f"\nWorker {args.worker_id} done: {total_games} games played")

    # Final Elo compute and results
    if total_games > 0:
        show_results(args.api_url, models)


if __name__ == "__main__":
    main()
