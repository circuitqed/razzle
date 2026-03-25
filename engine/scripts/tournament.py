#!/usr/bin/env python3
"""
Round-robin tournament between model iterations to compute Elo ratings.

Downloads models from the training API and plays them against each other.
Each pair plays N games (half as each color) to control for first-mover advantage.

Usage:
    python scripts/tournament.py --api-url https://knightball.org/api --step 10 --games 20 --simulations 256
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator
from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.training.api_client import TrainingAPIClient
from razzle.training.elo import EloRating, expected_score, update_rating


def play_game(mcts1: MCTS, mcts2: MCTS, max_moves: int = 300) -> float:
    """Play a single game. Returns +1 if mcts1 wins, -1 if mcts2 wins, 0 for draw."""
    state = GameState.new_game()
    players = {0: mcts1, 1: mcts2}

    for _ in range(max_moves):
        if state.is_terminal():
            break

        mcts = players[state.current_player]
        legal = get_legal_moves(state)
        if not legal:
            break

        move = mcts.search(state)
        state.apply_move(move)

    winner = state.get_winner()
    if winner == 0:
        return 1.0
    elif winner == 1:
        return -1.0
    return 0.0


def run_match(
    model1_path: Path,
    model2_path: Path,
    num_games: int,
    simulations: int,
    device: str,
) -> dict:
    """Run a match between two models, alternating colors."""
    net1 = RazzleNet.load(model1_path, device='cpu')
    net2 = RazzleNet.load(model2_path, device='cpu')

    eval1 = BatchedEvaluator(net1, device=device, batch_size=32)
    eval2 = BatchedEvaluator(net2, device=device, batch_size=32)

    config = MCTSConfig(num_simulations=simulations, temperature=0.1)

    m1_wins = 0
    m2_wins = 0
    draws = 0

    for i in range(num_games):
        # Alternate colors
        if i % 2 == 0:
            mcts_p0 = MCTS(config, eval1)
            mcts_p1 = MCTS(config, eval2)
            result = play_game(mcts_p0, mcts_p1)
            if result > 0:
                m1_wins += 1
            elif result < 0:
                m2_wins += 1
            else:
                draws += 1
        else:
            mcts_p0 = MCTS(config, eval2)
            mcts_p1 = MCTS(config, eval1)
            result = play_game(mcts_p0, mcts_p1)
            if result > 0:
                m2_wins += 1
            elif result < 0:
                m1_wins += 1
            else:
                draws += 1

    return {'m1_wins': m1_wins, 'm2_wins': m2_wins, 'draws': draws}


def main():
    parser = argparse.ArgumentParser(description='Round-robin tournament for Elo ratings')
    parser.add_argument('--api-url', type=str, default='https://knightball.org/api')
    parser.add_argument('--step', type=int, default=10, help='Iteration step between models (default: 10)')
    parser.add_argument('--max-iter', type=int, default=200, help='Maximum iteration to include')
    parser.add_argument('--games', type=int, default=20, help='Games per matchup (default: 20)')
    parser.add_argument('--simulations', type=int, default=256, help='MCTS sims per move (default: 256)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--models-dir', type=Path, default=Path('/tmp/tournament_models'))
    parser.add_argument('--api-key', type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('TRAINING_API_KEY', '')
    if api_key:
        os.environ['TRAINING_API_KEY'] = api_key

    client = TrainingAPIClient(base_url=args.api_url)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    # Get list of available models
    import requests
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.get(f"{args.api_url}/training/models", params={"limit": 500}, headers=headers, timeout=30)
    resp.raise_for_status()
    all_models = resp.json().get('models', [])

    # Filter to every Nth iteration
    tournament_models = []
    for m in all_models:
        it = m['iteration']
        if it % args.step == 0 and it <= args.max_iter and it > 0:
            tournament_models.append(m)

    tournament_models.sort(key=lambda m: m['iteration'])
    print(f"Tournament models ({len(tournament_models)}):")
    for m in tournament_models:
        print(f"  {m['version']} (iter {m['iteration']})")

    if len(tournament_models) < 2:
        print("Need at least 2 models for a tournament")
        return

    # Download models
    print(f"\nDownloading models to {args.models_dir}...")
    model_paths = {}
    for m in tournament_models:
        version = m['version']
        path = args.models_dir / f"{version}.pt"
        if not path.exists():
            print(f"  Downloading {version}...")
            client.download_model(version, path)
        else:
            print(f"  {version} (cached)")
        model_paths[version] = path

    # Round-robin tournament
    versions = [m['version'] for m in tournament_models]
    pairs = list(itertools.combinations(versions, 2))
    print(f"\n{'='*60}")
    print(f"Round-robin: {len(pairs)} matchups, {args.games} games each")
    print(f"MCTS simulations: {args.simulations}")
    print(f"{'='*60}\n")

    results = {}
    for i, (v1, v2) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {v1} vs {v2}...", end=" ", flush=True)
        t0 = time.time()
        result = run_match(
            model_paths[v1], model_paths[v2],
            num_games=args.games,
            simulations=args.simulations,
            device=args.device,
        )
        elapsed = time.time() - t0
        results[(v1, v2)] = result
        print(f"{result['m1_wins']}-{result['m2_wins']}-{result['draws']} ({elapsed:.0f}s)")

    # Compute Elo ratings
    print(f"\n{'='*60}")
    print("Computing Elo ratings...")
    print(f"{'='*60}\n")

    ratings = {v: EloRating() for v in versions}

    # Multiple passes to stabilize ratings
    for _ in range(10):
        for (v1, v2), result in results.items():
            total = result['m1_wins'] + result['m2_wins'] + result['draws']
            if total == 0:
                continue
            score1 = (result['m1_wins'] + 0.5 * result['draws']) / total
            score2 = 1.0 - score1

            for _ in range(total):
                update_rating(ratings[v1], ratings[v2].rating, score1)
                update_rating(ratings[v2], ratings[v1].rating, score2)

    # Print results
    sorted_models = sorted(versions, key=lambda v: ratings[v].rating, reverse=True)

    print(f"{'Model':<25} {'Elo':>6} {'Games':>6}")
    print("-" * 40)
    for v in sorted_models:
        r = ratings[v]
        print(f"{v:<25} {r.rating:>6.0f} {r.games_played:>6}")

    # Print matchup matrix
    print(f"\n{'='*60}")
    print("Matchup details:")
    print(f"{'='*60}")
    for (v1, v2), result in sorted(results.items()):
        total = result['m1_wins'] + result['m2_wins'] + result['draws']
        wr = result['m1_wins'] / total * 100 if total > 0 else 0
        print(f"  {v1} vs {v2}: {result['m1_wins']}-{result['m2_wins']}-{result['draws']} ({wr:.0f}%)")


if __name__ == '__main__':
    main()
