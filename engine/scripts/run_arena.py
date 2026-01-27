#!/usr/bin/env python3
"""
Arena Tournament Runner - Measure model strength via ELO ratings.

Runs matches between models and computes ELO ratings based on results.
Uses the existing model_arena.run_match() function for game play.

Usage:
    # Compare every 10th iteration at multiple search depths
    python scripts/run_arena.py --stride 10 --games 20 --simulations 400 800 1600

    # Single depth (faster)
    python scripts/run_arena.py --stride 10 --games 20 --simulations 800

    # Compare specific models
    python scripts/run_arena.py --models initial iter_050 iter_099 --games 40

    # Just recompute ELO from existing matches
    python scripts/run_arena.py --mode compute-only

    # Clear all arena data and start fresh
    python scripts/run_arena.py --mode clear
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.model_arena import run_match, MatchResult
from razzle.training.elo import (
    compute_all_ratings,
    format_rating,
    get_iteration_from_version,
    EloRating,
)


# Default models directory
MODELS_DIR = Path(__file__).parent.parent / "output" / "models"


def discover_models(models_dir: Path = MODELS_DIR) -> list[str]:
    """
    Discover all model files in the models directory.

    Returns:
        List of model version strings sorted by iteration number.
    """
    models = []

    if not models_dir.exists():
        print(f"Warning: Models directory not found: {models_dir}")
        return models

    # Look for model files
    for model_path in models_dir.glob("*.pt"):
        version = model_path.stem  # Remove .pt extension
        models.append(version)

    # Sort by iteration number
    def sort_key(version: str) -> int:
        iteration = get_iteration_from_version(version)
        return iteration if iteration >= 0 else 999999

    models.sort(key=sort_key)
    return models


def filter_models_by_stride(models: list[str], stride: int) -> list[str]:
    """
    Filter models to include only every Nth iteration.

    Always includes 'initial' and the last model.

    Args:
        models: List of model versions
        stride: Include every Nth iteration

    Returns:
        Filtered list of model versions
    """
    if stride <= 1:
        return models

    filtered = []
    for version in models:
        iteration = get_iteration_from_version(version)
        if iteration == 0:  # Always include initial
            filtered.append(version)
        elif iteration > 0 and iteration % stride == 0:
            filtered.append(version)

    # Always include the last model if not already included
    if models and models[-1] not in filtered:
        filtered.append(models[-1])

    return filtered


def get_model_path(version: str, models_dir: Path = MODELS_DIR) -> Path:
    """Get the full path to a model file."""
    return models_dir / f"{version}.pt"


def match_exists(
    model1: str,
    model2: str,
    simulations: int,
    matches: list[dict]
) -> bool:
    """Check if a match between these models at this simulation count exists."""
    for match in matches:
        if match["simulations"] != simulations:
            continue
        if (match["model1_version"] == model1 and match["model2_version"] == model2) or \
           (match["model1_version"] == model2 and match["model2_version"] == model1):
            return True
    return False


def run_tournament(
    models: list[str],
    num_games: int,
    simulations_list: list[int],
    models_dir: Path = MODELS_DIR,
    api_url: Optional[str] = None,
    device: str = 'cuda',
) -> dict[int, list[dict]]:
    """
    Run tournament between models at each simulation count.

    Args:
        models: List of model versions to compare
        num_games: Number of games per match
        simulations_list: List of simulation counts to test
        models_dir: Directory containing model files
        api_url: Optional API URL to save results (None = local database)
        device: Device to use for inference

    Returns:
        Dictionary mapping simulation count to list of match results
    """
    import requests
    from server import persistence

    # Initialize database if using local storage
    if api_url is None:
        persistence.init_db()

    all_results: dict[int, list[dict]] = {s: [] for s in simulations_list}

    # Get existing matches to skip already-played ones
    existing_matches: dict[int, list[dict]] = {}
    for sims in simulations_list:
        if api_url:
            try:
                resp = requests.get(f"{api_url}/arena/matches", params={"simulations": sims})
                existing_matches[sims] = resp.json() if resp.ok else []
            except Exception:
                existing_matches[sims] = []
        else:
            existing_matches[sims] = persistence.get_all_arena_matches(simulations=sims)

    # Generate all pairs to play
    pairs = []
    for i, model1 in enumerate(models):
        for model2 in models[i + 1:]:
            pairs.append((model1, model2))

    total_matches = len(pairs) * len(simulations_list)
    completed = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"ARENA TOURNAMENT")
    print(f"{'='*60}")
    print(f"Models: {len(models)}")
    print(f"Matches to play: {len(pairs)} pairs x {len(simulations_list)} depths = {total_matches}")
    print(f"Games per match: {num_games}")
    print(f"Simulation counts: {simulations_list}")
    print(f"{'='*60}\n")

    for sims in simulations_list:
        print(f"\n--- Running matches at {sims} simulations ---\n")

        for model1, model2 in pairs:
            # Check if match already exists
            if match_exists(model1, model2, sims, existing_matches[sims]):
                print(f"  Skipping {model1} vs {model2} (already played)")
                skipped += 1
                continue

            # Get model paths
            path1 = get_model_path(model1, models_dir)
            path2 = get_model_path(model2, models_dir)

            if not path1.exists():
                print(f"  Warning: Model not found: {path1}")
                continue
            if not path2.exists():
                print(f"  Warning: Model not found: {path2}")
                continue

            print(f"  Playing: {model1} vs {model2} ({num_games} games, {sims} sims)...")

            try:
                result = run_match(
                    str(path1),
                    str(path2),
                    num_games=num_games,
                    simulations=sims,
                    device=device,
                    verbose=False
                )

                match_data = {
                    "model1_version": model1,
                    "model2_version": model2,
                    "model1_wins": result.model1_wins,
                    "model2_wins": result.model2_wins,
                    "draws": result.draws,
                    "simulations": sims,
                }

                # Save to database or API
                if api_url:
                    try:
                        resp = requests.post(f"{api_url}/arena/matches", json=match_data)
                        if not resp.ok:
                            print(f"    Warning: Failed to save to API: {resp.text}")
                    except Exception as e:
                        print(f"    Warning: Failed to save to API: {e}")
                else:
                    persistence.save_arena_match(
                        model1_version=model1,
                        model2_version=model2,
                        model1_wins=result.model1_wins,
                        model2_wins=result.model2_wins,
                        draws=result.draws,
                        simulations=sims,
                    )

                all_results[sims].append(match_data)

                # Print result
                print(f"    Result: {model1} {result.model1_wins} - {result.model2_wins} {model2} "
                      f"(draws: {result.draws})")

            except Exception as e:
                print(f"    Error: {e}")

            completed += 1

    print(f"\n{'='*60}")
    print(f"Tournament complete: {completed} matches played, {skipped} skipped")
    print(f"{'='*60}\n")

    return all_results


def compute_and_save_ratings(
    simulations: Optional[int] = None,
    api_url: Optional[str] = None,
    anchor_model: str = "initial",
) -> dict[int, dict[str, EloRating]]:
    """
    Compute ELO ratings from match history and save to database.

    Args:
        simulations: Compute ratings only for this simulation count (None = all)
        api_url: Optional API URL (None = local database)
        anchor_model: Model to anchor at 1000 ELO

    Returns:
        Dictionary mapping simulation count to model ratings
    """
    import requests
    from server import persistence

    if api_url is None:
        persistence.init_db()

    # Get all simulation counts
    if simulations is not None:
        sim_counts = [simulations]
    else:
        # Discover from matches
        if api_url:
            try:
                resp = requests.get(f"{api_url}/arena/matches")
                all_matches = resp.json() if resp.ok else []
            except Exception:
                all_matches = []
        else:
            all_matches = persistence.get_all_arena_matches()

        sim_counts = list(set(m["simulations"] for m in all_matches))
        sim_counts.sort()

    all_ratings: dict[int, dict[str, EloRating]] = {}

    for sims in sim_counts:
        # Get matches for this simulation count
        if api_url:
            try:
                resp = requests.get(f"{api_url}/arena/matches", params={"simulations": sims})
                matches = resp.json() if resp.ok else []
            except Exception:
                matches = []
        else:
            matches = persistence.get_all_arena_matches(simulations=sims)

        if not matches:
            continue

        # Compute ratings
        ratings = compute_all_ratings(matches, anchor_model=anchor_model)
        all_ratings[sims] = ratings

        # Save ratings to database
        for version, rating in ratings.items():
            iteration = get_iteration_from_version(version)
            if api_url:
                try:
                    data = {
                        "model_version": version,
                        "iteration": iteration,
                        "simulations": sims,
                        "elo_rating": rating.rating,
                        "games_played": rating.games_played,
                    }
                    requests.post(f"{api_url}/arena/ratings", json=data)
                except Exception:
                    pass
            else:
                persistence.save_arena_rating(
                    model_version=version,
                    iteration=iteration,
                    elo_rating=rating.rating,
                    games_played=rating.games_played,
                    simulations=sims,
                )

    return all_ratings


def print_ratings_table(ratings: dict[int, dict[str, EloRating]]):
    """Print a formatted table of ELO ratings."""
    for sims in sorted(ratings.keys()):
        sim_ratings = ratings[sims]
        if not sim_ratings:
            continue

        print(f"\n{'='*60}")
        print(f"ELO RATINGS at {sims} simulations")
        print(f"{'='*60}")
        print(f"{'Rank':<6} {'Model':<20} {'ELO':<10} {'Games':<10}")
        print(f"{'-'*6} {'-'*20} {'-'*10} {'-'*10}")

        # Sort by ELO rating
        sorted_models = sorted(
            sim_ratings.items(),
            key=lambda x: x[1].rating,
            reverse=True
        )

        for rank, (version, rating) in enumerate(sorted_models, 1):
            print(f"{rank:<6} {version:<20} {rating.rating:>8.1f}  {rating.games_played:>8}")

        print(f"{'='*60}")


def clear_arena_data(api_url: Optional[str] = None):
    """Clear all arena data."""
    import requests
    from server import persistence

    if api_url:
        try:
            resp = requests.delete(f"{api_url}/arena/clear")
            if resp.ok:
                result = resp.json()
                print(f"Cleared {result['matches_deleted']} matches, {result['ratings_deleted']} ratings")
            else:
                print(f"Error: {resp.text}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        persistence.init_db()
        result = persistence.clear_arena_data()
        print(f"Cleared {result['matches_deleted']} matches, {result['ratings_deleted']} ratings")


def main():
    parser = argparse.ArgumentParser(
        description='Run arena tournament to measure model strength via ELO ratings'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='run',
        choices=['run', 'compute-only', 'clear'],
        help='Mode: run (play matches), compute-only (just compute ELO), clear (delete all data)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific model versions to compare (e.g., initial iter_050 iter_099)'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Compare every Nth iteration (default: 10)'
    )

    parser.add_argument(
        '--games',
        type=int,
        default=20,
        help='Number of games per match (default: 20)'
    )

    parser.add_argument(
        '--simulations',
        type=int,
        nargs='+',
        default=[800],
        help='MCTS simulations per move (can specify multiple, default: 800)'
    )

    parser.add_argument(
        '--models-dir',
        type=str,
        default=str(MODELS_DIR),
        help=f'Directory containing model files (default: {MODELS_DIR})'
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default=None,
        help='API URL to save results (default: local database)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for inference (default: cuda)'
    )

    parser.add_argument(
        '--anchor',
        type=str,
        default='initial',
        help='Model to anchor at 1000 ELO (default: initial)'
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    # Check for CUDA
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    if args.mode == 'clear':
        print("Clearing all arena data...")
        clear_arena_data(api_url=args.api_url)
        return

    if args.mode == 'compute-only':
        print("Computing ELO ratings from existing matches...")
        ratings = compute_and_save_ratings(
            api_url=args.api_url,
            anchor_model=args.anchor,
        )
        print_ratings_table(ratings)
        return

    # Discover or use specified models
    if args.models:
        models = args.models
    else:
        models = discover_models(models_dir)
        models = filter_models_by_stride(models, args.stride)

    if len(models) < 2:
        print("Error: Need at least 2 models to compare")
        sys.exit(1)

    print(f"Models to compare: {models}")

    # Run tournament
    run_tournament(
        models=models,
        num_games=args.games,
        simulations_list=args.simulations,
        models_dir=models_dir,
        api_url=args.api_url,
        device=args.device,
    )

    # Compute and display ratings
    print("\nComputing ELO ratings...")
    ratings = compute_and_save_ratings(
        api_url=args.api_url,
        anchor_model=args.anchor,
    )
    print_ratings_table(ratings)


if __name__ == '__main__':
    main()
