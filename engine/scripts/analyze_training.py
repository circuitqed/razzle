#!/usr/bin/env python3
"""
Training Analysis Script for Razzle Dazzle.

Computes various metrics to evaluate training progress:
- Policy quality metrics (accuracy, entropy, top-k)
- Value prediction accuracy
- Game statistics (length, diversity, win rates)
- Elo estimation between model versions

Usage:
    python analyze_training.py --api-url https://razzledazzle.lazybrains.com/api
    python analyze_training.py --model output/trainer_local/models/iter_020.pt --games 100
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.network import RazzleNet, NUM_ACTIONS, END_TURN_ACTION


@dataclass
class GameStats:
    """Statistics for a single game."""
    length: int
    result: float  # 1.0 = P0 wins, -1.0 = P1 wins, 0.0 = draw
    opening_moves: tuple  # First N moves for diversity analysis
    avg_policy_entropy: float
    avg_top1_confidence: float
    model_version: str


@dataclass
class TrainingMetrics:
    """Aggregated training metrics."""
    # Game statistics
    total_games: int
    avg_game_length: float
    std_game_length: float
    min_game_length: int
    max_game_length: int

    # Win rates
    p0_win_rate: float
    p1_win_rate: float
    draw_rate: float

    # Policy quality (from MCTS visit counts)
    avg_policy_entropy: float
    avg_top1_confidence: float
    avg_top3_concentration: float  # How much probability in top 3 moves

    # Opening diversity
    unique_openings: int
    most_common_opening: tuple
    opening_concentration: float  # % of games with most common opening

    # Per-model breakdown
    games_per_model: dict


def entropy(probs: np.ndarray) -> float:
    """Compute entropy of a probability distribution."""
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log(probs))


def visit_counts_to_policy(visit_counts: dict) -> np.ndarray:
    """Convert sparse visit counts to dense policy array."""
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    total = sum(visit_counts.values())
    if total > 0:
        for move_str, count in visit_counts.items():
            # Keys might be strings from JSON
            move = int(move_str)
            # Handle END_TURN move (-1)
            idx = END_TURN_ACTION if move == -1 else move
            if 0 <= idx < NUM_ACTIONS:
                policy[idx] = count / total
    return policy


def analyze_game(moves: list, visit_counts: list, result: float, model_version: str) -> GameStats:
    """Analyze a single game."""
    entropies = []
    top1_confidences = []

    for vc in visit_counts:
        if not vc:
            continue
        policy = visit_counts_to_policy(vc)
        if policy.sum() > 0:
            entropies.append(entropy(policy))
            top1_confidences.append(policy.max())

    # Opening = first 4 moves (or fewer if game is short)
    opening = tuple(moves[:4]) if len(moves) >= 4 else tuple(moves)

    return GameStats(
        length=len(moves),
        result=result,
        opening_moves=opening,
        avg_policy_entropy=np.mean(entropies) if entropies else 0.0,
        avg_top1_confidence=np.mean(top1_confidences) if top1_confidences else 0.0,
        model_version=model_version,
    )


def compute_metrics(games: list[dict]) -> TrainingMetrics:
    """Compute aggregate metrics from a list of games."""
    if not games:
        raise ValueError("No games to analyze")

    game_stats = []
    for g in games:
        stats = analyze_game(
            g['moves'],
            g['visit_counts'],
            g['result'],
            g.get('model_version', 'unknown'),
        )
        game_stats.append(stats)

    lengths = [s.length for s in game_stats]
    results = [s.result for s in game_stats]

    # Opening analysis
    opening_counts = Counter(s.opening_moves for s in game_stats)
    most_common = opening_counts.most_common(1)[0] if opening_counts else ((), 0)

    # Per-model breakdown
    games_per_model = Counter(s.model_version for s in game_stats)

    return TrainingMetrics(
        total_games=len(game_stats),
        avg_game_length=np.mean(lengths),
        std_game_length=np.std(lengths),
        min_game_length=min(lengths),
        max_game_length=max(lengths),

        p0_win_rate=sum(1 for r in results if r > 0) / len(results),
        p1_win_rate=sum(1 for r in results if r < 0) / len(results),
        draw_rate=sum(1 for r in results if r == 0) / len(results),

        avg_policy_entropy=np.mean([s.avg_policy_entropy for s in game_stats]),
        avg_top1_confidence=np.mean([s.avg_top1_confidence for s in game_stats]),
        avg_top3_concentration=0.0,  # TODO: compute from visit counts

        unique_openings=len(opening_counts),
        most_common_opening=most_common[0],
        opening_concentration=most_common[1] / len(game_stats) if game_stats else 0,

        games_per_model=dict(games_per_model),
    )


def get_model_device(model: RazzleNet):
    """Get the device the model is on."""
    return next(model.parameters()).device


def analyze_model_policy_accuracy(
    model: RazzleNet,
    games: list[dict],
    max_positions: int = 1000,
) -> dict:
    """
    Analyze how well a model's policy matches MCTS visit counts.

    Returns dict with:
    - top1_accuracy: How often model's top move matches MCTS top move
    - top3_accuracy: How often MCTS move is in model's top 3
    - top5_accuracy: How often MCTS move is in model's top 5
    - avg_kl_divergence: Average KL divergence between model and MCTS policies
    """
    import torch

    device = get_model_device(model)
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    kl_divergences = []
    total_positions = 0

    for game in games:
        state = GameState.new_game()

        for move, visit_counts in zip(game['moves'], game['visit_counts']):
            if total_positions >= max_positions:
                break

            if not visit_counts:
                state.apply_move(move)
                continue

            # Get MCTS policy from visit counts
            mcts_policy = visit_counts_to_policy(visit_counts)
            mcts_top_move = np.argmax(mcts_policy)

            # Get model's policy
            state_tensor = torch.tensor(state.to_tensor(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy_logits, _ = model(state_tensor.to(device))
                model_policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

            # Top-k accuracy
            model_top_moves = np.argsort(model_policy)[::-1]
            if model_top_moves[0] == mcts_top_move:
                top1_correct += 1
            if mcts_top_move in model_top_moves[:3]:
                top3_correct += 1
            if mcts_top_move in model_top_moves[:5]:
                top5_correct += 1

            # KL divergence (MCTS || Model)
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            mcts_p = mcts_policy + eps
            model_p = model_policy + eps
            mcts_p /= mcts_p.sum()
            model_p /= model_p.sum()
            kl = np.sum(mcts_p * np.log(mcts_p / model_p))
            kl_divergences.append(kl)

            total_positions += 1
            state.apply_move(move)

        if total_positions >= max_positions:
            break

    return {
        'total_positions': total_positions,
        'top1_accuracy': top1_correct / total_positions if total_positions > 0 else 0,
        'top3_accuracy': top3_correct / total_positions if total_positions > 0 else 0,
        'top5_accuracy': top5_correct / total_positions if total_positions > 0 else 0,
        'avg_kl_divergence': np.mean(kl_divergences) if kl_divergences else 0,
    }


def analyze_value_accuracy(
    model: RazzleNet,
    games: list[dict],
    max_positions: int = 1000,
) -> dict:
    """
    Analyze how well model's value predictions match game outcomes.

    Returns dict with:
    - correlation: Correlation between predicted values and actual outcomes
    - mean_absolute_error: Average |predicted - actual|
    - calibration: Binned accuracy (when model predicts 0.7, does player win 70%?)
    """
    import torch

    device = get_model_device(model)
    predictions = []
    actuals = []

    for game in games:
        if len(predictions) >= max_positions:
            break

        state = GameState.new_game()
        result = game['result']

        for i, move in enumerate(game['moves']):
            if len(predictions) >= max_positions:
                break

            # Get model's value prediction
            state_tensor = torch.tensor(state.to_tensor(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, value = model(state_tensor.to(device))
                pred_value = value.item()

            # Actual value from perspective of current player
            current_player = state.current_player
            if current_player == 0:
                actual_value = result
            else:
                actual_value = -result

            predictions.append(pred_value)
            actuals.append(actual_value)

            state.apply_move(move)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Correlation
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, actuals)[0, 1]
    else:
        correlation = 0.0

    # Mean absolute error
    mae = np.mean(np.abs(predictions - actuals))

    # Calibration: bin predictions and check actual win rates
    calibration = {}
    bins = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]
    for low, high in bins:
        mask = (predictions >= low) & (predictions < high)
        if mask.sum() > 0:
            # Convert actuals to win probability (1 for win, 0 for loss, 0.5 for draw)
            win_probs = (actuals[mask] + 1) / 2
            calibration[f'{low:.1f} to {high:.1f}'] = {
                'count': int(mask.sum()),
                'predicted_avg': float(predictions[mask].mean()),
                'actual_win_rate': float(win_probs.mean()),
            }

    return {
        'total_positions': len(predictions),
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'mean_absolute_error': float(mae),
        'calibration': calibration,
    }


def analyze_move_diversity(games: list[dict], position_depth: int = 6) -> dict:
    """
    Analyze diversity of moves at each position.

    Returns move frequency distribution for first N positions.
    """
    move_counts = defaultdict(Counter)

    for game in games:
        for i, move in enumerate(game['moves'][:position_depth]):
            move_counts[i][move] += 1

    diversity = {}
    for pos, counts in sorted(move_counts.items()):
        total = sum(counts.values())
        top_moves = counts.most_common(5)
        diversity[f'position_{pos}'] = {
            'unique_moves': len(counts),
            'total_games': total,
            'top_moves': [(m, c, c/total) for m, c, in top_moves],
            'entropy': entropy(np.array(list(counts.values())) / total),
        }

    return diversity


def fetch_games_from_api(api_url: str, limit: int = 500) -> list[dict]:
    """Fetch all games from the training API (both pending and used)."""
    import requests

    # Try the new /all endpoint first
    try:
        response = requests.get(
            f"{api_url}/training/games/all",
            params={'limit': limit},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get('games', [])
    except Exception:
        pass

    # Fall back to pending games endpoint (won't mark as used if mark_used=false)
    try:
        response = requests.get(
            f"{api_url}/training/games",
            params={'limit': limit, 'mark_used': 'false'},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get('games', [])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch games: {e}")


def load_games_from_file(filepath: Path) -> list[dict]:
    """Load games from a JSON or JSONL file."""
    filepath = Path(filepath)

    # JSONL format (one JSON object per line)
    if filepath.suffix == '.jsonl':
        games = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
        return games

    # Regular JSON format
    with open(filepath) as f:
        data = json.load(f)
    # Support both raw list and {"games": [...]} format
    if isinstance(data, list):
        return data
    return data.get('games', [])


def print_metrics(metrics: TrainingMetrics):
    """Pretty print training metrics."""
    print("\n" + "=" * 60)
    print("TRAINING ANALYSIS REPORT")
    print("=" * 60)

    print("\n📊 GAME STATISTICS")
    print("-" * 40)
    print(f"  Total games analyzed: {metrics.total_games}")
    print(f"  Game length: {metrics.avg_game_length:.1f} ± {metrics.std_game_length:.1f} moves")
    print(f"  Range: {metrics.min_game_length} - {metrics.max_game_length} moves")

    print("\n🏆 WIN RATES")
    print("-" * 40)
    print(f"  Player 0 wins: {metrics.p0_win_rate:.1%}")
    print(f"  Player 1 wins: {metrics.p1_win_rate:.1%}")
    print(f"  Draws: {metrics.draw_rate:.1%}")

    print("\n🎯 POLICY QUALITY (from MCTS)")
    print("-" * 40)
    print(f"  Avg policy entropy: {metrics.avg_policy_entropy:.3f}")
    print(f"  Avg top-1 confidence: {metrics.avg_top1_confidence:.1%}")

    print("\n🎲 OPENING DIVERSITY")
    print("-" * 40)
    print(f"  Unique openings (first 4 moves): {metrics.unique_openings}")
    print(f"  Most common opening: {metrics.most_common_opening}")
    print(f"  Concentration: {metrics.opening_concentration:.1%} of games")

    print("\n📈 GAMES PER MODEL")
    print("-" * 40)
    for model, count in sorted(metrics.games_per_model.items()):
        print(f"  {model}: {count} games")


def print_policy_accuracy(accuracy: dict):
    """Pretty print policy accuracy metrics."""
    print("\n🎯 MODEL POLICY ACCURACY")
    print("-" * 40)
    print(f"  Positions analyzed: {accuracy['total_positions']}")
    print(f"  Top-1 accuracy: {accuracy['top1_accuracy']:.1%}")
    print(f"  Top-3 accuracy: {accuracy['top3_accuracy']:.1%}")
    print(f"  Top-5 accuracy: {accuracy['top5_accuracy']:.1%}")
    print(f"  Avg KL divergence: {accuracy['avg_kl_divergence']:.3f}")


def print_value_accuracy(accuracy: dict):
    """Pretty print value accuracy metrics."""
    print("\n📉 MODEL VALUE ACCURACY")
    print("-" * 40)
    print(f"  Positions analyzed: {accuracy['total_positions']}")
    print(f"  Prediction-outcome correlation: {accuracy['correlation']:.3f}")
    print(f"  Mean absolute error: {accuracy['mean_absolute_error']:.3f}")

    print("\n  Calibration (predicted → actual win rate):")
    for bin_name, data in accuracy['calibration'].items():
        print(f"    {bin_name}: pred={data['predicted_avg']:.2f} → actual={data['actual_win_rate']:.2f} (n={data['count']})")


def print_move_diversity(diversity: dict):
    """Pretty print move diversity analysis."""
    print("\n🎲 MOVE DIVERSITY BY POSITION")
    print("-" * 40)
    for pos_name, data in diversity.items():
        print(f"\n  {pos_name}:")
        print(f"    Unique moves: {data['unique_moves']}")
        print(f"    Entropy: {data['entropy']:.3f}")
        print(f"    Top moves: ", end="")
        top_strs = [f"{m}({p:.0%})" for m, _, p in data['top_moves'][:3]]
        print(", ".join(top_strs))


def main():
    parser = argparse.ArgumentParser(description='Analyze Razzle Dazzle training progress')
    parser.add_argument('--api-url', type=str, default='https://razzledazzle.lazybrains.com/api',
                        help='Training API URL')
    parser.add_argument('--file', type=Path, default=None,
                        help='Load games from JSON file instead of API')
    parser.add_argument('--model', type=Path, default=None,
                        help='Model file to analyze (for policy/value accuracy)')
    parser.add_argument('--games', type=int, default=200,
                        help='Number of games to analyze')
    parser.add_argument('--model-version', type=str, default=None,
                        help='Filter games by model version (e.g., iter_020)')
    parser.add_argument('--min-iteration', type=int, default=None,
                        help='Filter games by minimum iteration')
    parser.add_argument('--positions', type=int, default=500,
                        help='Max positions for model analysis')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON instead of formatted text')

    args = parser.parse_args()

    # Fetch games
    if args.file:
        print(f"Loading games from {args.file}...")
        try:
            games = load_games_from_file(args.file)
            games = games[:args.games]  # Limit
            print(f"Loaded {len(games)} games")
        except Exception as e:
            print(f"Error loading games: {e}")
            return 1
    else:
        print(f"Fetching up to {args.games} games from {args.api_url}...")
        try:
            games = fetch_games_from_api(args.api_url, limit=args.games)
            print(f"Fetched {len(games)} games")
        except Exception as e:
            print(f"Error fetching games: {e}")
            return 1

    if not games:
        print("No games found!")
        return 1

    # Filter by model version if specified
    if args.model_version:
        games = [g for g in games if g.get('model_version') == args.model_version]
        print(f"Filtered to {len(games)} games from model {args.model_version}")

    # Filter by minimum iteration if specified
    if args.min_iteration:
        games = [g for g in games if g.get('iteration', 0) >= args.min_iteration]
        print(f"Filtered to {len(games)} games from iteration >= {args.min_iteration}")

    if not games:
        print("No games match the filter criteria!")
        return 1

    # Compute basic metrics
    metrics = compute_metrics(games)

    # Compute move diversity
    diversity = analyze_move_diversity(games)

    # If model provided, compute policy and value accuracy
    policy_accuracy = None
    value_accuracy = None
    if args.model and args.model.exists():
        print(f"\nLoading model: {args.model}")
        try:
            model = RazzleNet.load(args.model)
            model.eval()  # Set to evaluation mode

            print("Analyzing policy accuracy...")
            policy_accuracy = analyze_model_policy_accuracy(model, games, args.positions)

            print("Analyzing value accuracy...")
            value_accuracy = analyze_value_accuracy(model, games, args.positions)
        except Exception as e:
            import traceback
            print(f"Error loading/analyzing model: {e}")
            traceback.print_exc()

    # Output results
    if args.json:
        output = {
            'metrics': {
                'total_games': metrics.total_games,
                'avg_game_length': metrics.avg_game_length,
                'std_game_length': metrics.std_game_length,
                'p0_win_rate': metrics.p0_win_rate,
                'p1_win_rate': metrics.p1_win_rate,
                'draw_rate': metrics.draw_rate,
                'avg_policy_entropy': metrics.avg_policy_entropy,
                'avg_top1_confidence': metrics.avg_top1_confidence,
                'unique_openings': metrics.unique_openings,
                'opening_concentration': metrics.opening_concentration,
                'games_per_model': metrics.games_per_model,
            },
            'diversity': diversity,
        }
        if policy_accuracy:
            output['policy_accuracy'] = policy_accuracy
        if value_accuracy:
            output['value_accuracy'] = value_accuracy
        print(json.dumps(output, indent=2))
    else:
        print_metrics(metrics)
        print_move_diversity(diversity)
        if policy_accuracy:
            print_policy_accuracy(policy_accuracy)
        if value_accuracy:
            print_value_accuracy(value_accuracy)

    print("\n" + "=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
