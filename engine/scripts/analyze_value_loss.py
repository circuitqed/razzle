#!/usr/bin/env python3
"""
Analyze value prediction accuracy to understand why value loss is so low.

Investigates:
1. Value loss by game phase (early/mid/late)
2. Prediction accuracy vs actual outcomes
3. Whether certain openings always lead to same result
4. Distribution of errors
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from razzle.ai.network import RazzleNet, NetworkConfig
from razzle.core.state import GameState
from razzle.training.api_client import TrainingAPIClient


def main():
    parser = argparse.ArgumentParser(description='Analyze value prediction accuracy')
    parser.add_argument('--api-url', default='https://razzledazzle.lazybrains.com/api')
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--filters', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=15)
    args = parser.parse_args()

    print(f"Connecting to {args.api_url}...")
    client = TrainingAPIClient(args.api_url)

    # Get latest model
    dashboard = client.get_dashboard()
    if not dashboard or not dashboard.get('latest_model'):
        print("No model available")
        return

    model_version = dashboard['latest_model']['version']
    print(f"Model: {model_version}")

    # Download and load model
    model_path = Path(f'/tmp/model_{model_version}.pt')
    if not model_path.exists():
        client.download_model(model_version, model_path)

    config = NetworkConfig(num_filters=args.filters, num_blocks=args.blocks)
    network = RazzleNet(config)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_key = 'state_dict' if 'state_dict' in checkpoint else 'network_state'
    network.load_state_dict(checkpoint[state_key])
    network.eval()

    # Fetch games
    print(f"Fetching {args.num_games} games...")
    games, _ = client.fetch_pending_games(limit=args.num_games, mark_used=False)
    print(f"Got {len(games)} games")

    if not games:
        print("No games available")
        return

    # Analyze each game
    all_data = []  # (move_fraction, pred_value, target_value, game_result, opening_hash)
    opening_results = defaultdict(list)  # opening_hash -> list of results

    for game in games:
        state = GameState.new_game()
        game_length = len(game.moves)
        game_result = game.result  # +1 if P0 won, -1 if P1 won

        # Hash first 4 moves as "opening"
        opening_moves = tuple(game.moves[:4]) if len(game.moves) >= 4 else tuple(game.moves)
        opening_hash = hash(opening_moves)
        opening_results[opening_hash].append(game_result)

        states_in_game = []
        players_in_game = []

        for move_idx, move in enumerate(game.moves):
            states_in_game.append(state.to_tensor())
            players_in_game.append(state.current_player)
            state.apply_move(move)

        # Batch predict
        if not states_in_game:
            continue

        states_tensor = torch.from_numpy(np.stack(states_in_game))
        with torch.no_grad():
            _, values, _ = network(states_tensor)
            pred_values = values.squeeze(-1).numpy()

        # Record data
        for i, (pred, player) in enumerate(zip(pred_values, players_in_game)):
            # Target from current player's perspective
            if player == 0:
                target = game_result
            else:
                target = -game_result

            move_fraction = i / game_length
            all_data.append((move_fraction, float(pred), float(target), game_result, opening_hash))

    all_data = np.array(all_data, dtype=[
        ('move_frac', 'f4'), ('pred', 'f4'), ('target', 'f4'),
        ('result', 'f4'), ('opening', 'i8')
    ])

    print(f"\nTotal positions analyzed: {len(all_data)}")

    # 1. Value loss by game phase
    print("\n" + "=" * 60)
    print("VALUE LOSS BY GAME PHASE")
    print("=" * 60)

    phases = [
        ("Opening (0-20%)", 0.0, 0.2),
        ("Early (20-40%)", 0.2, 0.4),
        ("Mid (40-60%)", 0.4, 0.6),
        ("Late (60-80%)", 0.6, 0.8),
        ("Endgame (80-100%)", 0.8, 1.0),
    ]

    for name, lo, hi in phases:
        mask = (all_data['move_frac'] >= lo) & (all_data['move_frac'] < hi)
        if np.sum(mask) == 0:
            continue
        phase_data = all_data[mask]
        errors = phase_data['pred'] - phase_data['target']
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        mean_pred = np.mean(phase_data['pred'])
        mean_target = np.mean(phase_data['target'])
        extremity = np.mean(np.abs(phase_data['pred']))

        print(f"{name:25s}: MSE={mse:.4f}, MAE={mae:.4f}, "
              f"pred_mean={mean_pred:+.3f}, target_mean={mean_target:+.3f}, "
              f"extremity={extremity:.3f}, n={np.sum(mask)}")

    # 2. Analyze prediction accuracy
    print("\n" + "=" * 60)
    print("PREDICTION ACCURACY")
    print("=" * 60)

    # How often does sign match?
    sign_match = np.sign(all_data['pred']) == np.sign(all_data['target'])
    print(f"Sign match (pred & target same sign): {100*np.mean(sign_match):.1f}%")

    # How often is prediction correct direction?
    # (positive pred when P0 wins, negative when P1 wins)
    correct_direction = (all_data['pred'] > 0) == (all_data['result'] > 0)
    # But this ignores player perspective... let's use target instead
    print(f"Correct direction (pred sign matches outcome): {100*np.mean(sign_match):.1f}%")

    # By phase
    print("\nSign match by phase:")
    for name, lo, hi in phases:
        mask = (all_data['move_frac'] >= lo) & (all_data['move_frac'] < hi)
        if np.sum(mask) == 0:
            continue
        phase_data = all_data[mask]
        sign_match = np.sign(phase_data['pred']) == np.sign(phase_data['target'])
        print(f"  {name:25s}: {100*np.mean(sign_match):.1f}%")

    # 3. Opening diversity analysis
    print("\n" + "=" * 60)
    print("OPENING ANALYSIS")
    print("=" * 60)

    # How deterministic are openings?
    unique_openings = len(opening_results)
    print(f"Unique openings (first 4 moves): {unique_openings}")

    # For each opening, is the result always the same?
    deterministic_openings = 0
    mixed_openings = 0
    for opening, results in opening_results.items():
        if len(results) >= 2:
            if all(r == results[0] for r in results):
                deterministic_openings += 1
            else:
                mixed_openings += 1

    print(f"Openings with 2+ games:")
    print(f"  Deterministic (same result): {deterministic_openings}")
    print(f"  Mixed (different results): {mixed_openings}")
    if deterministic_openings + mixed_openings > 0:
        print(f"  Determinism rate: {100*deterministic_openings/(deterministic_openings+mixed_openings):.1f}%")

    # 4. Error distribution
    print("\n" + "=" * 60)
    print("ERROR DISTRIBUTION")
    print("=" * 60)

    errors = all_data['pred'] - all_data['target']
    print(f"Error stats: mean={np.mean(errors):+.4f}, std={np.std(errors):.4f}")
    print(f"Error percentiles:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  {p}th: {np.percentile(errors, p):+.4f}")

    # Histogram of errors
    print("\nError histogram:")
    bins = np.linspace(-2, 2, 21)
    counts, _ = np.histogram(errors, bins=bins)
    max_count = max(counts)
    for i in range(len(counts)):
        lo, hi = bins[i], bins[i+1]
        bar_len = int(counts[i] / max_count * 30) if max_count > 0 else 0
        print(f"[{lo:+5.2f}, {hi:+5.2f}): {'#' * bar_len:30s} {counts[i]:5d}")

    # 5. Investigate suspiciously accurate early predictions
    print("\n" + "=" * 60)
    print("EARLY GAME ACCURACY DEEP DIVE")
    print("=" * 60)

    early_mask = all_data['move_frac'] < 0.2
    early_data = all_data[early_mask]

    if len(early_data) > 0:
        # For early positions, the network shouldn't know the outcome
        # If it predicts correctly, is it because of opening determinism?

        early_correct = np.sign(early_data['pred']) == np.sign(early_data['target'])
        print(f"Early game positions: {len(early_data)}")
        print(f"Early game sign accuracy: {100*np.mean(early_correct):.1f}%")

        # What's the average |prediction| for correct vs incorrect?
        correct_extremity = np.mean(np.abs(early_data['pred'][early_correct]))
        if np.sum(~early_correct) > 0:
            incorrect_extremity = np.mean(np.abs(early_data['pred'][~early_correct]))
        else:
            incorrect_extremity = 0

        print(f"Avg |pred| when correct: {correct_extremity:.3f}")
        print(f"Avg |pred| when incorrect: {incorrect_extremity:.3f}")

        # Are early predictions just random but happen to match?
        # If predictions were random ±1, we'd expect 50% accuracy
        # Let's see what the baseline would be
        print(f"\nBaseline (random ±1): 50% accuracy")
        print(f"Actual early accuracy: {100*np.mean(early_correct):.1f}%")

        # Is early accuracy due to just predicting the majority class?
        majority_class = 1 if np.mean(early_data['target']) > 0 else -1
        majority_accuracy = np.mean((np.sign(early_data['target']) == majority_class))
        print(f"Majority class baseline: {100*majority_accuracy:.1f}%")


if __name__ == '__main__':
    main()
