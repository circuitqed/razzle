#!/usr/bin/env python3
"""
Analyze value prediction distribution.

Downloads a model and fetches recent games from the API, then computes
value predictions for all positions and displays a histogram.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from razzle.ai.network import RazzleNet, NetworkConfig
from razzle.core.state import GameState
from razzle.training.api_client import TrainingAPIClient


def main():
    parser = argparse.ArgumentParser(description='Analyze value predictions')
    parser.add_argument('--api-url', default='https://razzledazzle.lazybrains.com/api',
                        help='API server URL')
    parser.add_argument('--model-version', default='latest',
                        help='Model version to analyze (default: latest)')
    parser.add_argument('--num-games', type=int, default=50,
                        help='Number of games to analyze')
    parser.add_argument('--device', default='cpu',
                        help='Device for inference (default: cpu)')
    parser.add_argument('--filters', type=int, default=256,
                        help='Network filters (default: 256)')
    parser.add_argument('--blocks', type=int, default=15,
                        help='Network blocks (default: 15)')
    parser.add_argument('--bins', type=int, default=20,
                        help='Number of histogram bins')
    args = parser.parse_args()

    print(f"Connecting to {args.api_url}...")
    client = TrainingAPIClient(args.api_url)

    # Get model info
    print(f"Fetching model info...")
    dashboard = client.get_dashboard()
    if not dashboard:
        print("Failed to fetch dashboard")
        return

    if args.model_version == 'latest':
        model_info = dashboard.get('latest_model')
        if not model_info:
            print("No model available")
            return
        model_version = model_info['version']
    else:
        model_version = args.model_version

    print(f"Model: {model_version}")

    # Download model
    print(f"Downloading model...")
    model_path = Path(f'/tmp/model_{model_version}.pt')
    if not model_path.exists():
        success = client.download_model(model_version, model_path)
        if not success:
            print("Failed to download model")
            return

    # Load model
    print(f"Loading model with {args.filters} filters, {args.blocks} blocks...")
    config = NetworkConfig(num_filters=args.filters, num_blocks=args.blocks)
    network = RazzleNet(config)

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    # Handle both old and new checkpoint formats
    state_key = 'state_dict' if 'state_dict' in checkpoint else 'network_state'
    network.load_state_dict(checkpoint[state_key])
    network = network.to(args.device)
    network.eval()

    # Fetch pending games (don't mark as used)
    print(f"Fetching up to {args.num_games} pending games...")
    games, total_pending = client.fetch_pending_games(limit=args.num_games, mark_used=False)
    print(f"Fetched {len(games)} games ({total_pending} pending total)")

    if not games:
        print("No games available")
        return

    # Collect all positions
    print("Replaying games to extract positions...")
    all_states = []
    all_move_numbers = []  # Track position in game

    for game in games:
        state = GameState.new_game()
        game_length = len(game.moves)

        for move_idx, move in enumerate(game.moves):
            all_states.append(state.to_tensor())
            all_move_numbers.append(move_idx / game_length)  # Normalized position
            state.apply_move(move)

    print(f"Total positions: {len(all_states)}")

    # Compute predictions in batches
    print("Computing value predictions...")
    states_tensor = torch.from_numpy(np.stack(all_states)).to(args.device)

    all_values = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(states_tensor), batch_size):
            batch = states_tensor[i:i+batch_size]
            _, values, _ = network(batch)
            all_values.extend(values.squeeze(-1).cpu().numpy().tolist())

    all_values = np.array(all_values)
    all_move_numbers = np.array(all_move_numbers)

    # Print histogram
    print("\n" + "=" * 60)
    print("VALUE PREDICTION HISTOGRAM")
    print("=" * 60)

    bin_edges = np.linspace(-1, 1, args.bins + 1)
    counts, _ = np.histogram(all_values, bins=bin_edges)

    max_count = max(counts)
    bar_width = 40

    for i in range(len(counts)):
        low, high = bin_edges[i], bin_edges[i+1]
        bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
        bar = '#' * bar_len
        print(f"[{low:+5.2f}, {high:+5.2f}): {bar:40s} {counts[i]:5d}")

    print("-" * 60)
    print(f"Total positions: {len(all_values)}")
    print(f"Mean: {np.mean(all_values):+.4f}")
    print(f"Std:  {np.std(all_values):.4f}")
    print(f"Extremity (mean |v|): {np.mean(np.abs(all_values)):.4f}")
    print(f"Min: {np.min(all_values):+.4f}, Max: {np.max(all_values):+.4f}")

    # Analyze by game position
    print("\n" + "=" * 60)
    print("VALUE EXTREMITY BY GAME PHASE")
    print("=" * 60)

    # Split into early (0-33%), mid (33-66%), late (66-100%)
    early_mask = all_move_numbers < 0.33
    mid_mask = (all_move_numbers >= 0.33) & (all_move_numbers < 0.66)
    late_mask = all_move_numbers >= 0.66

    for name, mask in [("Early (0-33%)", early_mask), ("Mid (33-66%)", mid_mask), ("Late (66-100%)", late_mask)]:
        if np.sum(mask) > 0:
            phase_values = all_values[mask]
            extremity = np.mean(np.abs(phase_values))
            print(f"{name:20s}: extremity={extremity:.4f}, mean={np.mean(phase_values):+.4f}, n={np.sum(mask)}")

    # Show distribution of extreme vs moderate predictions
    print("\n" + "=" * 60)
    print("PREDICTION CONFIDENCE BREAKDOWN")
    print("=" * 60)

    extreme = np.abs(all_values) > 0.9
    high = (np.abs(all_values) > 0.5) & (np.abs(all_values) <= 0.9)
    moderate = np.abs(all_values) <= 0.5

    print(f"Extreme (|v| > 0.9): {np.sum(extreme):5d} ({100*np.mean(extreme):.1f}%)")
    print(f"High    (0.5 < |v| <= 0.9): {np.sum(high):5d} ({100*np.mean(high):.1f}%)")
    print(f"Moderate (|v| <= 0.5): {np.sum(moderate):5d} ({100*np.mean(moderate):.1f}%)")


if __name__ == '__main__':
    main()
