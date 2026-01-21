#!/usr/bin/env python3
"""
Generate training analysis plots for Razzle Dazzle.

Creates visualizations of:
- Loss progression over iterations
- Policy/value accuracy
- Game length trends
- Opening diversity
- Value calibration

Usage:
    python plot_training.py --db server/games.db --output plots/
    python plot_training.py --db server/games.db --model /tmp/iter_199.pt
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_db_connection(db_path: Path):
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_models(conn) -> list:
    """Fetch all training models."""
    rows = conn.execute("""
        SELECT iteration, version, final_loss, final_policy_loss, final_value_loss,
               games_trained_on, created_at
        FROM training_models
        ORDER BY iteration
    """).fetchall()
    return [dict(row) for row in rows]


def fetch_games(conn, limit: int = 5000) -> list:
    """Fetch training games."""
    rows = conn.execute("""
        SELECT id, moves, result, model_version, created_at
        FROM training_games
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()

    games = []
    for row in rows:
        g = dict(row)
        g['moves'] = json.loads(g['moves'])
        games.append(g)
    return games


def get_iter_num(v):
    """Extract iteration number from version string."""
    if v == 'initial':
        return 0
    try:
        return int(v.split('_')[1])
    except:
        return 0


def plot_loss_progression(models: list, output_dir: Path):
    """Plot loss over iterations."""
    iterations = [m['iteration'] for m in models if m['final_loss']]
    total_loss = [m['final_loss'] for m in models if m['final_loss']]
    policy_loss = [m['final_policy_loss'] for m in models if m['final_loss']]
    value_loss = [m['final_value_loss'] for m in models if m['final_loss']]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Total and policy loss
    ax1.plot(iterations, total_loss, 'b-', label='Total Loss', linewidth=2)
    ax1.plot(iterations, policy_loss, 'g-', label='Policy Loss', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Value loss (separate scale)
    ax2.plot(iterations, value_loss, 'r-', label='Value Loss', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_progression.png', dpi=150)
    plt.close()
    print(f"  Saved: loss_progression.png")


def plot_game_length_distribution(games: list, output_dir: Path):
    """Plot game length distribution."""
    lengths = [len(g['moves']) for g in games]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
    ax1.axvline(np.median(lengths), color='orange', linestyle='--', label=f'Median: {np.median(lengths):.1f}')
    ax1.set_xlabel('Game Length (moves)')
    ax1.set_ylabel('Count')
    ax1.set_title('Game Length Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Trend over training
    version_lengths = {}
    for g in games:
        v = g['model_version']
        if v not in version_lengths:
            version_lengths[v] = []
        version_lengths[v].append(len(g['moves']))

    versions = sorted(version_lengths.keys(), key=get_iter_num)
    iters = [get_iter_num(v) for v in versions]
    avg_lengths = [np.mean(version_lengths[v]) for v in versions]

    ax2.plot(iters, avg_lengths, 'purple', linewidth=2)
    ax2.fill_between(iters, avg_lengths, alpha=0.3, color='purple')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Avg Game Length')
    ax2.set_title('Average Game Length Over Training')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'game_length.png', dpi=150)
    plt.close()
    print(f"  Saved: game_length.png")


def plot_win_rates(games: list, output_dir: Path):
    """Plot win rates over training."""
    version_results = {}
    for g in games:
        v = g['model_version']
        if v not in version_results:
            version_results[v] = {'p0_wins': 0, 'p1_wins': 0, 'total': 0}
        version_results[v]['total'] += 1
        if g['result'] > 0:
            version_results[v]['p0_wins'] += 1
        elif g['result'] < 0:
            version_results[v]['p1_wins'] += 1

    versions = sorted(version_results.keys(), key=get_iter_num)
    iterations = [get_iter_num(v) for v in versions]
    p0_rates = [version_results[v]['p0_wins'] / version_results[v]['total'] * 100 for v in versions]
    p1_rates = [version_results[v]['p1_wins'] / version_results[v]['total'] * 100 for v in versions]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(iterations, p0_rates, 'b-', label='Player 0 Win Rate', linewidth=2, alpha=0.8)
    ax.plot(iterations, p1_rates, 'r-', label='Player 1 Win Rate', linewidth=2, alpha=0.8)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rates Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'win_rates.png', dpi=150)
    plt.close()
    print(f"  Saved: win_rates.png")


def plot_games_per_iteration(games: list, output_dir: Path):
    """Plot number of games generated per model version."""
    version_counts = Counter(g['model_version'] for g in games)

    versions = sorted(version_counts.keys(), key=get_iter_num)
    iterations = [get_iter_num(v) for v in versions]
    counts = [version_counts[v] for v in versions]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(iterations, counts, color='steelblue', alpha=0.8, width=0.8)
    ax.axhline(np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Games Generated')
    ax.set_title('Games Generated Per Model Version')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'games_per_iteration.png', dpi=150)
    plt.close()
    print(f"  Saved: games_per_iteration.png")


def plot_opening_diversity(games: list, output_dir: Path):
    """Plot opening move diversity."""
    first_moves = [g['moves'][0] if g['moves'] else None for g in games]
    first_moves = [m for m in first_moves if m is not None]
    move_counts = Counter(first_moves)

    top_moves = move_counts.most_common(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of top opening moves
    moves = [str(m[0]) for m in top_moves]
    counts = [m[1] for m in top_moves]

    ax1.barh(moves, counts, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Move (encoded)')
    ax1.set_title('Top 10 Opening Moves')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # Diversity over training
    version_diversity = {}
    for g in games:
        v = g['model_version']
        if g['moves']:
            if v not in version_diversity:
                version_diversity[v] = set()
            version_diversity[v].add(g['moves'][0])

    versions = sorted(version_diversity.keys(), key=get_iter_num)
    iterations = [get_iter_num(v) for v in versions]
    diversity = [len(version_diversity[v]) for v in versions]

    ax2.plot(iterations, diversity, 'g-o', linewidth=2, markersize=4, alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Unique Opening Moves')
    ax2.set_title('Opening Move Diversity Over Training')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'opening_diversity.png', dpi=150)
    plt.close()
    print(f"  Saved: opening_diversity.png")


def plot_training_summary(models: list, games: list, output_dir: Path):
    """Create a summary dashboard."""
    fig = plt.figure(figsize=(16, 12))

    # Loss progression
    ax1 = fig.add_subplot(2, 2, 1)
    iterations = [m['iteration'] for m in models if m['final_loss']]
    total_loss = [m['final_loss'] for m in models if m['final_loss']]
    policy_loss = [m['final_policy_loss'] for m in models if m['final_loss']]

    ax1.plot(iterations, total_loss, 'b-', label='Total', linewidth=2)
    ax1.plot(iterations, policy_loss, 'g-', label='Policy', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Game length trend
    ax2 = fig.add_subplot(2, 2, 2)
    version_lengths = {}
    for g in games:
        v = g['model_version']
        if v not in version_lengths:
            version_lengths[v] = []
        version_lengths[v].append(len(g['moves']))

    versions = sorted(version_lengths.keys(), key=get_iter_num)
    iters = [get_iter_num(v) for v in versions]
    avg_lengths = [np.mean(version_lengths[v]) for v in versions]

    ax2.plot(iters, avg_lengths, 'purple', linewidth=2)
    ax2.fill_between(iters, avg_lengths, alpha=0.3, color='purple')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Avg Game Length')
    ax2.set_title('Game Length Over Training')
    ax2.grid(True, alpha=0.3)

    # Games per iteration
    ax3 = fig.add_subplot(2, 2, 3)
    version_counts = Counter(g['model_version'] for g in games)
    versions = sorted(version_counts.keys(), key=get_iter_num)
    iters = [get_iter_num(v) for v in versions]
    counts = [version_counts[v] for v in versions]

    ax3.bar(iters, counts, color='teal', alpha=0.7, width=0.8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Games')
    ax3.set_title('Games Per Iteration')
    ax3.grid(True, alpha=0.3, axis='y')

    # Summary stats
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    stats_text = f"""
    Training Summary
    ================

    Total Iterations: {len(models)}
    Total Games: {len(games)}

    Loss:
      Initial: {total_loss[0]:.3f}
      Final: {total_loss[-1]:.3f}
      Reduction: {(1 - total_loss[-1]/total_loss[0])*100:.1f}%

    Game Length:
      Initial: {avg_lengths[0]:.1f} moves
      Final: {avg_lengths[-1]:.1f} moves

    Win Rates:
      P0: {sum(1 for g in games if g['result'] > 0) / len(games) * 100:.1f}%
      P1: {sum(1 for g in games if g['result'] < 0) / len(games) * 100:.1f}%
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=150)
    plt.close()
    print(f"  Saved: training_summary.png")


def plot_model_accuracy(model_path: Path, games: list, output_dir: Path):
    """Plot model value accuracy."""
    try:
        import torch
        from razzle.ai.network import RazzleNet
        from razzle.core.state import GameState
    except ImportError as e:
        print(f"  Skipping model accuracy plots: {e}")
        return

    print(f"  Loading model: {model_path}")
    model = RazzleNet.load(model_path)
    model.eval()
    device = next(model.parameters()).device

    predictions = []
    actuals = []

    for game in games[:100]:
        state = GameState.new_game()
        result = game['result']

        for move in game['moves'][:20]:
            state_tensor = torch.tensor(state.to_tensor(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, value = model(state_tensor.to(device))
                pred_value = value.item()

            current_player = state.current_player
            actual_value = result if current_player == 0 else -result

            predictions.append(pred_value)
            actuals.append(actual_value)
            state.apply_move(move)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1.scatter(predictions, actuals, alpha=0.3, s=10)
    ax1.plot([-1, 1], [-1, 1], 'r--', label='Perfect calibration')
    corr = np.corrcoef(predictions, actuals)[0, 1]
    ax1.set_xlabel('Predicted Value')
    ax1.set_ylabel('Actual Outcome')
    ax1.set_title(f'Value Prediction vs Outcome (r={corr:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calibration plot
    bins = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]
    bin_centers = []
    actual_rates = []

    for low, high in bins:
        mask = (predictions >= low) & (predictions < high)
        if mask.sum() > 0:
            bin_centers.append((low + high) / 2)
            win_rate = (actuals[mask] + 1) / 2
            actual_rates.append(win_rate.mean())

    ax2.bar(bin_centers, actual_rates, width=0.35, alpha=0.8, color='steelblue')
    ax2.plot([-1, 1], [0, 1], 'r--', label='Perfect calibration')
    ax2.set_xlabel('Predicted Value')
    ax2.set_ylabel('Actual Win Rate')
    ax2.set_title('Value Calibration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_accuracy.png', dpi=150)
    plt.close()
    print(f"  Saved: model_accuracy.png")


def main():
    parser = argparse.ArgumentParser(description='Generate training analysis plots')
    parser.add_argument('--db', type=Path, default=Path('server/games.db'),
                        help='Path to games database')
    parser.add_argument('--output', type=Path, default=Path('plots'),
                        help='Output directory for plots')
    parser.add_argument('--model', type=Path, default=None,
                        help='Model file for accuracy analysis')
    parser.add_argument('--games', type=int, default=5000,
                        help='Max games to analyze')

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.db}...")
    conn = get_db_connection(args.db)

    models = fetch_models(conn)
    games = fetch_games(conn, args.games)

    print(f"Loaded {len(models)} models and {len(games)} games")
    print("\nGenerating plots...")

    plot_loss_progression(models, args.output)
    plot_game_length_distribution(games, args.output)
    plot_win_rates(games, args.output)
    plot_games_per_iteration(games, args.output)
    plot_opening_diversity(games, args.output)
    plot_training_summary(models, games, args.output)

    if args.model and args.model.exists():
        plot_model_accuracy(args.model, games, args.output)

    print(f"\nAll plots saved to {args.output}/")
    conn.close()


if __name__ == '__main__':
    main()
