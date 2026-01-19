#!/usr/bin/env python3
"""
Generate training progress plots from training logs.

Creates:
- Loss curves (total, policy, value)
- Win rate by player
- Game length distribution
- Training throughput

Outputs PNG files and JSON summary for web app.
"""

import argparse
import json
from pathlib import Path

# Use non-interactive backend for server-side rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_training_log(log_path: Path) -> dict:
    """Load and parse training log."""
    with open(log_path) as f:
        return json.load(f)


def plot_loss_curves(data: dict, output_dir: Path) -> None:
    """Plot loss curves over iterations."""
    iterations = data['iterations']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Extract data
    iters = [it['iteration'] for it in iterations]

    # Total loss per epoch (flatten all epochs)
    all_epochs = []
    for it in iterations:
        for ep in it['epochs']:
            all_epochs.append({
                'iteration': it['iteration'],
                'epoch': ep['epoch'],
                'loss': ep['loss'],
                'policy_loss': ep['policy_loss'],
                'value_loss': ep['value_loss']
            })

    epoch_nums = range(len(all_epochs))

    # Total loss
    axes[0].plot(epoch_nums, [e['loss'] for e in all_epochs], 'b-', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)

    # Policy loss
    axes[1].plot(epoch_nums, [e['policy_loss'] for e in all_epochs], 'g-', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Policy Loss')
    axes[1].set_title('Policy Loss')
    axes[1].grid(True, alpha=0.3)

    # Value loss
    axes[2].plot(epoch_nums, [e['value_loss'] for e in all_epochs], 'r-', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Value Loss')
    axes[2].set_title('Value Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=100)
    plt.close()


def plot_win_rates(data: dict, output_dir: Path) -> None:
    """Plot win rates by player over iterations."""
    iterations = data['iterations']

    fig, ax = plt.subplots(figsize=(10, 5))

    iters = [it['iteration'] for it in iterations]
    p1_wins = [it['p1_wins'] for it in iterations]
    p2_wins = [it['p2_wins'] for it in iterations]
    draws = [it['draws'] for it in iterations]

    total_games = [p1 + p2 + d for p1, p2, d in zip(p1_wins, p2_wins, draws)]
    p1_rate = [p1/t*100 if t > 0 else 0 for p1, t in zip(p1_wins, total_games)]
    p2_rate = [p2/t*100 if t > 0 else 0 for p2, t in zip(p2_wins, total_games)]
    draw_rate = [d/t*100 if t > 0 else 0 for d, t in zip(draws, total_games)]

    width = 0.8
    ax.bar(iters, p1_rate, width, label='P1 Wins', color='#2ecc71')
    ax.bar(iters, p2_rate, width, bottom=p1_rate, label='P2 Wins', color='#e74c3c')
    ax.bar(iters, draw_rate, width, bottom=[p1+p2 for p1, p2 in zip(p1_rate, p2_rate)],
           label='Draws', color='#95a5a6')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Percentage')
    ax.set_title('Game Outcomes by Iteration')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'win_rates.png', dpi=100)
    plt.close()


def plot_game_lengths(data: dict, output_dir: Path) -> None:
    """Plot game length statistics over iterations."""
    iterations = data['iterations']

    fig, ax = plt.subplots(figsize=(10, 5))

    iters = [it['iteration'] for it in iterations]
    avg_lens = [it['avg_game_length'] for it in iterations]
    min_lens = [it.get('min_game_length', 0) for it in iterations]
    max_lens = [it.get('max_game_length', 0) for it in iterations]

    ax.plot(iters, avg_lens, 'b-o', label='Average', linewidth=2, markersize=8)
    ax.fill_between(iters, min_lens, max_lens, alpha=0.2, color='blue', label='Min-Max Range')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Game Length (moves)')
    ax.set_title('Game Length Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'game_lengths.png', dpi=100)
    plt.close()


def plot_training_summary(data: dict, output_dir: Path) -> None:
    """Create a summary dashboard image."""
    fig = plt.figure(figsize=(16, 10))

    iterations = data['iterations']

    # Loss curve (main plot)
    ax1 = fig.add_subplot(2, 2, 1)
    final_losses = [it['final_loss'] for it in iterations]
    iters = [it['iteration'] for it in iterations]
    ax1.plot(iters, final_losses, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Loss per Iteration')
    ax1.grid(True, alpha=0.3)

    # Win rates
    ax2 = fig.add_subplot(2, 2, 2)
    p1_wins = [it['p1_wins'] for it in iterations]
    p2_wins = [it['p2_wins'] for it in iterations]
    draws = [it['draws'] for it in iterations]
    ax2.stackplot(iters, p1_wins, p2_wins, draws,
                  labels=['P1 Wins', 'P2 Wins', 'Draws'],
                  colors=['#2ecc71', '#e74c3c', '#95a5a6'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Games')
    ax2.set_title('Game Outcomes')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Game lengths
    ax3 = fig.add_subplot(2, 2, 3)
    avg_lens = [it['avg_game_length'] for it in iterations]
    ax3.plot(iters, avg_lens, 'g-o', linewidth=2, markersize=8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Moves')
    ax3.set_title('Average Game Length')
    ax3.grid(True, alpha=0.3)

    # Stats text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    total_games = data['total_games']
    total_examples = data['total_examples']
    total_time = data['total_time_sec'] / 60
    start_loss = iterations[0]['epochs'][0]['loss'] if iterations else 0
    final_loss = iterations[-1]['final_loss'] if iterations else 0

    stats_text = f"""
Training Summary
================

Total Games: {total_games:,}
Training Examples: {total_examples:,}
Total Time: {total_time:.1f} min

Loss: {start_loss:.3f} → {final_loss:.3f}
Reduction: {(1 - final_loss/start_loss)*100:.1f}%

Iterations: {len(iterations)}
Games/Iteration: {total_games // len(iterations) if iterations else 0}
"""
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=100)
    plt.close()


def generate_json_summary(data: dict, output_dir: Path) -> None:
    """Generate JSON summary for web app."""
    iterations = data['iterations']

    summary = {
        'run_id': data.get('run_id', 'unknown'),
        'total_games': data['total_games'],
        'total_examples': data['total_examples'],
        'total_time_min': data['total_time_sec'] / 60,
        'num_iterations': len(iterations),
        'start_loss': iterations[0]['epochs'][0]['loss'] if iterations else None,
        'final_loss': iterations[-1]['final_loss'] if iterations else None,
        'iterations': [{
            'iteration': it['iteration'],
            'p1_wins': it['p1_wins'],
            'p2_wins': it['p2_wins'],
            'draws': it['draws'],
            'avg_game_length': it['avg_game_length'],
            'final_loss': it['final_loss'],
            'final_policy_loss': it['final_policy_loss'],
            'final_value_loss': it['final_value_loss'],
        } for it in iterations],
        'plots': {
            'summary': 'training_summary.png',
            'loss_curves': 'loss_curves.png',
            'win_rates': 'win_rates.png',
            'game_lengths': 'game_lengths.png',
        }
    }

    with open(output_dir / 'training_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate training progress plots')
    parser.add_argument('--log', type=Path, default=Path('output/output/training_log.json'),
                        help='Path to training log JSON')
    parser.add_argument('--output', type=Path, default=Path('output/plots'),
                        help='Output directory for plots')

    args = parser.parse_args()

    if not args.log.exists():
        print(f"Training log not found: {args.log}")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading training log: {args.log}")
    data = load_training_log(args.log)

    print(f"Generating plots to: {args.output}")
    plot_loss_curves(data, args.output)
    plot_win_rates(data, args.output)
    plot_game_lengths(data, args.output)
    plot_training_summary(data, args.output)
    generate_json_summary(data, args.output)

    print("Generated:")
    print("  - training_summary.png")
    print("  - loss_curves.png")
    print("  - win_rates.png")
    print("  - game_lengths.png")
    print("  - training_metrics.json")


if __name__ == '__main__':
    main()
