#!/usr/bin/env python3
"""
Monitor training progress in real-time.

Reads from training_log.json and displays progress.
Can run in watch mode to continuously update.

Usage:
    python scripts/monitor_training.py                    # Show current status
    python scripts/monitor_training.py --watch            # Continuously monitor
    python scripts/monitor_training.py --output output2   # Different directory
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def load_log(output_dir: Path) -> dict:
    """Load training log."""
    log_path = output_dir / "training_log.json"
    if not log_path.exists():
        return None
    with open(log_path, 'r') as f:
        return json.load(f)


def print_header(log: dict):
    """Print training header info."""
    print("=" * 70)
    print(f"  RAZZLE DAZZLE TRAINING MONITOR")
    print(f"  Run ID: {log.get('run_id', 'unknown')}")
    print(f"  Started: {log.get('start_time', 'unknown')}")
    print("=" * 70)


def print_config(log: dict):
    """Print training configuration."""
    config = log.get('config', {})
    if not config:
        return

    print("\nConfiguration:")
    print(f"  Games/iter: {config.get('games_per_iter', '?')}")
    print(f"  MCTS sims: {config.get('simulations', '?')}")
    print(f"  Epochs: {config.get('epochs', '?')}")
    print(f"  Network: {config.get('filters', '?')} filters, {config.get('blocks', '?')} blocks")
    print(f"  Device: {config.get('device', '?')}")


def print_summary(log: dict):
    """Print training summary."""
    iterations = log.get('iterations', [])
    if not iterations:
        print("\n  No iterations completed yet.")
        return

    total_games = log.get('total_games', 0)
    total_examples = log.get('total_examples', 0)
    total_time = log.get('total_time_sec', 0)

    print(f"\nProgress: {len(iterations)} iterations completed")
    print(f"  Total games: {total_games:,}")
    print(f"  Total examples: {total_examples:,}")
    print(f"  Total time: {format_time(total_time)}")

    if len(iterations) > 0 and total_time > 0:
        avg_time = total_time / len(iterations)
        games_per_sec = total_games / total_time if total_time > 0 else 0
        print(f"  Avg time/iter: {format_time(avg_time)}")
        print(f"  Throughput: {games_per_sec:.2f} games/sec")


def print_latest(log: dict):
    """Print latest iteration details."""
    iterations = log.get('iterations', [])
    if not iterations:
        return

    latest = iterations[-1]

    print(f"\nLatest iteration ({latest['iteration']}):")
    print(f"  Games: P1 {latest['p1_wins']} / P2 {latest['p2_wins']} / Draw {latest['draws']}")

    # Game length with min/max/std if available
    avg_len = latest['avg_game_length']
    min_len = latest.get('min_game_length', 0)
    max_len = latest.get('max_game_length', 0)
    std_len = latest.get('std_game_length', 0)
    if min_len and max_len:
        print(f"  Game length: {avg_len:.1f} avg (min={min_len}, max={max_len}, std={std_len:.1f})")
    else:
        print(f"  Avg length: {avg_len:.1f} moves")

    print(f"  Examples: {latest['training_examples']:,}")

    if latest.get('final_loss'):
        print(f"  Loss: {latest['final_loss']:.4f} (policy={latest.get('final_policy_loss', 0):.4f}, value={latest.get('final_value_loss', 0):.4f})")

    if latest.get('selfplay_time_sec'):
        print(f"  Time: self-play {format_time(latest['selfplay_time_sec'])}, train {format_time(latest.get('training_time_sec', 0))}")

    if latest.get('win_rate_vs_random') is not None:
        print(f"  Win rate vs random: {latest['win_rate_vs_random']:.1%}")


def print_game_length_trend(log: dict, n: int = 10):
    """Print game length trend as ASCII chart (key early learning indicator)."""
    iterations = log.get('iterations', [])
    if len(iterations) < 2:
        return

    print(f"\nGame Length Trend (last {min(n, len(iterations))} iters) - Early Learning Indicator:")

    recent = iterations[-n:]
    lengths = [it.get('avg_game_length', 0) for it in recent]
    max_len = max(lengths) if lengths else 1
    min_len = min(lengths) if lengths else 0
    bar_width = 35

    for it in recent:
        avg_len = it.get('avg_game_length', 0)
        min_g = it.get('min_game_length', 0)
        max_g = it.get('max_game_length', 0)

        # Normalize to bar width
        if max_len > min_len:
            bar_len = int(((avg_len - min_len) / (max_len - min_len)) * bar_width)
        else:
            bar_len = bar_width // 2

        bar = '=' * bar_len + '>' + '.' * (bar_width - bar_len - 1)

        # Show range if available
        if min_g and max_g:
            print(f"  {it['iteration']:3d} [{bar}] {avg_len:5.1f} ({min_g}-{max_g})")
        else:
            print(f"  {it['iteration']:3d} [{bar}] {avg_len:5.1f}")

    # Show trend direction
    if len(lengths) >= 3:
        first_half = sum(lengths[:len(lengths)//2]) / (len(lengths)//2)
        second_half = sum(lengths[len(lengths)//2:]) / (len(lengths) - len(lengths)//2)
        diff = second_half - first_half
        if abs(diff) > 5:
            direction = "INCREASING" if diff > 0 else "DECREASING"
            print(f"       Trend: {direction} ({diff:+.1f} moves)")


def print_loss_trend(log: dict, n: int = 10):
    """Print loss trend as ASCII chart."""
    iterations = log.get('iterations', [])
    if len(iterations) < 2:
        return

    # Skip if no loss data
    if not any(it.get('final_loss', 0) for it in iterations):
        return

    print(f"\nLoss Trend (last {min(n, len(iterations))} iterations):")

    recent = iterations[-n:]
    max_loss = max(it.get('final_loss', 0) for it in recent) or 1
    bar_width = 40

    for it in recent:
        loss = it.get('final_loss', 0)
        bar_len = int((loss / max_loss) * bar_width)
        bar = '#' * bar_len + '.' * (bar_width - bar_len)
        print(f"  {it['iteration']:3d} [{bar}] {loss:.4f}")


def print_win_balance(log: dict):
    """Print win balance indicator."""
    iterations = log.get('iterations', [])
    if not iterations:
        return

    total_p1 = sum(it['p1_wins'] for it in iterations)
    total_p2 = sum(it['p2_wins'] for it in iterations)
    total = total_p1 + total_p2

    if total == 0:
        return

    p1_pct = total_p1 / total * 100
    p2_pct = total_p2 / total * 100

    # Create balance bar
    bar_width = 40
    p1_bar = int(p1_pct / 100 * bar_width)

    print(f"\nWin balance:")
    bar = '=' * p1_bar + '|' + '-' * (bar_width - p1_bar - 1)
    print(f"  P1 [{bar}] P2")
    print(f"     {p1_pct:.1f}%{' ' * (bar_width - 10)}{p2_pct:.1f}%")


def monitor(output_dir: Path, watch: bool = False, interval: float = 5.0):
    """Run the monitor."""
    while True:
        if watch:
            clear_screen()

        log = load_log(output_dir)

        if log is None:
            print(f"Waiting for training to start...")
            print(f"Looking for: {output_dir / 'training_log.json'}")
        else:
            print_header(log)
            print_config(log)
            print_summary(log)
            print_latest(log)
            print_game_length_trend(log)  # Key early indicator
            print_loss_trend(log)
            print_win_balance(log)

        if watch:
            print(f"\n[Refreshing every {interval}s - Ctrl+C to exit]")
            print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

        if not watch:
            break

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


def main():
    parser = argparse.ArgumentParser(description='Monitor Razzle Dazzle training')
    parser.add_argument('--output', type=Path, default=Path('output'), help='Training output directory')
    parser.add_argument('--watch', '-w', action='store_true', help='Continuously monitor')
    parser.add_argument('--interval', '-i', type=float, default=5.0, help='Refresh interval (seconds)')

    args = parser.parse_args()
    monitor(args.output, args.watch, args.interval)


if __name__ == '__main__':
    main()
