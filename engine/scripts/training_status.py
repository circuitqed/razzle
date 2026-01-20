#!/usr/bin/env python3
"""
Training status dashboard for Razzle Dazzle distributed training.

Displays real-time status of distributed training including:
- Worker status and health
- Games generated/collected
- Training progress
- Cost estimates

Usage:
    # One-time status check
    python scripts/training_status.py --output output/distributed

    # Watch mode (continuous updates)
    python scripts/training_status.py --output output/distributed --watch

    # Detailed worker info
    python scripts/training_status.py --output output/distributed --verbose
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_time_ago(iso_time: str) -> str:
    """Format ISO timestamp as 'X ago'."""
    try:
        dt = datetime.fromisoformat(iso_time)
        delta = datetime.now() - dt
        return format_duration(delta.total_seconds()) + " ago"
    except:
        return "unknown"


def load_collector_status(output_dir: Path) -> Optional[dict]:
    """Load collector status from file."""
    status_file = output_dir / "collector_status.json"
    if not status_file.exists():
        return None

    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except:
        return None


def load_training_log(output_dir: Path) -> Optional[dict]:
    """Load training log from file."""
    log_file = output_dir / "training_log.json"
    if not log_file.exists():
        return None

    try:
        with open(log_file, 'r') as f:
            return json.load(f)
    except:
        return None


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_status(output_dir: Path, verbose: bool = False):
    """Print current training status."""
    collector_status = load_collector_status(output_dir)
    training_log = load_training_log(output_dir)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nRazzle Dazzle Training Status - {now}")

    # Overall status
    if collector_status:
        print_header("Collector Status")

        status = collector_status.get('status', 'unknown')
        status_color = {
            'running': '\033[92m',  # Green
            'training': '\033[93m',  # Yellow
            'stopped': '\033[91m',  # Red
        }.get(status, '')
        reset = '\033[0m' if status_color else ''

        print(f"  Status: {status_color}{status.upper()}{reset}")
        print(f"  Iteration: {collector_status.get('iteration', 0)}")
        print(f"  Training count: {collector_status.get('training_count', 0)}")

        # Games
        total = collector_status.get('total_games_collected', 0)
        batch = collector_status.get('games_in_batch', 0)
        threshold = collector_status.get('training_threshold', 100)
        print(f"  Games collected: {total} total, {batch}/{threshold} in current batch")

        # Current model
        print(f"  Current model: {collector_status.get('current_model', 'none')}")

        # Timing
        if collector_status.get('last_training'):
            print(f"  Last training: {format_time_ago(collector_status['last_training'])}")

        start_time = collector_status.get('start_time')
        if start_time:
            uptime = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
            print(f"  Uptime: {format_duration(uptime)}")

        # Cost
        cost = collector_status.get('cost_estimate_usd', 0)
        print(f"  Estimated cost: ${cost:.2f}")

        # Workers
        workers = collector_status.get('workers', [])
        if workers:
            print_header(f"Workers ({len(workers)})")

            # Summary line
            running = sum(1 for w in workers if w.get('status') == 'running')
            unreachable = sum(1 for w in workers if w.get('status') == 'unreachable')
            total_games = sum(w.get('games_collected', 0) for w in workers)
            total_pending = sum(w.get('games_pending', 0) for w in workers)

            print(f"  {running} running, {unreachable} unreachable")
            print(f"  {total_games} games collected, {total_pending} pending\n")

            # Per-worker details
            for w in workers:
                wid = w.get('worker_id', '?')
                wstatus = w.get('status', 'unknown')
                collected = w.get('games_collected', 0)
                pending = w.get('games_pending', 0)
                rate = w.get('games_per_hour', 0)
                model = w.get('model_version', 'unknown')

                status_char = {
                    'running': '\033[92m*\033[0m',  # Green
                    'stopped': '\033[91m-\033[0m',  # Red
                    'unreachable': '\033[91mX\033[0m',  # Red
                }.get(wstatus, '?')

                last_seen = w.get('last_seen')
                seen_str = format_time_ago(last_seen) if last_seen else "never"

                print(f"  [{status_char}] Worker {wid}: {collected} collected, "
                      f"{pending} pending, {rate:.1f}/hr, seen {seen_str}")

                if verbose:
                    print(f"        Model: {model}")
                    if w.get('error_count', 0) > 0:
                        print(f"        Errors: {w['error_count']}")
    else:
        print("\n  Collector status not available")
        print(f"  (Looking for {output_dir / 'collector_status.json'})")

    # Training log
    if training_log:
        iterations = training_log.get('iterations', [])
        if iterations:
            print_header("Training Progress")

            print(f"  Run ID: {training_log.get('run_id', 'unknown')}")
            print(f"  Total games: {training_log.get('total_games', 0)}")
            print(f"  Total examples: {training_log.get('total_examples', 0):,}")
            print(f"  Total time: {format_duration(training_log.get('total_time_sec', 0))}")

            # Latest iteration
            latest = iterations[-1]
            print(f"\n  Latest iteration ({latest.get('iteration', 0)}):")
            print(f"    Games: P1 {latest.get('p1_wins', 0)} / "
                  f"P2 {latest.get('p2_wins', 0)} / "
                  f"Draw {latest.get('draws', 0)}")
            print(f"    Avg length: {latest.get('avg_game_length', 0):.1f} moves")
            print(f"    Loss: {latest.get('final_loss', 0):.4f} "
                  f"(policy={latest.get('final_policy_loss', 0):.4f}, "
                  f"value={latest.get('final_value_loss', 0):.4f})")

            # Loss trend
            if len(iterations) >= 2:
                print("\n  Loss trend (last 5):")
                for it in iterations[-5:]:
                    loss = it.get('final_loss', 0)
                    bar_len = int(loss * 30)
                    bar = '#' * min(bar_len, 40)
                    print(f"    Iter {it.get('iteration', 0):3d}: {loss:.4f} {bar}")

    print("")


def watch_status(output_dir: Path, interval: float = 5.0, verbose: bool = False):
    """Continuously watch training status."""
    try:
        while True:
            clear_screen()
            print_status(output_dir, verbose)
            print(f"\nRefreshing every {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(
        description='Training status dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check status once
    python scripts/training_status.py --output output/distributed

    # Watch mode with 10s refresh
    python scripts/training_status.py --output output/distributed --watch --interval 10

    # Verbose worker details
    python scripts/training_status.py --output output/distributed --verbose
        """
    )

    parser.add_argument('--output', type=Path, default=Path('output/distributed'),
                        help='Output directory containing status files')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Continuously watch and refresh')
    parser.add_argument('--interval', '-i', type=float, default=5.0,
                        help='Refresh interval in seconds (for watch mode)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed worker information')

    args = parser.parse_args()

    if not args.output.exists():
        print(f"Output directory not found: {args.output}")
        print("Make sure training is running or specify correct --output path")
        sys.exit(1)

    if args.watch:
        watch_status(args.output, args.interval, args.verbose)
    else:
        print_status(args.output, args.verbose)


if __name__ == '__main__':
    main()
