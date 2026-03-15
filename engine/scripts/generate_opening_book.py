#!/usr/bin/env python3
"""
Generate an opening book from training games.

Queries the training_games DB, replays each game through GameState,
and builds a position→move distribution map for early-game positions.

Usage:
    python scripts/generate_opening_book.py \
        --db-path server/data/games.db \
        --min-iter 600 --min-visits 20 --depth 8 \
        --output data/opening_book.json
"""

import argparse
import json
import sqlite3
import sys
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add engine root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from razzle.core.state import GameState


def state_key(state: GameState) -> str:
    """Compute position key matching GameState.__hash__ fields."""
    return (
        f"{state.pieces[0]}:{state.balls[0]}:"
        f"{state.pieces[1]}:{state.balls[1]}:"
        f"{state.current_player}:{state.touched_mask}:{int(state.has_passed)}"
    )


def parse_iteration(model_version: str) -> int | None:
    """Extract iteration number from model_version string like 'iter_600'."""
    if not model_version:
        return None
    m = re.search(r'iter_(\d+)', model_version)
    return int(m.group(1)) if m else None


def generate_book(
    db_path: str,
    min_iter: int = 600,
    min_visits: int = 20,
    depth: int = 8,
    prune_pct: float = 0.02,
) -> dict:
    """Generate opening book from training games.

    Args:
        db_path: Path to SQLite database with training_games table.
        min_iter: Minimum model iteration to include (filters weak models).
        min_visits: Minimum visits for a position to be included in the book.
        depth: Maximum number of moves (actions) to track per game.
        prune_pct: Prune moves with less than this fraction of total visits.

    Returns:
        Book dict with metadata and positions.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find iteration range
    rows = conn.execute(
        "SELECT DISTINCT model_version FROM training_games WHERE model_version IS NOT NULL"
    ).fetchall()

    versions = []
    for row in rows:
        it = parse_iteration(row["model_version"])
        if it is not None and it >= min_iter:
            versions.append(row["model_version"])

    if not versions:
        print(f"No games found with iteration >= {min_iter}")
        conn.close()
        return {"metadata": {}, "positions": {}}

    # Build SQL filter for matching versions
    placeholders = ",".join("?" * len(versions))
    query = f"""
        SELECT moves, result FROM training_games
        WHERE model_version IN ({placeholders})
    """
    cursor = conn.execute(query, versions)

    # Position tracking: key -> {move_action: [count, win_sum]}
    positions: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    games_processed = 0
    min_iter_seen = float('inf')
    max_iter_seen = 0

    for row in cursor:
        moves = json.loads(row["moves"])
        result = float(row["result"])  # 1.0 = player 0 wins, -1.0 = player 1 wins

        state = GameState.new_game()
        turns_completed = 0

        # Normalize result to [0, 1]: 1.0 -> 1.0, -1.0 -> 0.0
        p0_win = (result + 1.0) / 2.0

        for action in moves:
            if turns_completed >= depth:
                break

            # Record this position→move pair
            key = state_key(state)
            # Win rate from perspective of current player
            if state.current_player == 0:
                win_rate = p0_win
            else:
                win_rate = 1.0 - p0_win

            positions[key][action][0] += 1
            positions[key][action][1] += win_rate

            # Track turn changes (knight moves and end-turn switch player)
            prev_player = state.current_player
            state.apply_move(action)
            if state.current_player != prev_player:
                turns_completed += 1

        games_processed += 1
        if games_processed % 10000 == 0:
            print(f"  Processed {games_processed} games, {len(positions)} unique positions...")

    conn.close()

    # Determine actual iteration range from versions used
    for v in versions:
        it = parse_iteration(v)
        if it is not None:
            min_iter_seen = min(min_iter_seen, it)
            max_iter_seen = max(max_iter_seen, it)

    print(f"\nProcessed {games_processed} games from iter_{int(min_iter_seen)}-iter_{max_iter_seen}")
    print(f"Found {len(positions)} unique positions before filtering")

    # Filter and format
    book_positions = {}
    for key, move_dict in positions.items():
        total = sum(v[0] for v in move_dict.values())
        if total < min_visits:
            continue

        # Build move entries: [action_index, count, win_rate]
        moves_list = []
        for action, (count, win_sum) in sorted(move_dict.items(), key=lambda x: -x[1][0]):
            freq = count / total
            if freq < prune_pct:
                continue
            win_rate = round(win_sum / count, 3)
            moves_list.append([action, count, win_rate])

        if moves_list:
            book_positions[key] = {
                "moves": moves_list,
                "total": total,
            }

    print(f"Book contains {len(book_positions)} positions (min_visits={min_visits})")

    return {
        "metadata": {
            "model_range": f"iter_{int(min_iter_seen)}-iter_{max_iter_seen}",
            "games": games_processed,
            "min_visits": min_visits,
            "depth": depth,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "positions": book_positions,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate opening book from training games")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--min-iter", type=int, default=600, help="Minimum model iteration")
    parser.add_argument("--min-visits", type=int, default=20, help="Minimum position visits")
    parser.add_argument("--depth", type=int, default=8, help="Max moves per game to track")
    parser.add_argument("--output", default="data/opening_book.json", help="Output file path")
    parser.add_argument("--prune-pct", type=float, default=0.02, help="Prune moves below this frequency")
    args = parser.parse_args()

    if not Path(args.db_path).exists():
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    print(f"Generating opening book from {args.db_path}")
    print(f"  min_iter={args.min_iter}, min_visits={args.min_visits}, depth={args.depth}")

    book = generate_book(
        db_path=args.db_path,
        min_iter=args.min_iter,
        min_visits=args.min_visits,
        depth=args.depth,
        prune_pct=args.prune_pct,
    )

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(book, f, separators=(",", ":"))

    size_kb = output_path.stat().st_size / 1024
    print(f"\nWrote {output_path} ({size_kb:.1f} KB)")
    print(f"  {len(book['positions'])} positions")


if __name__ == "__main__":
    main()
