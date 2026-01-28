#!/usr/bin/env python3
"""
Manage the training games archive.

Usage:
    # Show archive stats
    python scripts/manage_archive.py stats

    # List all batches
    python scripts/manage_archive.py list

    # List batches for a specific model
    python scripts/manage_archive.py list --model iter_050

    # Export games from API to archive (one-time migration)
    python scripts/manage_archive.py import-from-api --api-url http://localhost:8000

    # Train on archived games
    python scripts/manage_archive.py train --model output/models/iter_099_bigvalue.pt --epochs 5
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.training.game_archive import GameArchive


def cmd_stats(args):
    """Show archive statistics."""
    archive = GameArchive(args.archive_dir)
    stats = archive.get_stats()

    print(f"Archive: {args.archive_dir}")
    print(f"  Total batches:  {stats['total_batches']}")
    print(f"  Total games:    {stats['total_games']:,}")
    print(f"  Total examples: {stats['total_examples']:,}")
    print(f"  Models:         {', '.join(stats['models']) or 'none'}")

    if stats['total_examples'] > 0:
        # Estimate size
        size_bytes = stats['total_examples'] * (7*8*7 + 3137 + 1 + 3137) * 4  # float32
        size_mb = size_bytes / 1024 / 1024
        print(f"  Est. size:      {size_mb:.1f} MB uncompressed")


def cmd_list(args):
    """List all batches."""
    archive = GameArchive(args.archive_dir)
    batches = archive.list_batches(model_version=args.model)

    if not batches:
        print("No batches in archive")
        return

    print(f"{'Batch ID':<45} {'Model':<15} {'Games':>8} {'Examples':>10} {'Created'}")
    print("-" * 100)
    for b in batches:
        print(f"{b['batch_id']:<45} {b['model_version']:<15} {b['num_games']:>8} {b['num_examples']:>10} {b['created_at'][:19]}")

    total_games = sum(b['num_games'] for b in batches)
    total_examples = sum(b['num_examples'] for b in batches)
    print("-" * 100)
    print(f"{'Total':<45} {'':<15} {total_games:>8} {total_examples:>10}")


def cmd_import_from_api(args):
    """Import games from the training API into the archive."""
    import requests
    import numpy as np
    from razzle.training.selfplay import GameRecord

    archive = GameArchive(args.archive_dir)

    print(f"Fetching games from {args.api_url}...")

    # Fetch all used games
    response = requests.get(
        f"{args.api_url}/training/games",
        params={"status": "used", "limit": 10000}
    )
    response.raise_for_status()
    data = response.json()

    games = data.get("games", [])
    if not games:
        print("No games found in API")
        return

    print(f"Found {len(games)} games")

    # Group by model version
    by_model = {}
    for game in games:
        model = game.get("model_version", "unknown")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(game)

    # Archive each model's games
    for model, model_games in by_model.items():
        print(f"\nArchiving {len(model_games)} games from {model}...")

        # Convert to training examples
        all_states = []
        all_policies = []
        all_values = []
        all_masks = []
        game_results = []

        for game in model_games:
            # Reconstruct training examples from game data
            moves = game.get("moves", [])
            policies = game.get("policies", [])
            result = game.get("result", 0)
            players = game.get("players", [])

            # We need the actual state tensors, which aren't stored in the API
            # This is a limitation - we'd need to replay the games to get states
            print(f"  Warning: Cannot import game {game.get('id')} - state tensors not stored in API")

        print(f"  Note: API doesn't store state tensors. Games must be archived during generation.")


def cmd_train(args):
    """Train on archived games."""
    import torch
    import numpy as np
    from razzle.ai.network import RazzleNet
    from razzle.training.trainer import Trainer, TrainingConfig

    archive = GameArchive(args.archive_dir)
    stats = archive.get_stats()

    if stats['total_examples'] == 0:
        print("No examples in archive")
        return

    print(f"Loading {stats['total_examples']:,} examples from archive...")
    states, policies, values, legal_masks = archive.load_batches(
        model_versions=args.models.split(',') if args.models else None
    )
    print(f"Loaded {len(states):,} examples")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model {args.model} on {device}...")
    net = RazzleNet.load(args.model, device=device)

    # Train
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    trainer = Trainer(net, config)
    history = trainer.train(states, policies, values, legal_masks, verbose=True)

    # Save
    if args.output:
        net.save(args.output)
        print(f"\nSaved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Manage training games archive")
    parser.add_argument('--archive-dir', default='games_archive',
                       help='Archive directory (default: games_archive)')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # stats
    subparsers.add_parser('stats', help='Show archive statistics')

    # list
    list_parser = subparsers.add_parser('list', help='List batches')
    list_parser.add_argument('--model', help='Filter by model version')

    # import-from-api
    import_parser = subparsers.add_parser('import-from-api', help='Import from training API')
    import_parser.add_argument('--api-url', default='http://localhost:8000',
                              help='Training API URL')

    # train
    train_parser = subparsers.add_parser('train', help='Train on archived games')
    train_parser.add_argument('--model', required=True, help='Model to train')
    train_parser.add_argument('--output', help='Output model path')
    train_parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--models', help='Comma-separated model versions to include')

    args = parser.parse_args()

    if args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'import-from-api':
        cmd_import_from_api(args)
    elif args.command == 'train':
        cmd_train(args)


if __name__ == '__main__':
    main()
