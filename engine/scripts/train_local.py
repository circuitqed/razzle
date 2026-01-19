#!/usr/bin/env python3
"""
Local training script for Razzle Dazzle.

Runs self-play and training on local GPU.
Logs metrics to training_log.json for monitoring.

Modes:
  - Default: Run both self-play and training
  - --selfplay-only: Only generate games, skip training (for parallel workers)
  - --train-only --games-dir DIR: Only train on existing games (for aggregated training)
"""

import argparse
import pickle
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet, NetworkConfig, create_network
from razzle.training.selfplay import SelfPlay, games_to_training_data
from razzle.training.trainer import Trainer, TrainingConfig
from razzle.training.logger import TrainingLogger


def load_games_from_dir(games_dir: Path):
    """Load games from a directory of pickle files."""
    all_games = []
    for pkl_file in sorted(games_dir.glob("*.pkl")):
        with open(pkl_file, 'rb') as f:
            games = pickle.load(f)
            if isinstance(games, list):
                all_games.extend(games)
    return all_games


def main():
    parser = argparse.ArgumentParser(description='Local training for Razzle Dazzle')
    parser.add_argument('--iterations', type=int, default=10, help='Training iterations')
    parser.add_argument('--games-per-iter', type=int, default=100, help='Games per iteration')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs per iteration')
    parser.add_argument('--filters', type=int, default=64, help='Network filters')
    parser.add_argument('--blocks', type=int, default=6, help='Network residual blocks')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=Path('output'), help='Output directory')
    parser.add_argument('--resume', type=Path, help='Resume from checkpoint')

    # Mode flags for parallel training
    parser.add_argument('--selfplay-only', action='store_true', help='Only generate games, skip training')
    parser.add_argument('--train-only', action='store_true', help='Only train on existing games')
    parser.add_argument('--games-dir', type=Path, help='Directory with games to train on (for --train-only)')

    args = parser.parse_args()

    # Validate mode flags
    if args.train_only and args.selfplay_only:
        parser.error("Cannot use both --train-only and --selfplay-only")
    if args.train_only and not args.games_dir:
        parser.error("--train-only requires --games-dir")

    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize logger with training config
    logger = TrainingLogger(args.output, config={
        'iterations': args.iterations,
        'games_per_iter': args.games_per_iter,
        'simulations': args.simulations,
        'epochs': args.epochs,
        'filters': args.filters,
        'blocks': args.blocks,
        'device': args.device
    }, device=args.device)

    # Create or load network
    if args.resume:
        print(f"Resuming from {args.resume}")
        network = RazzleNet.load(args.resume, device=args.device)
        # Try to extract iteration number from filename (e.g., model_iter_005.pt -> 5)
        try:
            start_iter = int(args.resume.stem.split('_')[-1]) + 1
        except ValueError:
            # Filename doesn't end with a number (e.g., model_start.pt)
            start_iter = 0
    else:
        print(f"Creating new network: {args.filters} filters, {args.blocks} blocks")
        network = create_network(args.filters, args.blocks, args.device)
        start_iter = 0

    print(f"Network has {network.num_parameters():,} parameters")
    print(f"Device: {args.device}")
    print(f"Logging to: {args.output / 'training_log.json'}")

    # Train-only mode: load existing games and train
    if args.train_only:
        print(f"\nTrain-only mode: loading games from {args.games_dir}")
        games = load_games_from_dir(args.games_dir)
        print(f"Loaded {len(games)} games")

        states, policies, values = games_to_training_data(games)
        print(f"Training examples: {len(states)}")

        print(f"\nTraining for {args.epochs} epochs...")
        config = TrainingConfig(epochs=args.epochs, device=args.device)
        trainer = Trainer(network, config)
        history = trainer.train(states, policies, values, verbose=True)

        checkpoint_path = args.output / "trained_model.pt"
        network.save(checkpoint_path)
        print(f"Saved model to {checkpoint_path}")
        return

    for iteration in range(start_iter, start_iter + args.iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}")
        print(f"{'='*50}")

        # Start iteration timing
        logger.start_iteration(iteration)

        # Self-play
        print(f"\nGenerating {args.games_per_iter} self-play games...")
        games_dir = args.output / f"games_iter_{iteration:03d}"

        selfplay = SelfPlay(
            network=network,
            device=args.device,
            num_simulations=args.simulations,
            batch_size=1
        )

        logger.start_selfplay()
        games = selfplay.generate_games(
            args.games_per_iter,
            output_dir=games_dir,
            verbose=False
        )
        iter_metrics = logger.end_selfplay(games)

        print(f"Games: P1 wins {iter_metrics.p1_wins}, P2 wins {iter_metrics.p2_wins}, Draws {iter_metrics.draws}")
        print(f"Game length: {iter_metrics.avg_game_length:.1f} avg (min={iter_metrics.min_game_length}, max={iter_metrics.max_game_length}, std={iter_metrics.std_game_length:.1f})")
        print(f"Self-play time: {iter_metrics.selfplay_time_sec:.1f}s")

        # Convert to training data
        states, policies, values = games_to_training_data(games)
        print(f"Training examples: {len(states)}")

        # Skip training in selfplay-only mode
        if args.selfplay_only:
            print("Self-play only mode - skipping training")
            # Save games for later aggregation
            games_pkl = games_dir / "all_games.pkl"
            with open(games_pkl, 'wb') as f:
                pickle.dump(games, f)
            print(f"Saved games to {games_pkl}")
            continue

        # Train
        print(f"\nTraining for {args.epochs} epochs...")
        config = TrainingConfig(
            epochs=args.epochs,
            device=args.device
        )
        trainer = Trainer(network, config)

        logger.start_training()
        history = trainer.train(states, policies, values, verbose=True)
        logger.end_training(iter_metrics, history)

        print(f"Training time: {iter_metrics.training_time_sec:.1f}s")
        print(f"Total iteration time: {iter_metrics.total_time_sec:.1f}s")

        # Save checkpoint
        checkpoint_path = args.output / f"model_iter_{iteration:03d}.pt"
        network.save(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("\nTraining complete!")
    logger.print_status()


if __name__ == '__main__':
    main()
