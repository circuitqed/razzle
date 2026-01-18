#!/usr/bin/env python3
"""
Local training script for Razzle Dazzle.

Runs self-play and training on local GPU.
"""

import argparse
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet, NetworkConfig, create_network
from razzle.training.selfplay import SelfPlay, games_to_training_data
from razzle.training.trainer import Trainer, TrainingConfig


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

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Create or load network
    if args.resume:
        print(f"Resuming from {args.resume}")
        network = RazzleNet.load(args.resume, device=args.device)
        start_iter = int(args.resume.stem.split('_')[-1]) + 1
    else:
        print(f"Creating new network: {args.filters} filters, {args.blocks} blocks")
        network = create_network(args.filters, args.blocks, args.device)
        start_iter = 0

    print(f"Network has {network.num_parameters():,} parameters")
    print(f"Device: {args.device}")

    for iteration in range(start_iter, start_iter + args.iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}")
        print(f"{'='*50}")

        # Self-play
        print(f"\nGenerating {args.games_per_iter} self-play games...")
        games_dir = args.output / f"games_iter_{iteration:03d}"

        selfplay = SelfPlay(
            network=network,
            device=args.device,
            num_simulations=args.simulations,
            batch_size=1
        )

        games = selfplay.generate_games(
            args.games_per_iter,
            output_dir=games_dir,
            verbose=False
        )

        # Analyze games
        wins_p1 = sum(1 for g in games if g.result == 1.0)
        wins_p2 = sum(1 for g in games if g.result == -1.0)
        draws = sum(1 for g in games if g.result == 0.0)
        avg_length = sum(len(g.moves) for g in games) / len(games)

        print(f"Games: P1 wins {wins_p1}, P2 wins {wins_p2}, Draws {draws}")
        print(f"Average game length: {avg_length:.1f} moves")

        # Convert to training data
        states, policies, values = games_to_training_data(games)
        print(f"Training examples: {len(states)}")

        # Train
        print(f"\nTraining for {args.epochs} epochs...")
        config = TrainingConfig(
            epochs=args.epochs,
            device=args.device
        )
        trainer = Trainer(network, config)
        history = trainer.train(states, policies, values, verbose=True)

        # Save checkpoint
        checkpoint_path = args.output / f"model_iter_{iteration:03d}.pt"
        network.save(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
