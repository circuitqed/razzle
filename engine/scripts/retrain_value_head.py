#!/usr/bin/env python3
"""
Retrain the value head with larger architecture.

This script:
1. Loads a model with transferred tower/policy weights but fresh value head
2. Generates self-play games
3. Trains on them, focusing on improving the value head

The tower and policy are already trained, so this should converge quickly
on better value predictions.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time

from razzle.ai.network import RazzleNet
from razzle.ai.evaluator import BatchedEvaluator
from razzle.training.selfplay import SelfPlay
from razzle.training.trainer import Trainer, TrainingConfig


def generate_games(model_path: str, num_games: int, simulations: int = 400, device: str = 'cuda'):
    """Generate self-play games."""
    print(f"\nGenerating {num_games} self-play games...")

    net = RazzleNet.load(model_path, device=device)
    net.eval()

    evaluator = BatchedEvaluator(net, device=device)
    selfplay = SelfPlay(
        network=net,
        device=device,
        num_simulations=simulations,
        temperature=1.0,
        temperature_moves=15,
        random_opening_fraction=0.3,  # More opening diversity
        random_opening_moves=4,
    )

    all_examples = []
    game_lengths = []
    results = []

    start = time.time()
    for i in range(num_games):
        record = selfplay.play_game()
        examples = record.training_examples(use_ball_shaping=False)
        all_examples.extend(examples)
        game_lengths.append(len(record.moves))
        results.append(record.result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  Game {i+1}/{num_games} ({rate:.1f} games/min) - "
                  f"avg length: {np.mean(game_lengths):.1f}, "
                  f"examples: {len(all_examples)}")

    elapsed = time.time() - start
    print(f"\nGenerated {num_games} games in {elapsed:.1f}s ({num_games/elapsed*60:.1f} games/min)")
    print(f"Total examples: {len(all_examples)}")
    print(f"Avg game length: {np.mean(game_lengths):.1f}")
    print(f"Results: P0 wins={sum(1 for r in results if r > 0)}, "
          f"P1 wins={sum(1 for r in results if r < 0)}, "
          f"draws={sum(1 for r in results if r == 0)}")

    return all_examples


def train_on_examples(net, examples, config, epochs=1):
    """Train on examples for specified epochs."""
    # Prepare data
    states = np.array([e[0] for e in examples])
    policies = np.array([e[1] for e in examples])
    values = np.array([e[2] for e in examples])
    legal_masks = np.array([e[3] for e in examples])

    # Update config epochs
    config.epochs = epochs

    trainer = Trainer(net, config)
    history = trainer.train(states, policies, values, legal_masks, verbose=True)

    return history[-1] if history else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models_local/iter_099_bigvalue.pt',
                       help='Model to train')
    parser.add_argument('--games', type=int, default=200,
                       help='Games to generate per iteration')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Training iterations')
    parser.add_argument('--simulations', type=int, default=400,
                       help='MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Training epochs per iteration')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='models_local/retrained.pt',
                       help='Output model path')
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Model: {args.model}")

    # Load model
    net = RazzleNet.load(args.model, device=args.device)
    print(f"Loaded model: {net.config.num_filters} filters, {net.config.num_blocks} blocks")
    print(f"Value head: {net.config.value_filters} filters, {net.config.value_hidden} hidden")

    config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=256,
        value_weight=1.0,
        illegal_penalty_weight=0.1,
    )

    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print('='*60)

        # Generate games with current model
        examples = generate_games(
            args.model if iteration == 0 else args.output,
            args.games,
            simulations=args.simulations,
            device=args.device,
        )

        # Train on generated games
        print(f"\nTraining for {args.epochs} epochs...")
        net = RazzleNet.load(
            args.model if iteration == 0 else args.output,
            device=args.device
        )

        metrics = train_on_examples(net, examples, config, epochs=args.epochs)
        print(f"Final metrics: loss={metrics.get('loss', 0):.4f}, "
              f"policy={metrics.get('policy_loss', 0):.4f}, "
              f"value={metrics.get('value_loss', 0):.4f}")

        # Save model
        net.save(args.output)
        print(f"Saved to {args.output}")

        # Quick evaluation
        print("\nQuick value head test...")
        net.eval()
        from razzle.core.state import GameState
        from razzle.ai.evaluator import BatchedEvaluator

        evaluator = BatchedEvaluator(net, device=args.device)
        state = GameState.new_game()
        _, value = evaluator.evaluate(state)
        print(f"  Opening position value: {value:.3f} (should be ~0)")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final model saved to: {args.output}")


if __name__ == '__main__':
    main()
