#!/usr/bin/env python3
"""
Standalone trainer for Razzle Dazzle distributed training.

This script:
1. Polls the training API for pending games
2. When enough games are collected, converts to training data
3. Trains the neural network
4. Uploads the new model to the API

Can run locally (CPU/GPU) or on a cloud GPU instance.

Usage:
    python trainer.py --api-url http://server:8000 --device cuda --threshold 50
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Optional

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet, create_network, NUM_ACTIONS
from razzle.core.state import GameState
from razzle.training.trainer import Trainer, TrainingConfig
from razzle.training.api_client import TrainingAPIClient, TrainingGame


def games_to_training_data(
    games: list[TrainingGame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert API games to training arrays.

    This reconstructs board states from moves and converts sparse
    visit counts to dense policy vectors.

    Returns (states, policies, values) arrays.
    """
    all_states = []
    all_policies = []
    all_values = []

    for game in games:
        # Replay the game to get states
        state = GameState.new_game()
        states = []
        policies = []

        for i, (move, visit_counts) in enumerate(zip(game.moves, game.visit_counts)):
            # Get state tensor
            states.append(state.to_tensor())

            # Convert sparse visit counts to dense policy
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            total_visits = sum(visit_counts.values())
            if total_visits > 0:
                for m, count in visit_counts.items():
                    policy[m] = count / total_visits
            policies.append(policy)

            # Apply move
            state.apply_move(move)

        # Calculate values for each position
        for i in range(len(states)):
            player_to_move = i % 2

            # Value from perspective of player to move
            if game.result == 0:
                value = 0.0
            elif player_to_move == 0:
                value = game.result
            else:
                value = -game.result

            all_states.append(states[i])
            all_policies.append(policies[i])
            all_values.append(value)

    return (
        np.stack(all_states),
        np.stack(all_policies),
        np.array(all_values, dtype=np.float32),
    )


class DistributedTrainer:
    """
    Trainer that fetches games from API and uploads models.
    """

    def __init__(
        self,
        api_url: str,
        device: str = 'cuda',
        threshold: int = 50,
        poll_interval: int = 30,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        filters: int = 64,
        blocks: int = 6,
        output_dir: Path = Path('output/trainer'),
    ):
        self.api_url = api_url
        self.device = device
        self.threshold = threshold
        self.poll_interval = poll_interval
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.filters = filters
        self.blocks = blocks
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # API client
        self.api_client = TrainingAPIClient(base_url=api_url)

        # Network
        self.network: Optional[RazzleNet] = None
        self.iteration = 0

        # State
        self.shutdown_event = Event()
        self.total_games_trained = 0
        self.start_time = datetime.now()

    def _load_or_create_network(self) -> bool:
        """Load latest model from API or create new one."""
        try:
            model_info = self.api_client.get_latest_model()
            if model_info:
                # Download and load
                model_path = self.models_dir / f"{model_info.version}.pt"
                print(f"[Trainer] Downloading model: {model_info.version}")
                self.api_client.download_model(model_info.version, model_path)
                self.network = RazzleNet.load(model_path, device=self.device)
                self.iteration = model_info.iteration
                print(f"[Trainer] Loaded model: {model_info.version} (iteration {self.iteration})")
            else:
                # Create new network
                self.network = create_network(self.filters, self.blocks, self.device)
                self.iteration = 0
                print(f"[Trainer] Created new network")
            return True
        except Exception as e:
            print(f"[Trainer] Error loading network: {e}")
            return False

    def train_on_games(self, games: list[TrainingGame]) -> dict:
        """
        Train on a batch of games.

        Returns training metrics.
        """
        print(f"[Trainer] Converting {len(games)} games to training data...")
        states, policies, values = games_to_training_data(games)
        print(f"[Trainer] Training examples: {len(states)}")

        # Create trainer
        config = TrainingConfig(
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            device=self.device,
        )
        trainer = Trainer(self.network, config)

        # Train
        history = trainer.train(states, policies, values, verbose=True)

        # Get final metrics
        final = history[-1] if history else {}

        return {
            'games': len(games),
            'examples': len(states),
            'final_loss': final.get('loss', 0),
            'final_policy_loss': final.get('policy_loss', 0),
            'final_value_loss': final.get('value_loss', 0),
            'epochs': len(history),
        }

    def save_and_upload_model(self, metrics: dict) -> str:
        """Save model locally and upload to API."""
        self.iteration += 1
        version = f"iter_{self.iteration:03d}"

        # Save locally
        model_path = self.models_dir / f"{version}.pt"
        self.network.save(model_path)
        print(f"[Trainer] Saved model: {model_path}")

        # Upload to API
        try:
            self.api_client.upload_model(
                version=version,
                iteration=self.iteration,
                file_path=model_path,
                games_trained_on=metrics.get('games'),
                final_loss=metrics.get('final_loss'),
                final_policy_loss=metrics.get('final_policy_loss'),
                final_value_loss=metrics.get('final_value_loss'),
            )
            print(f"[Trainer] Uploaded model: {version}")
        except Exception as e:
            print(f"[Trainer] Failed to upload model: {e}")

        return version

    def run(self):
        """Main trainer loop."""
        print(f"[Trainer] Starting")
        print(f"[Trainer] API URL: {self.api_url}")
        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Training threshold: {self.threshold} games")
        print(f"[Trainer] Poll interval: {self.poll_interval}s")

        # Wait for API
        print(f"[Trainer] Waiting for API server...")
        if not self.api_client.wait_for_server(timeout=120):
            print(f"[Trainer] API server not available")
            return

        print(f"[Trainer] API server connected")

        # Load network
        if not self._load_or_create_network():
            return

        print(f"[Trainer] Starting training loop")

        try:
            while not self.shutdown_event.is_set():
                # Check for pending games
                try:
                    games, total_pending = self.api_client.fetch_pending_games(
                        limit=self.threshold * 2,  # Fetch more than threshold
                        mark_used=False,  # Don't mark yet, just check count
                    )

                    if len(games) >= self.threshold:
                        print(f"\n[Trainer] Training on {len(games)} games (iteration {self.iteration + 1})")

                        # Now actually fetch and mark as used
                        games, _ = self.api_client.fetch_pending_games(
                            limit=len(games),
                            mark_used=True,
                        )

                        # Train
                        start_time = time.time()
                        metrics = self.train_on_games(games)
                        train_time = time.time() - start_time

                        print(f"[Trainer] Training completed in {train_time:.1f}s")
                        self.total_games_trained += len(games)

                        # Save and upload
                        version = self.save_and_upload_model(metrics)

                        print(f"[Trainer] Iteration {self.iteration} complete: "
                              f"loss={metrics['final_loss']:.4f}, "
                              f"games={metrics['games']}, "
                              f"examples={metrics['examples']}")

                    else:
                        pending = len(games)
                        print(f"[Trainer] Waiting for games: {pending}/{self.threshold} pending")

                except Exception as e:
                    print(f"[Trainer] Error fetching games: {e}")

                # Wait before next poll
                self.shutdown_event.wait(self.poll_interval)

        except Exception as e:
            print(f"[Trainer] Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print(f"[Trainer] Stopped. Total games trained: {self.total_games_trained}")

    def shutdown(self):
        """Signal the trainer to stop."""
        print(f"[Trainer] Shutdown requested")
        self.shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(description='Distributed trainer')
    parser.add_argument('--api-url', type=str, required=True,
                        help='Training API URL (e.g., http://server:8000)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Number of games before training')
    parser.add_argument('--poll-interval', type=int, default=30,
                        help='Seconds between checking for games')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--filters', type=int, default=64,
                        help='Network filter count')
    parser.add_argument('--blocks', type=int, default=6,
                        help='Network residual blocks')
    parser.add_argument('--output', type=Path, default=Path('output/trainer'),
                        help='Output directory')

    args = parser.parse_args()

    # Ensure unbuffered output
    os.environ['PYTHONUNBUFFERED'] = '1'

    trainer = DistributedTrainer(
        api_url=args.api_url,
        device=args.device,
        threshold=args.threshold,
        poll_interval=args.poll_interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        filters=args.filters,
        blocks=args.blocks,
        output_dir=args.output,
    )

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        trainer.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    trainer.run()


if __name__ == '__main__':
    main()
