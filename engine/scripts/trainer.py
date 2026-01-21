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

from razzle.ai.network import RazzleNet, create_network, NUM_ACTIONS, END_TURN_ACTION
from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.training.trainer import Trainer as NetworkTrainer, TrainingConfig
from razzle.training.api_client import TrainingAPIClient, TrainingGame


def compute_difficulty_target(raw_policy: np.ndarray, mcts_policy: np.ndarray) -> float:
    """
    Compute difficulty as KL divergence between raw and MCTS policies.

    Returns value in [0, 1] where:
    - 0 = MCTS completely agrees with network (easy)
    - 1 = MCTS found very different answer (hard)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    raw_policy = np.clip(raw_policy, eps, 1.0)
    mcts_policy = np.clip(mcts_policy, eps, 1.0)

    # Only compute KL over moves where mcts_policy > eps
    # KL divergence: sum(mcts * log(mcts / raw))
    mask = mcts_policy > eps
    if not mask.any():
        return 0.0

    kl_div = np.sum(mcts_policy[mask] * np.log(mcts_policy[mask] / raw_policy[mask]))

    # Normalize to [0, 1] - KL of 2.0 maps to difficulty 1.0
    KL_NORMALIZATION = 2.0
    return min(1.0, kl_div / KL_NORMALIZATION)


def games_to_training_data(
    games: list[TrainingGame],
    temperature: float = 1.0,
    network: Optional[RazzleNet] = None,
    device: str = 'cpu',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Convert API games to training arrays.

    This reconstructs board states from moves and converts sparse
    visit counts to dense policy vectors.

    IMPORTANT: Player tracking must replay the game state to get the actual
    current_player at each position. In Razzle Dazzle, turns DON'T strictly
    alternate - ball passes keep the same player, only knight moves and
    end_turn switch players.

    Args:
        games: List of training games from API
        temperature: Temperature used during self-play (for policy conversion)
        network: Optional network for computing difficulty targets.
                 If provided, computes KL divergence between raw and MCTS policies.
        device: Device for network inference when computing difficulty.

    Returns (states, policies, values, legal_masks, difficulties) arrays.
            difficulties is None if network is not provided.
    """
    all_states = []
    all_policies = []
    all_values = []
    all_legal_masks = []
    all_difficulties = [] if network is not None else None

    # Helper to map move to policy index
    def move_to_index(m: int) -> int:
        return END_TURN_ACTION if m == -1 else m

    for game in games:
        # Replay the game to get states and track actual player
        state = GameState.new_game()
        states = []
        policies = []
        legal_masks = []
        players = []  # Track actual current_player at each position

        for move, visit_counts in zip(game.moves, game.visit_counts):
            # Record state and player BEFORE applying move
            states.append(state.to_tensor())
            players.append(state.current_player)

            # Generate legal move mask for this state
            legal_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
            legal_moves = get_legal_moves(state)
            for m in legal_moves:
                legal_mask[move_to_index(m)] = 1.0
            legal_masks.append(legal_mask)

            # Convert sparse visit counts to dense policy with temperature
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            total_visits = sum(visit_counts.values())
            if total_visits > 0:
                # Apply temperature to visit counts (same as MCTS.get_policy)
                if temperature > 0 and temperature != 1.0:
                    moves = list(visit_counts.keys())
                    visits_array = np.array([visit_counts[m] for m in moves], dtype=np.float32)
                    visits_array = np.power(visits_array, 1.0 / temperature)
                    visits_sum = visits_array.sum()
                    if visits_sum > 0:
                        for idx, m in enumerate(moves):
                            policy[move_to_index(m)] = visits_array[idx] / visits_sum
                else:
                    # temperature == 1.0: simple proportional
                    for m, count in visit_counts.items():
                        policy[move_to_index(m)] = count / total_visits
            policies.append(policy)

            # Apply move to advance state (this updates current_player correctly)
            state.apply_move(move)

        # Calculate values for each position using ACTUAL player
        for i in range(len(states)):
            player_to_move = players[i]  # Use tracked player, NOT i % 2

            # Value from perspective of player to move
            if game.result == 0:
                value = 0.0
            elif player_to_move == 0:
                # Player 0 was to move; game.result is +1 if P0 won, -1 if P1 won
                value = game.result
            else:
                # Player 1 was to move; flip the result
                value = -game.result

            all_states.append(states[i])
            all_policies.append(policies[i])
            all_values.append(value)
            all_legal_masks.append(legal_masks[i])

    # Compute difficulty targets if network provided
    if network is not None and all_states:
        network.eval()
        states_tensor = torch.from_numpy(np.stack(all_states)).to(device)

        with torch.no_grad():
            log_policies, _, _ = network(states_tensor)
            raw_policies = torch.exp(log_policies).cpu().numpy()

        # Compute KL divergence for each position
        mcts_policies = np.stack(all_policies)
        for i in range(len(all_states)):
            difficulty = compute_difficulty_target(raw_policies[i], mcts_policies[i])
            all_difficulties.append(difficulty)

    return (
        np.stack(all_states),
        np.stack(all_policies),
        np.array(all_values, dtype=np.float32),
        np.stack(all_legal_masks),
        np.array(all_difficulties, dtype=np.float32) if all_difficulties else None,
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

        # Network and trainer (created once, reused across iterations)
        self.network: Optional[RazzleNet] = None
        self.network_trainer: Optional[NetworkTrainer] = None  # Reuse to preserve optimizer state
        self.iteration = 0

        # State
        self.shutdown_event = Event()
        self.total_games_trained = 0
        self.start_time = datetime.now()

        # Games log for analysis
        self.games_log_path = self.output_dir / 'games_log.jsonl'

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
                # No model available, create new network
                self.network = create_network(self.filters, self.blocks, self.device)
                self.iteration = 0
                print(f"[Trainer] No model found, created new network")
        except Exception as e:
            # API error (e.g., 500) - fall back to creating new network
            print(f"[Trainer] API error checking for model: {e}")
            print(f"[Trainer] Falling back to new network")
            try:
                self.network = create_network(self.filters, self.blocks, self.device)
                self.iteration = 0
            except Exception as e2:
                print(f"[Trainer] Error creating network: {e2}")
                return False

        # Create trainer once (preserves optimizer state across iterations)
        config = TrainingConfig(
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            device=self.device,
        )
        self.network_trainer = NetworkTrainer(self.network, config)
        print(f"[Trainer] Created trainer with Adam optimizer (lr={self.learning_rate})")
        return True

    def _save_games_to_log(self, games: list[TrainingGame], iteration: int):
        """Save games to JSONL log for later analysis."""
        with open(self.games_log_path, 'a') as f:
            for game in games:
                record = {
                    'iteration': iteration,
                    'model_version': game.model_version,
                    'worker_id': game.worker_id,
                    'moves': game.moves,
                    'result': game.result,
                    'visit_counts': game.visit_counts,
                }
                f.write(json.dumps(record) + '\n')

    def train_on_games(self, games: list[TrainingGame]) -> dict:
        """
        Train on a batch of games.

        Returns training metrics.
        """
        print(f"[Trainer] Converting {len(games)} games to training data...")
        states, policies, values, legal_masks, difficulties = games_to_training_data(
            games,
            network=self.network,
            device=self.device,
        )
        print(f"[Trainer] Training examples: {len(states)}")
        if difficulties is not None:
            avg_difficulty = difficulties.mean()
            print(f"[Trainer] Average difficulty target: {avg_difficulty:.3f}")

        # Reuse trainer to preserve optimizer momentum across iterations
        history = self.network_trainer.train(
            states, policies, values,
            legal_masks=legal_masks,
            difficulties=difficulties,
            verbose=True
        )

        # Get final metrics
        final = history[-1] if history else {}

        return {
            'games': len(games),
            'examples': len(states),
            'final_loss': final.get('loss', 0),
            'final_policy_loss': final.get('policy_loss', 0),
            'final_value_loss': final.get('value_loss', 0),
            'final_difficulty_loss': final.get('difficulty_loss', 0),
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

                        # Save games to log for analysis
                        self._save_games_to_log(games, self.iteration + 1)

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
    parser.add_argument('--filters', type=int, default=None,
                        help='Network filter count (overrides --network-size)')
    parser.add_argument('--blocks', type=int, default=None,
                        help='Network residual blocks (overrides --network-size)')
    parser.add_argument('--network-size', type=str, default='medium', choices=['small', 'medium', 'large'],
                        help='Network size preset: small (64f/6b), medium (128f/10b), large (256f/15b)')
    parser.add_argument('--output', type=Path, default=Path('output/trainer'),
                        help='Output directory')

    args = parser.parse_args()

    # Resolve network size presets
    NETWORK_PRESETS = {
        'small': (64, 6),      # ~900K params, fast inference
        'medium': (128, 10),   # ~3.5M params, balanced
        'large': (256, 15),    # ~15M params, stronger but slower
    }

    if args.filters is not None and args.blocks is not None:
        filters, blocks = args.filters, args.blocks
    else:
        filters, blocks = NETWORK_PRESETS[args.network_size]
        if args.filters is not None:
            filters = args.filters
        if args.blocks is not None:
            blocks = args.blocks

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
        filters=filters,
        blocks=blocks,
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
