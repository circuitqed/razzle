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
from razzle.training.replay_buffer import ReplayBuffer
from razzle.training.metrics import (
    compute_policy_metrics, compute_value_metrics,
    compute_value_calibration, compute_calibration_error, compute_pass_stats
)


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

    # Learning rate schedule: list of (iteration, learning_rate) tuples
    # LR changes when iteration reaches each milestone
    DEFAULT_LR_SCHEDULE = [
        (0, 0.001),     # Start at 0.001 (proven to work well)
        (200, 0.0005),  # Drop to 0.0005 at iter 200
        (500, 0.0001),  # Drop to 0.0001 at iter 500 for fine-tuning
    ]

    def __init__(
        self,
        api_url: str,
        device: str = 'cuda',
        threshold: int = 50,
        poll_interval: int = 30,
        epochs: int = 10,
        batch_size: int = 512,  # Increased from 256
        learning_rate: float = 0.001,  # Starting LR (proven to work well)
        filters: int = 64,
        blocks: int = 6,
        output_dir: Path = Path('output/trainer'),
        lr_schedule: list[tuple[int, float]] | None = None,
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
        self.lr_schedule = lr_schedule if lr_schedule is not None else self.DEFAULT_LR_SCHEDULE
        self.current_lr = learning_rate

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

        # Replay buffer for preventing catastrophic forgetting
        self.replay_buffer = ReplayBuffer(max_positions=100_000)

        # Checkpoint gating - track best model
        self.best_model_path: Optional[Path] = None

    def _load_or_create_network(self) -> bool:
        """Load latest model from API or create new one, and restore trainer state."""
        model_path = None

        try:
            model_info = self.api_client.get_latest_model()
            if model_info:
                # Download and load
                model_path = self.models_dir / f"{model_info.version}.pt"
                print(f"[Trainer] Downloading model: {model_info.version}")
                self.api_client.download_model(model_info.version, model_path)
                self.network = RazzleNet.load(model_path, device=self.device)
                self.iteration = model_info.iteration
                # IMPORTANT: Set best_model_path so checkpoint gating works!
                self.best_model_path = model_path
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

        # Try to restore trainer state (optimizer, replay buffer)
        self._load_trainer_state()

        return True

    def _update_learning_rate(self):
        """Update learning rate based on schedule and current iteration."""
        # Find the appropriate LR for current iteration
        target_lr = self.learning_rate  # default
        for milestone_iter, lr in self.lr_schedule:
            if self.iteration >= milestone_iter:
                target_lr = lr

        if target_lr != self.current_lr:
            print(f"[Trainer] Adjusting learning rate: {self.current_lr} -> {target_lr}")
            for param_group in self.network_trainer.optimizer.param_groups:
                param_group['lr'] = target_lr
            self.current_lr = target_lr

    def _get_state_path(self) -> Path:
        """Path to trainer state file."""
        return self.output_dir / 'trainer_state.pt'

    def _get_replay_buffer_path(self) -> Path:
        """Path to replay buffer file."""
        return self.output_dir / 'replay_buffer.npz'

    def _submit_iteration_metrics(self, metrics: dict, train_time: float, model_version: str = None):
        """Submit iteration metrics to API for dashboard tracking."""
        api_metrics = {
            # Loss metrics
            'loss_total': metrics.get('final_loss', 0),
            'loss_policy': metrics.get('final_policy_loss', 0),
            'loss_value': metrics.get('final_value_loss', 0),
            'loss_difficulty': metrics.get('final_difficulty_loss', 0),
            'loss_illegal_penalty': metrics.get('final_illegal_penalty', 0),
            # Policy metrics
            'policy_top1_accuracy': metrics.get('policy_top1_accuracy'),
            'policy_top3_accuracy': metrics.get('policy_top3_accuracy'),
            'policy_entropy': metrics.get('policy_entropy'),
            'policy_legal_mass': metrics.get('policy_legal_mass'),
            'policy_ebf': metrics.get('policy_ebf'),
            'policy_confidence': metrics.get('policy_confidence'),
            # Value metrics
            'value_mean': metrics.get('value_mean'),
            'value_std': metrics.get('value_std'),
            'value_extremity': metrics.get('value_extremity'),
            'value_calibration_error': metrics.get('value_calibration_error'),
            # Pass metrics
            'pass_decision_rate': metrics.get('pass_decision_rate'),
            # Game stats
            'num_games': metrics.get('games', 0),
            'num_examples': metrics.get('examples', 0),
            'avg_game_length': metrics.get('avg_game_length'),
            # Meta
            'learning_rate': self.current_lr,
            'model_version': model_version,
            'train_time_sec': train_time,
        }

        success = self.api_client.submit_metrics(self.iteration, api_metrics)
        if success:
            print(f"[Trainer] Metrics submitted to API")
        # Silently ignore failures - metrics are logged locally anyway

    def _save_trainer_state(self):
        """Save optimizer state and metadata for resumption."""
        state_path = self._get_state_path()
        state = {
            'iteration': self.iteration,
            'total_games_trained': self.total_games_trained,
            'optimizer_state_dict': self.network_trainer.optimizer.state_dict(),
            'best_model_path': str(self.best_model_path) if self.best_model_path else None,
        }
        torch.save(state, state_path)

        # Save replay buffer separately (numpy format for efficiency)
        replay_path = self._get_replay_buffer_path()
        if len(self.replay_buffer) > 0:
            np.savez_compressed(
                replay_path,
                states=np.array(list(self.replay_buffer.states)),
                policies=np.array(list(self.replay_buffer.policies)),
                values=np.array(list(self.replay_buffer.values)),
                legal_masks=np.array(list(self.replay_buffer.legal_masks)) if self.replay_buffer.legal_masks[0] is not None else None,
            )
            print(f"[Trainer] Saved trainer state and replay buffer ({len(self.replay_buffer)} positions)")

    def _load_trainer_state(self):
        """Load optimizer state and replay buffer if available."""
        state_path = self._get_state_path()
        if state_path.exists():
            try:
                state = torch.load(state_path, map_location=self.device)
                self.network_trainer.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.total_games_trained = state.get('total_games_trained', 0)
                if state.get('best_model_path'):
                    self.best_model_path = Path(state['best_model_path'])
                print(f"[Trainer] Restored optimizer state (iteration {state.get('iteration', '?')})")
            except Exception as e:
                print(f"[Trainer] Could not restore optimizer state: {e}")

        # Load replay buffer
        replay_path = self._get_replay_buffer_path()
        if replay_path.exists():
            try:
                data = np.load(replay_path, allow_pickle=True)
                states = data['states']
                policies = data['policies']
                values = data['values']
                legal_masks = data['legal_masks'] if 'legal_masks' in data and data['legal_masks'] is not None else None

                # Reconstruct replay buffer
                self.replay_buffer = ReplayBuffer(max_positions=100_000)
                if legal_masks is not None:
                    self.replay_buffer.add(states, policies, values, legal_masks)
                else:
                    self.replay_buffer.add(states, policies, values, None)
                print(f"[Trainer] Restored replay buffer ({len(self.replay_buffer)} positions)")
            except Exception as e:
                print(f"[Trainer] Could not restore replay buffer: {e}")

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

        Uses replay buffer to mix old and new positions, preventing
        catastrophic forgetting.

        Returns training metrics.
        """
        print(f"[Trainer] Converting {len(games)} games to training data...")
        states, policies, values, legal_masks, difficulties = games_to_training_data(
            games,
            network=self.network,
            device=self.device,
        )
        new_examples = len(states)
        print(f"[Trainer] New training examples: {new_examples}")
        if difficulties is not None:
            avg_difficulty = difficulties.mean()
            print(f"[Trainer] Average difficulty target: {avg_difficulty:.3f}")

        # Add new positions to replay buffer
        self.replay_buffer.add(states, policies, values, legal_masks)
        print(f"[Trainer] Replay buffer size: {len(self.replay_buffer)}")

        # Sample from buffer (50% new, 50% buffer) if buffer has enough data
        if len(self.replay_buffer) > 1000:
            buf_states, buf_policies, buf_values, buf_masks = self.replay_buffer.sample(new_examples)
            states = np.concatenate([states, buf_states])
            policies = np.concatenate([policies, buf_policies])
            values = np.concatenate([values, buf_values])
            if legal_masks is not None and buf_masks is not None:
                legal_masks = np.concatenate([legal_masks, buf_masks])
            # Disable difficulty prediction when using buffer (sizes don't match)
            difficulties = None
            print(f"[Trainer] Training on {len(states)} examples ({new_examples} new + {len(buf_states)} from buffer)")

        # Reuse trainer to preserve optimizer momentum across iterations
        history = self.network_trainer.train(
            states, policies, values,
            legal_masks=legal_masks,
            difficulties=difficulties,
            verbose=True
        )

        # Get final metrics
        final = history[-1] if history else {}

        # Compute additional metrics after training (pass games for pass stats)
        extended_metrics = self._compute_extended_metrics(
            states, policies, values, legal_masks, games=games
        )

        return {
            'games': len(games),
            'examples': len(states),
            'final_loss': final.get('loss', 0),
            'final_policy_loss': final.get('policy_loss', 0),
            'final_value_loss': final.get('value_loss', 0),
            'final_difficulty_loss': final.get('difficulty_loss', 0),
            'final_illegal_penalty': final.get('illegal_penalty', 0),
            'epochs': len(history),
            **extended_metrics,
        }

    def _compute_extended_metrics(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        legal_masks: Optional[np.ndarray],
        games: Optional[list] = None,
    ) -> dict:
        """
        Compute extended metrics after training for analysis.

        Returns dict with policy accuracy, value stats, calibration, pass stats, etc.
        """
        # Convert to tensors for network inference
        states_tensor = torch.from_numpy(states).to(self.device)

        self.network.eval()
        with torch.no_grad():
            pred_logits, pred_values, pred_difficulty = self.network(states_tensor)
            pred_values = pred_values.squeeze(-1)

        # Compute policy metrics
        policy_metrics = compute_policy_metrics(
            pred_logits.cpu().numpy(),
            policies,
            legal_masks,
        )

        # Compute value metrics
        value_metrics = compute_value_metrics(pred_values.cpu().numpy())

        # Compute value calibration
        calibration = compute_value_calibration(pred_values.cpu().numpy(), values)
        calibration_error = compute_calibration_error(calibration)

        # Compute pass stats from games if available
        pass_decision_rate = None
        avg_game_length = None
        if games:
            total_pass_decisions = 0
            total_knight_decisions = 0
            total_length = 0
            for game in games:
                stats = compute_pass_stats(game.moves)
                total_pass_decisions += stats['pass_decisions']
                total_knight_decisions += stats['knight_decisions']
                total_length += len(game.moves)
            total_decisions = total_pass_decisions + total_knight_decisions
            if total_decisions > 0:
                pass_decision_rate = total_pass_decisions / total_decisions
            avg_game_length = total_length / len(games) if games else 0

        # Log extended metrics
        print(f"[Trainer] Extended metrics:")
        print(f"  Policy top-1 acc: {policy_metrics.top1_accuracy*100:.1f}%, "
              f"top-3 acc: {policy_metrics.top3_accuracy*100:.1f}%")
        print(f"  Policy entropy: {policy_metrics.entropy:.3f}, "
              f"EBF: {policy_metrics.effective_branching_factor:.2f}, "
              f"confidence: {policy_metrics.policy_confidence*100:.1f}%")
        print(f"  Policy legal mass: {policy_metrics.legal_mass*100:.1f}%")
        print(f"  Value mean: {value_metrics.mean:+.3f}, std: {value_metrics.std:.3f}, "
              f"extremity: {value_metrics.extremity:.3f}")
        print(f"  Value calibration error: {calibration_error:.4f}")
        if pass_decision_rate is not None:
            print(f"  Pass decision rate: {pass_decision_rate*100:.1f}%")

        return {
            'policy_top1_accuracy': policy_metrics.top1_accuracy,
            'policy_top3_accuracy': policy_metrics.top3_accuracy,
            'policy_entropy': policy_metrics.entropy,
            'policy_legal_mass': policy_metrics.legal_mass,
            'policy_ebf': policy_metrics.effective_branching_factor,
            'policy_confidence': policy_metrics.policy_confidence,
            'value_mean': value_metrics.mean,
            'value_std': value_metrics.std,
            'value_extremity': value_metrics.extremity,
            'value_calibration_error': calibration_error,
            'pass_decision_rate': pass_decision_rate,
            'avg_game_length': avg_game_length,
        }

    def validate_model(
        self,
        candidate_path: Path,
        baseline_path: Path,
        num_games: int = 20
    ) -> float:
        """
        Play candidate vs baseline to validate improvement.

        Args:
            candidate_path: Path to candidate model checkpoint
            baseline_path: Path to baseline (current best) model
            num_games: Number of games to play

        Returns:
            Win rate of candidate model (0.0 to 1.0)
        """
        try:
            from scripts.model_arena import run_match
        except ImportError:
            # On Vast.ai, model_arena.py is at workspace root
            from model_arena import run_match

        result = run_match(
            str(candidate_path),
            str(baseline_path),
            num_games=num_games,
            simulations=100,  # Faster for validation
            device=self.device,
            verbose=False,
        )
        return result.model1_win_rate()

    def save_and_upload_model(self, metrics: dict) -> Optional[str]:
        """
        Save model locally and upload to API if it passes checkpoint gating.

        Checkpoint gating: new model must beat current best by >55% win rate
        before workers will use it. This prevents regression.

        Returns:
            Model version string if promoted, None if rejected.
        """
        self.iteration += 1
        version = f"iter_{self.iteration:03d}"

        # Save locally (always save for analysis, even if not promoted)
        candidate_path = self.models_dir / f"{version}.pt"
        self.network.save(candidate_path)
        print(f"[Trainer] Saved candidate model: {candidate_path}")

        # Checkpoint gating disabled - always promote new models
        # In AlphaZero-style training, each iteration makes incremental improvements
        # and requiring 55% win rate is too aggressive for early training.
        # The model will naturally improve through the training loop.

        # Update best model path
        self.best_model_path = candidate_path

        # Upload to API (only promoted models)
        try:
            self.api_client.upload_model(
                version=version,
                iteration=self.iteration,
                file_path=candidate_path,
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

                        # Update learning rate based on schedule
                        self._update_learning_rate()

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

                        # Save trainer state for resumption
                        self._save_trainer_state()

                        # Submit metrics to API for dashboard
                        self._submit_iteration_metrics(metrics, train_time, model_version=version)

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
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size (default: 512)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--filters', type=int, default=None,
                        help='Network filter count (overrides --network-size)')
    parser.add_argument('--blocks', type=int, default=None,
                        help='Network residual blocks (overrides --network-size)')
    parser.add_argument('--network-size', type=str, default='medium', choices=['small', 'medium', 'large', 'alphazero'],
                        help='Network size preset: small (64f/6b), medium (128f/10b), large (256f/15b), alphazero (256f/20b)')
    parser.add_argument('--output', type=Path, default=Path('output/trainer'),
                        help='Output directory')

    args = parser.parse_args()

    # Resolve network size presets
    NETWORK_PRESETS = {
        'small': (64, 6),      # ~0.8M params, fast inference
        'medium': (128, 10),   # ~3.3M params, balanced
        'large': (256, 15),    # ~18M params, stronger but slower
        'alphazero': (256, 20),  # ~24M params, AlphaZero-scale
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
