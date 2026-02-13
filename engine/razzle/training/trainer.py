"""
Neural network trainer for Razzle Dazzle.

Implements the training loop for the policy-value network.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from ..ai.network import RazzleNet


@dataclass
class TrainingConfig:
    """Configuration for training.

    Uses AlphaZero-style equal weighting for policy and value losses.
    The quartic value term helps with calibration but should be small.
    """
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4  # Standard L2 regularization
    epochs: int = 10
    policy_weight: float = 1.0
    value_weight: float = 5.0  # Balanced with 4x value head capacity + gradient clipping
    value_weight_quartic: float = 1.0  # Small quartic term for calibration
    difficulty_weight: float = 0.5
    illegal_penalty_weight: float = 1.0
    max_grad_norm: float = 1.0  # Gradient clipping (0 = disabled)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class RazzleDataset(Dataset):
    """PyTorch dataset for Razzle Dazzle training data."""

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        legal_masks: np.ndarray,
        difficulties: Optional[np.ndarray] = None
    ):
        self.states = torch.from_numpy(states)
        self.policies = torch.from_numpy(policies)
        self.values = torch.from_numpy(values)
        self.legal_masks = torch.from_numpy(legal_masks)
        # Difficulty targets: predicted KL divergence between raw and MCTS policies
        self.difficulties = torch.from_numpy(difficulties) if difficulties is not None else None

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        base = (self.states[idx], self.policies[idx], self.values[idx], self.legal_masks[idx])
        if self.difficulties is not None:
            return base + (self.difficulties[idx],)
        return base


class Trainer:
    """
    Trains the neural network on self-play data.
    """

    def __init__(
        self,
        network: RazzleNet,
        config: Optional[TrainingConfig] = None
    ):
        self.network = network
        self.config = config or TrainingConfig()

        self.network = self.network.to(self.config.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # No LR scheduler - Adam adapts per-parameter learning rates automatically
        # A constant base LR works well with Adam for most cases
        self.scheduler = None

    def train_epoch(
        self,
        dataloader: DataLoader,
        has_difficulties: bool = False
    ) -> dict:
        """Train for one epoch."""
        self.network.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_value_loss_quartic = 0.0
        total_difficulty_loss = 0.0
        total_illegal_penalty = 0.0
        num_batches = 0

        for batch in dataloader:
            # Unpack batch: always has legal_masks, optionally has difficulties
            if has_difficulties:
                states, target_policies, target_values, legal_masks, target_difficulties = batch
                target_difficulties = target_difficulties.to(self.config.device)
            else:
                states, target_policies, target_values, legal_masks = batch
                target_difficulties = None

            states = states.to(self.config.device)
            target_policies = target_policies.to(self.config.device)
            target_values = target_values.to(self.config.device)
            legal_masks = legal_masks.to(self.config.device)

            # Forward pass
            log_policies, values, difficulties = self.network(states)
            values = values.squeeze(-1)
            difficulties = difficulties.squeeze(-1)

            # Convert log probs to probs for the penalty term
            policies = torch.exp(log_policies)

            # Masked cross-entropy on legal moves only
            # legal_masks: 1 for legal, 0 for illegal
            masked_target = target_policies * legal_masks
            masked_log_policies = log_policies * legal_masks
            policy_loss = -torch.sum(masked_target * masked_log_policies, dim=1).mean()

            # Illegal move penalty: sum of probability mass on illegal moves
            illegal_masks = 1.0 - legal_masks
            illegal_prob_mass = torch.sum(policies * illegal_masks, dim=1).mean()
            illegal_penalty = self.config.illegal_penalty_weight * illegal_prob_mass

            # Value loss: MSE (quadratic) + quartic term for calibration
            value_diff = values - target_values
            value_loss_quadratic = torch.mean(value_diff ** 2)
            value_loss_quartic = torch.mean(value_diff ** 4)
            value_loss = value_loss_quadratic  # For logging, report quadratic only
            value_loss_combined = (
                self.config.value_weight * value_loss_quadratic +
                self.config.value_weight_quartic * value_loss_quartic
            )

            # Difficulty loss: Binary cross-entropy (target is in [0, 1])
            if target_difficulties is not None:
                difficulty_loss = F.binary_cross_entropy(difficulties, target_difficulties)
            else:
                difficulty_loss = torch.tensor(0.0, device=self.config.device)

            # Combined loss
            loss = (
                self.config.policy_weight * policy_loss +
                value_loss_combined +
                self.config.difficulty_weight * difficulty_loss +
                illegal_penalty
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_value_loss_quartic += value_loss_quartic.item()
            total_difficulty_loss += difficulty_loss.item()
            total_illegal_penalty += illegal_penalty.item()
            num_batches += 1

        metrics = {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'value_loss_quartic': total_value_loss_quartic / num_batches,
            'illegal_penalty': total_illegal_penalty / num_batches,
        }
        if has_difficulties:
            metrics['difficulty_loss'] = total_difficulty_loss / num_batches

        return metrics

    def train(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        legal_masks: np.ndarray,
        difficulties: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> list[dict]:
        """
        Train on given data for configured number of epochs.

        Args:
            states: Board state tensors (N, 7, 8, 7)
            policies: Policy targets (N, NUM_ACTIONS)
            values: Value targets (N,)
            legal_masks: Legal move masks (N, NUM_ACTIONS).
                        1 for legal moves, 0 for illegal.
            difficulties: Optional difficulty targets (N,).
                         Values in [0, 1] where higher = harder position.
                         If provided, trains the difficulty prediction head.
            verbose: Print progress

        Returns list of metrics per epoch.
        """
        dataset = RazzleDataset(states, policies, values, legal_masks, difficulties)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        has_difficulties = difficulties is not None
        history = []

        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(dataloader, has_difficulties=has_difficulties)
            if self.scheduler:
                self.scheduler.step()

            metrics['epoch'] = epoch + 1
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            history.append(metrics)

            if verbose:
                msg = (
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"policy={metrics['policy_loss']:.4f}, "
                    f"value={metrics['value_loss']:.4f}, "
                    f"illegal={metrics['illegal_penalty']:.4f}"
                )
                if 'difficulty_loss' in metrics:
                    msg += f", difficulty={metrics['difficulty_loss']:.4f}"
                msg += f", lr={metrics['lr']:.6f}"
                print(msg)

        return history

    def save_checkpoint(self, path: Path, extra: Optional[dict] = None) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'network_config': self.network.config,
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> dict:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler and checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        return checkpoint
