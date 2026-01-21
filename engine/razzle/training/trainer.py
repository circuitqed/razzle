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
    """Configuration for training."""
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 10
    policy_weight: float = 1.0
    value_weight: float = 1.0
    difficulty_weight: float = 0.5  # Weight for difficulty prediction loss
    illegal_penalty_weight: float = 1.0  # Lagrange multiplier for illegal move constraint
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class RazzleDataset(Dataset):
    """PyTorch dataset for Razzle Dazzle training data."""

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        difficulties: Optional[np.ndarray] = None
    ):
        self.states = torch.from_numpy(states)
        self.policies = torch.from_numpy(policies)
        self.values = torch.from_numpy(values)
        # Legal masks: 1 for legal moves, 0 for illegal
        # If not provided, assume all moves could be legal (backward compatibility)
        if legal_masks is not None:
            self.legal_masks = torch.from_numpy(legal_masks)
        else:
            self.legal_masks = None
        # Difficulty targets: predicted KL divergence between raw and MCTS policies
        if difficulties is not None:
            self.difficulties = torch.from_numpy(difficulties)
        else:
            self.difficulties = None

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        base = (self.states[idx], self.policies[idx], self.values[idx])
        if self.legal_masks is not None and self.difficulties is not None:
            return base + (self.legal_masks[idx], self.difficulties[idx])
        elif self.legal_masks is not None:
            return base + (self.legal_masks[idx],)
        elif self.difficulties is not None:
            return base + (self.difficulties[idx],)
        else:
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
        has_legal_masks: bool = False,
        has_difficulties: bool = False
    ) -> dict:
        """Train for one epoch."""
        self.network.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_difficulty_loss = 0.0
        total_illegal_penalty = 0.0
        num_batches = 0

        for batch in dataloader:
            # Unpack batch based on what's included
            if has_legal_masks and has_difficulties:
                states, target_policies, target_values, legal_masks, target_difficulties = batch
                legal_masks = legal_masks.to(self.config.device)
                target_difficulties = target_difficulties.to(self.config.device)
            elif has_legal_masks:
                states, target_policies, target_values, legal_masks = batch
                legal_masks = legal_masks.to(self.config.device)
                target_difficulties = None
            elif has_difficulties:
                states, target_policies, target_values, target_difficulties = batch
                legal_masks = None
                target_difficulties = target_difficulties.to(self.config.device)
            else:
                states, target_policies, target_values = batch
                legal_masks = None
                target_difficulties = None

            states = states.to(self.config.device)
            target_policies = target_policies.to(self.config.device)
            target_values = target_values.to(self.config.device)

            # Forward pass
            log_policies, values, difficulties = self.network(states)
            values = values.squeeze(-1)
            difficulties = difficulties.squeeze(-1)

            if legal_masks is not None:
                # Masked cross-entropy on legal moves only
                # legal_masks: 1 for legal, 0 for illegal
                # We compute CE only where legal_masks == 1

                # Convert log probs to probs for the penalty term
                policies = torch.exp(log_policies)

                # Cross-entropy on legal moves: -sum(target * log_pred) over legal moves
                # Mask both target and log_policies
                masked_target = target_policies * legal_masks
                masked_log_policies = log_policies * legal_masks

                # Normalize target over legal moves for proper CE
                # (target should already sum to 1 over legal moves, but ensure numerical stability)
                policy_loss = -torch.sum(masked_target * masked_log_policies, dim=1).mean()

                # Illegal move penalty: sum of probability mass on illegal moves
                # illegal_masks = 1 - legal_masks
                illegal_masks = 1.0 - legal_masks
                illegal_prob_mass = torch.sum(policies * illegal_masks, dim=1).mean()

                illegal_penalty = self.config.illegal_penalty_weight * illegal_prob_mass
            else:
                # Backward compatible: standard cross-entropy over all moves
                policy_loss = -torch.sum(target_policies * log_policies, dim=1).mean()
                illegal_penalty = torch.tensor(0.0, device=self.config.device)

            # Value loss: MSE
            value_loss = F.mse_loss(values, target_values)

            # Difficulty loss: Binary cross-entropy (target is in [0, 1])
            if target_difficulties is not None:
                difficulty_loss = F.binary_cross_entropy(difficulties, target_difficulties)
            else:
                difficulty_loss = torch.tensor(0.0, device=self.config.device)

            # Combined loss
            loss = (
                self.config.policy_weight * policy_loss +
                self.config.value_weight * value_loss +
                self.config.difficulty_weight * difficulty_loss +
                illegal_penalty
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_difficulty_loss += difficulty_loss.item()
            total_illegal_penalty += illegal_penalty.item()
            num_batches += 1

        metrics = {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
        }
        if has_legal_masks:
            metrics['illegal_penalty'] = total_illegal_penalty / num_batches
        if has_difficulties:
            metrics['difficulty_loss'] = total_difficulty_loss / num_batches

        return metrics

    def train(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        legal_masks: Optional[np.ndarray] = None,
        difficulties: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> list[dict]:
        """
        Train on given data for configured number of epochs.

        Args:
            states: Board state tensors (N, 6, 8, 7)
            policies: Policy targets (N, NUM_ACTIONS)
            values: Value targets (N,)
            legal_masks: Optional legal move masks (N, NUM_ACTIONS).
                        1 for legal moves, 0 for illegal.
                        If provided, enables masked cross-entropy + illegal penalty.
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

        has_legal_masks = legal_masks is not None
        has_difficulties = difficulties is not None
        history = []

        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(
                dataloader,
                has_legal_masks=has_legal_masks,
                has_difficulties=has_difficulties
            )
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
                    f"value={metrics['value_loss']:.4f}"
                )
                if 'illegal_penalty' in metrics:
                    msg += f", illegal={metrics['illegal_penalty']:.4f}"
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
