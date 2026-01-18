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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class RazzleDataset(Dataset):
    """PyTorch dataset for Razzle Dazzle training data."""

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray
    ):
        self.states = torch.from_numpy(states)
        self.policies = torch.from_numpy(policies)
        self.values = torch.from_numpy(values)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.policies[idx], self.values[idx]


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

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.9
        )

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.network.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for states, target_policies, target_values in dataloader:
            states = states.to(self.config.device)
            target_policies = target_policies.to(self.config.device)
            target_values = target_values.to(self.config.device)

            # Forward pass
            log_policies, values = self.network(states)
            values = values.squeeze(-1)

            # Policy loss: cross-entropy (using log probs from network)
            policy_loss = -torch.sum(target_policies * log_policies, dim=1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(values, target_values)

            # Combined loss
            loss = (
                self.config.policy_weight * policy_loss +
                self.config.value_weight * value_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }

    def train(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        verbose: bool = True
    ) -> list[dict]:
        """
        Train on given data for configured number of epochs.

        Returns list of metrics per epoch.
        """
        dataset = RazzleDataset(states, policies, values)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        history = []

        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(dataloader)
            self.scheduler.step()

            metrics['epoch'] = epoch + 1
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            history.append(metrics)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"policy={metrics['policy_loss']:.4f}, "
                    f"value={metrics['value_loss']:.4f}, "
                    f"lr={metrics['lr']:.6f}"
                )

        return history

    def save_checkpoint(self, path: Path, extra: Optional[dict] = None) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'network_config': self.network.config,
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> dict:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        return checkpoint
