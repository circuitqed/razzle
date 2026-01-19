"""
Neural network for Razzle Dazzle position evaluation.

Architecture: Residual CNN with policy and value heads.
Input: (batch, 6, 8, 7) - board planes
Output:
  - policy: (batch, 3136) - log probabilities over moves (56*56)
  - value: (batch, 1) - position evaluation [-1, 1]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.bitboard import ROWS, COLS, NUM_SQUARES


# Total possible moves (any square to any square)
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES  # 56 * 56 = 3136


@dataclass
class NetworkConfig:
    """Configuration for the neural network."""
    num_input_planes: int = 6
    num_filters: int = 64
    num_blocks: int = 6
    policy_filters: int = 2
    value_filters: int = 1
    value_hidden: int = 64


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection."""

    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class RazzleNet(nn.Module):
    """
    Neural network for Razzle Dazzle.

    Architecture follows AlphaZero: residual tower with policy and value heads.
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        super().__init__()
        self.config = config or NetworkConfig()
        c = self.config

        # Input convolution
        self.conv_in = nn.Conv2d(c.num_input_planes, c.num_filters, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(c.num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(c.num_filters) for _ in range(c.num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(c.num_filters, c.policy_filters, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(c.policy_filters)
        self.policy_fc = nn.Linear(c.policy_filters * ROWS * COLS, NUM_ACTIONS)

        # Value head
        self.value_conv = nn.Conv2d(c.num_filters, c.value_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(c.value_filters)
        self.value_fc1 = nn.Linear(c.value_filters * ROWS * COLS, c.value_hidden)
        self.value_fc2 = nn.Linear(c.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 6, 8, 7)

        Returns:
            policy: Log probabilities over actions (batch, 3136)
            value: Position evaluation (batch, 1) in range [-1, 1]
        """
        # Input block
        x = F.relu(self.bn_in(self.conv_in(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference mode prediction (no gradients)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> RazzleNet:
        """Load model from file."""
        # Use weights_only=False since we're loading our own checkpoints
        # which contain NetworkConfig dataclass
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_network(
    num_filters: int = 64,
    num_blocks: int = 6,
    device: str = 'cpu'
) -> RazzleNet:
    """Create a new network with given configuration."""
    config = NetworkConfig(
        num_filters=num_filters,
        num_blocks=num_blocks
    )
    model = RazzleNet(config)
    return model.to(device)
