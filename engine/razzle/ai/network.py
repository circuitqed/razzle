"""
Neural network for Razzle Dazzle position evaluation.

Architecture: Residual CNN with policy, value, and difficulty heads.
Follows AlphaZero design: large residual tower, small projection heads.

Input: (batch, 7, 8, 7) - board planes (pieces, balls, touched_mask, player, has_passed)
Output:
  - policy: (batch, 3137) - log probabilities over moves (56*56 + END_TURN)
  - value: (batch, 1) - position evaluation [-1, 1]
  - difficulty: (batch, 1) - predicted search difficulty [0, 1]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.bitboard import ROWS, COLS, NUM_SQUARES


# Total possible moves (any square to any square, plus END_TURN)
# Moves are encoded as src * 56 + dst for knight moves and passes (0-3135)
# Index 3136 is reserved for END_TURN (-1 in game logic)
END_TURN_ACTION = NUM_SQUARES * NUM_SQUARES  # 3136
NUM_ACTIONS = END_TURN_ACTION + 1  # 3137 total actions


@dataclass
class NetworkConfig:
    """Configuration for the neural network.

    AlphaZero-style architecture: most parameters in the residual tower,
    with small policy and value heads. Three presets available via PRESETS dict.

    Policy head: Conv(filters→policy_filters, 1x1) → BN → ReLU → FC → actions
      If policy_hidden > 0, adds a bottleneck: FC(flat→hidden) → ReLU → FC(hidden→actions)
      This is needed for small networks where the direct FC would exceed the tower size.

    Value head: Conv(filters→value_filters, 1x1) → BN → ReLU → FC(flat→hidden) → ReLU → FC(hidden→1) → tanh
    """
    num_input_planes: int = 7
    num_filters: int = 256      # Width of residual tower
    num_blocks: int = 20        # Depth of residual tower
    policy_filters: int = 2     # AZ uses 2
    value_filters: int = 1      # AZ uses 1
    value_hidden: int = 256     # AZ uses 256
    policy_hidden: int = 0      # 0 = direct FC (AZ default), >0 = bottleneck hidden layer


# Presets matching AlphaZero architecture at different scales
PRESETS = {
    'small': NetworkConfig(
        num_filters=32, num_blocks=6,
        policy_filters=2, value_filters=1,
        value_hidden=128, policy_hidden=32,
    ),
    'medium': NetworkConfig(
        num_filters=96, num_blocks=12,
        policy_filters=2, value_filters=1,
        value_hidden=256, policy_hidden=0,
    ),
    'large': NetworkConfig(
        num_filters=256, num_blocks=20,
        policy_filters=2, value_filters=1,
        value_hidden=256, policy_hidden=0,
    ),
}


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
        policy_flat = c.policy_filters * ROWS * COLS
        if c.policy_hidden > 0:
            self.policy_fc1 = nn.Linear(policy_flat, c.policy_hidden)
            self.policy_fc2 = nn.Linear(c.policy_hidden, NUM_ACTIONS)
        else:
            self.policy_fc = nn.Linear(policy_flat, NUM_ACTIONS)

        # Value head
        self.value_conv = nn.Conv2d(c.num_filters, c.value_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(c.value_filters)
        value_flat = c.value_filters * ROWS * COLS
        self.value_fc1 = nn.Linear(value_flat, c.value_hidden)
        self.value_fc2 = nn.Linear(c.value_hidden, 1)

        # Difficulty head (predicts how much MCTS will change the policy)
        self.difficulty_conv = nn.Conv2d(c.num_filters, c.value_filters, 1, bias=False)
        self.difficulty_bn = nn.BatchNorm2d(c.value_filters)
        self.difficulty_fc1 = nn.Linear(value_flat, c.value_hidden)
        self.difficulty_fc2 = nn.Linear(c.value_hidden, 1)

        # Initialize final layers with small weights to prevent saturation
        nn.init.normal_(self.value_fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_fc2.bias)
        nn.init.normal_(self.difficulty_fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.difficulty_fc2.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 7, 8, 7)

        Returns:
            policy: Log probabilities over actions (batch, 3137)
            value: Position evaluation (batch, 1) in range [-1, 1]
            difficulty: Predicted search difficulty (batch, 1) in range [0, 1]
        """
        # Input block
        x = F.relu(self.bn_in(self.conv_in(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        tower = x  # Save tower output for all heads

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(tower)))
        p = p.view(p.size(0), -1)
        if hasattr(self, 'policy_fc1'):
            p = F.relu(self.policy_fc1(p))
            p = self.policy_fc2(p)
        else:
            p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(tower)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Difficulty head (backward compatible - may not exist in old models)
        if hasattr(self, 'difficulty_fc2'):
            d = F.relu(self.difficulty_bn(self.difficulty_conv(tower)))
            d = d.view(d.size(0), -1)
            d = F.relu(self.difficulty_fc1(d))
            d = torch.sigmoid(self.difficulty_fc2(d))  # Output in [0, 1]
        else:
            d = torch.full((x.size(0), 1), 0.5, device=x.device)

        return p, v, d

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference mode prediction (no gradients).

        Returns:
            policy: Log probabilities over actions
            value: Position evaluation
            difficulty: Predicted search difficulty
        """
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
        """Load model from file.

        Handles backward compatibility with old models that don't have
        the difficulty head or have different policy head structure.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']

        # Backward compat: old configs don't have policy_hidden
        if not hasattr(config, 'policy_hidden'):
            config.policy_hidden = 0

        model = cls(config)
        state_dict = checkpoint['state_dict']
        model_state = model.state_dict()

        # Handle backward compatibility: filter to matching keys
        missing_keys = set(model_state.keys()) - set(state_dict.keys())
        extra_keys = set(state_dict.keys()) - set(model_state.keys())

        if missing_keys or extra_keys:
            filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
            model.load_state_dict(filtered_state, strict=False)
        else:
            model.load_state_dict(state_dict)

        return model

    def num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_network(
    preset: Optional[str] = None,
    num_filters: int = 0,
    num_blocks: int = 0,
    device: str = 'cpu'
) -> RazzleNet:
    """Create a new network.

    Args:
        preset: One of 'small' (~236K), 'medium' (~2.4M), 'large' (~24M).
                If provided, num_filters and num_blocks are ignored.
        num_filters: Tower width (ignored if preset is set).
        num_blocks: Tower depth (ignored if preset is set).
        device: Device to place model on.
    """
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}")
        config = PRESETS[preset]
    elif num_filters > 0 and num_blocks > 0:
        config = NetworkConfig(num_filters=num_filters, num_blocks=num_blocks)
    else:
        config = NetworkConfig()
    model = RazzleNet(config)
    return model.to(device)
