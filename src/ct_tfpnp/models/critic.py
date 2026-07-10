"""
Value network for TFPnP (ResNet without batch norm).

The value network V^pi(s) estimates the expected cumulative reward from state s.
It shares the same input format as the policy network but uses a ResNet variant
without batch normalisation, which is empirically more stable for value estimation
in this RL setting (Wei et al.).

Input: same 5-channel state as the policy network (x, z, u, noise_lvl, iter_frac)
Output: scalar value estimate V(s), shape (B, 1)
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TReLU(nn.Module):
    """Truncated ReLU: max(0, min(x, 1)) — bounded activation."""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class ResBlock_wobn(nn.Module):
    """ResNet basic block without batch normalisation, using weight normalisation
    and truncated ReLU as per Wei et al. (2022)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(channels, channels, 3, padding=1, bias=True))
        self.trelu = TReLU()
        self.conv2 = weight_norm(nn.Conv2d(channels, channels, 3, padding=1, bias=True))

    def forward(self, x):
        return x + self.conv2(self.trelu(self.conv1(x)))


class ResNet_wobn(nn.Module):
    """
    Value network: maps ADMM state → scalar value estimate V(s).

    Uses weight-normalised convolutions and truncated ReLU activations
    throughout, matching Wei et al. (2022)'s design for stability in
    the RL value-estimation setting.

    Args:
        in_channels: number of input channels (default 5)
        base_channels: feature width (default 64)
        n_blocks: number of residual blocks (default 8)
    """

    def __init__(self, in_channels: int = 5, base_channels: int = 64, n_blocks: int = 8):
        super().__init__()

        self.head = weight_norm(nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=True))
        self.head_act = TReLU()
        self.body = nn.Sequential(*[ResBlock_wobn(base_channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Linear(base_channels, 1)

    def forward(self, x, z, u, noise_level, iter_frac):
        """
        Args:
            x, z, u:     ADMM variables, shape (B, 1, H, W)
            noise_level: shape (B,)
            iter_frac:   shape (B,)

        Returns:
            value: shape (B, 1)
        """
        B, _, H, W = x.shape
        noise_map = noise_level.view(B, 1, 1, 1).expand(B, 1, H, W)
        iter_map = iter_frac.view(B, 1, 1, 1).expand(B, 1, H, W)

        inp = torch.cat([x, z, u, noise_map, iter_map], dim=1)

        out = self.head_act(self.head(inp))
        out = self.body(out)
        out = self.pool(out).view(B, -1)
        return self.tail(out)