"""
Value network for TFPnP.

V^pi(s) estimates the expected cumulative reward from state s. It takes the same
5-channel state as the policy (x, z, u, noise_lvl, iter_frac) and returns a scalar.

Wei et al. (Table 1) use weight normalisation and TReLU in the value network. 
This implementation uses plain convolutions with no normalisation and standard ReLU. 
"""

import torch
import torch.nn as nn


class ResBlock_wobn(nn.Module):
    """
    ResNet basic block without batch normalisation.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        """
        Residual forward pass: x + conv2(relu(conv1(x))).
        """
        return x + self.conv2(self.relu(self.conv1(x)))


class ResNet_wobn(nn.Module):
    """
    Value network: maps the ADMM state to a scalar value estimate V(s).

    Args:
        in_channels   : number of input channels (default 5)
        base_channels : feature width (default 64)
        n_blocks      : number of residual blocks (default 8)
        stride        : stem stride
    """

    def __init__(self, in_channels: int = 5, base_channels: int = 64, n_blocks: int = 8, stride: int = 1):
       
        super().__init__()

        self.head = nn.Conv2d(in_channels, base_channels, 3, stride=stride, padding=1, bias=True)
        self.body = nn.Sequential(*[ResBlock_wobn(base_channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Linear(base_channels, 1)

    def forward(self, x, z, u, noise_level, iter_frac):
        """
        Args:
            x, z, u     : ADMM variables, shape (B, 1, H, W).
            noise_level : fractional sinogram noise level, shape (B,).
            iter_frac   : fraction of the iteration budget used, shape (B,).

        Returns:
            value: shape (B, 1).
        """

        # set shape and noise/iter maps
        B, _, H, W = x.shape
        noise_map = noise_level.view(B, 1, 1, 1).expand(B, 1, H, W)
        iter_map = iter_frac.view(B, 1, 1, 1).expand(B, 1, H, W)

        # concatenate state channels 
        inp = torch.cat([x, z, u, noise_map, iter_map], dim=1)

        # forward through the network
        out = self.head(inp)
        out = self.body(out)
        out = self.pool(out).view(B, -1)

        # return the scalar value estimate
        return self.tail(out)
