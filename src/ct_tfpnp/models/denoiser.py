"""
DRUNet denoiser wrapper — architecture matches KAIR pretrained weights exactly.
No batch norm (the grayscale DRUNet variant uses plain conv-relu-conv residual blocks).

Weights: drunet_gray.pth from https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
"""

import torch
import torch.nn as nn
from pathlib import Path


class ResBlock(nn.Module):
    """Conv-ReLU-Conv residual block (no batch norm)."""
    def __init__(self, channels):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.res(x)


class DRUNet(nn.Module):
    """
    DRUNet matching KAIR's drunet_gray.pth weights.
    Structure: head → 3 encoder stages → bottleneck → 3 decoder stages → tail
    Each stage has 4 residual blocks. Encoder uses strided conv to downsample,
    decoder uses ConvTranspose to upsample with additive skip connections.

    Skip connection convention (matching KAIR):
        x = m_up(x + skip)
    The skip is added BEFORE the up block, not after the ConvTranspose.
    """
    def __init__(self, in_channels=2, out_channels=1, nc=[64, 128, 256, 512]):
        super().__init__()

        self.m_head = nn.Conv2d(in_channels, nc[0], 3, padding=1, bias=False)

        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0]) for _ in range(4)],
            nn.Conv2d(nc[0], nc[1], 2, stride=2, bias=False),
        )
        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1]) for _ in range(4)],
            nn.Conv2d(nc[1], nc[2], 2, stride=2, bias=False),
        )
        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2]) for _ in range(4)],
            nn.Conv2d(nc[2], nc[3], 2, stride=2, bias=False),
        )

        self.m_body = nn.Sequential(*[ResBlock(nc[3]) for _ in range(4)])

        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(nc[3], nc[2], 2, stride=2, bias=False),
            *[ResBlock(nc[2]) for _ in range(4)],
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2, bias=False),
            *[ResBlock(nc[1]) for _ in range(4)],
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2, bias=False),
            *[ResBlock(nc[0]) for _ in range(4)],
        )

        self.m_tail = nn.Conv2d(nc[0], out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        x1 = self.m_head(x)        # (B, 64,  H,   W)
        x2 = self.m_down1(x1)      # (B, 128, H/2, W/2)
        x3 = self.m_down2(x2)      # (B, 256, H/4, W/4)
        x4 = self.m_down3(x3)      # (B, 512, H/8, W/8)

        x = self.m_body(x4)        # (B, 512, H/8, W/8)

        x = self.m_up3(x + x4)     # skip BEFORE up block → (B, 256, H/4, W/4)
        x = self.m_up2(x + x3)     # skip BEFORE up block → (B, 128, H/2, W/2)
        x = self.m_up1(x + x2)     # skip BEFORE up block → (B, 64,  H,   W)

        return self.m_tail(x + x1)  # skip before final conv → (B, 1, H, W)


class DRUNetDenoiser(nn.Module):
    """
    Noise-level-conditional DRUNet denoiser.

    Handles automatic normalisation: input can be in any non-negative range
    (e.g. µ units [0, ~2]). Internally normalised to [0, 1] for DRUNet,
    then scaled back to the original range.

    Usage:
        denoiser = DRUNetDenoiser("/path/to/drunet_gray.pth")
        x_clean = denoiser(x_noisy, sigma=15.0)
    """
    def __init__(self, pretrained_path: str | None = None):
        super().__init__()
        self.net = DRUNet(in_channels=2, out_channels=1)
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DRUNet weights not found at {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        print(f"Loaded DRUNet weights from {path}")

    def forward(self, z: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        squeezed = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeezed = True

        B, _, H, W = z.shape

        scale = z.view(B, -1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-8)
        z_01 = z / scale

        if isinstance(sigma, (int, float)):
            sigma_norm = torch.full((B, 1, H, W), sigma / 50.0,
                                     device=z.device, dtype=z.dtype)
        else:
            sigma_t = sigma.to(z.device, z.dtype)
            sigma_norm = (sigma_t / 50.0).view(B, 1, 1, 1).expand(B, 1, H, W)

        inp = torch.cat([z_01, sigma_norm], dim=1)
        out = self.net(inp) * scale

        if squeezed:
            out = out.squeeze(0)
        return out

