"""
DRUNet denoiser wrapper — architecture matches KAIR pretrained weights exactly.
No batch norm (the grayscale DRUNet variant uses plain conv-relu-conv residual blocks).

Weights: drunet_gray.pth from https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class ResBlock(nn.Module):
    """
    Conv-ReLU-Conv residual block (no batch norm).
    """
    def __init__(self, channels):
        super().__init__()

        # Conv-ReLU-Conv residual block (no batch norm)
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        """
        Residual forward pass: x + f(x).
        """
        return x + self.res(x)


class DRUNet(nn.Module):
    """
    DRUNet matching KAIR's drunet_gray.pth weights.

    Structure: head -> 3 encoder stages -> bottleneck -> 3 decoder stages -> tail.
    Each stage has 4 residual blocks. The encoder downsamples with strided conv,
    the decoder upsamples with ConvTranspose and additive skip connections.

    Skip-connection convention (matching KAIR): x = m_up(x + skip), i.e. the skip
    is added BEFORE the up block, not after the ConvTranspose.

    Args:
            in_channels:  input channels (2 = image + noise-level map).
            out_channels: output channels (1 for grayscale).
            nc:           channel widths of the four resolution levels.
    """
    def __init__(self, in_channels=2, out_channels=1, nc=[64, 128, 256, 512]):
        
        super().__init__()

        # define head
        self.m_head = nn.Conv2d(in_channels, nc[0], 3, padding=1, bias=False)

        # define encoder stages
        self.m_down1 = nn.Sequential(*[ResBlock(nc[0]) for _ in range(4)], nn.Conv2d(nc[0], nc[1], 2, stride=2, bias=False),)
        self.m_down2 = nn.Sequential(*[ResBlock(nc[1]) for _ in range(4)], nn.Conv2d(nc[1], nc[2], 2, stride=2, bias=False),)
        self.m_down3 = nn.Sequential(*[ResBlock(nc[2]) for _ in range(4)], nn.Conv2d(nc[2], nc[3], 2, stride=2, bias=False),)

        # define bottleneck
        self.m_body = nn.Sequential(*[ResBlock(nc[3]) for _ in range(4)])

        # define decoder stages
        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], 2, stride=2, bias=False), *[ResBlock(nc[2]) for _ in range(4)],)
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2, bias=False), *[ResBlock(nc[1]) for _ in range(4)],)
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2, bias=False), *[ResBlock(nc[0]) for _ in range(4)],)

        # define tail
        self.m_tail = nn.Conv2d(nc[0], out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        """
        Args:
            x: shape (B, 2, H, W) - image channel plus noise-level map.

        Returns:
            Denoised image, shape (B, 1, H, W).
        """
        # forward pass through encoder-decoder with skip connections
        x1 = self.m_head(x)        # (B,  64, H,   W)
        x2 = self.m_down1(x1)      # (B, 128, H/2, W/2)
        x3 = self.m_down2(x2)      # (B, 256, H/4, W/4)
        x4 = self.m_down3(x3)      # (B, 512, H/8, W/8)

        x = self.m_body(x4)        # (B, 512, H/8, W/8)

        x = self.m_up3(x + x4)     # skip added before the up block
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)

        return self.m_tail(x + x1)  # skip before final conv → (B, 1, H, W)


class DRUNetDenoiser(nn.Module):
    """
    Noise-level-conditional DRUNet denoiser.

    Args:
        pretrained_path: path to drunet_gray.pth.
        sigma_scale:     divisor mapping sigma onto DRUNet's noise-level map.
        verbose:         print a line on weight load.

    Usage:
        denoiser = DRUNetDenoiser("/path/to/drunet_gray.pth")
        x_clean = denoiser(x_noisy, sigma=1.5)
    """
    def __init__(self, pretrained_path: str | None = None, sigma_scale: float = 50.0, verbose: bool = True):
        
        super().__init__()
        
        self.net = DRUNet(in_channels=2, out_channels=1)
        self.sigma_scale = sigma_scale
        self.verbose = verbose

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path: str):
        """
        Load KAIR DRUNet weights, raising if the file is missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DRUNet weights not found at {path}")
        
        # Load the state dict and apply it to the network
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)

        if self.verbose:
            print(f"Loaded DRUNet weights from {path}")

    def forward(self, z: torch.Tensor, sigma: float | torch.Tensor):
        """
        Denoise `z` at strength `sigma`.

        Args:
            z:     image, shape (B, 1, H, W) or (1, H, W). Any real range.
            sigma: denoising strength, scalar or tensor of shape (B,).

        Returns:
            Denoised image, same shape as z.
        """
        # add a channel dimension if z is 3D (single image)
        squeezed = z.dim() == 3
        if squeezed:
            z = z.unsqueeze(0)

        # set up batch size and image dimensions
        B, _, H, W = z.shape

        # Normalise each image to roughly [-1, 1] for the pretrained net
        scale = z.view(B, -1).abs().max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-8)
        z_01 = z / scale

        # Constant noise-level map, DRUNet's second input channel
        if isinstance(sigma, (int, float)):
            sigma_t = torch.full((B,), float(sigma), device=z.device, dtype=z.dtype)
        else:
            sigma_t = sigma.to(z.device, z.dtype).reshape(-1)
            if sigma_t.numel() == 1:
                sigma_t = sigma_t.expand(B)
        
        # scale to match expected range
        sigma_map = (sigma_t / self.sigma_scale).view(B, 1, 1, 1).expand(B, 1, H, W)

        # concatenate image and noise-level map along channel dimension
        inp = torch.cat([z_01, sigma_map], dim=1)

        # DRUNet downsamples by 8 pad to a multiple of 8 and crop back
        pad_h = (-H) % 8
        pad_w = (-W) % 8
        if pad_h or pad_w:
            inp = F.pad(inp, (0, pad_w, 0, pad_h), mode="replicate")

        out = self.net(inp)[..., :H, :W] * scale

        if squeezed:
            out = out.squeeze(0)
        return out

