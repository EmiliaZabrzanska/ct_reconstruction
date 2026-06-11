"""
Evaluation metrics for CT reconstruction quality.

PSNR and SSIM are the standard metrics used in the TFPnP paper (Table 3–5).
Also includes utility functions used across all notebooks.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict

try:
    from piq import haarpsi as _haarpsi
    HAS_HAARPSI = True
except ImportError:
    HAS_HAARPSI = False


# ── Tensor-based metrics (differentiable) ─────────────────────────────────

def psnr(x_hat: torch.Tensor, x_gt: torch.Tensor,
         data_range: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio in dB (differentiable).

    Args:
        x_hat:      reconstruction, shape (..., H, W) or (B, 1, H, W)
        x_gt:       ground truth, same shape
        data_range: value range of x_gt (default 1.0 for normalised images)

    Returns:
        Scalar PSNR (mean over batch if batched), in dB
    """
    mse = ((x_hat - x_gt) ** 2).mean()
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(torch.tensor(data_range**2) / mse)


def ssim(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM).

    Args:
        x_hat:       reconstruction, shape (B, 1, H, W)
        x_gt:        ground truth, shape (B, 1, H, W)
        data_range:  pixel value range
        window_size: Gaussian window size (default 11)
        sigma:       Gaussian window std (default 1.5)

    Returns:
        Mean SSIM over batch, scalar in [0, 1]
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(x_hat.device)

    pad = window_size // 2

    def conv(img):
        return F.conv2d(img, kernel, padding=pad, groups=1)

    mu1 = conv(x_hat)
    mu2 = conv(x_gt)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv(x_hat * x_hat) - mu1_sq
    sigma2_sq = conv(x_gt * x_gt) - mu2_sq
    sigma12 = conv(x_hat * x_gt) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return (numerator / denominator).mean()


def evaluate_reconstruction(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """Compute PSNR, SSIM, and HaarPSI for a batch of reconstructions."""
    with torch.no_grad():
        psnr_val = psnr(x_hat, x_gt, data_range).item()
        ssim_val = ssim(x_hat, x_gt, data_range).item()
        result = {"psnr": psnr_val, "ssim": ssim_val}
        if HAS_HAARPSI:
            # Clamp to [0, data_range] — HaarPSI requires this
            x_hat_c = x_hat.clamp(0, data_range)
            x_gt_c = x_gt.clamp(0, data_range)
            result["haarpsi"] = haarpsi_score(x_hat_c, x_gt_c, data_range)
    return result


# ── Numpy-friendly utilities (used in notebooks and training) ─────────────

def psnr_np(gt, recon) -> float:
    """
    PSNR using GT data range. Accepts tensors or numpy arrays.
    Detaches and moves to CPU automatically.
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(recon, torch.Tensor):
        recon = recon.detach().cpu().numpy()
    gt = np.asarray(gt, dtype=np.float64).squeeze()
    recon = np.asarray(recon, dtype=np.float64).squeeze()
    mse_val = np.mean((gt - recon) ** 2)
    data_range = float(gt.max() - gt.min())
    if mse_val < 1e-14:
        return float('inf')
    return 10.0 * math.log10(data_range**2 / mse_val)


def ls_scale(gt: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """
    Least-squares optimal scalar alignment: argmin_α ||gt - α·recon||².
    Returns α·recon.
    """
    alpha = (gt * recon).sum() / ((recon * recon).sum() + 1e-12)
    return recon * alpha


def haarpsi_score(x_hat: torch.Tensor, x_gt: torch.Tensor,
                  data_range: float = 1.0) -> float:
    """
    HaarPSI — correlates most with radiologist preference (Biguri et al.).

    Args:
        x_hat: reconstruction, shape (B, 1, H, W)
        x_gt:  ground truth, shape (B, 1, H, W)
        data_range: pixel value range

    Returns:
        HaarPSI score as float
    """
    if not HAS_HAARPSI:
        raise ImportError("piq not installed — run: pip install piq")
    return _haarpsi(x_hat.float(), x_gt.float(), data_range=data_range).item()