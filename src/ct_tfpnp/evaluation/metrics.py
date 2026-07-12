"""
Evaluation metrics for CT reconstruction quality.

PSNR, SSIM and HaarPSI, all computed on torch tensors (the *_np helpers just
accept numpy input and return Python floats).
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

def default_data_range(x_gt: torch.Tensor):
    """
    The project-wide data range convention: the peak-to-peak range of the ground truth.
    """
    return float(x_gt.max() - x_gt.min())


def ls_scale(gt: torch.Tensor, recon: torch.Tensor):
    """
    Least-squares optimal scalar alignment: argmin_alpha ||gt - alpha*recon||^2.

    The eval/plot scripts apply ls_scale to the output of every
    method (FBP, TV, DRUNet, FBPConvNet, fixed PnP-ADMM, TFPnP), so the comparison
    between them is fair.

    Args:
        gt:    ground truth.
        recon: reconstruction, same shape.

    Returns:
        alpha * recon.
    """

    # find scaling factor
    alpha = (gt * recon).sum() / ((recon * recon).sum() + 1e-12)

    # scale reconstruction
    return recon * alpha


# differentiable metrics (torch tensors)

def psnr(x_hat: torch.Tensor, x_gt: torch.Tensor, data_range: float = None):
    """
    Peak Signal-to-Noise Ratio in dB (differentiable).

    Args:
        x_hat:      reconstruction, any shape.
        x_gt:       ground truth, same shape.
        data_range: peak signal value. 

    Returns:
        Scalar PSNR in dB, as a tensor on x_hat's device.
    """
    # compute data range if not provided
    if data_range is None:
            data_range = default_data_range(x_gt)

    # compute MSE
    mse = ((x_hat - x_gt) ** 2).mean()
    if mse.item() == 0.0:
        return torch.tensor(float("inf"), device=x_hat.device)
    
    # compute PSNR
    return 10 * torch.log10(torch.tensor(data_range ** 2, device=x_hat.device, dtype=mse.dtype) / mse)


def ssim(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    data_range: float = None,
    window_size: int = 11,
    sigma: float = 1.5,
):
    """
    Structural Similarity Index (differentiable).

    Args:
        x_hat:       reconstruction, shape (B, 1, H, W).
        x_gt:        ground truth, shape (B, 1, H, W).
        data_range:  peak signal value.
        window_size: Gaussian window size (default 11).
        sigma:       Gaussian window std (default 1.5).

    Returns:
        Mean SSIM over the map and the batch, scalar tensor.
    """
    # compute data range if not provided
    if data_range is None:
        data_range = default_data_range(x_gt)
    
    if x_hat.shape[1] != 1:
        raise ValueError(f"ssim expects single-channel input, got {x_hat.shape[1]} channels")

    # constants for SSIM
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # create Gaussian window
    coords = torch.arange(window_size, dtype=x_hat.dtype, device=x_hat.device) - window_size // 2

    # compute Gaussian kernel
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g).view(1, 1, window_size, window_size)

    # convolution function
    def conv(img):
        return F.conv2d(img, kernel)

    # compute means of x_hat and x_gt
    mu1 = conv(x_hat)
    mu2 = conv(x_gt)

    # compute squares and products of means
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # compute variances and covariance
    sigma1_sq = conv(x_hat * x_hat) - mu1_sq
    sigma2_sq = conv(x_gt * x_gt) - mu2_sq
    sigma12 = conv(x_hat * x_gt) - mu1_mu2

    # compute SSIM map
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    # return mean SSIM over the map and the batch
    return (numerator / denominator).mean()


def haarpsi_score(x_hat: torch.Tensor, x_gt: torch.Tensor,
                  data_range: float = None):
    """
    HaarPSI correlates most with radiologist preference.
    Inputs are clamped to [0, data_range], which piq's implementation requires.

    Args:
        x_hat:      reconstruction, shape (B, 1, H, W).
        x_gt:       ground truth, same shape.
        data_range: peak signal value. Defaults to default_data_range(x_gt).

    Returns:
        HaarPSI score in [0, 1].
    """
    # ensure piq is installed
    if not HAS_HAARPSI:
        raise ImportError("piq not installed — run: pip install piq")
    
    # compute data range if not provided
    if data_range is None:
        data_range = default_data_range(x_gt)

    # compute HaarPSI score using piq's implementation and return as float
    return float(_haarpsi(x_hat.float().clamp(0, data_range), x_gt.float().clamp(0, data_range), data_range=data_range).item())


def evaluate_reconstruction(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    data_range: float = None,
):
    """
    Compute PSNR, SSIM and HaarPSI for one batch.

    Args:
        x_hat:      reconstruction, shape (B, 1, H, W).
        x_gt:       ground truth, same shape.
        data_range: peak signal value. Defaults to default_data_range(x_gt).

    Returns:
        {'psnr': ..., 'ssim': ..., 'haarpsi': ...}
    """
    # compute data range if not provided
    if data_range is None:
        data_range = default_data_range(x_gt)
    
    # compute PSNR, SSIM and HaarPSI
    with torch.no_grad():
        result = {
            "psnr": float(psnr(x_hat, x_gt, data_range)),
            "ssim": float(ssim(x_hat, x_gt, data_range)),
        }
        if HAS_HAARPSI:
            result["haarpsi"] = haarpsi_score(x_hat, x_gt, data_range)

    return result


# Numpy-friendly utilities (used in notebooks and training)

def _as_4d(t):
    """
    Coerce a (H, W) / (1, H, W) / (1, 1, H, W) tensor or array to (1, 1, H, W).
    """
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(np.asarray(t))
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return t

def psnr_np(gt, recon, data_range: float = None):
    """
    PSNR as a Python float. 
    Accepts tensors or arrays and detaches automatically.
    """
    # compute ground truth and reconstruction as 4D tensors
    gt = _as_4d(gt).detach()
    recon = _as_4d(recon).detach().to(gt.device)

    # compute data range if not provided
    if data_range is None:
        data_range = default_data_range(gt)

    # copmute MSE
    mse_val = float(((gt - recon) ** 2).mean())

    if mse_val < 1e-14:
        return float('inf')
    
    # compute PSNR
    return 10.0 * math.log10(data_range**2 / mse_val)


def ssim_np(gt, recon, data_range: float = None):
    """
    SSIM as a Python float, using the same convention as `ssim`.
    """
    # compute ground truth and reconstruction as 4D tensors
    gt = _as_4d(gt).detach()
    recon = _as_4d(recon).detach().to(gt.device)

    # compute data range if not provided
    if data_range is None:
        data_range = default_data_range(gt)

    # compute SSIM and return as float
    return float(ssim(recon.clamp(0, data_range), gt, data_range=data_range))


def haarpsi_np(gt, recon, data_range: float = None):
    """
    HaarPSI as a Python float, using the same convention as `haarpsi_score`.
    """
    # ensure piq is installed
    if not HAS_HAARPSI:
        raise ImportError("piq not installed — required for haarpsi_np")
    
    # compute ground truth and reconstruction as 4D tensors
    gt = _as_4d(gt).detach()
    recon = _as_4d(recon).detach().to(gt.device)

    # compute data range if not provided
    if data_range is None:
        data_range = float(gt.max())

    # compute HaarPSI score
    return haarpsi_score(recon, gt, data_range=data_range)