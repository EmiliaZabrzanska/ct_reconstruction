"""
Filtered Backprojection (FBP) using LION's CT operator.

Applies a Ram-Lak (ramp) filter in the frequency domain to the sinogram
before backprojection. This is the standard FBP reconstruction algorithm
and serves as the x₀ initialisation for PnP-ADMM.

This implementation is designed to work with LION's tomosipo-backed operator
and keeps everything in PyTorch/CUDA throughout.
"""

import torch
import numpy as np


def ramp_filter(sino: torch.Tensor) -> torch.Tensor:
    """
    Apply Ram-Lak (ramp) filter to a sinogram in the frequency domain.

    Args:
        sino: sinogram tensor of shape (1, n_angles, n_det)

    Returns:
        filtered sinogram, same shape
    """
    n_det = sino.shape[-1]

    # Build ramp filter in frequency domain
    freqs = torch.fft.fftfreq(n_det, device=sino.device)
    ramp  = torch.abs(freqs)

    # Apply filter to each projection independently
    sino_fft      = torch.fft.fft(sino, dim=-1)
    sino_filtered = torch.fft.ifft(sino_fft * ramp, dim=-1).real

    return sino_filtered


def fbp(sino: torch.Tensor, op, n_angles: int = 30) -> torch.Tensor:
    """
    Filtered Backprojection reconstruction.

    For parallel beam with unit spacing (LION convention):
    x_FBP = (π / N_θ) × A† × (ramp * y)

    Args:
        sino:     sinogram, shape (1, n_angles, n_det)
        op:       LION CT operator
        n_angles: number of projection angles (default 30)

    Returns:
        FBP reconstruction, shape (1, H, W), clipped to [0, ∞)
    """
    sino_filtered = ramp_filter(sino)
    recon         = op.adjoint(sino_filtered)

    # FBP normalisation for parallel beam with unit spacing
    recon = recon * (np.pi / (2.0 * n_angles))

    return recon.clamp(min=0)