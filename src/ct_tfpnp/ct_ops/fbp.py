"""
Filtered Backprojection (FBP) using LION's CT operator.

Applies a ramp filter in the frequency domain to the sinogram before
backprojection. This is the standard analytic CT reconstruction and serves as
the x0 initialisation for PnP-ADMM.

Everything stays in PyTorch/CUDA throughout, so the FBP is differentiable and
can be used inside an autograd graph if needed.
"""

import math

import torch



def ramp_filter(sino: torch.Tensor):
    """
    Apply a ramp filter to a sinogram along the detector axis.

    Args:
        sino: sinogram tensor of shape (1, n_angles, n_det)

    Returns:
        filtered sinogram, same shape
    """
    # define number of detector pixels
    n_det = sino.shape[-1]

    # Build ramp filter in frequency domain
    freqs = torch.fft.fftfreq(n_det, device=sino.device)
    ramp  = torch.abs(freqs)

    # Apply filter to each projection independently along detector axis
    sino_fft = torch.fft.fft(sino, dim=-1)

    # inverse FFT to get filtered sinogram back in spatial domain
    sino_filtered = torch.fft.ifft(sino_fft * ramp, dim=-1).real

    return sino_filtered


def fbp(sino: torch.Tensor, op, clamp_nonneg: bool = True):
    """
    Filtered Backprojection reconstruction.

    For parallel beam with unit spacing (LION convention):
    x_FBP = (π / N_θ) × A† × (ramp * y)

    Args:
        sino:     sinogram, shape (1, n_angles, n_det)
        op:       LION CT operator
        clamp_nonneg: whether to clamp the reconstruction to [0, ∞)

    Returns:
        FBP reconstruction, shape (1, H, W), clipped to [0, ∞)
    """
    n_angles = sino.shape[-2]
    scaling = math.pi / (2.0 * n_angles)

    # backproject filtered sinogram
    recon = op.adjoint(ramp_filter(sino)) * scaling

    # clamp to non-negative vals
    return recon.clamp(min=0) if clamp_nonneg else recon


def calibrate_to_data(x: torch.Tensor, y: torch.Tensor, op):
    """
    Rescale a reconstruction by the least-squares factor that best explains the
    measurements: alpha = <Ax, y> / <Ax, Ax>, returning alpha * x.

    Args:
        x:  reconstruction, shape (1, H, W).
        y:  measured sinogram, shape (1, n_angles, n_det).
        op: LION CT operator.

    Returns:
        Rescaled reconstruction, same shape as x.
    """
    # forward project x to get Ax
    Ax = op.forward(x)
    
    # compute least-squares scaling factor 
    alpha = (Ax * y).sum() / ((Ax * Ax).sum() + 1e-12)

    # return scaled reconstruction
    return x * alpha