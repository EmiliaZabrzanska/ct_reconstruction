"""
Total Variation (TV) minimisation for CT reconstruction.

Solves:  min_x  (1/2)||Ax - y||² + λ·TV(x)

using proximal gradient descent (ISTA) with LION's CT operator.
TV is the standard regularisation baseline in CT reconstruction
(Sidky & Pan 2008, Chambolle & Pock 2011).

Usage:
    from ct_tfpnp.ct_ops.tv import tv_reconstruction
    x_tv = tv_reconstruction(y, op, lam=0.01, n_iters=200)
"""

import torch


def _tv_proximal(x: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Proximal operator for anisotropic Total Variation.

    Applies soft-thresholding to finite differences (image gradients),
    then reconstructs via the divergence. This is equivalent to one step
    of the Chambolle projection algorithm.

    Args:
        x:   image tensor, shape (1, H, W)
        lam: regularisation strength × step size

    Returns:
        TV-denoised image, same shape
    """
    # Finite differences (image gradients)
    dx = x[..., :, 1:] - x[..., :, :-1]   # horizontal gradients
    dy = x[..., 1:, :] - x[..., :-1, :]   # vertical gradients

    # Soft-thresholding (anisotropic TV proximal)
    dx = torch.sign(dx) * torch.clamp(dx.abs() - lam, min=0)
    dy = torch.sign(dy) * torch.clamp(dy.abs() - lam, min=0)

    # Divergence (adjoint of gradient operator)
    div = torch.zeros_like(x)
    div[..., :, 1:]  += dx
    div[..., :, :-1] -= dx
    div[..., 1:, :]  += dy
    div[..., :-1, :] -= dy

    return x + lam * div


def tv_reconstruction(
    y: torch.Tensor,
    op,
    x0: torch.Tensor = None,
    lam: float = 0.01,
    step_size: float = 0.1,
    n_iters: int = 200,
    verbose: bool = False,
) -> torch.Tensor:
    """
    TV-regularised CT reconstruction via proximal gradient descent (ISTA).

    Each iteration:
      1. Data fidelity gradient: A^T(Ax - y)  (normalised)
      2. Gradient descent step: x ← x - α·grad
      3. TV proximal step: soft-threshold finite differences
      4. Non-negativity: clamp to [0, ∞)

    Args:
        y:         sinogram measurements, shape (1, n_angles, n_det)
        op:        LION CT operator
        x0:        initial reconstruction (default: zeros)
        lam:       TV regularisation strength (default 0.01)
        step_size: gradient descent step size (default 0.1)
        n_iters:   number of iterations (default 200)
        verbose:   print progress every 50 iterations

    Returns:
        TV reconstruction, shape (1, H, W)
    """
    if x0 is not None:
        x = x0.clone()
    else:
        x = torch.zeros(1, y.shape[-1], y.shape[-1],
                         device=y.device, dtype=y.dtype)

    with torch.no_grad():
        for k in range(n_iters):
            # Data fidelity gradient: A^T(Ax - y)
            residual = op.forward(x) - y
            grad = op.adjoint(residual)
            # Normalise to prevent gradient explosion (same as ADMM z-step)
            grad = grad / (grad.abs().max() + 1e-8)

            # Gradient step
            x = x - step_size * grad

            # TV proximal step
            x = _tv_proximal(x, lam * step_size)

            # Non-negativity
            x = x.clamp(min=0)

            if verbose and (k + 1) % 50 == 0:
                print(f"  TV iter {k+1}/{n_iters}")

    return x