"""
Total Variation (TV) minimisation for CT reconstruction.

Solves:  min_x  (1/2)||Ax - y||^2 + lam * TV(x),  x >= 0

using proximal gradient descent with LION's CT operator.

As in `admm.py`, the backprojected residual is normalised to unit max magnitude
each iteration, to compensate for LION's unscaled adjoint.

Usage:
    from ct_tfpnp.ct_ops.tv import tv_reconstruction
    x_tv = tv_reconstruction(y, op, lam=0.01, n_iters=200)
"""

import torch

def _grad(x: torch.Tensor):
    """
    Forward-difference gradient. 
    
    (1, H, W) -> (2, 1, H, W): [d/dx, d/dy].
    """
    # calcualte gradient along x and y axes
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)

    # horizonral differences
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1] 

    # vertical differences
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.stack((gx, gy), dim=0)


def _div(p: torch.Tensor):
    """
    Discrete divergence: the negative adjoint of `_grad`, so <grad x, p> = -<x, div p>.
    """
    # unpack the gradient
    px, py = p[0], p[1]

    # set up empty tensors for the divergence
    dx = torch.zeros_like(px)
    dy = torch.zeros_like(py)

    # compute backward differences of px along the horizontal axis
    dx[..., :, 0] = px[..., :, 0]
    dx[..., :, 1:-1] = px[..., :, 1:-1] - px[..., :, :-2]
    dx[..., :, -1] = -px[..., :, -2]

    # compute backward differences of py along the vertical axis
    dy[..., 0, :] = py[..., 0, :]
    dy[..., 1:-1, :] = py[..., 1:-1, :] - py[..., :-2, :]
    dy[..., -1, :] = -py[..., -2, :]

    return dx + dy


def tv_prox(x: torch.Tensor, lam: float, n_iters: int = 20, tau: float = 0.125):
    """
    Proximal operator of isotropic TV: 
    
    argmin_u (1/2)||u - x||^2 + lam * TV(u).

    Args:
        x:       image, shape (1, H, W).
        lam:     regularisation weight.
        n_iters: dual iterations.
        tau:     dual step size, must satisfy tau <= 1/8.

    Returns:
        TV-denoised image, same shape as x.
    """
    # add negativity check
    if lam <= 0:
        return x

    # initialise dual variable
    p = torch.zeros((2, *x.shape), device=x.device, dtype=x.dtype)

    # iterate to solce dual problem
    for _ in range(n_iters):
        
        # gradient of dual objective
        g = _grad(_div(p) - x / lam)
        
        # project onto unit ball
        norm = g.pow(2).sum(dim=0, keepdim=True).sqrt()

        # update dual variable
        p = (p + tau * g) / (1.0 + tau * norm)

    # return primal variable
    return x - lam * _div(p)

def tv_reconstruction(
    y: torch.Tensor,
    op,
    image_shape: tuple = None,
    x0: torch.Tensor = None,
    lam: float = 0.01,
    step_size: float = 0.1,
    n_iters: int = 200,
    prox_iters: int = 20,
    nonneg: bool = True,
    verbose: bool = False,
):
    """
    TV-regularised CT reconstruction via proximal gradient descent.

    Each iteration:
      1. Data fidelity gradient: A^T(Ax - y) normalised to unit max magnitude
      2. Gradient descent step x <- x - step_size * grad
      3. TV proximal step
      4. Non-negativity projection

    Args:
        image_shape: image domain shape, e.g. (1, 512, 512). 
        x0:          initial image. 
        lam:         TV regularisation strength.
        step_size:   gradient step size.
        n_iters:     ISTA iterations.
        prox_iters:  inner Chambolle iterations per prox call.
        nonneg:      project onto x >= 0 each iteration.
        verbose:     print the data-fidelity residual and TV every 50 iterations.

    Returns:
        TV reconstruction, shape (1, H, W).
    """
    # set device and dtype
    device, dtype = y.device, y.dtype

    # initialise x to x0 or zeros
    if x0 is not None:
        x = x0.clone()
    else:
        if image_shape is None:
            image_shape = getattr(op, "domain_shape", None)
        if image_shape is None:
            raise ValueError(
                "image_shape could not be inferred from `op`; pass it explicitly, "
                "e.g. image_shape=(1, 512, 512)."
            )
        x = torch.zeros(*image_shape, device=device, dtype=dtype)

    with torch.no_grad():
        for k in range(n_iters):
            # compute data-fidelity gradient
            residual = op.forward(x) - y
            grad = op.adjoint(residual)
            # Normalise to prevent gradient explosion (same as ADMM z-step)
            grad = grad / (grad.abs().max() + 1e-8)

            # Gradient step
            x = x - step_size * grad

            # TV proximal step
            x = tv_prox(x, lam * step_size, n_iters=prox_iters)

            # Non-negativity
            if nonneg:
                x = x.clamp(min=0)

            # print progress every 50 iterations
            if verbose and (k + 1) % 50 == 0:
                resid = 0.5 * (op.forward(x) - y).pow(2).sum().item()
                tv = _grad(x).pow(2).sum(dim=0).sqrt().sum().item()
                print(f"  TV iter {k+1}/{n_iters} ||Ax-y||^2/2 = {resid:.4e}  TV = {tv:.4e}")

    return x