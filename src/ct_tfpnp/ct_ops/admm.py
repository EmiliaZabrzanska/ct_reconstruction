"""
Differentiable ADMM step for sparse-view CT.

Implements the three ADMM sub-steps (x, z, u) following the variable convention
of Wei et al. 2022 (TFPnP paper), equations 7-9:

    x^{k+1} = H_{sigma}(z^k - u^k)                           [x-step: denoiser prior]
    z^{k+1} = argmin_z (1/2)||Az-y||^2 + (mu/2)||z-(x+u)||^2  [z-step: data fidelity]
    u^{k+1} = u^k + x^{k+1} - z^{k+1}                        [u-step: dual update]

Note on convention: this follows the paper's variable naming where x is the
denoiser variable and z is the data fidelity variable. Some implementations
(including earlier versions of this file) use the opposite convention.

The z-step proximal operator has no closed form for CT, so it is solved by
gradient descent (inexact proximal step). This is deliberately differentiable,
unlike CG, which allows the model-based pi_2 policy gradient to backpropagate
through the z-step with respect to sigma and mu.

LION imports used:
    from LION.CTtools.ct_utils import make_operator
    # op.forward(x) = Ax   (Radon transform)
    # op.adjoint(y) = A^T y (backprojection)
"""

import torch
import torch.nn as nn


class ADMMStep(nn.Module):
    """
    One full ADMM iteration (x, z, u steps) following TFPnP paper convention.

    Args:
        op:         LION CT operator with .forward() and .adjoint() methods
        denoiser:   callable with interface denoiser(v, sigma) -> denoised_v
        n_x_steps:  number of gradient descent steps for z-update (default 6)
        x_lr:       base step size for z gradient descent (default 1.0)
    """

    def __init__(self, op, denoiser, n_x_steps: int = 6, x_lr: float = 1.0):
        super().__init__()
        self.op = op
        self.denoiser = denoiser
        self.n_x_steps = n_x_steps
        self.x_lr = x_lr

    def x_step(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        x^{k+1} = H_{sigma}(z^k - u^k)

        Applies the denoiser to (z - u), enforcing the implicit image prior.
        The denoiser acts as a proximal operator for the regulariser defined
        by the denoiser itself (Venkatakrishnan et al. 2013).

        Args:
            z:     current z variable,    shape (B, 1, H, W)
            u:     current dual variable, shape (B, 1, H, W)
            sigma: denoising strength (scalar or per-batch tensor)

        Returns:
            x_new: shape (B, 1, H, W)
        """
        v = z - u
        return self.denoiser(v, sigma)

    def z_step(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        mu: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        z^{k+1} = argmin_z (1/2)||Az - y||^2 + (mu/2)||z - (x + u)||^2

        Solved by inexact gradient descent (n_x_steps iterations).
        Using GD rather than CG makes this step differentiable w.r.t. mu,
        allowing the model-based pi_2 policy gradient.

        Gradient: A^T(Az - y) + mu*(z - (x + u))

        The backprojection gradient is normalised to unit magnitude to
        compensate for LION's unscaled adjoint operator.

        Args:
            x:  current x variable,       shape (B, 1, H, W)
            u:  current dual variable,    shape (B, 1, H, W)
            y:  sinogram measurements,    shape (B, 1, n_angles, det_count)
            mu: penalty parameter (scalar or per-batch tensor)

        Returns:
            z_new: shape (B, 1, H, W)
        """
        target = x + u
        z = x.clone()
        lr = self.x_lr / (1.0 + mu)

        for _ in range(self.n_x_steps):
            residual  = self.op.forward(z) - y           # both in raw sinogram units
            ATres     = self.op.adjoint(residual)
            ATres     = ATres / (ATres.abs().max() + 1e-8)
            grad      = ATres + mu * (z - target)
            z         = z - lr * grad

        return z

    def u_step(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        u^{k+1} = u^k + x^{k+1} - z^{k+1}

        Accumulates the constraint violation (x - z). This drives x and z
        to agree as the algorithm converges (constraint: x = z at optimum).

        Args:
            u: current dual variable, shape (B, 1, H, W)
            x: updated x variable,   shape (B, 1, H, W)
            z: updated z variable,   shape (B, 1, H, W)

        Returns:
            u_new: shape (B, 1, H, W)
        """
        return u + x - z

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        sigma: float | torch.Tensor,
        mu: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One full ADMM iteration.

        Args:
            x:     current x variable (denoiser output), shape (B, 1, H, W)
            z:     current z variable (data fidelity),   shape (B, 1, H, W)
            u:     current dual variable,                shape (B, 1, H, W)
            y:     sinogram measurements,                shape (B, 1, n_angles, det_count)
            sigma: denoising strength for x-step
            mu:    penalty parameter for z-step

        Returns:
            (x_new, z_new, u_new): updated ADMM variables
        """
        x_new = self.x_step(z, u, sigma)
        z_new = self.z_step(x_new, u, y, mu)
        u_new = self.u_step(u, x_new, z_new)
        return x_new, z_new, u_new