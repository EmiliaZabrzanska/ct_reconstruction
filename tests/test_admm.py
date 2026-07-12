"""
Tests for ct_tfpnp.ct_ops.admm.
"""

import pytest
import torch

from ct_tfpnp.ct_ops.admm import ADMMStep


class TestSubSteps:

    # test if constraint violation is collected in u-step
    def test_u_step_accumulates_the_constraint_violation(self, admm_step):
        u = torch.zeros(1, 8, 8)
        x = torch.full((1, 8, 8), 2.0)
        z = torch.full((1, 8, 8), 0.5)
        assert torch.allclose(admm_step.u_step(u, x, z), x - z)

    # test if x-step denoises z - u by the denoiser
    def test_x_step_denoises_z_minus_u(self, admm_step, denoiser):

        # set fake z and u 
        z, u = torch.rand(1, 16, 16), torch.rand(1, 16, 16)

        # check if x steo equals denoiser(z - u, sigma)
        assert torch.allclose(admm_step.x_step(z, u, sigma=10.0), denoiser(z - u, 10.0))

    # if steps are big, z-step should approach x + u
    def test_z_step_approaches_the_target_as_mu_grows(self, admm_step, gt, op):

        # define target and fake x and u 
        y = op.forward(gt)
        x, u = gt.clone(), torch.zeros_like(gt)
        target = x + u

        # define z with large mu
        z = admm_step.z_step(x, u, y, mu=1e6)

        # check if z approaches target
        assert (z - target).abs().max() < 1e-2


class TestForward:

    # ensure all outputs have the correct shape
    def test_returns_three_tensors_of_the_input_shape(self, admm_step, gt, op):

        # define fake x, z, u and y
        y = op.forward(gt)
        x, z, u = gt.clone(), gt.clone(), torch.zeros_like(gt)

        # compute outputs
        xn, zn, un = admm_step(x, z, u, y, sigma=5.0, mu=20.0)

        # check shapes match
        assert xn.shape == zn.shape == un.shape == gt.shape

    # ensure differentialbility for pi2 policy gradient
    def test_is_differentiable_wrt_sigma_and_mu(self, admm_step, gt, op):
        
        # define y, sigma, mu 
        y = op.forward(gt)
        sigma = torch.tensor(5.0, requires_grad=True)
        mu = torch.tensor(20.0, requires_grad=True)

        # define fake x, z, u
        x, z, u = gt.clone(), gt.clone(), torch.zeros_like(gt)

        # compute outputs
        for _ in range(2):
            x, z, u = admm_step(x, z, u, y, sigma=sigma, mu=mu)

        # use backward to compute gradients
        x.sum().backward()

        # check if gradients are non-zero
        assert sigma.grad is not None and sigma.grad.abs().item() > 0
        assert mu.grad is not None and mu.grad.abs().item() > 0
