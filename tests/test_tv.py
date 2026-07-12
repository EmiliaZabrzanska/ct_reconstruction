"""
Tests for ct_tfpnp.ct_ops.tv.
"""

import pytest
import torch

from ct_tfpnp.ct_ops.tv import _div, _grad, tv_prox, tv_reconstruction


def total_variation(x):
    # check TV is computed correctly
    return _grad(x).pow(2).sum(dim=0).sqrt().sum()


def prox_objective(u, x, lam):
    # check proximal objective is computed correctly
    return 0.5 * ((u - x) ** 2).sum() + lam * total_variation(u)


class TestGradAndDiv:

    def test_adjoint_identity(self):
        # define both sides of adjoint identity <grad x, p> = -<x, div p>
        torch.manual_seed(0)
        x = torch.randn(1, 12, 12)
        p = torch.randn(2, 1, 12, 12)
        lhs = (_grad(x) * p).sum()
        rhs = -(x * _div(p)).sum()

        # compare 
        assert lhs.item() == pytest.approx(rhs.item(), abs=1e-5)


class TestTVProx:
    @pytest.fixture
    # create noisy image with clean square in middle
    def noisy(self):
        torch.manual_seed(1)
        x = torch.zeros(1, 24, 24)
        x[..., 6:18, 6:18] = 1.0                
        return x + 0.15 * torch.randn(1, 24, 24)

    def test_decreases_the_objective(self, noisy):
        # define lambda and prox objective
        lam = 0.05
        u = tv_prox(noisy, lam)

        # check if objective decreased after prox op
        assert prox_objective(u, noisy, lam) < prox_objective(noisy, noisy, lam)

    # check if increasing lambda increases smoothing
    def test_stronger_lam_smooths_more(self, noisy):
        assert total_variation(tv_prox(noisy, 0.20)) < total_variation(tv_prox(noisy, 0.02))

    # chekc if output shape matches input
    def test_preserves_shape(self, noisy):
        assert tv_prox(noisy, 0.05).shape == noisy.shape


class TestTVReconstruction:

    # check that output is non-negative
    def test_is_non_negative(self, gt, op, image_shape):
        y = op.forward(gt)
        assert tv_reconstruction(y, op, image_shape=image_shape, n_iters=5).min() >= 0