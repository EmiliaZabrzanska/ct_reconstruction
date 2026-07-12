"""
Tests for ct_tfpnp.models.denoiser.
"""

import pytest
import torch

from ct_tfpnp.models.denoiser import DRUNet, DRUNetDenoiser


@pytest.fixture(scope="module")

# define small net
def small_net():
    torch.manual_seed(0)
    d = DRUNetDenoiser(pretrained_path=None, verbose=False)
    d.net = DRUNet(in_channels=2, out_channels=1, nc=[4, 8, 16, 32])
    return d


class TestDRUNetArchitecture:

    # check forward shape
    def test_forward_shape(self):
        net = DRUNet(in_channels=2, out_channels=1, nc=[4, 8, 16, 32])
        assert net(torch.rand(2, 2, 32, 32)).shape == (2, 1, 32, 32)


class TestDRUNetDenoiser:

    # check 4D input
    def test_preserves_shape_4d(self, small_net):
        x = torch.rand(2, 1, 32, 32)
        assert small_net(x, sigma=10.0).shape == x.shape

    # check 3D input
    def test_preserves_shape_3d(self, small_net):
        x = torch.rand(1, 32, 32)
        assert small_net(x, sigma=10.0).shape == x.shape

    # check differentiability for pi2 
    def test_is_differentiable_wrt_sigma(self, small_net):

        # deifne x and sigma
        x = torch.rand(1, 1, 32, 32)
        sigma = torch.tensor([10.0], requires_grad=True)

        # check net gradient
        small_net(x, sigma).sum().backward()
        
        assert sigma.grad is not None and sigma.grad.abs().item() > 0
