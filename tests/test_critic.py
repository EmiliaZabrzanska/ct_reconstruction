"""
Tests for ct_tfpnp.models.critic.
"""

import torch

from ct_tfpnp.models.critic import ResBlock_wobn, ResNet_wobn

# set image size
H = W = 32

# make state for critic
def make_state(B, noise=0.05, iter_frac=0.0):
    img = torch.rand(B, 1, H, W)
    return (img, img.clone(), torch.zeros_like(img),
            torch.full((B,), noise), torch.full((B,), iter_frac))


class TestResBlock:

    # check shape is preserved
    def test_preserves_shape(self):
        x = torch.randn(2, 6, 8, 8)
        assert ResBlock_wobn(6)(x).shape == x.shape


class TestResNetWobn:

    # check differentiability
    def test_is_differentiable(self):
        net = ResNet_wobn(n_blocks=2, base_channels=8)
        net(*make_state(2)).sum().backward()
        assert net.head.weight.grad.abs().sum() > 0