"""
Tests for ct_tfpnp.models.policy.
"""

import pytest
import torch

from ct_tfpnp.models.policy import ResNetActor_ADMM

# set image size
H = W = 64      

# make state for testing
def make_state(B, noise, iter_frac):
    torch.manual_seed(0)

    # create image
    img = torch.rand(1, 1, H, W).repeat(B, 1, 1, 1)

    # create state tuple
    return (img, img.clone(), torch.zeros_like(img), torch.full((B,), noise), torch.full((B,), iter_frac))


@pytest.fixture(scope="module")

# define policy network
def net():
    torch.manual_seed(0)
    return ResNetActor_ADMM(in_channels=5, n_action_steps=5)


class TestOutputContract:

    # check shapes match
    def test_shapes(self, net):
        logits, sigma, mu = net(*make_state(4, 0.05, 0.0))
        assert logits.shape == (4, 2)
        assert sigma.shape == (4, 5)
        assert mu.shape == (4, 5)

    # check outputs stay in range
    def test_sigma_and_mu_stay_inside_their_ranges(self, net):

        # define params
        _, sigma, mu = net(*make_state(4, 0.10, 0.5))

        # define ranges
        s_lo, s_hi = net.sigma_range
        m_lo, m_hi = net.mu_range

        # check outputs are in range
        assert (sigma >= s_lo).all() and (sigma <= s_hi).all()
        assert (mu >= m_lo).all() and (mu <= m_hi).all()
