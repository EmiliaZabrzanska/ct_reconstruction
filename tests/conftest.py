"""
Shared fixtures for the ct_tfpnp test suite.

Design note
-----------
None of these tests require LION, tomosipo, astra or a GPU, as a `FakeCTOperator` is created to stand in for the real CT operator.
"""

import pytest
import torch
import torch.nn as nn

# set seed
torch.manual_seed(0)


class FakeCTOperator:
    """
    A small dense linear operator with the LION operator's interface.

    Allows the test suite to run without LION, tomosipo or astra, and without a GPU.§

    Args:
        image_shape: domain shape, (C, H, W).
        n_angles:    projection angles.
        n_det:       detector bins.
        seed:        RNG seed for the matrix entries.
    """

    def __init__(self, image_shape=(1, 16, 16), n_angles=8, n_det=24, seed=0):

        # set shape attributes
        self.domain_shape = tuple(image_shape)
        self.range_shape = (1, n_angles, n_det)

        # create image and sinogram
        n_img = image_shape[0] * image_shape[1] * image_shape[2]
        n_sino = n_angles * n_det

        # create local generator
        g = torch.Generator().manual_seed(seed)
        
        # define matrix
        self.A = torch.rand(n_sino, n_img, generator=g) / (n_img ** 0.5)

    def forward(self, x):
        """
        Ax. 
        
        Accepts (C, H, W) or (B, C, H, W) with B == 1.
        """
        # reshape to 4D
        is4d = x.dim() == 4

        # reshape back to 3D and apply A
        out = (self.A @ x.reshape(-1)).reshape(*self.range_shape)

        # reshape back to 4D if needed
        return out.unsqueeze(0) if is4d else out

    def adjoint(self, y):
        """
        A^T y. 
        
        Accepts (1, n_angles, n_det) or a 4-D version with B == 1.
        """
        # reshape to 4D
        is4d = y.dim() == 4

        # reshape back to 3D and apply A^T
        out = (self.A.T @ y.reshape(-1)).reshape(*self.domain_shape)

        # reshape back to 4D if needed
        return out.unsqueeze(0) if is4d else out

    @property
    def norm_sq(self):
        """
        ||A||^2, i.e. the largest eigenvalue of A^T A. 
        """
        # compute squared norm of A
        return torch.linalg.matrix_norm(self.A, ord=2).item() ** 2


class FakeDenoiser(nn.Module):
    """
    A differentiable stand-in for DRUNet.
    """

    def forward(self, z, sigma):

        # find scale factor
        s = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma))

        # scale input to match DRUNet range
        return z / (1.0 + s.to(z.dtype) / 50.0)


class FakePolicy(nn.Module):
    """
    A tiny policy with the real one's interface and output contract, conditioning only on (noise_level, iter_frac).
    """

    def __init__(self, m=3, sigma_range=(1.0, 5.0), mu_range=(10.0, 100.0)):
        super().__init__()
        self.n_action_steps = m
        self.sigma_range = tuple(sigma_range)
        self.mu_range = tuple(mu_range)
        self.head = nn.Linear(2, 2 + 2 * m)

    def forward(self, x, z, u, noise_level, iter_frac):

        #define batch size and condition vector
        B = x.shape[0]
        cond = torch.stack([noise_level.reshape(B), iter_frac.reshape(B)], dim=1)
        out = self.head(cond)

        # split into stop logits and raw sigma/mu
        stop_logits = out[:, :2]
        raw = torch.sigmoid(out[:, 2:])

        # scale sigma and mu to configured ranges
        m = self.n_action_steps
        s_lo, s_hi = self.sigma_range
        m_lo, m_hi = self.mu_range
        sigma_seq = s_lo + raw[:, :m] * (s_hi - s_lo)
        mu_seq = m_lo + raw[:, m:] * (m_hi - m_lo)

        return stop_logits, sigma_seq, mu_seq


class FakeCritic(nn.Module):
    """
    A tiny value network with the real one's interface.
    """

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(5, 1)

    def forward(self, x, z, u, noise_level, iter_frac):

        # define batch size and feature vector
        B = x.shape[0]
        feats = torch.stack([
            x.reshape(B, -1).mean(1),
            z.reshape(B, -1).mean(1),
            u.reshape(B, -1).mean(1),
            noise_level.reshape(B),
            iter_frac.reshape(B),
        ], dim=1)
        return self.head(feats)


@pytest.fixture(scope="session")
def op():
    """
    A small CT-like linear operator.
    """
    return FakeCTOperator()


@pytest.fixture(scope="session")
def image_shape(op):
    """
    The operator's image-domain shape.
    """
    return op.domain_shape


@pytest.fixture
def gt(image_shape):
    """
    A non-negative phantom.
    """
    # create square
    x = torch.zeros(*image_shape)
    x[..., 4:12, 4:12] = 1.0
    x[..., 6:10, 6:10] = 1.6

    # add noise
    x += 0.05 * torch.rand(*image_shape)

    return x.clamp(min=0)


@pytest.fixture
def sino(op, gt):
    """
    The clean sinogram A @ gt.
    """
    return op.forward(gt)


@pytest.fixture
def denoiser():
    """
    A differentiable stand-in for DRUNet.
    """
    return FakeDenoiser()


@pytest.fixture
def admm_step(op, denoiser):
    """
    An ADMMStep wired to the fake operator and denoiser.
    """
    from ct_tfpnp.ct_ops.admm import ADMMStep
    return ADMMStep(op=op, denoiser=denoiser, n_x_steps=3)


@pytest.fixture
def policy():
    """
    A fake policy with m = 3, to keep the ADMM rollouts short.
    """
    return FakePolicy(m=3)


@pytest.fixture
def critic():
    """
    A fake value network.
    """
    return FakeCritic()