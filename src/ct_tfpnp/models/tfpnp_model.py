"""
TFPnP model wrapper, makes the policy/critic pair compatible with LIONmodel.
"""

from copy import deepcopy
from LION.models.LIONmodel import LIONmodel
from LION.utils.parameter import LIONParameter

from ct_tfpnp.models.policy import ResNetActor_ADMM
from ct_tfpnp.models.critic import ResNet_wobn


class TFPnPModel(LIONmodel):
    """
    Wraps the TFPnP policy, critic and target critic as a single LIONmodel, so
    that TFPnPSolver can use LION's checkpointing and training infrastructure.

    Args:
        model_parameters: LIONParameter; see `default_parameters`.
        geometry:         LION Geometry (stored by LIONmodel; unused here).
    """

    def __init__(self, model_parameters=None, geometry=None):
        if model_parameters is None:
            model_parameters = TFPnPModel.default_parameters()

        super().__init__(model_parameters, geometry)

        # create policy and critic networks
        mp = self.model_parameters
        self.policy = ResNetActor_ADMM(
            in_channels=mp.in_channels,
            n_action_steps=mp.m,
            sigma_range=(mp.sigma_min, mp.sigma_max),
            mu_range=(mp.mu_min, mp.mu_max),
        )
        self.critic = ResNet_wobn(
            in_channels=mp.in_channels,
            base_channels=mp.critic_base_channels,
            n_blocks=mp.critic_n_blocks,
            stride=mp.critic_stride,
        )

        # EMA target for the TD bootstrap; never receives gradients
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    @staticmethod
    def default_parameters():
        """
        Default architecture parameters.

        sigma_min/max are in this project's units (denoiser sigma_scale=50).

        The policy uses BatchNorm, as in the paper's Table 1. 

        The critic uses plain convolutions with no normalisation.
        """
        return LIONParameter(
            in_channels=5,
            m=5,
            sigma_min=1.0,
            sigma_max=5.0,
            mu_min=10.0,
            mu_max=100.0,
            critic_base_channels=64,
            critic_n_blocks=8,
            critic_stride=1,
            normalisator=None,
        )

    def forward(self, x, z, u, noise_level, iter_frac):
        """
        Policy forward pass.
        """
        # compute next action (sigma, mu) for the current state (x, z, u)
        return self.policy(x, z, u, noise_level, iter_frac)
