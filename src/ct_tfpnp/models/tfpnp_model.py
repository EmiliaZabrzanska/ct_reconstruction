"""
TFPnP model wrapper — makes the policy/critic pair compatible with LIONmodel.
"""

import torch
from copy import deepcopy
from LION.models.LIONmodel import LIONmodel
from LION.utils.parameter import LIONParameter

from ct_tfpnp.models.policy import ResNetActor_ADMM
from ct_tfpnp.models.critic import ResNet_wobn


class TFPnPModel(LIONmodel):
    """
    Wraps the TFPnP policy and critic networks as a single LIONmodel.

    This allows TFPnPSolver to use LION's checkpointing, saving, and
    training infrastructure.
    """

    def __init__(self, model_parameters=None, geometry=None):
        if model_parameters is None:
            model_parameters = TFPnPModel.default_parameters()

        super().__init__(model_parameters, geometry)

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
        )
        self.target_critic = deepcopy(self.critic)
        # Target critic doesn't need gradients
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    @staticmethod
    def default_parameters():
        return LIONParameter(
            in_channels=5,
            m=5,
            sigma_min=1.0,
            sigma_max=5.0,
            mu_min=10.0,
            mu_max=100.0,
            critic_base_channels=64,
            critic_n_blocks=8,
            normalisator=None,
        )

    def forward(self, x, z, u, noise_level, iter_frac):
        """Policy forward pass."""
        return self.policy(x, z, u, noise_level, iter_frac)

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            return ('Wei, Kaixuan, et al. "Tuning-free plug-and-play proximal '
                    'algorithm for inverse imaging problems." '
                    'Journal of Machine Learning Research 23.1 (2022).')
        elif cite_format == "BibTeX":
            return ('@article{wei2022tfpnp,\n'
                    '  title={Tuning-free plug-and-play proximal algorithm},\n'
                    '  author={Wei, Kaixuan and Aviles-Rivero, Angelica I and '
                    'Liang, Jingwei and Fu, Ying and Huang, Hua and Sch{\\"{o}}nlieb, '
                    'Carola-Bibiane},\n'
                    '  journal={JMLR},\n'
                    '  year={2022}\n}')