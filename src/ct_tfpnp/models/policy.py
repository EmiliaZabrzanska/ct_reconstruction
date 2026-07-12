"""
Policy network for TFPnP.

Takes the current ADMM state (x, z, u) plus auxiliary scalars (noise level,
iteration fraction) and outputs:
  - a1: termination logits  (2-way: [continue, stop])
  - a2: continuous parameters (sigma[m], mu[m]) for the next m ADMM steps

Architecture (Wei et al. Table 1):
  - ResNet-18 feature extractor (shared between sub-policies)
  - Head 1 -> 2 logits  (continue / stop)
  - Head 2 -> 2m sigmoid outputs (m sigmas, m mus), affinely scaled to range

Input channels:
  - x  (1 channel)
  - z  (1 channel)
  - u  (1 channel)
  - noise_level broadcast to HxW  (1 channel)
  - iteration_frac broadcast to HxW  (1 channel)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNetActor_ADMM(nn.Module):
    """
    Policy network: maps the ADMM state to (termination logits, sigma_seq, mu_seq).

    Args:
        in_channels:    input channels (default 5: x, z, u, noise_lvl, iter_frac).
        n_action_steps: ADMM steps per decision (m in the paper, default 5).
        sigma_range:    (min, max) for the denoising strength sigma, in this
                        project's units (denoiser sigma_scale=50). 
        mu_range:       (min, max) for the ADMM penalty parameter mu, this project's
                        [10, 100] reflects LION's unnormalised operator and the
                        max-normalised z-step gradient, which rescales the
                        effective mu.
    """

    def __init__(
        self,
        in_channels: int = 5,
        n_action_steps: int = 5,
        sigma_range: tuple = (1.0, 5.0),     # DRUNet-calibrated
        mu_range: tuple = (10.0, 100.0),     # LION-native µ scale
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_action_steps = n_action_steps
        self.sigma_range = tuple(sigma_range)
        self.mu_range = tuple(mu_range)

        # Backbone: ResNet-18, trained from scratch
        backbone = tv_models.resnet18(weights=None)

        # replace first conv to have in_channels
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # remove final FC layer, use the 512-dim feature vector
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Sub-policy 1: termination (discrete, stochastic)
        self.termination_head = nn.Linear(512, 2)

        # Sub-policy 2: continuous parameters (deterministic) 
        self.param_head = nn.Linear(512, n_action_steps * 2)

    def forward(self, x, z, u, noise_level, iter_frac):
        """
        Args:
            x, z, u:     ADMM variables, shape (B, 1, H, W).
            noise_level: fractional sinogram noise level, shape (B,).
            iter_frac:   fraction of the iteration budget used, shape (B,).

        Returns:
            stop_logits: shape (B, 2) — pass through softmax for probabilities.
            sigma_seq:   shape (B, n_action_steps) — denoising strengths.
            mu_seq:      shape (B, n_action_steps) — penalty parameters.
        """
        # set x shape
        B, _, H, W = x.shape

        # Broadcast scalars to spatial maps
        noise_map = noise_level.view(B, 1, 1, 1).expand(B, 1, H, W)
        iter_map = iter_frac.view(B, 1, 1, 1).expand(B, 1, H, W)

        # Concatenate inputs to form 5-channel input tensor
        inp = torch.cat([x, z, u, noise_map, iter_map], dim=1) 

        # extract features using ResNet-18 backbone
        feats = self.feature_extractor(inp).view(B, -1)  # (B, 512)

        # load termination logits
        stop_logits = self.termination_head(feats)  # (B, 2)

        # load continuous parameters
        raw_params = torch.sigmoid(self.param_head(feats))  # (B, n*2), in [0,1]
        sigma_norm = raw_params[:, : self.n_action_steps]   # (B, n)
        mu_norm = raw_params[:, self.n_action_steps :]      # (B, n)

        # Scale to valid ranges
        sigma_min, sigma_max = self.sigma_range
        mu_min, mu_max = self.mu_range
        sigma_seq = sigma_min + sigma_norm * (sigma_max - sigma_min)
        mu_seq = mu_min + mu_norm * (mu_max - mu_min)

        return stop_logits, sigma_seq, mu_seq
