"""
Policy network for TFPnP.

The policy network takes the current ADMM state (x, z, u) plus auxiliary
scalar information (noise level, iteration count) and outputs:
  - a1: termination probability  (stop_prob, scalar)
  - a2: continuous parameters    (sigma[5], mu[5]) for the next m=5 ADMM steps

Architecture (from Wei et al. Table 1):
  - ResNet-18 feature extractor (shared between sub-policies)
  - Two separate FC heads:
      * Head 1 → 2-class softmax  (continue / stop)
      * Head 2 → 10-dim sigmoid   (5x sigma, 5x mu), scaled to valid ranges

Input channels:
  - x  (1 channel)
  - z  (1 channel)
  - u  (1 channel)
  - noise_level broadcast to HxW  (1 channel)
  - iteration_frac broadcast to HxW  (1 channel)
  Total: 5 channels
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNetActor_ADMM(nn.Module):
    """
    Policy network: maps ADMM state → (termination_prob, sigma_seq, mu_seq).

    Args:
        in_channels: number of input channels (default 5: x, z, u, noise_lvl, iter_frac)
        n_action_steps: number of ADMM steps per decision (m in the paper, default 5)
        sigma_range: (min, max) for denoising strength sigma
        mu_range: (min, max) for penalty parameter mu
    """

    def __init__(
        self,
        in_channels: int = 5,
        n_action_steps: int = 5,
        sigma_range: tuple = (1.0, 5.0),     # DRUNet-calibrated (was paper's [1, 50])
        mu_range: tuple = (10.0, 100.0),     # LION-native µ scale (was paper's [0.01, 1.0])
    ):
        super().__init__()
        self.n_action_steps = n_action_steps
        self.sigma_range = sigma_range         
        self.mu_range = mu_range              
        self.sigma_min, self.sigma_max = sigma_range
        self.mu_min, self.mu_max = mu_range

        # --- Backbone: ResNet-18 modified per Wei et al. (2022) Table 1 ---
        backbone = tv_models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        # Remove the final FC layer; use the 512-dim feature vector
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # --- Sub-policy 1: termination (discrete, stochastic) ---
        # Outputs 2 logits → softmax → P(stop), P(continue)
        self.termination_head = nn.Linear(512, 2)

        # --- Sub-policy 2: continuous parameters (deterministic) ---
        # Outputs n_action_steps * 2 values (sigma and mu per step)
        self.param_head = nn.Linear(512, n_action_steps * 2)

    def forward(self, x, z, u, noise_level, iter_frac):
        """
        Args:
            x, z, u:     ADMM variables, shape (B, 1, H, W)
            noise_level: scalar noise level per image, shape (B,)
            iter_frac:   fraction of max iterations elapsed, shape (B,)

        Returns:
            stop_logits: shape (B, 2) — pass through softmax for probabilities
            sigma_seq:   shape (B, n_action_steps) — denoising strengths
            mu_seq:      shape (B, n_action_steps) — penalty parameters
        """
        B, _, H, W = x.shape

        # Broadcast scalars to spatial maps
        noise_map = noise_level.view(B, 1, 1, 1).expand(B, 1, H, W)
        iter_map = iter_frac.view(B, 1, 1, 1).expand(B, 1, H, W)

        inp = torch.cat([x, z, u, noise_map, iter_map], dim=1)  # (B, 5, H, W)

        feats = self.feature_extractor(inp)   # (B, 512, 1, 1)
        feats = feats.view(B, -1)             # (B, 512)

        stop_logits = self.termination_head(feats)  # (B, 2)

        raw_params = torch.sigmoid(self.param_head(feats))  # (B, n*2), in [0,1]
        sigma_norm = raw_params[:, : self.n_action_steps]   # (B, n)
        mu_norm = raw_params[:, self.n_action_steps :]      # (B, n)

        # Scale to valid ranges
        sigma_seq = self.sigma_min + sigma_norm * (self.sigma_max - self.sigma_min)
        mu_seq = self.mu_min + mu_norm * (self.mu_max - self.mu_min)

        return stop_logits, sigma_seq, mu_seq