"""
TFPnP solver — inherits from LIONsolver.

Implements Algorithm 1 of Wei et al. (2022) using LION's training
infrastructure for checkpointing, validation, and device management.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pathlib
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter
from LION.CTtools.ct_geometry import Geometry

from ct_tfpnp.models.tfpnp_model import TFPnPModel
from ct_tfpnp.models.denoiser import DRUNetDenoiser
from ct_tfpnp.ct_ops.admm import ADMMStep
from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp
from ct_tfpnp.evaluation.metrics import psnr_np, ls_scale
from ct_tfpnp.training.replay_buffer import ReplayBuffer
from ct_tfpnp.training.losses import (
    compute_critic_loss,
    compute_policy_loss_discrete,
    compute_policy_loss_continuous,
)
from ct_tfpnp.training.environment import collect_episode


class TFPnPSolverParams(LIONParameter):
    """Parameters for TFPnP training."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'm'):                  self.m = 5
        if not hasattr(self, 'N'):                  self.N = 6
        if not hasattr(self, 'eta'):                self.eta = 0.05
        if not hasattr(self, 'gamma'):              self.gamma = 0.99
        if not hasattr(self, 'target_ema'):         self.target_ema = 1e-3
        if not hasattr(self, 'n_grad_steps'):       self.n_grad_steps = 5
        if not hasattr(self, 'batch_size'):         self.batch_size = 8
        if not hasattr(self, 'pi2_batch_size'):     self.pi2_batch_size = 2
        if not hasattr(self, 'buffer_size'):        self.buffer_size = 5000
        if not hasattr(self, 'noise_std'):          self.noise_std = None
        if not hasattr(self, 'lr_critic'):          self.lr_critic = 1e-4
        if not hasattr(self, 'pi2_loss_scale'):     self.pi2_loss_scale = 0.01
        if not hasattr(self, 'lr_pi2'):             self.lr_pi2 = 1e-6
        if not hasattr(self, 'pi2_warmup_epochs'):  self.pi2_warmup_epochs = 5


class TFPnPSolver(LIONsolver):
    """
    TFPnP solver: RL-based adaptive PnP-ADMM parameter selection.

    Inherits from LIONsolver to use LION's checkpointing, validation,
    and training loop infrastructure. Overrides train_step() and
    mini_batch_step() for the RL training procedure.
    """

    def __init__(
        self,
        model: TFPnPModel,
        optimizer,
        loss_fn=None,
        geometry: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
        solver_params: TFPnPSolverParams = None,
        save_folder: pathlib.Path = None,
        denoiser_path: str = None,
    ):
        if solver_params is None:
            solver_params = self.default_parameters()

        if loss_fn is None:
            loss_fn = lambda x, y: torch.tensor(0.0)

        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            geometry=geometry,
            verbose=verbose,
            device=device,
            solver_params=solver_params,
            save_folder=save_folder,
        )

        sp = self.solver_params

        # ── Denoiser (frozen) ─────────────────────────────────────────
        assert denoiser_path is not None, "Must provide denoiser_path"
        self.denoiser = DRUNetDenoiser(pretrained_path=denoiser_path).to(self.device)
        for p in self.denoiser.parameters():
            p.requires_grad_(False)
        self.denoiser.eval()

        # ── ADMM step ─────────────────────────────────────────────────
        self.admm_step = ADMMStep(op=self.op, denoiser=self.denoiser, n_x_steps=6)

        # ── Critic optimizer (separate from policy optimizer) ─────────
        self.critic_optim = torch.optim.Adam(
            self.model.critic.parameters(), lr=sp.lr_critic)

        # ── Replay buffer ─────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(capacity=sp.buffer_size)

        # ── Validation ────────────────────────────────────────────────
        self.val_images = None
        self.best_val_psnr = -float('inf')

        # ── Training log ──────────────────────────────────────────────
        self.train_log = defaultdict(list)

    @staticmethod
    def default_parameters() -> TFPnPSolverParams:
        return TFPnPSolverParams()

    def set_validation(self, val_images):
        """Set validation images for checkpoint selection."""
        self.val_images = val_images
        print(f"Validation set: {len(val_images)} images")

    def train_step(self):
        """
        Override LIONsolver.train_step() for RL training.

        One epoch = iterate over all training images:
          1. Collect episode (run policy through ADMM environment)
          2. Update networks from replay buffer (critic + π₁ + π₂)
          3. Validate on held-out images and save best checkpoint
        """
        if self.train_loader is None:
            raise ValueError("Training dataloader not set")

        self.model.train()
        sp = self.solver_params
        epoch_psnrs = []

        for _, (sino_batch, gt_batch) in enumerate(tqdm(self.train_loader)):
            gt_batch = gt_batch.to(self.device)

            for b in range(gt_batch.shape[0]):
                gt = gt_batch[b]  # (1, H, W)

                ep_stats = collect_episode(
                    self.model.policy, self.admm_step, self.replay_buffer,
                    gt, self.op,
                    noise_std=sp.noise_std, m=sp.m, N=sp.N, eta=sp.eta,
                )
                epoch_psnrs.append(ep_stats['final_psnr'])

            if len(self.replay_buffer) >= sp.batch_size:
                enable_pi2 = self.current_epoch >= sp.pi2_warmup_epochs
                losses = self._update_networks(enable_pi2=enable_pi2)
                for k, v in losses.items():
                    self.train_log[k].append(v)

        mean_psnr = np.mean(epoch_psnrs) if epoch_psnrs else 0.0
        self.train_log['epoch_psnr'].append(mean_psnr)

        # ── Validation ────────────────────────────────────────────────
        if self.val_images is not None:
            val_psnr = self.validate(self.val_images, n_images=30)
            self.train_log['val_psnr'].append(val_psnr)

            if val_psnr > self.best_val_psnr:
                self.best_val_psnr = val_psnr
                torch.save({
                    'epoch': len(self.train_log['epoch_psnr']),
                    'model_state_dict': self.model.state_dict(),
                    'val_psnr': val_psnr,
                    'train_psnr': mean_psnr,
                }, self.save_folder / "checkpoint_best_val.pth")
                best_marker = " ★"
            else:
                best_marker = ""

            if self.verbose:
                print(f"  Train PSNR: {mean_psnr:.2f} dB | "
                      f"Val PSNR: {val_psnr:.2f} dB | "
                      f"Buffer: {len(self.replay_buffer)}{best_marker}")
        else:
            if self.verbose:
                print(f"  Mean PSNR: {mean_psnr:.2f} dB | "
                      f"Buffer: {len(self.replay_buffer)}")

        return mean_psnr

    def validate(self, val_images, n_images=30):
        """
        Run policy on validation images (no gradients, no buffer updates).
        Returns mean PSNR across validation set.
        """
        self.model.eval()
        sp = self.solver_params
        val_psnrs = []

        n = min(n_images, len(val_images))

        for idx in range(n):
            gt = val_images[idx].to(self.device)

            # Generate noisy sinogram (fixed 5% noise for consistent validation)
            sino_clean = self.op.forward(gt)
            SCALE = sino_clean.max() / gt.max()
            noise_std = 0.05
            sino_scaled = sino_clean / SCALE
            noise = noise_std * sino_scaled.std() * torch.randn_like(sino_scaled)
            y = (sino_scaled + noise) * SCALE

            # FBP init
            x = lion_fbp(y, self.op)
            x = ls_scale(gt, x).clamp(min=0)
            z = x.clone()
            u = torch.zeros_like(x)

            noise_level = torch.tensor([noise_std], device=self.device)

            with torch.no_grad():
                for t in range(sp.N):
                    iter_frac = torch.tensor([t / sp.N], device=self.device)
                    stop_logits, sigma_seq, mu_seq = self.model.policy(
                        x.unsqueeze(0), z.unsqueeze(0), u.unsqueeze(0),
                        noise_level, iter_frac)

                    stop_prob = torch.softmax(stop_logits, dim=-1)[0, 1].item()
                    if t > 0 and stop_prob > 0.5:
                        break

                    for i in range(sp.m):
                        x, z, u = self.admm_step(x, z, u, y,
                                                  sigma=sigma_seq[0, i],
                                                  mu=mu_seq[0, i])

            x_scaled = ls_scale(gt, x).clamp(min=0)
            val_psnrs.append(psnr_np(gt, x_scaled))

        self.model.train()
        return float(np.mean(val_psnrs))

    def _update_networks(self, enable_pi2=True):
        """
        Algorithm 1 lines 12–15: update critic, π₁, π₂, and target critic.
        """
        sp = self.solver_params
        all_info = defaultdict(list)

        for step in range(sp.n_grad_steps):
            if len(self.replay_buffer) < sp.batch_size:
                break
            batch = self.replay_buffer.sample(sp.batch_size)

            # ── Critic (eq. 15) ───────────────────────────────────────
            self.critic_optim.zero_grad()
            c_loss, c_info = compute_critic_loss(
                self.model.critic, self.model.target_critic, batch, sp.gamma)
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 1.0)
            self.critic_optim.step()
            for k, v in c_info.items():
                all_info[k].append(v)

            # ── π₁: REINFORCE (eq. 16) ────────────────────────────────
            self.optimizer.zero_grad()
            p1_loss, p1_info = compute_policy_loss_discrete(
                self.model.policy, self.model.critic, batch, sp.gamma)
            p1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 1.0)
            self.optimizer.step()
            for k, v in p1_info.items():
                all_info[k].append(v)

            # ── π₂: model-based DDPG (eq. 17) — every other step ─────
            if enable_pi2 and step % 2 == 0:
                torch.cuda.empty_cache()
                pi2_batch = self.replay_buffer.sample(
                    min(sp.pi2_batch_size, len(self.replay_buffer)))
                self.optimizer.zero_grad()
                p2_loss, p2_info = compute_policy_loss_continuous(
                    self.model.policy, self.model.critic, self.admm_step,
                    pi2_batch, m=sp.m, N=sp.N)
                (p2_loss * sp.pi2_loss_scale).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.policy.param_head.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(
                    self.model.policy.feature_extractor.parameters(), 0.1)
                self.optimizer.step()
                for k, v in p2_info.items():
                    all_info[k].append(v)
                torch.cuda.empty_cache()

            # ── EMA target critic (line 15) ───────────────────────────
            with torch.no_grad():
                for p, pt in zip(self.model.critic.parameters(),
                                  self.model.target_critic.parameters()):
                    pt.data.mul_(1 - sp.target_ema).add_(sp.target_ema * p.data)

        return {k: float(np.nanmean(v)) for k, v in all_info.items()}

    def mini_batch_step(self, sino_batch, target_batch):
        """Required by LIONsolver ABC but not used."""
        return torch.tensor(0.0, device=self.device)