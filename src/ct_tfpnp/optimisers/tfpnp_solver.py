"""
TFPnP solver — inherits from LIONsolver.

Implements Algorithm 1 of Wei et al. (2022) using LION's training
infrastructure for checkpointing, validation, and device management.
"""

from logging import info

import torch
import torch.nn.functional as F
import numpy as np
import pathlib
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import json

from LION.optimizers.LIONsolver import LIONsolver
from LION.utils.parameter import LIONParameter
from LION.CTtools.ct_geometry import Geometry

from ct_tfpnp.models.tfpnp_model import TFPnPModel
from ct_tfpnp.models.denoiser import DRUNetDenoiser
from ct_tfpnp.ct_ops.admm import ADMMStep
from ct_tfpnp.training.losses import (
    compute_critic_loss,
    compute_policy_loss_discrete,
    compute_policy_loss_continuous,
)
from ct_tfpnp.training.environment import run_episode
from ct_tfpnp.training.replay_buffer import ReplayBuffer

# set default parameters for TFPnP solver
_DEFAULTS = {
    "m": 5,                     # ADMM steps per policy decision
    "N": 6,                     # maximum decisions per episode
    "eta": 0.05,                # per-step cost in the reward
    "gamma": 0.99,              # discount factor
    "target_ema": 1e-3,         # EMA rate for the target critic
    "n_grad_steps": 5,          # gradient steps per collected batch
    "batch_size": 8,            # replay batch for the critic and pi_1
    "pi2_batch_size": 2,        # replay batch for pi_2 
    "buffer_size": 5000,
    "noise_std": None,          # None = sample a noise level per episode
    "lr_critic": 1e-4,
    "pi2_loss_scale": 0.01,
    "lr_pi2": 1e-6,           
    "pi2_warmup_epochs": 5,     # epochs of critic/pi_1-only training first
    "pi2_include_reward": True, # include r(s,a) in eq. 17's Q, as the paper does
    "reward_type": "psnr",
    "reward_alpha": 0.0,
    "val_noise_std": 0.05,      # fixed noise level for validation
    "val_seed": 12345,          # fixed noise realisation
    "n_val_images": 30,
}

class TFPnPSolverParams(LIONParameter):
    """
    Parameters for TFPnP training.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # create attributes for parameters from _DEFAULTS§
        for key, value in _DEFAULTS.items():
            if not hasattr(self, key):
                setattr(self, key, value)

class TFPnPSolver(LIONsolver):
    """
    TFPnP solver: RL-based adaptive parameter selection for PnP-ADMM.

    Overrides train_step() with the RL procedure

    Args:
        model:         TFPnPModel (policy, critic and target critic).
        optimizer:     optimiser over the POLICY parameters only.
        loss_fn:       unused.
        geometry:      LION Geometry.
        verbose:       print per-epoch summaries.
        device:        torch device.
        solver_params: TFPnPSolverParams.
        save_folder:   directory for checkpoints and metrics_history.json.
        denoiser_path: path to drunet_gray.pth.
        admm_kwargs:   forwarded to ADMMStep.
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
        admm_kwargs: dict = None,
    ):
        if solver_params is None:
            solver_params = self.default_parameters()

        if loss_fn is None:
            def loss_fn(x, y):
                return torch.tensor(0.0)

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

        # ensure optimizer only contains policy parameters (critic has its own optimiser)
        critic_ids = {id(p) for p in self.model.critic.parameters()}
        for group in self.optimizer.param_groups:
            if any(id(p) in critic_ids for p in group["params"]):
                raise ValueError(
                    "`optimizer` must contain only policy parameters")
            
        # ensure batch sizes are > 1 for BatchNorm in the policy
        if sp.batch_size < 2 or sp.pi2_batch_size < 2:
            raise ValueError(
                f"batch_size ({sp.batch_size}) and pi2_batch_size "
                f"({sp.pi2_batch_size}) must both be > 1")

        # Frozen denoiser: the PnP prior is fixed, only the policy is learned
        if denoiser_path is None:
            raise ValueError("denoiser_path is required")
        self.denoiser = DRUNetDenoiser(pretrained_path=denoiser_path).to(self.device)
        for p in self.denoiser.parameters():
            p.requires_grad_(False)
        self.denoiser.eval()

        # ADMM step
        self.admm_step = ADMMStep(op=self.op, denoiser=self.denoiser, n_x_steps=6)

        # Critic optimizer (separate from policy optimizer)
        self.critic_optim = torch.optim.Adam(self.model.critic.parameters(), lr=sp.lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=sp.buffer_size)

        # Validation
        self.val_images = None
        self.best_val_psnr = -float('inf')

        # Training log
        self.train_log = defaultdict(list)

    @staticmethod
    def default_parameters() -> TFPnPSolverParams:
        """
        Return the default solver parameters.
        """
        return TFPnPSolverParams()

    def set_validation(self, val_images):
        """
        Register the held-out images used for checkpoint selection.
        """
        self.val_images = val_images
        if self.verbose:
            print(f"Validation set: {len(val_images)} images")

    def train_step(self):
        """
        One epoch of RL training (overrides LIONsolver.train_step).

        For every training image: collect an episode into the replay buffer,
        take `n_grad_steps` updates of the critic, pi_1 and (after warm-up) pi_2.

        Ends with validation and best-checkpoint selection.

        Returns:
            Mean training PSNR over the epoch.
        """
        # Check that the training dataloader is set
        if self.train_loader is None:
            raise ValueError("Training dataloader not set")

        # Set model to training mode
        self.model.train()
        sp = self.solver_params
        epoch_psnrs = []

        # set gt images to the device
        for _, gt_batch in tqdm(self.train_loader, disable=not self.verbose):
            gt_batch = gt_batch.to(self.device)

            for b in range(gt_batch.shape[0]):

                # Collect one episode into the replay buffer
                stats = run_episode(
                    self.model.policy, self.admm_step, gt_batch[b], self.op,
                    replay_buffer=self.replay_buffer,
                    noise_std=sp.noise_std, m=sp.m, N=sp.N, eta=sp.eta,
                    reward_type=sp.reward_type, reward_alpha=sp.reward_alpha,
                    stochastic=True,
                )
                epoch_psnrs.append(stats["final_psnr"])

            # Update networks after every batch of episodes 
            if len(self.replay_buffer) >= sp.batch_size:
                enable_pi2 = self.current_epoch >= sp.pi2_warmup_epochs
                for k, v in self._update_networks(enable_pi2=enable_pi2).items():
                    self.train_log[k].append(v)

        # Compute mean PSNR over the epoch
        mean_psnr = float(np.mean(epoch_psnrs)) if epoch_psnrs else 0.0
        self.train_log["epoch_psnr"].append(mean_psnr)

        # validation
        if self.val_images is not None:
            val_psnr = self.validate(self.val_images, n_images=sp.n_val_images)
            self.train_log["val_psnr"].append(val_psnr)

            # checkpoint selection
            is_best = val_psnr > self.best_val_psnr
            if is_best:
                self.best_val_psnr = val_psnr
                self._save_checkpoint(val_psnr, mean_psnr)

            if self.verbose:
                print(f"  Train PSNR: {mean_psnr:.2f} dB | Val PSNR: {val_psnr:.2f} dB "
                      f"| Buffer: {len(self.replay_buffer)}{' *' if is_best else ''}")
        elif self.verbose:
            print(f"  Mean PSNR: {mean_psnr:.2f} dB | Buffer: {len(self.replay_buffer)}")

        self._save_metrics_history()
        return mean_psnr

    def validate(self, val_images, n_images=30):
        """
        Run policy on validation images (no gradients, no buffer updates).
        
        Args:
            val_images: list of ground-truth tensors, shape (1, H, W).
            n_images:   how many to evaluate.

        Returns:
            Mean PSNR over the validation subset.
        """
        # set model to eval mode
        self.model.eval()
        sp = self.solver_params
        psnrs = []

        for idx in range(min(n_images, len(val_images))):

            # Run one episode with the current policy
            stats = run_episode(
                self.model.policy, self.admm_step,
                val_images[idx].to(self.device), self.op,
                replay_buffer=None,
                noise_std=sp.val_noise_std, m=sp.m, N=sp.N, eta=sp.eta,
                reward_type=sp.reward_type, reward_alpha=sp.reward_alpha,
                stochastic=False, seed=sp.val_seed + idx,
            )
            psnrs.append(stats["final_psnr"])

        self.model.train()
        return float(np.mean(psnrs))
    
    def _update_networks(self, enable_pi2=True):
        """
        Algorithm 1 lines 12–15: update critic, π₁, π₂, and target critic.

        Args:
            enable_pi2: whether the (expensive) model-based pi_2 update runs.

        Returns:
            Dict of mean loss/diagnostic values over the gradient steps.
        """
        sp = self.solver_params
        all_info = defaultdict(list)

        for step in range(sp.n_grad_steps):
            if len(self.replay_buffer) < sp.batch_size:
                break
            batch = self.replay_buffer.sample(sp.batch_size)

            # Critic — TD regression (eq. 15)
            self.critic_optim.zero_grad(set_to_none=True)
            c_loss, c_info = compute_critic_loss(self.model.critic, self.model.target_critic, batch, sp.gamma)
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 1.0)
            self.critic_optim.step()
            for k, v in c_info.items():
                all_info[k].append(v)

            # pi_1 — REINFORCE with a value baseline (eq. 16)
            self.optimizer.zero_grad(set_to_none=True)
            p1_loss, p1_info = compute_policy_loss_discrete(self.model.policy, self.model.critic, batch, sp.gamma)
            p1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 1.0)
            self.optimizer.step()
            for k, v in p1_info.items():
                all_info[k].append(v)

            # pi_2 — model-based DDPG (eq. 17). Every other step
            if enable_pi2 and step % 2 == 0:

                pi2_batch = self.replay_buffer.sample(sp.pi2_batch_size)
                self.optimizer.zero_grad(set_to_none=True)

                p2_loss, p2_info = compute_policy_loss_continuous(
                    self.model.policy, self.model.critic, self.admm_step,
                    pi2_batch, m=sp.m, N=sp.N, gamma=sp.gamma, eta=sp.eta,
                    include_reward=sp.pi2_include_reward)
                
                (p2_loss * sp.pi2_loss_scale).backward()

                torch.nn.utils.clip_grad_norm_(self.model.policy.param_head.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(self.model.policy.feature_extractor.parameters(), 0.1)
                
                self.optimizer.step()

                # The critic accumulated grads on the way through - clean to avoid issues
                self.critic_optim.zero_grad(set_to_none=True)
                for k, v in p2_info.items():
                    info[k].append(v)

            # Target critic — EMA update (line 15)
            with torch.no_grad():
                for p, pt in zip(self.model.critic.parameters(), self.model.target_critic.parameters()):
                    pt.data.mul_(1 - sp.target_ema).add_(sp.target_ema * p.data)

        return {k: float(np.nanmean(v)) for k, v in all_info.items()}


    def _save_checkpoint(self, val_psnr, train_psnr):
        """
        Save the best-so-far checkpoint, including optimiser state so runs can resume.
        """
        torch.save({
            "epoch": len(self.train_log["epoch_psnr"]),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "val_psnr": val_psnr,
            "train_psnr": train_psnr,
        }, self.save_folder / "checkpoint_best_val.pth")

    
    def _save_metrics_history(self):
        """
        Dump train_log plus the run config to metrics_history.json.

        Called every epoch so a killed job still leaves usable data behind.
        """
        sp = self.solver_params
        metrics = {k: list(v) for k, v in self.train_log.items()}

        n_epochs = len(self.train_log.get("epoch_psnr", []))
        if n_epochs:
            metrics["epoch"] = list(range(1, n_epochs + 1))

        config = {k: getattr(sp, k) for k in _DEFAULTS}
        config.update({
            "best_val_psnr": float(self.best_val_psnr),
            "sigma_range": list(self.model.policy.sigma_range),
            "mu_range": list(self.model.policy.mu_range),
            "denoiser_sigma_scale": self.denoiser.sigma_scale,
        })
        metrics["config"] = config

        with open(self.save_folder / "metrics_history.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def mini_batch_step(self, sino_batch, target_batch):
        """Required by LIONsolver ABC but not used."""
        return torch.tensor(0.0, device=self.device)