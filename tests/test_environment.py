"""
Tests for ct_tfpnp.training.environment.
"""

import pytest
import torch

from ct_tfpnp.training.environment import (
    _measure, _step_reward, admm_init, collect_episode, run_episode,
)

from ct_tfpnp.training.replay_buffer import ReplayBuffer

# define policy that always stops
class AlwaysStop(torch.nn.Module):

    def __init__(self, m=3):
        super().__init__()
        self.m = m

    def forward(self, x, z, u, noise_level, iter_frac):
        B = x.shape[0]
        return (torch.tensor([[-20.0, 20.0]]).repeat(B, 1),
                torch.full((B, self.m), 2.0),
                torch.full((B, self.m), 20.0))

# define policy that never stops
class NeverStop(AlwaysStop):

    def forward(self, x, z, u, noise_level, iter_frac):
        logits, sigma, mu = super().forward(x, z, u, noise_level, iter_frac)
        return logits.flip(-1), sigma, mu


class TestAdmmInit:
   
    # check initial x and z are equal
    def test_z_starts_equal_to_x(self, gt, op):
        y = op.forward(gt)
        x, z, _ = admm_init(y, op)
        assert torch.allclose(x, z)

    # check initial x is non-negative
    def test_is_non_negative(self, gt, op):
        x, _, _ = admm_init(op.forward(gt), op)
        assert x.min() >= 0

class TestStepReward:

    # chekc reward uses correct formula
    def test_psnr_reward_is_delta_minus_eta(self):
        prev = {"psnr": 20.0}
        curr = {"psnr": 22.0}
        assert _step_reward(prev, curr, "psnr", alpha=0.0, eta=0.05) == pytest.approx(1.95)


class TestRunEpisode:

    # check all keys are returned
    def test_returns_the_documented_keys(self, policy, admm_step, gt, op):

        # run episode
        ep = run_episode(policy, admm_step, gt, op, noise_std=0.05, m=3, N=4)

        # check keys exist
        for key in ("x", "x0", "y", "sigmas", "mus", "rewards", "final_psnr", "n_decisions", "n_admm_steps", "noise_std"):
            assert key in ep

    # check stop happens after block runs
    def test_stop_action_terminates_after_the_block_runs(self, admm_step, gt, op):

        # run stopping episode
        ep = run_episode(AlwaysStop(m=3), admm_step, gt, op, noise_std=0.05, m=3, N=6, stochastic=False)
        
        # check if stopped
        assert ep["n_decisions"] == 1
        assert ep["n_admm_steps"] == 3

    # check noise level is mixed
    def test_noise_level_is_sampled_when_unspecified(self, policy, admm_step, gt, op):

        # set nosie 
        levels = (0.05, 0.075, 0.10)

        # run episode
        seen = {run_episode(policy, admm_step, gt, op, m=2, N=2, noise_levels=levels)["noise_std"] for _ in range(20)}

        # check noise 
        assert seen.issubset(set(levels))

    @pytest.mark.parametrize("reward_type", ["psnr", "psnr_ssim", "ssim"])

    # check all reward types run
    def test_all_reward_types_run(self, policy, admm_step, gt, op, reward_type):

        # run episode
        ep = run_episode(policy, admm_step, gt, op, noise_std=0.05, m=2, N=2, reward_type=reward_type, reward_alpha=5.0)
        assert len(ep["rewards"]) >= 1
