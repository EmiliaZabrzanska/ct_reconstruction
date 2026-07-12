"""
Tests for ct_tfpnp.training.losses.
"""

import pytest
import torch

from ct_tfpnp.training.losses import (
    _batch_states, compute_critic_loss, compute_policy_loss_continuous,
    compute_policy_loss_discrete,
)

from ct_tfpnp.training.replay_buffer import Transition

# define transition function
def make_transition(gt, y, reward=1.0, stop=False, done=False, iter_frac=0.0, m=3):

    # compute image
    img = gt.clone()

    # compute state and action
    state = (img, img.clone(), torch.zeros_like(img), torch.tensor([0.05]), torch.tensor([iter_frac]))
    action = (stop, torch.zeros(1, 2), torch.full((1, m), 2.0), torch.full((1, m), 20.0))

    # return transition
    return Transition(state, action, torch.tensor([reward]), state, y.cpu(), gt.cpu(), done)


@pytest.fixture

# define batch of random transitions
def batch(gt, op):
    y = op.forward(gt)
    return [
        make_transition(gt, y, reward=1.0, stop=False, done=False, iter_frac=0.0),
        make_transition(gt, y, reward=2.0, stop=True, done=True, iter_frac=0.33),
        make_transition(gt, y, reward=0.5, stop=False, done=False, iter_frac=0.66),
        make_transition(gt, y, reward=-0.2, stop=True, done=True, iter_frac=0.66),
    ]


class TestBatchStates:

    # check if stacking works
    def test_stacks_images_and_scalars_correctly(self, batch):

        # batch states
        x, z, u, noise, iters = _batch_states([t.state for t in batch], "cpu")

        # find batch size and check shapes
        B = len(batch)
        assert x.dim() == 4 and x.shape[0] == B
        assert z.shape == x.shape and u.shape == x.shape
        assert noise.shape == (B,)
        assert iters.shape == (B,)



class TestCriticLoss:

    # check differentiability
    def test_gradients_reach_the_critic(self, critic, batch):
        import copy

        # compute loss and backprop
        loss, _ = compute_critic_loss(critic, copy.deepcopy(critic), batch)
        loss.backward()

        # check criic gradients
        assert critic.head.weight.grad.abs().sum() > 0


class TestPolicyLossDiscrete:

    # check loss uses action
    def test_uses_the_sampled_action(self, policy, critic, gt, op):
        
        # define forward pass
        y = op.forward(gt)

        # define batches
        took_stop = [make_transition(gt, y, reward=1.0, stop=True)]
        took_continue = [make_transition(gt, y, reward=1.0, stop=False)]

        # compute losses
        loss_stop, info_stop = compute_policy_loss_discrete(policy, critic, took_stop)
        loss_cont, info_cont = compute_policy_loss_discrete(policy, critic, took_continue)

        # check losses differ
        assert loss_stop.item() != pytest.approx(loss_cont.item(), abs=1e-6)
        assert info_stop["stop_rate"] == 1.0
        assert info_cont["stop_rate"] == 0.0


class TestPolicyLossContinuous:

    # check if reward used chanes loss
    def test_reward_term_changes_the_objective(self, policy, critic, admm_step, batch):
        
        # compute losses with and without reward term
        with_r, info = compute_policy_loss_continuous(policy, critic, admm_step, batch, m=3, N=6, include_reward=True)
        without_r, _ = compute_policy_loss_continuous(policy, critic, admm_step, batch, m=3, N=6, include_reward=False)

        # check losses differ
        assert with_r.item() != pytest.approx(without_r.item(), abs=1e-6)
        assert "pi2_mean_reward" in info