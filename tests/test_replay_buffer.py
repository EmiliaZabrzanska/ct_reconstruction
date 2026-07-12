"""
Tests for ct_tfpnp.training.replay_buffer.
"""

import torch

from ct_tfpnp.training.replay_buffer import ReplayBuffer, Transition


def make_transition(reward=0.0, stop=False, done=False):
    # create image
    img = torch.zeros(1, 4, 4)
    # create state and action
    state = (img, img.clone(), torch.zeros_like(img), torch.tensor([0.05]), torch.tensor([0.0]))
    action = (stop, torch.zeros(1, 2), torch.ones(1, 3), torch.ones(1, 3))

    # make transition
    return Transition(state, action, torch.tensor([reward]), state, torch.zeros(1, 8, 12), img.clone(), done)


class TestTransition:

    # check grund truth is stored
    def test_gt_is_stored(self):

        # make transition
        t = make_transition()

        # check gt exists
        assert t.gt is not None


class TestReplayBuffer:

    # check buffer grows
    def test_len_grows_with_pushes(self):

        # define buffer
        buf = ReplayBuffer(capacity=10)

        # check length is 0
        assert len(buf) == 0

        # check length grows
        for i in range(4):
            buf.push(*make_transition(reward=i))
        assert len(buf) == 4
