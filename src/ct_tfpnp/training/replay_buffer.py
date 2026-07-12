"""
Replay buffer for TFPnP training.
"""

import random

from collections import namedtuple

# define a named tuple to store transitions in the replay buffer
Transition = namedtuple("Transition",("state", "action", "reward", "next_state", "sinogram", "gt", "done"),)

class ReplayBuffer:
    """
    Simple circular replay buffer storing ADMM transitions.
    """

    def __init__(self, capacity=10000):
        """
        Args:
            capacity: maximum number of transitions retained.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Insert one Transition, overwriting the oldest entry when full.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Draw `batch_size` transitions uniformly without replacement.
        """
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        """Number of transitions currently stored."""
        return len(self.memory)