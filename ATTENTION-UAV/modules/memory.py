# -*- coding: utf-8 -*-
import numpy as np


class Memory:
    """均匀采样经验回放缓冲区"""
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.counter = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.counter % self.capacity
        self.data[index, :] = transition
        self.counter += 1

    def sample(self, batch_size):
        sample_index = np.random.choice(
            min(self.counter, self.capacity), size=batch_size
        )
        batch = self.data[sample_index, :]
        return batch

    def __len__(self):
        return min(self.counter, self.capacity)

    @property
    def is_ready(self):
        return self.counter >= self.capacity
