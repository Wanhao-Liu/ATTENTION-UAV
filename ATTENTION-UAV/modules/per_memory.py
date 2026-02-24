# -*- coding: utf-8 -*-
import numpy as np


class SumTree:
    """二叉 Sum Tree，用于 O(log n) 的优先级采样"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """基于 SumTree 的优先经验回放"""
    def __init__(self, capacity, dims, alpha=0.6, beta_start=0.4,
                 beta_end=1.0, beta_steps=100000, eps=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.dims = dims
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.eps = eps
        self.max_priority = 1.0
        self.counter = 0

    def _get_beta(self):
        fraction = min(self.counter / max(self.beta_steps, 1), 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
        self.counter += 1

    def sample(self, batch_size):
        batch = np.zeros((batch_size, self.dims))
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size)
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            idx, p, data = self.tree.get(s)
            tree_indices[i] = idx
            priorities[i] = p
            batch[i] = data

        # 重要性采样权重
        beta = self._get_beta()
        n = self.tree.n_entries
        min_prob = np.min(priorities) / self.tree.total
        min_prob = max(min_prob, 1e-8)
        max_weight = (n * min_prob) ** (-beta)

        probs = priorities / self.tree.total
        probs = np.clip(probs, 1e-8, None)
        is_weights = (n * probs) ** (-beta)
        is_weights = is_weights / max_weight

        return batch, tree_indices, is_weights.astype(np.float32)

    def update_priorities(self, tree_indices, td_errors):
        td_errors = np.abs(td_errors) + self.eps
        clipped = np.minimum(td_errors, 100.0)
        priorities = clipped ** self.alpha
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, float(p))
            self.max_priority = max(self.max_priority, float(p))

    def __len__(self):
        return self.tree.n_entries

    @property
    def is_ready(self):
        return self.tree.n_entries >= 1000
