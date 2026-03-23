"""Prioritized Experience Replay with SumTree."""

import numpy as np


class SumTree:
    """
    Array-based binary sum tree for O(log n) proportional sampling.

    Leaf nodes at indices [capacity, 2*capacity).
    Internal nodes store sums of children. Root at index 1.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data_pointer = 0
        self.size = 0
        self.max_priority = 1.0

    @property
    def total(self) -> float:
        return self.tree[1]

    def add(self, priority: float):
        """Add item at data_pointer with given priority."""
        tree_idx = self.data_pointer + self.capacity
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update priority at tree_idx and propagate up."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] += delta

    def sample(self, value: float) -> int:
        """Sample a leaf tree_idx proportional to priority."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx

    def data_index(self, tree_idx: int) -> int:
        """Convert tree index to data array index."""
        return tree_idx - self.capacity


class PrioritizedReplayBuffer:
    """
    PER buffer with SumTree. States stored as uint8 to save ~4x memory.
    Stores next_action_mask for proper Double DQN masking in multi-topology.
    """

    def __init__(self, capacity: int, state_shape: tuple,
                 num_actions: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_anneal_steps: int = 100000, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon

        self.tree = SumTree(capacity)

        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_action_masks = np.zeros((capacity, num_actions), dtype=np.bool_)

        self._ptr = 0
        self._size = 0

    def __len__(self):
        return self._size

    def add(self, state, action, reward, next_state, done, next_action_mask):
        """Store transition with max priority."""
        idx = self._ptr

        self.states[idx] = (state * 255.0).clip(0, 255).astype(np.uint8)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = (next_state * 255.0).clip(0, 255).astype(np.uint8)
        self.dones[idx] = done
        self.next_action_masks[idx] = next_action_mask

        priority = self.tree.max_priority ** self.alpha
        self.tree.add(priority)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, step: int):
        """
        Sample batch proportional to priorities.

        Returns:
            batch: dict of numpy arrays
            is_weights: importance-sampling weights (batch_size,)
            tree_indices: for priority updates (batch_size,)
        """
        beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start)
            * step / self.beta_anneal_steps,
        )

        tree_indices = np.zeros(batch_size, dtype=np.int64)
        data_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            tree_idx = self.tree.sample(value)
            data_idx = self.tree.data_index(tree_idx)

            # Clamp data_idx to valid range
            data_idx = max(0, min(data_idx, self._size - 1))

            tree_indices[i] = tree_idx
            data_indices[i] = data_idx
            priorities[i] = max(self.tree.tree[tree_idx], 1e-10)

        probs = priorities / max(self.tree.total, 1e-10)
        min_prob = probs.min()
        max_weight = (self._size * min_prob) ** (-beta)
        is_weights = (self._size * probs) ** (-beta) / max_weight

        batch = {
            "states": self.states[data_indices].astype(np.float32) / 255.0,
            "actions": self.actions[data_indices],
            "rewards": self.rewards[data_indices],
            "next_states": self.next_states[data_indices].astype(np.float32) / 255.0,
            "dones": self.dones[data_indices],
            "next_action_masks": self.next_action_masks[data_indices],
        }

        return batch, is_weights.astype(np.float32), tree_indices

    def update_priorities(self, tree_indices, td_errors):
        """Update priorities from TD errors."""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(tree_idx), float(priority))
            self.tree.max_priority = max(self.tree.max_priority, float(priority))
