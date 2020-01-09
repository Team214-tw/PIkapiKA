import random
import numpy as np
from .SumTree import SumTree
import torch as T
import time
import os


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.001
    a = 0.7
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def save(self, filepath):
        size = self.tree.n_entries
        memory = {
            "p": self.tree.priority[:size],
            "data": self.tree.data[:size],
        }
        T.save(memory, filepath)

    def load(self, filepath):
        if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
            try:
                memory = T.load(filepath)
                for i in range(len(memory['data'])):
                    self.tree.add(memory['p'][i], memory['data'][i])
                print('memory loaded from', filepath)
                os.remove(filepath)
            except:
                pass
                # print('memory loaded from', filepath, 'failed')
        else:
            pass
            # print('no memory found at', filepath)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            while True:
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                if not isinstance(data, int):
                    break
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries *
                             sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def clear(self):
        self.tree.write = 0
        self.tree.tree.fill(0)
        self.tree.data.fill(0)
        self.tree.priority.fill(0)
        self.tree.n_entries = 0
