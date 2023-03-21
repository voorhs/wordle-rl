import cpprb
import numpy as np
import torch


class PrioritizedReplayBuffer:
    is_prioritized = True

    def __init__(
        self, state_size, buffer_size=int(1e5), batch_size=64,
        alpha=0.4, beta=0.9, beta_growth_rate=1.001, update_beta_every=3000,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        self.buffer = cpprb.PrioritizedReplayBuffer(
            buffer_size, {
                "state": {"shape": state_size},
                "action": {"dtype": np.int64},
                "reward": {},
                "next_state": {"shape": state_size},
                "done": {"dtype": np.bool8},
                # "indexes" and "weights" are generated automatically
            },
            alpha=alpha
        )

        self.beta = beta
        self.beta_growth_rate = beta_growth_rate
        self.update_beta_every = update_beta_every
        self.batch_size = batch_size

        self.t = 0
        self.device = device

        if beta_growth_rate < 1:
            raise ValueError(
                f'`beta_growth_rate` must be >=1: {beta_growth_rate}')
        if not (batch_size <= buffer_size):
            raise ValueError(
                f'Invalid sizes provided: `batch_size`={batch_size}, `buffer_size`={buffer_size}')

    def add(self, **observation):
        self.buffer.add(**observation)

        self.t = (self.t + 1) % self.update_beta_every
        if self.t == 0:
            self.beta = min(1, self.beta * self.beta_growth_rate)

    def sample(self):
        batch = self.buffer.sample(self.batch_size, self.beta)

        # send to device
        batch_device = {}
        for key, value in batch.items():
            if key == 'indexes':
                continue
            batch_device[key] = torch.from_numpy(value).to(self.device)

        return batch_device

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def update_priorities(self, indexes, tds):
        self.buffer.update_priorities(indexes, tds)

    def on_episode_end(self):
        self.buffer.on_episode_end()


class ReplayBuffer:
    def __init__(
        self, state_size, buffer_size=int(1e5), batch_size=64,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        self.buffer = cpprb.ReplayBuffer(
            buffer_size, {
                "state": {"shape": state_size},
                "action": {"dtype": np.int64},
                "reward": {},
                "next_state": {"shape": state_size},
                "done": {"dtype": np.bool8},
            }
        )

        self.batch_size = batch_size
        self.device = device

        if not (batch_size <= buffer_size):
            raise ValueError(
                f'Invalid sizes provided: `batch_size`={batch_size}, `buffer_size`={buffer_size}')

    def add(self, **observation):
        self.buffer.add(**observation)

    def sample(self):
        batch = self.buffer.sample(self.batch_size)

        # send to device
        batch_device = {}
        for key, value in batch.items():
            batch_device[key] = torch.from_numpy(value).to(self.device)

        return batch_device

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def on_episode_end(self):
        self.buffer.on_episode_end()