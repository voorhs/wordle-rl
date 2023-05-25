import cpprb
import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(
        self, state_size, buffer_size=int(1e5), batch_size=64,
        alpha=0.4, beta=1, beta_growth_rate=1, update_beta_every=3000,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        n_step=1, gamma=1
    ):
        buffer_fields = {
            "state": {"shape": state_size},
            "action": {"dtype": np.int64},
            "reward": {},
            "next_state": {"shape": state_size},
            "done": {"dtype": np.bool8},
            # "indexes" and "weights" are generated automatically
        }

        self.n_step = n_step
        if n_step > 1:
            self.buffer = cpprb.PrioritizedReplayBuffer(
                buffer_size, buffer_fields, alpha=alpha,
                Nstep = {
                    # to automatically sum discounted rewards over next `n_step` steps
                    "size": n_step,
                    "gamma": gamma,
                    "rew": "reward",
                    # to automatically return `n_step`th state in `sample()`
                    "next": "next_state"
                }
            )
        else:
            self.buffer = cpprb.PrioritizedReplayBuffer(
                buffer_size, buffer_fields, alpha=alpha
            )

        self.beta = beta
        self.beta_growth_rate = beta_growth_rate
        self.update_beta_every = update_beta_every
        self.batch_size = batch_size

        # number of replays seen (but maybe not stored, because of `buffer_size` restriction)
        self.n_seen = 0
        self.device = device

        if beta_growth_rate < 1:
            raise ValueError(
                f'`beta_growth_rate` must be >=1: {beta_growth_rate}')
        if not (batch_size <= buffer_size):
            raise ValueError(
                f'Invalid sizes provided: `batch_size`={batch_size}, `buffer_size`={buffer_size}')

    def add(self, **observation):
        self.buffer.add(**observation)

        self.n_seen += 1
        if self.n_seen % self.update_beta_every == 0:
            self.beta = min(1, self.beta * self.beta_growth_rate)

    def sample(self):
        batch = self.buffer.sample(self.batch_size, self.beta)

        # send to device
        batch_device = {}
        for key, value in batch.items():
            if key == 'indexes':
                # batch['indexes'] is only used for updating priorities
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
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        n_step=1, gamma=1
    ):
        buffer_fields = {
            "state": {"shape": state_size},
            "action": {"dtype": np.int64},
            "reward": {},
            "next_state": {"shape": state_size},
            "done": {"dtype": np.bool8},
        }

        self.n_step = n_step
        if n_step > 1:
            self.buffer = cpprb.ReplayBuffer(
                buffer_size, buffer_fields,
                Nstep = {
                    # to automatically sum discounted rewards over next `n_step` steps
                    "size": n_step,
                    "gamma": gamma,
                    "rew": "reward",
                    # to automatically return `n_step`th state in `sample()`
                    "next": "next_state"
                }
            )
        else:
            self.buffer = cpprb.ReplayBuffer(
                buffer_size, buffer_fields
            )

        self.batch_size = batch_size
        self.device = device
        
        # number of replays seen (but maybe not stored, because of `buffer_size` restriction)
        self.n_seen = 0

        if not (batch_size <= buffer_size):
            raise ValueError(
                f'Invalid sizes provided: `batch_size`={batch_size}, `buffer_size`={buffer_size}')

    def add(self, **observation):
        self.n_seen += 1
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