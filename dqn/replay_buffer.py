# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/replay_buffer.py

import random
from collections import namedtuple
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
            self, action_size, buffer_size, batch_size, sample_size,
            seed, compute_weights, priority_rate=None):
        """
        Initialize a ReplayBuffer object.

        Params
        ------
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            sample_size (int): number of experiences to sample during a sampling iteration
            batch_size (int): size of each training batch
            seed (int): random seed
            priority_rate (dict): priority updating rate params:
                                    alpha, alpha_decay_rate, beta, beta_growth_rate

        Buffer
        ------
        Stores experience as named tuples in self.buffer:
            `state`, `next_state` (np.ndarray): vector representing state of environment
            `action` (int): index of word which agent chose (BaseAction.value in general)
            `done` (bool): indicator of end of episode after agent choosing `action`
            `reward` (float): reward from environment
        """
        self.seed = random.seed(seed)

        # environment configs
        self.action_size = action_size
        self.buffer_size = buffer_size

        # training configs
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.compute_weights = compute_weights

        # priority updating rate
        self.priority_rate = priority_rate
        if priority_rate is None:
            self.priority_rate = {
                'alpha': 0.9,
                'alpha_decay_rate': 0.95,
                'beta': 1,
                'beta_growth_rate': 1
            }
        self.alpha = self.priority_rate['alpha'] + 1e-3
        self.alpha_decay_rate = self.priority_rate['alpha_decay_rate']
        self.beta = self.priority_rate['beta']
        self.beta_growth_rate = self.priority_rate['beta_growth_rate']

        self.priorities_max = 1
        self.weights_max = 1

        # to conveniently create new experience records
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priority = namedtuple(
            "Data", field_names=["priority", "probability", "weight", "index"])

        # initialize experience

        # number of collected replays
        self.n_collected = 0

        # array of replays
        self.buffer = {}

        # array of auxiliary data
        self.aux = {i: self.priority(
            0, 0, 0, i) for i in range(buffer_size)}

        # to perform batch-training we sample batches
        # and simply iterate through list of batches
        self.sampled_batches = []
        self.current_batch = 0

    def update_priorities(self, tds, indices):
        """
        Update priorities based on temporal differencies
        between predicted Q value (local qnet) and
        expected reward (target qnet)
        """
        for td, index in zip(tds, indices):
            N = min(self.n_collected, self.buffer_size)

            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority

            if self.compute_weights:
                updated_weight = ((N * updated_priority) **
                                  (-self.beta)) / self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1

            updated_probability = td[0] ** self.alpha
            self.aux[index] = self.priority(
                updated_priority, updated_probability, updated_weight, index)

    def update_sample(self):
        """Randomly sample X batches of experiences."""
        # X is the number of steps before updating memory
        self.current_batch = 0

        random_values = random.choices(
            self.aux,
            [data.probability for data in list(self.aux.values())],
            k=self.sample_size)

        self.sampled_batches = [random_values[i:i + self.batch_size]
                                for i in range(0, len(random_values), self.batch_size)]

    def update_parameters(self):
        """Move priorities distribution towards uniform one."""
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1

        N = min(self.n_collected, self.buffer_size)

        sum_prob_after = 0
        for element in self.aux.values():
            probability = element.priority ** self.alpha
            sum_prob_after += probability

            weight = 1
            if self.compute_weights:
                weight = ((N * element.probability) **
                          (-self.beta)) / self.weights_max

            self.aux[element.index] = self.priority(
                element.priority, probability, weight, element.index)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.n_collected += 1

        # index to insert
        index = self.n_collected % self.buffer_size

        if self.n_collected > self.buffer_size:
            # in case buffer is overflowed

            if self.aux[index].priority == self.priorities_max:
                # to remove max priority we need to replace it
                # with second max priority
                self.aux[index] = self.priority(0, 0, 0, index)
                self.priorities_max = max(
                    self.aux.values(), key=lambda x: x.priority).priority

            if self.compute_weights:
                if self.aux[index].weight == self.weights_max:
                    # the same as for max priority
                    self.aux[index] = self.priority(0, 0, 0, index)
                    self.weights_max = max(
                        self.aux.values(), key=lambda x: x.weight).weight

        # recent experience has max priority and weight
        priority = self.priorities_max
        weight = self.weights_max

        # define probability
        probability = priority ** self.alpha

        # finish insertion
        self.buffer[index] = self.experience(
            state, action, reward, next_state, done)
        self.aux[index] = self.priority(priority, probability, weight, index)

    def sample(self):
        """Return batch of replays."""
        # update batch iterator info
        sampled_batch = self.sampled_batches[self.current_batch]
        self.current_batch += 1

        # batch elements
        experiences = []
        weights = []
        indices = []
        for replay in sampled_batch:
            experiences.append(self.buffer.get(replay.index))
            weights.append(replay.weight)
            indices.append(replay.index)

        # expand, convert to torch, send to cuda
        # почему здесь проверки на None?
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones, weights, indices)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
