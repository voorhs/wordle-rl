import numpy as np
import random

from dqn.model import QNetwork
from dqn.replay_buffer import ReplayBuffer
from environment.environment import BaseAction, BaseState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # for soft update of target parameters
LR = 5e-4                   # learning rate
UPDATE_NN_EVERY = 1         # how often to update the network

# prioritized experience replay
UPDATE_MEM_EVERY = 20          # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 3000    # how often to update the hyperparameters
SAMPLE_SIZE = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, action_constructor, seed=0, compute_weights=False):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            action_constructor: function taking `nn_output` and returning action instance
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_constructor = action_constructor

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).float().to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).float().to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.compute_weights = compute_weights

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, SAMPLE_SIZE, seed, compute_weights)

        # Initialize time steps
        self.t_step_nn = 0
        self.t_step_mem_par = 0
        self.t_step_mem = 0

    def step(self, state:BaseState, action:BaseAction, reward, next_state:BaseAction, done):
        """
        Runs internal processes:
        - update replay buffer state
        - run one epoch train
        """
        # save experience in replay memory
        self.memory.add(state.copy(), action, reward, next_state, done)

        # check for updates
        self.t_step_nn = (self.t_step_nn + 1) % UPDATE_NN_EVERY
        self.t_step_mem = (self.t_step_mem + 1) % UPDATE_MEM_EVERY
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY

        if self.t_step_mem_par == 0:
            self.memory.update_parameters()

        if self.t_step_nn == 0:
            if self.memory.n_collected > SAMPLE_SIZE:
                # if enough samples are available in memory
                # get random subset and learn
                self.learn(self.memory.sample(), GAMMA)
        
        if self.t_step_mem == 0:
            self.memory.update_sample()

    def act(self, state:BaseState, eps=.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        nn_output = None
        if random.random() > eps:
            # greedy action based on Q function
            self.qnetwork_local.eval()
            with torch.no_grad():
                nn_output = self.qnetwork_local(state.tovector())
            self.qnetwork_local.train()
        else:
            # exploration action
            nn_output = torch.randn(self.action_size)   # change to torch
            
        return self.action_constructor(nn_output.cpu().data.numpy())

    def learn(self, sampling, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            sampling (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices = sampling

        # Q-function
        q_target = self.qnetwork_target(next_states)\
                        .detach().max(1)[0].unsqueeze(1)

        # discounted return
        expected_values = rewards + gamma * q_target * (1 - dones)

        # predicted return
        output = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(output, expected_values)

        # in case replays have weights
        if self.compute_weights:
            with torch.no_grad():
                loss *= sum(np.multiply(weights, loss.data.cpu().numpy()))

        # train local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # update priorities basing on loss ??
        delta = abs(expected_values - output.detach()).numpy()
        self.memory.update_priorities(delta, indices)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)
