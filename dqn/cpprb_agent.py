# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/agent.py

import cpprb
import numpy as np
import random
from functools import partial

from dqn.model import QNetwork
from environment.environment import BaseAction, BaseState

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(
        self, state_size, action_size, action_constructor, replay_buffer, seed=0,
        gamma=1, tau=1e-3,
        optimizer=partial(Adam, lr=5e-4),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
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
        self.memory = replay_buffer

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).float().to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).float().to(device)
        self.device = device
        self.optimizer = optimizer(self.qnetwork_local.parameters())
        self.criterion = nn.MSELoss()
        self.loss = None
        # to implement pytorch-like logic of network regimes train and eval
        # if True, then `learn()` method won't be called
        self.eval = True
        self.eps = None

        # Initialize time steps
        self.t = 0

        # update params
        self.gamma = gamma
        self.tau = tau

        # validation
        if not (0 <= gamma <= 1):
            raise ValueError(
                f'Discount factor `gamma` must be float in [0,1], but given: {gamma}')
        if not (0 <= tau <= 1):
            raise ValueError(
                f'Soft update coefficient `tau` must be float in [0,1], but given: {tau}')

    def add(self, state: BaseState, action: BaseAction, reward, next_state: BaseState, done):
        """
        Runs internal processes:
        - update replay buffer state
        - run one epoch train
        """
        # save experience in replay memory
        self.memory.add(
            state=state.value,
            action=action.value,
            reward=reward,
            next_state=next_state.value,
            done=done)
        if done:
            self.memory.on_episode_end()

        # check for updates
        self.t += 1

        if self.t % 2 == 0:
            self.learn()
        
    def act(self, state: BaseState):
        """Returns actions for given state as per current policy.

        Params
        ======
            eps (float): epsilon, for epsilon-greedy action selection
        """
        nn_output = None
        if random.random() > self.eps:
            # greedy action based on Q function
            self.qnetwork_local.eval()
            with torch.no_grad():
                nn_output = self.qnetwork_local(
                    torch.from_numpy(state.value).float().unsqueeze(0).to(self.device)
                )
        else:
            # exploration action
            nn_output = torch.randn(self.action_size)

        return self.action_constructor(nn_output.cpu().data.numpy())

    def learn(self):
        """Update net params using batch sampled from replay buffer"""
        batch = self.memory.sample()

        # Q-function
        self.qnetwork_target.eval()
        q_target = self.qnetwork_target(
            batch['next_state']
        ).detach().max(1)[0].unsqueeze(1)

        # discounted return
        expected_values = batch['reward'] + self.gamma * q_target * (~batch['done'])

        # predicted return
        self.qnetwork_local.train()
        output = self.qnetwork_local(
            batch['state']
        ).gather(1, batch['action'].long())

        # MSE( Q_L(s_t, a_t); r_t + gamma * max_a Q_T(s_{t+1}, a) )
        if 'weights' in batch.keys():
            loss = torch.sum(batch['weights'] * (output - expected_values) ** 2)
        else:
            loss = torch.sum((output - expected_values) ** 2)

        # to print during training
        self.loss = torch.sqrt(loss).cpu().item()

        # train local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        if 'indexes' in batch.keys():
            # update priorities basing on TD-error
            tds = (expected_values - output.detach()).abs().cpu().numpy()
            self.memory.update_priorities(batch['indexes'], tds)

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
