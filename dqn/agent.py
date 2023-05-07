# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/agent.py

import numpy as np
import random
from functools import partial
from typing import List, Union

from dqn.model import QNetwork, BackboneQNetwork
from environment.environment import BaseState
from environment.action import BaseAction, ActionEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(
        self, state_size, action_instance:BaseAction, replay_buffer,
        gamma=1, tau=1e-3, optimize_interval=8,
        agent_path=None, **model_params
    ):
        self.state_size = state_size
        self.action = action_instance
        self.memory = replay_buffer

        # Q-Network
        self.define_model(**model_params)
        
        # load checkpoint
        if agent_path is not None:
            if 'local' in agent_path.keys():
                self.qnetwork_local.load_state_dict(torch.load(agent_path['local']))
            if 'target' in agent_path.keys():
                self.qnetwork_target.load_state_dict(torch.load(agent_path['target']))
            if 'buffer' in agent_path.keys():
                self.memory.buffer.load_transitions(agent_path['buffer'])
        
        if agent_path is not None and 'optimizer' in agent_path.keys():
            self.optimizer.load_state_dict(torch.load(agent_path['optimizer']))
        
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
        self.optimize_interval = optimize_interval

        # validation
        if not (0 <= gamma <= 1):
            raise ValueError(
                f'Discount factor `gamma` must be float in [0,1], but given: {gamma}')
        if not (0 <= tau <= 1):
            raise ValueError(
                f'Soft update coefficient `tau` must be float in [0,1], but given: {tau}')

    def define_model(self, **model_params):
        self.device = model_params.get('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model = model_params.get('model', QNetwork)
        self.qnetwork_local = model(self.state_size, self.action.size, **model_params).float().to(self.device)
        self.qnetwork_target = model(self.state_size, self.action.size, **model_params).float().to(self.device)
        self.optimizer = model_params.get('optimizer', partial(Adam, lr=5e-4))(self.qnetwork_local.parameters())

    def add(self, state: Union[np.ndarray, BaseState], action: BaseAction, reward, next_state: BaseState, done):
        """
        Runs internal processes:
        - update replay buffer state
        - run one epoch train
        """
        # save experience in replay memory
        self.memory.add(
            state=state if isinstance(state, np.ndarray) else state.value,
            action=action.index,
            reward=reward,
            next_state=next_state.value,
            done=done)
        if done:
            self.memory.on_episode_end()

        # check for updates
        self.t += 1

        if self.t % self.optimize_interval == 0 and not self.eval:
            self.learn()

    def act_single(self, state: BaseState):
        """Returns action for given state"""
        if isinstance(self.action, ActionEmbedding):
            return self.act_single_embedding(state)
        
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
            nn_output = torch.randn(1, self.action.size)

        return self.action(nn_output)
    
    def act_batch(self, states: List[np.ndarray]):
        """Returns actions for given batch of states"""
        play_batch_size = len(states)
        states = np.array(states)
        
        # for each game define what action to choose: greedy or exporative
        explore_ind = []
        greedy_ind = []
        events = np.random.uniform(low=0, high=1, size=play_batch_size)
        for i in range(play_batch_size):
            if events[i] < self.eps:
                explore_ind.append(i)
            else:
                greedy_ind.append(i)

        if isinstance(self.action, ActionEmbedding):
            return self.act_batch_embedding(states, explore_ind, greedy_ind)

        # to store result
        nn_output = torch.empty((play_batch_size, self.action.size)).to(self.device)
        
        # greedy action
        self.qnetwork_local.eval()
        with torch.no_grad():
            nn_output[greedy_ind] = self.qnetwork_local(
                torch.from_numpy(states[greedy_ind])
                    .float()
                    .to(self.device)
            )
        
        # explorative action
        nn_output[explore_ind] = torch.randn(len(explore_ind), self.action.size).to(self.device)

        # convert to `BaseAction` inheritant
        return self.action(nn_output)

    def act_single_embedding(self, state):
        index = None
        if random.random() > self.eps:
            # greedy action based on Q function
            self.qnetwork_local.eval()
            with torch.no_grad():
                greedy_act = self.qnetwork_local(
                    torch.from_numpy(state.value).float().unsqueeze(0).to(self.device)
                )
            index = self.action(greedy_act).index
        else:
            # exploration action
            index = torch.tensor(
                np.random.choice(len(self.action.vocabulary))
            )

        return self.action.act(index=index)

    def act_batch_embedding(self, states: np.ndarray, explore_ind, greedy_ind):
        """Ugly branch to incorporate ActionEmbedding cases."""
        
        # indices of words in vocabulary
        index = torch.empty(states.shape[0]).long()

        # make greedy actions
        if greedy_ind:
            self.qnetwork_local.eval()
            with torch.no_grad():
                # tuple (qfunc, embeddings)
                greedy_acts = self.qnetwork_local(
                    torch.from_numpy(states[greedy_ind])
                        .float()
                        .to(self.device)
                )
            # calculate indexes (knn)
            index[greedy_ind] = self.action(greedy_acts).index

        if explore_ind:
            # calculate explorative indexes
            index[explore_ind] = torch.from_numpy(
                np.random.choice(len(self.action.vocabulary), len(explore_ind))
            )
        
        # convert to ActionEmbedding object
        return self.action.act(index=index)

    def learn(self):
        """Update net params using batch sampled from replay buffer"""
        batch = self.memory.sample()

        # Q-function
        q_target = None
        self.qnetwork_target.eval()
        if not isinstance(self.action, ActionEmbedding):
            nn_output = self.qnetwork_target(batch['next_state']).detach()
            q_target = self.action(nn_output).qfunc
        else:
            q_target, _ = self.qnetwork_target(batch['next_state'])

        # discounted return
        expected_values = batch['reward'] + self.gamma ** self.memory.n_step * q_target * (~batch['done'])

        # predicted return
        q_local = None
        self.qnetwork_local.train()
        if not isinstance(self.action, ActionEmbedding):
            nn_output = self.qnetwork_local(batch['state'])
            q_local = self.action(nn_output, index=batch['action'].long()).qfunc
        else:
            embeddings = self.action.get_embeddings(batch['action'].long())
            q_local, _ = self.qnetwork_local(batch['state'], embeddings)

        # MSE( Q_L(s_t, a_t); r_t + gamma * max_a Q_T(s_{t+1}, a) )
        if 'weights' in batch.keys():
            loss = torch.sum(batch['weights'] * (q_local - expected_values) ** 2)
        else:
            loss = torch.sum((q_local - expected_values) ** 2)

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
            tds = (expected_values - q_local.detach()).abs().cpu().numpy()
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

    def dump(self, nickname, t):
        agent_path = {
            'local': f'{nickname}/local-{t}.pth',
            'target': f'{nickname}/target-{t}.pth',
            'buffer': f'{nickname}/buffer-{t}.npz',
            'optimizer': f'{nickname}/optimizer-{t}.pth',
        }
        torch.save(self.qnetwork_local.state_dict(), agent_path['local'])
        torch.save(self.qnetwork_target.state_dict(), agent_path['target'])
        self.memory.buffer.save_transitions(agent_path['buffer'])
        torch.save(self.optimizer.state_dict(), agent_path['optimizer'])

        return agent_path
    
    def load_backbone(self, model_path):
        # local network
        backbone_local = BackboneQNetwork(self.state_size, 130).float().to(self.device)
        backbone_local.load_state_dict(torch.load(model_path))

        self.qnetwork_local.layers[0].load_state_dict(backbone_local.fc1.state_dict())
        self.qnetwork_local.layers[1].load_state_dict(backbone_local.fc2.state_dict())
        
        # target network
        backbone_target = BackboneQNetwork(self.state_size, 130).float().to(self.device)
        backbone_target.load_state_dict(torch.load(model_path))
        
        self.qnetwork_target.layers[0].load_state_dict(backbone_target.fc1.state_dict())
        self.qnetwork_target.layers[1].load_state_dict(backbone_target.fc2.state_dict())

    def freeze_layers(self):
        self.qnetwork_local.freeze()
        self.qnetwork_target.freeze()