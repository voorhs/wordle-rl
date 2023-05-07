# deeply inspired by https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from collections import defaultdict
from functools import partial


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=None, hidden_size=256, n_hidden_layers=1, **kwargs):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        
        layers = [nn.Linear(state_size, hidden_size)]
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        for fc in self.layers:
            x = F.relu(fc(x))
        
        return self.output(x)
    
    def freeze(self):
        self.layers[0].requires_grad_(False)
        self.layers[1].requires_grad_(False)


class BackboneQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0, **kwargs):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)


class SkipConnectionQNetwork(BackboneQNetwork):
    def forward(self, x):
        y = F.relu(self.fc1(x))
        x = F.relu(self.fc2(y))
        
        return self.fc3(x+y)
    
    def freeze(self):
        self.fc1.requires_grad_(False)
        self.fc2.requires_grad_(False)
    

class StackingOverBackbone(nn.Module):
    def __init__(self, state_size, action_size, seed=0, **kwargs):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        # skip connection anywhere?
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 130)
        self.fc4 = nn.Linear(130, action_size)


class ConvexQNetwork(nn.Module):
    class _ParametrizePositive(nn.Module):
        def forward(self, weights):
            return F.softplus(weights)
    
    def __init__(self, state_size, emb_size, hidden_size, optim_steps) -> None:
        super().__init__()

        self.state_size = state_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.optim_steps = optim_steps

        self.Ws = nn.Linear(self.state_size, self.hidden_size)
        self.Wu_1 = nn.Linear(self.state_size, self.hidden_size) 
        self.Wz_1 = nn.Linear(self.emb_size, self.hidden_size)
        
        self.Wz_2 = nn.Linear(self.hidden_size, 1)
        self.Wu_2 = nn.Linear(self.hidden_size, 1)
        self.Wa = nn.Linear(self.emb_size, 1)

        P.register_parametrization(self.Wz_1, 'weight', self._ParametrizePositive())
        P.register_parametrization(self.Wz_2, 'weight', self._ParametrizePositive())

    def forward(self, s, a=None):
        # forward pass through network
        def minus_Q(s, a):
            u1 = F.relu(self.Ws(s))
            z1 = F.relu(self.Wz_1(a) + self.Wu_1(s))
            z2 = self.Wu_2(u1) + self.Wz_2(z1) + self.Wa(a)
            return z2
        
        if a is None:
            # solve optimization problem, so gradients wrt
            # net params are unneccessary to be computed
            for param in self.parameters():
                param.requires_grad = False

            # L-BFGS optimization
            a, self.conv = Optimizer(minus_Q, s, self.optim_steps, self.emb_size).solve()

            # enable gradients back
            for param in self.parameters():
                param.requires_grad = True
        
        # return Q(s,a), a
        return minus_Q(s, a) * (-1), a


class Optimizer:
    def __init__(self, obj, s, optim_steps, a_size):
        self.obj = obj
        self.s = s
        self.optim_steps = optim_steps
        self.a_size = a_size
    
    def _closure(self, s, a, conv, optimizer):
        """
        Argument for torch.optim.LBFGS.step
        """
        # forward pass to compute objective
        obj = self.obj(s, a)
        conv.append(obj.item())

        # delete previous gradients wrt self.a[closure.ind]
        optimizer.zero_grad()
        
        # compute gradients wrt self.a
        obj.backward()

        return obj

    def solve(self):
        """L-BFGS optimization."""
        res = []

        # for each state in batch
        conv = []
        for s in self.s:
            # initial approximation for action embedding
            a = nn.Parameter(torch.randn(self.a_size))

            # L-BFGS optimizer for one action, because torch computes
            # gradients for scalar only
            optimizer = torch.optim.LBFGS([a], max_iter=self.optim_steps)
            
            # to store convergence history for current batch elem
            conv.append([])
            
            # start optimization algorithm
            closure = partial(self._closure, s=s, a=a, conv=conv[-1], optimizer=optimizer)
            optimizer.step(closure)

            # collect resulting action
            res.append(a.detach())
        
        return torch.stack(res), conv
        