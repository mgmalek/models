import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from .rl import MLP

from typing import Tuple


class ContinuousActionPolicy(nn.Module):
    """A network to learn an entropy-regularised stochastic policy for
    continuous-action environments"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        units_per_layer: int,
        hidden_layers: int,
        learning_rate: float,
    ):
        super().__init__()
        self.action_dim = action_dim
        
        self.nn = MLP(
            input_dim=observation_dim,
            output_dim=2 * action_dim,
            units_per_layer=units_per_layer,
            hidden_layers=hidden_layers,
            nonlinearity=nn.ReLU,
            output_nonlinearity=False,
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state: torch.Tensor, rsample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        nn_out = self.nn(state)
        means = nn_out[:, self.action_dim:]
        logstds = nn_out[:, :self.action_dim]
        stds = logstds.exp()
        
        action_dist = Normal(means, stds)
        action = action_dist.rsample() if rsample else means

        # Stabler version of log-probability calculation
        # Reference: Spinning Up implementation of SAC
        logprobs = action_dist.log_prob(action).sum(-1, keepdims=True)
        logprobs -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(-1, keepdims=True)
        
        action = torch.tanh(action)
        
        return action, logprobs
    
    def step(self, logprobs: torch.Tensor, values: torch.Tensor,
              temperature: torch.Tensor, retain_graph: bool = False) -> torch.Tensor:
        self.optimiser.zero_grad()
        policy_loss = torch.mean(temperature * logprobs - values)
        policy_loss.backward(retain_graph=retain_graph)
        self.optimiser.step()
        return policy_loss.item()
    
    def disable_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad = True


class DiscreteActionPolicy(nn.Module):
    """A network to learn an entropy-regularised stochastic policy for
    discrete-action environments"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        units_per_layer: int,
        hidden_layers: int,
        learning_rate: float,
    ):
        super().__init__()
        self.action_dim = action_dim
        
        self.nn = MLP(
            input_dim=observation_dim,
            output_dim=2 * action_dim,
            units_per_layer=units_per_layer,
            hidden_layers=hidden_layers,
            nonlinearity=nn.ReLU,
            output_nonlinearity=False,
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state: torch.Tensor, rsample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        nn_out = self.nn(state)
        logprobs = torch.log_softmax(nn_out, -1)

        if rsample:
            one_hot_action = self.gumbel_softmax(logprobs, temperature=self.config.gumbel_temperature)
            one_hot_action = (one_hot_action - logprobs).detach() + logprobs # Straight-through gradients
        else:
            one_hot_action = F.one_hot(logprobs.argmax(-1), num_classes=logprobs.size(-1))
        
        logprobs = logprobs.gather(-1, one_hot_action.detach().argmax(-1, keepdims=True))

        return one_hot_action, logprobs
    
    def step(self, logprobs: torch.Tensor, values: torch.Tensor,
              temperature: torch.Tensor, retain_graph: bool = False) -> torch.Tensor:
        self.optimiser.zero_grad()
        policy_loss = torch.mean(temperature * logprobs - values)
        policy_loss.backward(retain_graph=retain_graph)
        self.optimiser.step()
        return policy_loss.item()
    
    def disable_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad = True
