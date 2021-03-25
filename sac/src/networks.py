import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from . import SACConfig

from typing import Tuple


class QFunction(nn.Module):
    """A network to approximate an action-value function"""
    
    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        input_dim = self.config.observation_dim + self.config.action_dim
        output_dim = 1
        self.nn = NeuralNet(input_dim, output_dim, config)
        self.optimiser = optim.Adam(self.parameters(), lr=config.lr)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([state, action], dim=-1)
        state_value = self.nn(inp)
        return state_value


class MultipleQFunctions(nn.Module):
    """A network that returns the (elementwise) minimum action-value function
    estimate of several QFunction networks"""

    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        self.networks = nn.ModuleList([
            QFunction(self.config)
            for _ in range(self.config.num_q_networks)
        ])

        self.optimiser = optim.Adam(self.networks.parameters(), self.config.lr)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values = [network(state, action) for network in self.networks]
        min_values = torch.stack(values).min(0).values
        return min_values
    
    def train(self, states: torch.Tensor, actions: torch.Tensor,
              rewards: torch.Tensor, is_done: torch.Tensor,
              next_values: torch.Tensor, next_action_logprobs: torch.Tensor,
              temperature: torch.Tensor, retain_graph: bool) -> torch.Tensor:
        targs = rewards + self.config.discount * (1 - is_done) * \
            (next_values - temperature * next_action_logprobs)
        
        # Update Q-Function Networks
        loss = 0.0
        for network in self.networks:
            preds = network(states, actions)
            loss += F.mse_loss(preds, targs)
        
        self.optimiser.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimiser.step()

        return loss.item()
    
    def train_polyak(self, other_network: nn.Module):
        with torch.no_grad():
            for param, other_param in zip(self.parameters(), other_network.parameters()):
                tau = self.config.targ_smoothing_coeff
                param.data.mul_(tau)
                param.data.add_((1-tau)*other_param.data)
    
    def disable_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad = True
    

class Policy(nn.Module):
    """A network to learn an entropy-regularised stochastic policy"""
    
    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        input_dim = self.config.observation_dim
        output_dim = 2 * self.config.action_dim
        self.nn = NeuralNet(input_dim, output_dim, self.config)
        self.optimiser = optim.Adam(self.parameters(), lr=self.config.lr)
    
    def forward(self, state: torch.Tensor, use_mean: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        nn_out = self.nn(state)
        means = nn_out[:, self.config.action_dim:]
        logstds = nn_out[:, :self.config.action_dim]
        stds = logstds.exp()
        
        action_dist = Normal(means, stds)
        action = means if use_mean else action_dist.rsample()

        # Stabler version from Spinning Up implementation
        logprob = action_dist.log_prob(action).sum(-1, keepdims=True)
        logprob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(-1, keepdims=True)
        
        action = torch.tanh(action)
        
        return action, logprob
    
    def train(self, logprobs: torch.Tensor, values: torch.Tensor,
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


class NeuralNet(nn.Module):
    """A simple parameterised MLP"""
    
    def __init__(self, input_dim: int, output_dim: int, config: SACConfig):
        super().__init__()
        features = [input_dim] + [config.units_per_layer]*config.hidden_layers
        self.layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(in_feats, out_feats), config.nonlinearity())
            for in_feats, out_feats in zip(features[:-1], features[1:])
        ])
        self.out_layer = nn.Linear(config.units_per_layer, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.out_layer(x)
        return x


class Temperature(nn.Module):

    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        log_temp = torch.Tensor([np.log(self.config.temperature)])
        self.log_temperature = nn.parameter.Parameter(log_temp)
        if self.config.adjust_temperature:
            self.optimiser = optim.Adam(self.parameters(), lr=self.config.lr)
        else:
            self.log_temperature.requires_grad = False
    
    def train(self, action_logprobs: torch.Tensor):
        loss = -self.log_temperature * (action_logprobs + self.config.entropy_target).mean()
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    
    def disable_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad = True


class SAC(nn.Module):

    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        self.q_networks = MultipleQFunctions(self.config)
        self.target_q_networks = MultipleQFunctions(self.config)
        self.policy_network = Policy(self.config)
        self.temperature = Temperature(self.config)

        self.q_networks.disable_grad()
        self.policy_network.disable_grad()
        self.temperature.disable_grad()
        self.target_q_networks.disable_grad()

        self.target_q_networks.load_state_dict(self.q_networks.state_dict())

    def train(self, states: torch.Tensor, actions: torch.Tensor,
              rewards: torch.Tensor, next_states: torch.Tensor,
              is_done: torch.Tensor):
        retain_graph = self.config.adjust_temperature
        with torch.no_grad():
            temperature = self.temperature.log_temperature.exp()
        
        # Train Q-Functions
        self.q_networks.enable_grad()

        with torch.no_grad():
            next_actions, next_action_logprobs = self.policy_network(next_states)
            next_values = self.target_q_networks(next_states, next_actions)
        
        self.q_networks.train(states, actions, rewards, is_done, next_values,
                              next_action_logprobs, temperature, retain_graph)
        
        self.q_networks.disable_grad()

        # Train Policy Network
        self.policy_network.enable_grad()

        sample_actions, sample_action_logprobs = self.policy_network(states)
        sample_action_values = self.q_networks(states, sample_actions)
        self.policy_network.train(sample_action_logprobs, sample_action_values, temperature, retain_graph)
        
        self.policy_network.disable_grad()

        # Update Temperature (if applicable)
        if self.config.adjust_temperature:
            self.temperature.enable_grad()
            self.temperature.train(sample_action_logprobs.detach().clone())
            self.temperature.disable_grad()
        
        # Update Target Q-networks using polyak averaging
        self.target_q_networks.train_polyak(self.q_networks)
