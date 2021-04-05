import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from .temperature import Temperature
from .q_function import MultipleQFunctions
from .policy import DiscreteActionPolicy, ContinuousActionPolicy
from .config import SACConfig


class SAC(nn.Module):

    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config

        self.q_networks = MultipleQFunctions(
            self.config.observation_dim,
            self.config.action_dim,
            units_per_layer=self.config.units_per_layer,
            hidden_layers=self.config.hidden_layers,
            num_q_networks=self.config.num_q_networks,
            learning_rate=self.config.lr,
            discount=self.config.discount,
            polyak_tau=self.config.polyak_tau,
        )

        self.target_q_networks = MultipleQFunctions(
            self.config.observation_dim,
            self.config.action_dim,
            units_per_layer=self.config.units_per_layer,
            hidden_layers=self.config.hidden_layers,
            num_q_networks=self.config.num_q_networks,
            learning_rate=self.config.lr,
            discount=self.config.discount,
            polyak_tau=self.config.polyak_tau,
        )

        if self.config.discrete_actions:
            self.policy_network = DiscreteActionPolicy(
                self.config.observation_dim,
                self.config.action_dim,
                self.config.units_per_layer,
                self.config.hidden_layers,
                self.config.lr,
            )
        else:
            self.policy_network = ContinuousActionPolicy(
                self.config.observation_dim,
                self.config.action_dim,
                self.config.units_per_layer,
                self.config.hidden_layers,
                self.config.lr,
            )

        self.temperature = Temperature(
            self.config.temperature,
            self.config.adjust_temperature,
            self.config.lr,
            entropy_target = -self.config.action_dim,
        )

        self.q_networks.disable_grad()
        self.policy_network.disable_grad()
        self.temperature.disable_grad()
        self.target_q_networks.disable_grad()

        self.target_q_networks.load_state_dict(self.q_networks.state_dict())

    def step(self, states: torch.Tensor, actions: torch.Tensor,
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
        
        q_loss = self.q_networks.step(states, actions, rewards, is_done, next_values,
                                       next_action_logprobs, temperature, retain_graph)
        self.q_networks.disable_grad()

        # Train Policy Network
        self.policy_network.enable_grad()
        sample_actions, sample_action_logprobs = self.policy_network(states)
        sample_action_values = self.q_networks(states, sample_actions)
        p_loss = self.policy_network.step(sample_action_logprobs,
                                          sample_action_values, temperature,
                                          retain_graph)
        self.policy_network.disable_grad()

        # Update Temperature (if applicable)
        if self.config.adjust_temperature:
            self.temperature.enable_grad()
            self.temperature.step(sample_action_logprobs.detach().clone())
            self.temperature.disable_grad()
        
        # Update Target Q-networks using polyak averaging
        self.target_q_networks.step_polyak(self.q_networks)

        return q_loss, p_loss
