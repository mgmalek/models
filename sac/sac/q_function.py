import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .rl import MLP


class QFunction(nn.Module):
    """A network to approximate an action-value function"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        units_per_layer: int,
        hidden_layers: int,
    ):
        super().__init__()

        self.nn = MLP(
            input_dim=observation_dim + action_dim,
            output_dim=1,
            units_per_layer=units_per_layer,
            hidden_layers=hidden_layers,
            nonlinearity=nn.ReLU,
            output_nonlinearity=False,
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([state, action], dim=-1)
        state_value = self.nn(inp)
        return state_value


class MultipleQFunctions(nn.Module):
    """A network that returns the (elementwise) minimum action-value function
    estimate of several QFunction networks"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        units_per_layer: int,
        hidden_layers: int,
        num_q_networks: int,
        learning_rate: float,
        discount: float,
        polyak_tau: float = 1.0,
    ):
        super().__init__()
        self.discount = discount
        self.polyak_tau = polyak_tau

        self.networks = nn.ModuleList([
            QFunction(observation_dim, action_dim, units_per_layer, hidden_layers)
            for _ in range(num_q_networks)
        ])

        self.optimiser = optim.Adam(self.networks.parameters(), learning_rate)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values = [network(state, action) for network in self.networks]
        min_value = torch.stack(values).min(0).values
        return min_value
    
    def step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        is_done: torch.Tensor,
        next_values: torch.Tensor,
        next_action_logprobs: torch.Tensor,
        temperature: torch.Tensor,
        retain_graph: bool
    ) -> torch.Tensor:
        targs = rewards + self.discount * (1 - is_done) * \
            (next_values - temperature * next_action_logprobs)
        
        # Update Q-Functions
        loss = 0.0
        for network in self.networks:
            preds = network(states, actions)
            loss += F.mse_loss(preds, targs)
        
        self.optimiser.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimiser.step()

        return loss.item()
    
    def step_polyak(self, other_network: nn.Module):
        with torch.no_grad():
            for param, other_param in zip(self.parameters(), other_network.parameters()):
                param.data.mul_(self.polyak_tau)
                param.data.add_((1-self.polyak_tau) * other_param.data)
    
    def disable_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad = True
