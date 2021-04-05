import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Temperature(nn.Module):

    def __init__(
        self,
        initial_temperature: float,
        adjust: bool = True,
        learning_rate: float = None,
        entropy_target: float = None,
    ):
        super().__init__()
        self.adjust = adjust
        self.entropy_target = entropy_target
        log_temp = torch.Tensor([np.log(initial_temperature)])
        self.log_temperature = nn.parameter.Parameter(log_temp)
        if self.adjust:
            self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            self.log_temperature.requires_grad = False
    
    def step(self, action_logprobs: torch.Tensor):
        loss = -self.log_temperature * (action_logprobs + self.entropy_target).mean()
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    
    def disable_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad = True
