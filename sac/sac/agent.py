import torch
import torch.nn as nn
import numpy as np

from .rl import BasicAgent


class SACAgent(BasicAgent):

    def __init__(self, policy: nn.Module, discrete_actions: bool):
        super().__init__()
        self.policy = policy
        self.discrete_actions = discrete_actions
    
    def get_action(self) -> np.array:
        obs = torch.as_tensor(self.latest_obs, dtype=torch.float).cuda()
        with torch.no_grad():
            should_sample = not self.test
            action, _ = self.policy(obs[None], rsample=should_sample)
            action = action[0].cpu()

        if self.discrete_actions:
            action = int(action.argmax())
        
        return action
