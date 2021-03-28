from abc import ABC

import torch
import torch.nn as nn
import numpy as np

from .config import SACConfig


class AbstractAgent(ABC):

    def __call__(self, observation: np.array) -> np.array:
        pass


class RandomAgent(AbstractAgent):

    def __init__(self, config: SACConfig):
        self.config = config

    def __call__(self, observation: np.array) -> np.array:
        if self.config.discrete_actions:
            return np.random.randint(0, self.config.action_dim)
        else:
            action_range = self.config.action_max - self.config.action_min
            action = np.random.random(self.config.action_dim) * action_range + self.config.action_min
            action = np.tanh(action)
            return action


class SACAgent(AbstractAgent):

    def __init__(self, policy: nn.Module, config: SACConfig):
        self.config = config
        self.policy = policy
        self.deterministic = False
    
    def __call__(self, observation: np.array) -> np.array:
        obs = torch.as_tensor(observation, dtype=torch.float).cuda()
        with torch.no_grad():
            should_rsample = not self.deterministic
            action, _ = self.policy(obs[None], rsample=should_rsample)
            action = action[0].cpu().numpy()
        
        if self.config.discrete_actions:
            action = int(action.argmax())

        return action
