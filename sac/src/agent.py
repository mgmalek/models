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
            action, _ = self.policy(obs[None], self.deterministic)
        return action[0].cpu().numpy()
