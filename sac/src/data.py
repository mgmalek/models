import random
from collections import namedtuple

import torch
import torch.nn.functional as F

from .config import SACConfig

from typing import List, Tuple


TrajectoryPoint = namedtuple('TrajectoryPoint', ['state', 'action', 'reward', 'next_state', 'is_done'])

class ReplayBuffer():
    
    def __init__(self, config: SACConfig):
        self.config = config
        self.capacity = self.config.buffer_size
        self.position = 0
        self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        point = self.data[index]
        state = torch.as_tensor(point.state, dtype=torch.float)
        action = torch.as_tensor(point.action, dtype=torch.float)
        reward = torch.as_tensor(point.reward, dtype=torch.float).unsqueeze(0)
        next_state = torch.as_tensor(point.next_state, dtype=torch.float)
        is_done = torch.as_tensor(point.is_done, dtype=torch.float).unsqueeze(0)
        return state, action, reward, next_state, is_done

    def extend(self, points: List[TrajectoryPoint]):
        for point in points:
            self.append(point)
    
    def append(self, point: TrajectoryPoint):
        if len(self.data) < self.capacity:
            self.data.append(point)
        else:
            self.data[self.position] = point
        
        self.position += 1
        if self.position >= self.capacity:
            self.position = 0
    
    def sample(self) -> Tuple[torch.Tensor, ...]:
        batch_size = min(self.config.batch_size, len(self))
        states, actions, rewards, next_states, is_done = zip(*random.sample(self.data, batch_size))
        states = torch.as_tensor(states, dtype=torch.float).cuda()
        actions = torch.as_tensor(actions, dtype=torch.float)
        if self.config.discrete_actions:
            actions = F.one_hot(actions.long(), num_classes=self.config.action_dim).float()
        actions = actions.reshape(-1, self.config.action_dim).cuda()
        rewards = torch.as_tensor(rewards, dtype=torch.float).reshape(-1, 1).cuda()
        next_states = torch.as_tensor(next_states, dtype=torch.float).cuda()
        is_done = torch.as_tensor(is_done, dtype=torch.float).reshape(-1, 1).cuda()
        return states, actions, rewards, next_states, is_done
