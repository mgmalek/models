import random
from collections import namedtuple

import torch
import torch.nn.functional as F

from typing import List, Tuple


TrajectoryPoint = namedtuple('TrajectoryPoint', ['state', 'action', 'reward', 'next_state', 'is_done'])

class ReplayBuffer():
    
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    ):
        self.capacity = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.position = 0
        self.data = []
    
    def extend(self, points: List[TrajectoryPoint]):
        for point in points:
            self.append(point)
    
    def append(self, point: TrajectoryPoint):
        if len(self.data) < self.capacity:
            self.data.append(point)
        else:
            self.data[self.position] = point
        
        self.position = (self.position + 1) % self.capacity

    def sample(self) -> Tuple[torch.Tensor, ...]:
        batch_size = min(self.batch_size, len(self.data))
        states, actions, rewards, next_states, is_done = zip(*random.sample(self.data, batch_size))

        states = torch.stack(states).to(self.device).float()
        next_states = torch.stack(next_states).to(self.device).float()
        actions = torch.as_tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=self.device).reshape(-1, 1)
        is_done = torch.as_tensor(is_done, dtype=torch.float, device=self.device).reshape(-1, 1)
        return states, actions, rewards, next_states, is_done
