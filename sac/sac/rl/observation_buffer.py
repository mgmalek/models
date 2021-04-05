import torch


class ObservationBuffer(object):
    """Buffer to store the last n observations seen by the agent"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def insert(self, item):
        assert len(self) <= self.capacity
        if len(self) == self.capacity:
            self.data = self.data[1:]

        self.data.append(item)
    
    @property
    def is_full(self):
        return len(self) == self.capacity
    
    def get_tensor(self):
        assert self.is_full
        return torch.cat(self.data, dim=0)
    
    def reset(self):
        self.data = []
