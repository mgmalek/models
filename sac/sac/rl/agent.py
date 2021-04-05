from abc import ABC
import torch


class AbstractAgent(ABC):

    def __init__(self):
        self.test = False

    def get_input_state(self) -> torch.Tensor:
        """Return a copy of the agent's representation of the environment's
        state (to be added to the replay buffer and used for training)"""
        NotImplemented
    
    def update_input_state(self, latest_obs: torch.Tensor):
        """Update the agent's internal representation of the environment's
        state, given the latest observation collected from the environment"""
        NotImplemented
    
    def get_action(self) -> torch.Tensor:
        """Return the action to be taken by the agent. This function has no
        parameters since the agent should maintain its own copy of the
        environment's state (passed to it by the update_input_state method)"""
        NotImplemented
    
    def step(self):
        """Update any required state after an environment step has been
        completed (e.g. decrease epsilon in an epsilon-greedy exploration
        strategy)"""
        NotImplemented
    
    def reset(self):
        """Perform any cleanup required between episodes (e.g. resetting
        internal state)"""
        NotImplemented


class BasicAgent(AbstractAgent):
    """A class that is useful for creating agents that just directly use
    the latest environment observations as their internal representation of
    state"""

    def __init__(self):
        self.test = False
        self.latest_obs = None

    def get_input_state(self) -> torch.Tensor:
        return self.latest_obs.detach().clone()
    
    def update_input_state(self, latest_obs: torch.Tensor):
        self.latest_obs = latest_obs
    
    def step(self):
        pass

    def reset(self):
        self.latest_obs = None


class RandomContinuousAgent(BasicAgent):
    
    def __init__(self, action_dim: int, action_min: float, action_max: float):
        super().__init__()
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
    
    def get_action(self):
        action_range = self.action_max - self.action_min
        return torch.rand(self.action_dim) * action_range + self.action_min


class RandomDiscreteAgent(BasicAgent):
    
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
    
    def get_action(self):
        unif_sample = torch.rand(self.action_dim)
        gumbel_sample = -torch.log(-torch.log(unif_sample))
        return gumbel_sample.argmax().item()