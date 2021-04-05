import gym
import torch.nn as nn

from typing import Optional, List


class SACConfig(object):
    
    def __init__(
        self,
        # Environment Config,
        env: gym.Env,
        temperature: float = 1.0,
        max_episode_length: int = 1000,
        reward_scale: float = 1.0,

        # Data Collection Config,
        random_steps: int = 10_000,
        initial_policy_steps: int = 1000,
        env_steps: int = 1,
        training_steps: int = 1,
        buffer_size: int = int(1e6),
        
        # Neural Network Config,
        hidden_layers: int = 2,
        units_per_layer: int = 256,
        num_q_networks: int = 2,
        adjust_temperature: bool = True,
        gumbel_temperature: float = 1.0,

        # Training Config,
        lr: float = 3e-4,
        discount: float = 0.99,
        batch_size: int = 256,
        nonlinearity: nn.Module = nn.ReLU,
        polyak_tau: float = 0.995,
        total_train_steps: int = 5_000_000,
    ):
        # Environment Config
        self.env = env
        self.observation_dim = env().observation_space.shape[0]
        action_space = env().action_space
        self.discrete_actions = isinstance(action_space, gym.spaces.Discrete)
        self.action_dim = action_space.n if self.discrete_actions else action_space.shape[0]
        self.action_min = action_space.low.min() if not self.discrete_actions else None
        self.action_max = action_space.high.max() if not self.discrete_actions else None
        self.temperature = temperature
        self.max_episode_length = max_episode_length
        self.reward_scale = reward_scale

        # Data Collection Config
        self.random_steps = random_steps
        self.initial_policy_steps = initial_policy_steps
        self.env_steps = env_steps
        self.training_steps = training_steps
        self.buffer_size = buffer_size
        
        # Neural Network Config
        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer
        self.num_q_networks = num_q_networks
        self.adjust_temperature = adjust_temperature
        self.entropy_target = -float(self.action_dim)
        self.gumbel_temperature = gumbel_temperature

        # Training Config
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.nonlinearity = nonlinearity
        self.polyak_tau = polyak_tau
        self.total_train_steps = total_train_steps
