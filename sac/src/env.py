import time
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

from .config import SACConfig
from .agent import AbstractAgent
from .data import TrajectoryPoint


class EnvWrapper(object):
    """A wrapper around an OpenAI Gym environment that returns a
    (s, a, r, s', d) tuple at each time step, and which automatically handles
    resetting the environment once it ends."""
    
    def __init__(self, config: SACConfig, agent: AbstractAgent):
        self.config = config
        self.agent = agent
        self.env = self.config.env()
        self.latest_obs = None
        self.total_steps = 0
        self.num_episodes = 0
        self.ep_returns = []
        self.rewards = []
        self.reset()

        env = config.env()
        _ = env.reset()
        env.render()
        env.close()
        del env
    
    def reset(self):
        self.prev_obs = self.env.reset()
        self.latest_obs = self.prev_obs
    
    def step(self, render=False):
        obs = self.latest_obs
        action = self.agent(obs)
        rescaled_action = self.scale_action(torch.as_tensor(action, dtype=torch.float))
        next_obs, reward, done, _ = self.env.step(rescaled_action)
        reward *= self.config.reward_scale
        self.latest_obs = deepcopy(next_obs)
        self.rewards.append(reward)
        
        done = done or (len(self.rewards) >= self.config.max_episode_length)
        
        point = TrajectoryPoint(
            state=deepcopy(obs),
            action=action,
            reward=reward,
            next_state=deepcopy(self.latest_obs),
            is_done=int(done)
        )

        self.total_steps += 1
        
        if done:
            self.reset()
            self.ep_returns.append(np.sum(self.rewards))
            self.rewards = []
            self.num_episodes += 1
        
        return point
    
    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        action_range = self.config.action_max - self.config.action_min
        scaled_action = (action + 1.0) / 2.0 * action_range + self.config.action_min
        return scaled_action
    
    def test(self, render=False) -> float:
        self.agent.deterministic = True
        
        rewards = []
        env = self.config.env()
        obs = env.reset()
        done = False
        idx = 0
        
        while not done and idx < self.config.max_episode_length:
            if render:
                img = env.render('rgb_array')
                clear_output(wait=True)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            
            action = self.agent(obs)
            action = self.scale_action(action)
            next_obs, reward, done, _ = env.step(action)
            obs = deepcopy(next_obs)
            idx += 1
            
            rewards.append(reward)
        
        self.agent.deterministic = False
        
        return np.sum(rewards)
