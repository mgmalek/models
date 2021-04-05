from copy import deepcopy

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from IPython.display import clear_output

from .agent import AbstractAgent
from .data import TrajectoryPoint


class EnvWrapper(object):
    """A wrapper around an OpenAI Gym environment that returns a
    (s, a, r, s', d) tuple at each time step, and which automatically handles
    resetting the environment once it ends."""
    
    def __init__(
        self,
        env: gym.Env,
        agent: AbstractAgent,
        max_episode_length: int = 1000,
        use_rgb_buffer: bool = False,
        min_reward: float = float("-Inf"),
        max_reward: float = float("Inf"),
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    ):
        self.env = env()
        self.agent = agent
        self.max_epsiode_length = max_episode_length
        self.use_rgb_buffer = use_rgb_buffer
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.device = device

        self.latest_obs = None
        self.reset_statistics()

        self.reset()

        if isinstance(self.env, gym.envs.mujoco.MujocoEnv):
            _ = self.env.reset()
            self.env.render()

    def reset_statistics(self):
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_returns = []
        self.current_rewards = []
    
    def update_agent(self, agent: AbstractAgent):
        self.agent = agent
        self.reset()

    def process_rgb_buffer(self, raw_obs: np.array) -> torch.Tensor:
        raw_obs = raw_obs.transpose(2, 0, 1).astype('float64') / 255
        tens = torch.as_tensor(raw_obs, dtype=torch.float, device=self.device)
        tens = (0.2126*tens[0] + 0.7152*tens[1] + 0.0722*tens[2])[None]
        return tens
    
    def save_latest_obs(self, obs: np.array):
        if self.use_rgb_buffer:
            self.latest_obs = self.process_rgb_buffer(self.env.render('rgb_array'))
        else:
            self.latest_obs = torch.as_tensor(obs, device=self.device)
    
    def reset(self):
        obs = self.env.reset()
        self.save_latest_obs(obs)
        self.agent.reset()
        self.agent.update_input_state(self.latest_obs)
        self.done = False
        self.current_rewards = []
    
    def step(self, render: bool = False):
        # Reset environment if a previous episode just finished
        if self.done:
            self.episode_returns.append(np.sum(self.current_rewards))
            self.total_episodes += 1
            self.reset()

        # Take the next step in the environment
        action = self.agent.get_action()
        new_obs, reward, done, _ = self.env.step(action)

        # Save the data generated from the environment step (i.e. obs, reward,
        # done), update the agent's state and return the relevanet data
        self.save_latest_obs(new_obs)
        reward = max(min(reward, self.max_reward), self.min_reward)
        self.current_rewards.append(reward)
        self.done = done or (len(self.current_rewards) >= self.max_epsiode_length)

        if render:
            self.render()
        
        agent_input = self.agent.get_input_state()
        self.agent.update_input_state(self.latest_obs)
        agent_next_input = self.agent.get_input_state()

        point = TrajectoryPoint(
            state=agent_input,
            action=np.asarray(action),
            reward=float(reward),
            next_state=agent_next_input,
            is_done=int(done),
        )

        self.agent.step()
        self.total_steps += 1

        return point
    
    def render(self):
        img = self.env.render('rgb_array')
        clear_output(wait=True)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def test(self, render=False) -> float:
        self.reset()

        self.agent.test = True
        while not self.done:
            self.step(render=render)
        self.agent.test = False
        
        return np.sum(self.current_rewards)
