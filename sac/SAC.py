#!/usr/bin/env python
# coding: utf-8

# # Soft Actor-Critic (SAC)

# ## TODO: Check whether PETS concerns about variances are relevant to policy network here
# ## TODO: Change observation to diff from last observation

# In[ ]:


import torch
import gym
from gym.envs.classic_control import PendulumEnv
from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, HumanoidEnv
from matplotlib import pyplot as plt

from src import SACConfig, RandomAgent, SACAgent, EnvWrapper, ReplayBuffer, SAC


# In[ ]:


config = SACConfig(
    env=HumanoidEnv,
    observation_dim=376,
    action_dim=17,
    temperature=1.0,
    total_train_steps=6_000_000,
    action_min=-1.0,
    action_max=1.0,
    max_episode_length=10_000,
    env_steps=1,
    training_steps=1,
    adjust_temperature=True,
    random_steps=10_000,
    initial_policy_steps=1_000,
)


# In[ ]:


# Initialise Networks, Agents, Dataset and Environment
sac = SAC(config).cuda()
random_agent = RandomAgent(config)
agent = SACAgent(sac.policy_network, config)
dataset = ReplayBuffer(config)
env_wrapper = EnvWrapper(config, random_agent)


# In[ ]:


# Collect Initial Data
dataset.extend([env_wrapper.step() for _ in range(config.random_steps)])

env_wrapper.agent = agent
dataset.extend([env_wrapper.step() for _ in range(config.initial_policy_steps)])


# In[ ]:


#env_wrapper.test(render=True)


# In[ ]:


test_returns = []
episode_idx = len(env_wrapper.ep_returns)
while env_wrapper.total_steps < config.total_train_steps:
    for _ in range(config.env_steps):
        dataset.extend([env_wrapper.step()])

    for batch_idx in range(config.training_steps):
        states, actions, rewards, next_states, is_done = dataset.sample()
        sac.train(states, actions, rewards, next_states, is_done)
    
    if (env_wrapper.total_steps % 1_000) < config.env_steps:
        test_return = env_wrapper.test(render=False)
        test_returns.append(test_return)
        print(f"Step: {env_wrapper.total_steps}\tEpisode: {env_wrapper.num_episodes}\tTest Return: {test_return:6.2f}\tTemperature: {sac.temperature.log_temperature.exp().item():8.4f}")
        torch.save(sac, "humanoid.pt")


# In[ ]:


#plt.plot(test_returns)


# In[ ]:


#env_wrapper.test(render=True)

