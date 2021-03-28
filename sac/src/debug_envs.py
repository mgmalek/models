import gym
import numpy as np

from typing import Union


class DebugEnv1(gym.Env):
    """
    Reference: https://andyljones.com/posts/rl-debugging.html#probe
    One action, zero observation, one timestep long, +1 reward every timestep
    
    This isolates the value network. If my agent can't learn that the value of
    the only observation it ever sees it 1, there's a problem with the value
    loss calculation or the optimizer.
    """
    
    def __init__(self, discrete_actions: bool = False):
        self.done = False
        self.discrete_actions = discrete_actions
    
    def reset(self):
        self.done = False
        return np.array([0.0])
    
    def step(self, action: np.array):
        next_state = np.array([0.0])
        reward = 1.0
        done = True
        return next_state, reward, done, {}
    
    def render(self):
        pass


class DebugEnv2(gym.Env):
    """
    Reference: https://andyljones.com/posts/rl-debugging.html#probe
    One action, random +1/-1 observation, one timestep long, obs-dependent
    +1/-1 reward every time
    
    If my agent can learn the value in (1.) but not this one - meaning it can
    learn a constant reward but not a predictable one! - it must be that
    backpropagation through my network is broken.
    """
    
    def __init__(self, discrete_actions: bool = False):
        self.done = False
        self.discrete_actions = discrete_actions
        self.latest_obs = None
    
    def _get_obs(self):
        return np.array([np.random.choice([-1.0, 1.0])])
    
    def reset(self):
        self.done = False
        self.latest_obs = self._get_obs()
        return self.latest_obs
    
    def step(self, action: np.array):
        reward = self.latest_obs[0]
        self.latest_obs = self._get_obs()
        next_state = self.latest_obs
        done = True
        return next_state, reward, done, {}
    
    def render(self):
        pass


class DebugEnv3(gym.Env):
    """
    Reference: https://andyljones.com/posts/rl-debugging.html#probe
    One action, zero-then-one observation, two timesteps long, +1 reward at
    the end
    
    If my agent can learn the value in (2.) but not this one, it must be that
    my reward discounting is broken.
    """
    
    def __init__(self, discrete_actions: bool = False):
        self.discrete_actions = discrete_actions
        self.idx = 0
    
    def reset(self):
        self.idx = 0
        return np.array([0.0])
    
    def step(self, action: np.array):
        if self.idx == 0:
            next_state, reward, done = np.array([1.0]), 0.0, False
            self.idx += 1
        elif self.idx == 1:
            next_state, reward, done = np.array([-1.0]), 1.0, True
        else:
            raise RuntimeError("Environment has already ended.")
        
        return next_state, reward, done, {}
    
    def render(self):
        pass


class DebugEnv4(gym.Env):
    """
    Reference: https://andyljones.com/posts/rl-debugging.html#probe
    Two actions, zero observation, one timestep long, action-dependent +1/-1
    reward
    
    If my agent can't learn to pick the better action, there's something wrong
    with either my advantage calculations, my policy loss or my policy update.
    That's three things, but it's easy to work out by hand the expected values
    for each one and check that the values produced by your actual code line
    up with them.
    """
    
    def __init__(self):
        self.done = False
        self.idx = 0
    
    def reset(self):
        return np.array([0.0])
    
    def step(self, action: Union[np.array, int]):
        if isinstance(action, int):
            # Discrete Actions {0, 1}
            assert action in {0, 1}
            action = float(action * 2 - 1)
        else:
            # Continuous Actions
            assert -1 <= action[0] <= 1
            action = action[0]
        
        reward = float(action)
        return np.array([0.0]), reward, True, {}
    
    def render(self):
        pass


class DebugEnv5(gym.Env):
    """
    Reference: https://andyljones.com/posts/rl-debugging.html#probe
    Two actions, random +1/-1 observation, one timestep long, action-and-obs
    dependent +1/-1 reward
    
    The policy and value networks interact here, so there's a couple of things
    to verify: that the policy network learns to pick the right action in each
    of the two states, and that the value network learns that the value of
    each state is +1. If everything's worked up until now, then if - for
    example - the value network fails to learn here, it likely means your
    batching process is feeding the value network stale experience.
    """
    
    def __init__(self, discrete_actions: bool = False):
        self.done = False
        self.discrete_actions = discrete_actions
        self.latest_obs = None
    
    def _get_obs(self):
        return np.array([np.random.choice([-1.0, 1.0])])
    
    def reset(self):
        self.done = False
        self.latest_obs = self._get_obs()
        return self.latest_obs
    
    def step(self, action: np.array):
        if isinstance(action, int):
            # Discrete Actions {0, 1}
            assert action in {0, 1}
            action = float(action * 2 - 1)
        else:
            # Continuous Actions
            assert -1 <= action[0] <= 1
            action = action[0]

        reward = float(action) * self.latest_obs[0]
        self.latest_obs = self._get_obs()
        next_state = self.latest_obs
        done = True
        return next_state, reward, done, {}
    
    def render(self):
        pass
