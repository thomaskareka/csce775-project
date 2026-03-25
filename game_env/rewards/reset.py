import gymnasium as gym
from . import register_reward

@register_reward("reset")
class ResetReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        return obs, 0, terminated, truncated, info
