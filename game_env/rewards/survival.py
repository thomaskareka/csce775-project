import gymnasium as gym
from . import register_reward

@register_reward("survival")
class SurvivalReward(gym.Wrapper):
    def __init__(self, env, value = 0.0):
        super().__init__(env)
        self.value = value

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward + self.value, terminated, truncated, info
