import gymnasium as gym
from . import register_reward

@register_reward("death_penalty")
class DeathPenalty(gym.Wrapper):
    def __init__(self, env, penalty = -1000):
        super().__init__(env)
        self.penalty = penalty
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated: reward += self.penalty

        return obs, reward, terminated, truncated, info
