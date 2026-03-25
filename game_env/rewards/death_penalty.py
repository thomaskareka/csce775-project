import gymnasium as gym
from . import register_reward

@register_reward("death_penalty")
class DeathPenalty(gym.Wrapper):
    def __init__(self, env, penalty = -1000):
        super().__init__(env)
        self.penalty = penalty
        self.prev_lives = 0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get('lives', self.prev_lives)
        if(lives < self.prev_lives):
            reward += self.penalty
            # print(f"death penalty applied: {reward}")
        self.prev_lives = lives

        return obs, reward, terminated, truncated, info
