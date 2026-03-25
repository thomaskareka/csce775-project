import gymnasium as gym
from . import register_reward

@register_reward("new_screen")
class NewScreenReward(gym.Wrapper):
    def __init__(self, env, weight = 0.01):
        super().__init__(env)
        self.value = weight
        self.last_pos = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = self.env.unwrapped.get_ram()[0x006D]
        if pos > self.last_pos:
            reward += self.value
        self.last_pos = pos
        return obs, reward, terminated, truncated, info
