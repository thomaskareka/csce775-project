import gymnasium as gym
from . import register_reward

@register_reward("screen_scroll")
class ScreenScrollReward(gym.Wrapper):
    def __init__(self, env, weight = 0.1):
        super().__init__(env)
        self.value = weight
        self.last_pos = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # scrolling = self.env.unwrapped.get_ram()[0x0775]
        x_low = info.get('xscrollLo', 0)
        x_high = info.get('xscrollHi', 0)
        pos = (x_high << 8) | x_low
        if(pos > self.last_pos):
            reward += self.value
        self.last_pos = pos
        return obs, reward, terminated, truncated, info
