import gymnasium as gym
from . import register_reward

@register_reward("screen_scroll_amount")
class ScreenScrollReward(gym.Wrapper):
    def __init__(self, env, weight = 0.1):
        super().__init__(env)
        self.value = weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scrolling = info.get('scrolling', 0)
        print(scrolling)
        r = (scrolling - 16) * self.value
        return obs, reward, terminated, truncated, info
