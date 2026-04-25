import gymnasium as gym
from . import register_reward

@register_reward("simple_screen_scroll")
class ScreenScrollReward(gym.Wrapper):
    def __init__(self, env, weight = 0.1):
        super().__init__(env)
        self.value = weight

# just if the screen is scrolling, no other logic
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # scrolling = self.env.unwrapped.get_ram()[0x0775]
        scrolling = info.get('scrolling', 0)
        print(scrolling)
        if scrolling > 17:
            reward += self.value
        return obs, reward, terminated, truncated, info
