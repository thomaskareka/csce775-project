import gymnasium as gym
from . import register_reward

@register_reward("player_position")
class PlayerPositionReward(gym.Wrapper):
    def __init__(self, env, weight = 0.01):
        super().__init__(env)
        self.value = weight
        self.last_pos = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.get_ram()
        pos = (int(ram[0x006D]) << 8 | int(ram[0x0086]))
        if pos > self.last_pos:
            reward += self.value * (pos - self.last_pos)
        self.last_pos = pos
        return obs, reward, terminated, truncated, info
