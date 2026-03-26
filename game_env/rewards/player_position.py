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
        delta = min(pos - self.last_pos, 5)
        if pos > self.last_pos and self.last_pos != 0 and self.max_pos < pos:
            reward += self.value * delta
        self.last_pos = pos
        if pos > self.max_pos:
            self.max_pos = pos
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.last_pos = 0
        self.max_pos = 0
        return self.env.reset(**kwargs)
