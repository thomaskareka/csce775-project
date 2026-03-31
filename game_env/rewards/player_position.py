import gymnasium as gym
from . import register_reward

@register_reward("player_position")
class PlayerPositionReward(gym.Wrapper):
    def __init__(self, env, weight = 0.01):
        super().__init__(env)
        self.value = weight
        #from frame 0 to frame 1, player is warped from 0 -> 40
        #having this as the initial max prevents the player from getting a reward from just existing at the start
        self.max_pos = 41

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.get_ram()
        pos = (int(ram[0x006D]) << 8 | int(ram[0x0086]))

        if pos > self.max_pos:
            delta = min(pos - self.max_pos, 5)
            reward += self.value * delta
            self.max_pos = pos
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.max_pos = 41
        return self.env.reset(**kwargs)
