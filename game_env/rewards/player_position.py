import gymnasium as gym
from . import register_reward
from utils.from_ram import get_x_pos_adjusted

@register_reward("player_position")
class PlayerPositionReward(gym.Wrapper):
    def __init__(self, env, weight = 0.01):
        super().__init__(env)
        self.value = weight
        #from frame 0 to frame 1, player is warped from 0 -> 40
        #having this as the initial max prevents the player from getting a reward from just existing at the start
        self.max_pos = 41
        self.last_level = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.get_ram()
        pos = get_x_pos_adjusted(ram)

        level_id = int(ram[0x075F]) * 4 + int(ram[0x0760])
        if level_id != self.last_level:
            self.last_level = level_id
            self.max_pos = 41 + level_id * 10000
            reward = 0
            return obs, reward, terminated, truncated, info


        if pos > self.max_pos:
            delta = min(pos - self.max_pos, 5)
            reward += self.value * delta
            self.max_pos = pos
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.max_pos = 41
        self.last_level = 0
        return self.env.reset(**kwargs)
