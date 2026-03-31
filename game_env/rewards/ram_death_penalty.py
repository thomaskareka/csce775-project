import gymnasium as gym
from . import register_reward

@register_reward("ram_death_penalty")
class RamDeathPenalty(gym.Wrapper):
    def __init__(self, env, penalty = -1.0):
        super().__init__(env)
        self.penalty = penalty
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.get_ram()
        # player state, 0x06 is player dies, 0x0b is dying
        is_dead = ram[0x000E] == 0x06 or ram[0x000E] == 0x0B
        #vertical screen position, 1 = viewport, 0 = above, > 1 = below
        is_below_screen = ram[0x00B5] > 0x01 and ram[0x000E] != 0x03 and ram[0x000E] != 0x07
        terminated = terminated or is_dead or is_below_screen
        if is_dead or is_below_screen:
            reward += self.penalty

        return obs, reward, terminated, truncated, info
