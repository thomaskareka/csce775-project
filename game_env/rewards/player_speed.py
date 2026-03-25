import gymnasium as gym
from . import register_reward

@register_reward("player_speed")
class PlayerSpeedReward(gym.Wrapper):
    def __init__(self, env, weight = 0.01):
        super().__init__(env)
        self.value = weight
        self.last_pos = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        speed = self.env.unwrapped.get_ram()[0x0057]
        if(speed > 0x00 and speed <= 0x028): #player x speed is capped at 40, 0xD8<0 is moving left, 0x28>0 is moving right
            reward += self.value * speed
        return obs, reward, terminated, truncated, info
