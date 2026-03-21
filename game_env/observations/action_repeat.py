import gymnasium, stable_retro
from . import register_observation

@register_observation("action_repeat")
class ActionRepeat(gymnasium.Wrapper):
    def __init__(self, env, count):
        super().__init__(env)
        self.repeat = count
    
    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False

        for i in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            if done or truncated: break
        
        return obs, total_reward, done, truncated, info


