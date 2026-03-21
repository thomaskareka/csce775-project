import gymnasium, stable_retro
import numpy as np
from collections import deque
from . import register_observation

@register_observation("frame_buffer")
class FrameBuffer(gymnasium.Wrapper):
    def __init__(self, env: stable_retro.RetroEnv, count):
        super().__init__(env)
        self.count = count
        self.frames = deque(maxlen=count)
        old_shape = env.observation_space.shape

        self.observation_space = gymnasium.spaces.Box(
            low=0, 
            high=255,
            shape=(count, *old_shape),
            dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for i in range(self.count):
            self.frames.append(obs)
        return np.stack(self.frames), info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames), reward, done, truncated, info