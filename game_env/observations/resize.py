import gymnasium, stable_retro, cv2
import numpy as np
from . import register_observation

@register_observation("resize")
class ResizeObservation(gymnasium.ObservationWrapper):
    def __init__(self, env: stable_retro.RetroEnv, width, height):
        super().__init__(env)
        self.width = width
        self.height = height

        old_space = env.observation_space
        old_shape = old_space.shape

        if len(old_shape) == 3:
            channels = old_shape[2]
            new_shape = (height, width, channels)
        else:
            new_shape = (height, width)

        self.observation_space = gymnasium.spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=new_shape,
            dtype=old_space.dtype
        )
        
    def observation(self, obs):
        return cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)