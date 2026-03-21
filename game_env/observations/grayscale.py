import gymnasium
import cv2
import numpy as np
from . import register_observation

@register_observation("grayscale")
class GrayScaleObservation(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        old_space = env.observation_space
        h, w, _ = old_space.shape

        self.observation_space = gymnasium.spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=(h, w),
            dtype=old_space.dtype
        )

    def observation(self, obs):
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)