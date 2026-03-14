import gymnasium
import numpy as np
from . import register_observation

@register_observation("entire_ram")
class EntireRamObservation(gymnasium.ObservationWrapper):
    RAM_END = 0x800
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(self.RAM_END,),
            dtype=np.uint8
        )
    def observation(self, observation):
        return self.env.unwrapped.get_ram()[:self.RAM_END]