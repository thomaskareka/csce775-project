# based on https://github.com/farama-foundation/stable-retro/blob/master/stable_retro/examples/discretizer.py

import gymnasium, stable_retro
import numpy as np
from . import register_observation

COMBOS = [
    [],
    ["LEFT"],
    ["RIGHT"],
    ["DOWN"],
    ["UP"],
    ["RIGHT", "A"],
    ["RIGHT", "B"],
    ["RIGHT", "A", "B"],
    ["LEFT", "A"],
    ["LEFT", "B"],
    ["LEFT", "A", "B"],
    ["A"],
    ["B"],
]


@register_observation("limited_action_space")
class LimitedActionSpace(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gymnasium.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in COMBOS:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gymnasium.spaces.Discrete(len(self._decode_discrete_action))
    
    def action(self, action):
        return self._decode_discrete_action[action].copy()

