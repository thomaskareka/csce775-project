REWARD_REGISTRY = {}

def register_reward(name):
    def decorator(cls):
        REWARD_REGISTRY[name] = cls
        return cls
    return decorator

def build_reward_pipeline(env, config):
    pipeline = config.get("reward", {}).get("pipeline", [])

    for step in pipeline:
        name = step["name"]
        params = {k: v for k, v in step.items() if k != "name"}

        if name not in REWARD_REGISTRY:
            raise ValueError(f"unknown reward item {name}")

        env = REWARD_REGISTRY[name](env, **params)

    return env

import os
import pkgutil
import importlib

package_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    if module_name != "__init__":
        importlib.import_module(f"{__name__}.{module_name}")