OBSERVATION_REGISTRY = {}

def register_observation(name):
    def decorator(cls):
        OBSERVATION_REGISTRY[name] = cls
        return cls
    return decorator

def build_observation_pipeline(env, config):
    pipeline = config.get("observation", {}).get("pipeline", [])
    for step in pipeline:
        name = step["name"]
        params = {k: v for k, v in step.items() if k != "name"}

        if name not in OBSERVATION_REGISTRY:
            raise ValueError(f"unknown observation item {name}")

        env = OBSERVATION_REGISTRY[name](env, **params)
    return env

import os
import pkgutil
import importlib

package_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    if module_name != "__init__":
        importlib.import_module(f"{__name__}.{module_name}")