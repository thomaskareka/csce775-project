ALGORITHM_REGISTRY = {}

def register_algorithm(name):
    def decorator(cls):
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator


def get_algorithm(name):
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"unknown algorithm: {name}")
    return ALGORITHM_REGISTRY[name]

#auto importer from chatgpt
import os
import pkgutil
import importlib

package_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    if module_name != "__init__":
        importlib.import_module(f"{__name__}.{module_name}")