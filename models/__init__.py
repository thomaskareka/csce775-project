MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def build_model(model_config):
    print(f"building model {model_config}")
    model = model_config["type"]

    if model not in MODEL_REGISTRY:
        raise ValueError(f"tried building invalid model type {model}")
    # calls the class constructor using all parameters from model config
    return MODEL_REGISTRY[model](**{
        key: value for key, value in model_config.items() if key != "type"
    })


#auto importer from chatgpt
import os
import pkgutil
import importlib

package_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    if module_name != "__init__":
        importlib.import_module(f"{__name__}.{module_name}")