import yaml
from pathlib import Path
from copy import deepcopy

def _deep_merge(base, override):
    out = deepcopy(base)

    for key, value in override.items():
        if (key in out and isinstance(out[key], dict) and isinstance(value, dict)):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)

    return out

def load_config(path: str | Path):
    path = Path(path)

    with open(path, "r") as file:
        config = yaml.safe_load(file)
    
    include = config.pop("include", None)
    if include is not None:
        base_path = (path.parent / include).resolve()
        with open(base_path, "r") as file:
            base_config = yaml.safe_load(file)
        
        config = _deep_merge(base_config, config)
    return config