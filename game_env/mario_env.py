import stable_retro
import gymnasium

def make_env(env_config: dict, seed:int = 1) -> stable_retro.RetroEnv:
    kwargs = {k: v for k, v in env_config.items() if v is not None}

    env = stable_retro.make(**kwargs)
    env.reset(seed=seed)
    return env