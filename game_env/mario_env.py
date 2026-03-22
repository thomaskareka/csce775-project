import stable_retro
import gymnasium

from game_env.observations import build_observation_pipeline
from game_env.rewards import build_reward_pipeline

def make_single_env(config: dict, rank:int, seed:int = 1) -> stable_retro.RetroEnv:
    def _init():
        env_config = config["env"]
        kwargs = {k: v for k, v in env_config.items() if v is not None}

        env = stable_retro.make(**kwargs, use_restricted_actions = stable_retro.Actions.DISCRETE)
        env.reset(seed=seed + rank)

        env = build_observation_pipeline(env, config)
        env = build_reward_pipeline(env, config)

        return env
    return _init


def make_env(config: dict, seed:int = 1, num_envs: int = 8):
    print(f"using {num_envs} envs")
    if num_envs == 1:
        return make_single_env(config, 0, seed)()
    return gymnasium.vector.AsyncVectorEnv([
        make_single_env(config, i, seed)
        for i in range(num_envs)
    ])