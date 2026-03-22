import gymnasium, torch, random
import numpy as np

from algorithms import get_algorithm
from models import build_model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_runner(config, env, device):
    model = build_model(config["model"]).to(device)
    algorithm_class = get_algorithm(config["algorithm"])

    algorithm = algorithm_class(
        model = model,
        env = env,
        device = device,
        config = config["training"]
    )

    return model, algorithm

def infer_model_config(config, env, num_envs):
    if num_envs > 1:
        obs_shape = env.single_observation_space.shape
        action_space = env.single_action_space
    else:
        obs_shape = env.observation_space.shape
        action_space = env.action_space
    
    print(obs_shape, action_space)

    model_cfg = config["model"]

    #input
    if len(obs_shape) == 1:
        model_cfg["input_dim"] = obs_shape[0]
    elif len(obs_shape) == 3:
        c, h, w = obs_shape
        
        if model_cfg["type"] == "simple_linear":
            model_cfg["input_dim"] = h * w * c
        elif model_cfg["type"] == "atari_cnn":
            model_cfg["input_shape"] = obs_shape
        else:
            model_cfg["height"] = h
            model_cfg["width"] = w
            model_cfg["input_channels"] = c
    else:
        raise ValueError(f"bad observation shape {obs_shape}")
    
    #output
    if isinstance(action_space, gymnasium.spaces.Discrete):
        model_cfg["action_type"] = "discrete"
        model_cfg["num_actions"] = action_space.n
    elif isinstance(action_space, gymnasium.spaces.MultiBinary):
        model_cfg["action_type"] = "multibinary"
        model_cfg["num_actions"] = action_space.n
    elif isinstance(action_space, gymnasium.spaces.Box):
        model_cfg["action_type"] = "continuous"
        model_cfg["action_dim"] = action_space.shape[0]

        model_cfg["action_low"] = action_space.low
        model_cfg["action_high"] = action_space.high
    else:
        print(action_space.shape)
        raise ValueError(f"bad action space {action_space}")

    return model_cfg