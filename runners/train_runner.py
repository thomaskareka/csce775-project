import os, json, time, torch, random, yaml, gymnasium
import numpy as np
from pathlib import Path
from datetime import datetime

from algorithms import get_algorithm
from game_env.mario_env import make_env
from models import build_model
from game_env.observations import build_observation_pipeline
from game_env.rewards import build_reward_pipeline

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class TrainRunner:
    def __init__(self, config:dict):
        self.config = config
        self.device = get_device()
        
        self.seed = config.get("seed", 1)
        self._set_seed(self.seed)

        self.exp_dir = self._create_experiment_directory()

        self.env = make_env(config["env"], seed=self.seed)

        self.env = build_observation_pipeline(self.env, self.config)
        self.env = build_reward_pipeline(self.env, self.config)

        self._predect_model_size()

        self.model = build_model(config["model"]).to(self.device)
        algorithm_class = get_algorithm(config["algorithm"])
        self.algorithm = algorithm_class(
            model=self.model,
            env=self.env,
            device=self.device,
            config=config["training"]
        )
    
    def train(self):
        print(f"starting training on {self.device}, {self.exp_dir}")
        self.algorithm.train(self.config["training"]["total_steps"], callback = self.save_checkpoint)
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, name="latest.pt"):
        checkpoint = {
            "config": self.config,
            "model_state": self.model.state_dict(),
            "algo_state": self.algorithm.state_dict()
        }
        save_path = self.exp_dir / name
        torch.save(checkpoint, save_path)
        
        print(f"checkpoint saved to {save_path}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        config = checkpoint["config"]
        runner = cls(config)

        runner.model.load_state_dict(checkpoint["model_state"])
        runner.algorithm.load_state_dict(checkpoint["algo_state"])

        return runner


#TODO: make this better
    def _predect_model_size(self):
        obs_shape = self.env.observation_space.shape
        action_space = self.env.action_space
        print(obs_shape, action_space)

        model_cfg = self.config["model"]

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

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _create_experiment_directory(self):
        base_dir = Path("experiments")
        base_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        algo_name = self.config["algorithm"]
        model_type = self.config["model"]["type"]

        exp_name = f"{algo_name}_{model_type}_{self.seed}_{timestamp}"

        exp_dir = base_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        config_path = exp_dir / "config.yaml"
        import yaml
        with open(config_path, "w") as file:
            yaml.dump(self.config, file)
        
        return exp_dir