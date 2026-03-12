import os, json, time, torch, random, yaml
import numpy as np
from pathlib import Path
from datetime import datetime

from algorithms import get_algorithm
from game_env.mario_env import make_env
from models import build_model

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
        self.algorithm.train(self.config["training"]["total_steps"])
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
        checkpoint = torch.load(checkpoint_path)

        config = checkpoint["config"]
        runner = cls(config)

        runner.model.load_state_dict(checkpoint["model_state"])
        runner.algorithm.load_state_dict(checkpoint["algo_state"])

        return runner

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