import torch
from pathlib import Path
from datetime import datetime

from game_env.mario_env import make_env
from utils import get_device
from .setup import set_seed, build_runner, infer_model_config

class TrainRunner:
    def __init__(self, config:dict):
        self.config = config
        self.device = get_device()
        
        self.seed = config.get("seed", 1)
        set_seed(self.seed)

        self.exp_dir = self._create_experiment_directory()

        self.num_envs = config["training"].get("num_envs", 1)
        self.env = make_env(config, num_envs=self.num_envs, seed=self.seed)

        self.config["model"] = infer_model_config(self.config, self.env, self.num_envs)

        self.model, self.algorithm = build_runner(self.config, self.env, self.device)
    
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

        print(config)
        runner = cls(config)
        runner.model.load_state_dict(checkpoint["model_state"])
        runner.algorithm.load_state_dict(checkpoint["algo_state"])

        return runner

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