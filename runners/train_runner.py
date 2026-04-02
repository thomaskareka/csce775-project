import torch
from pathlib import Path
from datetime import datetime

from game_env.mario_env import make_env
from utils import get_device
from utils.logger import Logger
from utils.metrics import ExperimentResults
from .setup import set_seed, build_runner, infer_model_config

class TrainRunner:
    def __init__(self, config:dict):
        self.config = config
        self.device = get_device()
        
        self.seed = config.get("seed", 1)
        set_seed(self.seed)

        self.exp_dir = self._create_experiment_directory()
        
        # Initialize logger for TensorBoard and results persistence
        self.logger = Logger(self.config, self.exp_dir)
        
        # Tracking for final results
        self.training_metrics = None

        self.num_envs = config["training"].get("num_envs", 1)
        self.env = make_env(config, num_envs=self.num_envs, seed=self.seed)

        self.config["model"] = infer_model_config(self.config, self.env, self.num_envs)

        self.model, self.algorithm = build_runner(self.config, self.env, self.device)
    
    def train(self):
        print(f"starting training on {self.device}, {self.exp_dir}")
        self.training_metrics = self.algorithm.train(
            self.config["training"]["total_steps"], 
            callback=self.save_checkpoint,
            logger=self.logger
        )
        self.save_checkpoint("final.pt")
        
        # Create and finalize experiment results
        self._finalize_results()
        self.logger.close()

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

    def _finalize_results(self):
        """Create and save ExperimentResults from training metrics."""
        # If algorithm returns metrics dict, use it; otherwise create default results
        if isinstance(self.training_metrics, dict):
            # Algorithms can optionally return a metrics dict with training statistics
            metrics_dict = self.training_metrics
        else:
            # Default empty dict if algorithm doesn't return metrics
            metrics_dict = {}
        
        # Get last loss from logger if available
        last_loss = metrics_dict.get("last_loss", 0.0)
        
        # Create experiment results
        results = ExperimentResults(
            experiment_name=self.exp_dir.name,
            algorithm=self.config["algorithm"],
            model_type=self.config["model"]["type"],
            seed=self.seed,
            timestamp=datetime.now().isoformat(),
            total_steps=self.config["training"]["total_steps"],
            final_loss=last_loss,
            num_episodes=metrics_dict.get("num_episodes", 0),
            mean_return=metrics_dict.get("mean_return", 0.0),
            std_return=metrics_dict.get("std_return", 0.0),
            mean_episode_length=metrics_dict.get("mean_episode_length", 0.0),
        )
        
        # Add Mario-specific metrics to extra_metrics
        if "mean_max_x_position" in metrics_dict:
            results.extra_metrics["mean_max_x_position"] = metrics_dict["mean_max_x_position"]
        if "mean_max_score" in metrics_dict:
            results.extra_metrics["mean_max_score"] = metrics_dict["mean_max_score"]
        if "mean_velocity" in metrics_dict:
            results.extra_metrics["mean_velocity"] = metrics_dict["mean_velocity"]
        if "success_rate" in metrics_dict:
            results.extra_metrics["success_rate"] = metrics_dict["success_rate"]
        
        # Finalize in logger (saves to JSON and logs hyperparams to TensorBoard)
        self.logger.finalize_results(results)

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