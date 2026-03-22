import torch
from pathlib import Path
from datetime import datetime
import numpy as np

from game_env.mario_env import make_env
from utils import get_device
from .setup import set_seed, build_runner, infer_model_config

class EvalRunner:
    def __init__(self, config:dict):
        self.config = config
        self.device = get_device()
        
        self.seed = config.get("seed", 1)
        set_seed(self.seed)

        self.num_envs = 1
        self.config["training"]["num_envs"] = 1
        self.env = make_env(config, num_envs=self.num_envs, seed=self.seed, force_render=True)

        self.config["model"] = infer_model_config(self.config, self.env, self.num_envs)

        self.model, self.algorithm = build_runner(self.config, self.env, self.device)
    

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        config = checkpoint["config"]
        
        runner = cls(config)
        runner.model.load_state_dict(checkpoint["model_state"])
        runner.algorithm.load_state_dict(checkpoint["algo_state"])

        runner.model.eval()
        return runner
    
    def evaluate(self, num_episodes = 10):
        episode_rewards = []
        episode_lengths = []

        for i in range(num_episodes):
            obs, _ = self.env.reset(seed = self.seed)
            done = False
            total_reward = 0.0
            length = 0
            
            while not done:
                action = self.algorithm.choose_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                total_reward += np.sign(reward).astype(np.float32)
                length += 1
            episode_rewards.append(total_reward)
            episode_lengths.append(length)

        metrics = {
            "num_episodes": len(episode_rewards),
            "mean_return": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "std_return": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "returns": episode_rewards,
            "lengths": episode_lengths,
        }

        print(
            f"{metrics['num_episodes']} episodes | "
            f"mean_return={metrics['mean_return']:.3f} | "
            f"std_return={metrics['std_return']:.3f} | "
            f"mean_length={metrics['mean_length']:.1f}"
        )
        return metrics

    
    