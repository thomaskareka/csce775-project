import torch
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

from game_env.mario_env import make_env
from utils import get_device
from utils.logger import Logger
from utils.metrics import ExperimentResults, EpisodeMetrics
from utils.metrics_aggregator import MetricsAggregator
from .setup import set_seed, build_runner, infer_model_config

class EvalRunner:
    def __init__(self, config:dict):
        self.config = config
        self.device = get_device()
        
        self.seed = config.get("seed", 1)
        set_seed(self.seed)
        self.num_envs = config["training"].get("num_envs", 1)

        self.env = make_env(config, num_envs=self.num_envs, seed=self.seed, force_render=self.num_envs==1)

        self.config["model"] = infer_model_config(self.config, self.env, self.num_envs)

        self.model, self.algorithm = build_runner(self.config, self.env, self.device)
        self.algorithm.epsilon = 0.0 # no exploration during evaluation
    

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, num_envs: int = 1):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        config = checkpoint["config"]
        config["training"]["num_envs"] = num_envs

        runner = cls(config)
        runner.model.load_state_dict(checkpoint["model_state"])
        runner.algorithm.load_state_dict(checkpoint["algo_state"])

        runner.model.eval()
        return runner
    
    def evaluate(self, num_episodes = 10, num_envs = 1, log_results = True, directory = None):
        if num_envs == 1:
            return self.single_env_evaluate(num_episodes) # simpler evaluation, mainly here for visual display and quicker testing
        
        finished_episodes: list[EpisodeMetrics] = []
        episode_idx = 0
        
        logger = Logger(self.config, exp_dir=directory)

        pbar = tqdm(total=num_episodes, desc="Evaluating", unit="episode")

        obs, _ = self.env.reset(seed=self.seed)

        while len(finished_episodes) < num_episodes:
            remaining = num_episodes - len(finished_episodes)
            batch_size = min(self.num_envs, remaining)

            active_episodes = [EpisodeMetrics() for _ in range(num_envs)] # track data from each episode
            active_mask = np.zeros(self.num_envs, dtype=bool)
            active_mask[:batch_size] = True

            wave_done = np.zeros(self.num_envs, dtype=bool)

            while not np.all(wave_done[:batch_size]):
                actions = self.algorithm.choose_action(obs)
                obs, rewards, terminated, truncated, info = self.env.step(actions)
                info = [
                    {key: value[i] for key, value in info.items()}
                    for i in range(num_envs)
                ]

                dones = np.logical_or(terminated, truncated)
                for i in range(self.num_envs):
                    if not active_mask[i]: #envs auto reset, if its done ignore it until all other envs are finished
                        #to prevent biasing towards shorter episodes
                        continue
                    active_episodes[i].update_from_info(info[i], rewards[i])
                    if dones[i]:
                        if log_results:
                            logger.log_episode_metric(
                                episode_idx=episode_idx,
                                metrics={
                                    "total_reward": active_episodes[i].total_reward,
                                    "episode_length": active_episodes[i].episode_length,
                                    "max_x_position": active_episodes[i].max_x_position,
                                    "score": active_episodes[i].score,
                                    "coins": active_episodes[i].coins,
                                },
                                group="EvalEpisodes"
                            )

                        episode_idx += 1
                        finished_episodes.append(active_episodes[i])

                        active_mask[i] = False
                        wave_done[i] = True
                        pbar.update(1)


        pbar.close()

        # Compute summary statistics
        returns = np.array([ep.total_reward for ep in finished_episodes], dtype=np.float32)
        lengths = np.array([ep.episode_length for ep in finished_episodes], dtype=np.float32)
        max_x_positions = np.array([ep.max_x_position for ep in finished_episodes], dtype=np.float32)
        scores = np.array([ep.score for ep in finished_episodes], dtype=np.float32)
        coins = np.array([ep.coins for ep in finished_episodes], dtype=np.float32)

        
        summary_metrics = {
            "num_episodes": returns.size,
            "mean_return": float(returns.mean()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "std_return": float(returns.std()),
            "mean_length": float(lengths.mean()),
            "mean_max_x": float(max_x_positions.mean()),
            "mean_score": float(scores.mean()),
            "mean_coins": float(coins.mean()),
            "max_return": float(returns.max()),
            "max_length": float(lengths.max()),
            "max_max_x": float(max_x_positions.max()),
            "max_score": float(scores.max()),
            "max_coins": float(coins.max()),
        }
        metrics = {
            "summary": summary_metrics,
            "episodes": {
                "returns": returns.tolist(),
                "lengths": lengths.tolist(),
                "max_x_positions": max_x_positions.tolist(),
                "scores": scores.tolist(),
                "coins": coins.tolist(),
            },
        }

        if log_results:
            logger.finalize_results(metrics)
            logger.close()

        print(
            f"{metrics['summary']['num_episodes']} episodes | "
            f"mean_return={metrics['summary']['mean_return']:.3f} | "
            f"std_return={metrics['summary']['std_return']:.3f} | "
            f"mean_length={metrics['summary']['mean_length']:.1f} | "
            f"mean_max_x={metrics['summary']['mean_max_x']:.1f} | "
            f"mean_score={metrics['summary']['mean_score']:.1f} | "
            f"mean_coins={metrics['summary']['mean_coins']:.1f}"
        )
        return metrics


    def single_env_evaluate(self, num_episodes=10):
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

                total_reward += np.clip(reward, -1, 1)
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