import torch
from pathlib import Path
import time
import numpy as np

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

        self.num_envs = 1
        self.config["training"]["num_envs"] = 1
        self.env = make_env(config, num_envs=self.num_envs, seed=self.seed, force_render=True)

        self.config["model"] = infer_model_config(self.config, self.env, self.num_envs)

        self.model, self.algorithm = build_runner(self.config, self.env, self.device)
        self.algorithm.epsilon = 0.0 # no exploration during evaluation
    

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
    
    def append_eval_results(self, experiment_dir: Path):
        """Load training results, run evaluation, and append evaluation metrics to results.json with TensorBoard logging."""
        results_path = experiment_dir / "results.json"
        
        if not results_path.exists():
            print(f"No results.json found at {results_path}. Skipping eval results append.")
            return
        
        # Initialize logger pointing to the same experiment directory
        logger = Logger(self.config, exp_dir=experiment_dir)
        
        # Load existing training results
        try:
            results = ExperimentResults.from_json(str(results_path))
        except Exception as e:
            print(f"Error loading results.json: {e}")
            logger.close()
            return
        
        # Run evaluation with metrics aggregation
        agg = MetricsAggregator(batch_size=10)
        episode_rewards = []
        episode_lengths = []
        episode_max_positions = []
        episode_scores = []
        
        for ep_idx in range(10):  # Default 10 eval episodes
            obs, _ = self.env.reset(seed=self.seed)
            done = False
            total_reward = 0.0
            length = 0
            max_x_position = 0.0
            max_score = 0.0
            terminated = False
            
            while not done:
                action = self.algorithm.choose_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += np.clip(reward, -1, 1)
                length += 1
                
                # Track Mario metrics
                x_pos = ((info.get('xscrollHi', 0) << 8) | info.get('xscrollLo', 0))
                max_x_position = max(max_x_position, x_pos)
                max_score = max(max_score, info.get('score', 0.0))
            
            # Calculate velocity and level completion
            velocity = max_x_position / length if length > 0 else 0.0
            level_completed = (terminated and max_x_position > 100)
            
            # Track metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            episode_max_positions.append(max_x_position)
            episode_scores.append(max_score)
            
            # Add to aggregator for batched logging
            ep_metrics = EpisodeMetrics(
                episode_idx=ep_idx,
                total_reward=total_reward,
                episode_length=length,
                max_x_position=max_x_position,
                max_score=max_score,
                velocity=velocity,
                level_completed=level_completed
            )
            agg.add_episode(ep_metrics)
        
        # Get eval summary statistics
        eval_summary = {
            "mean_return": float(np.mean(episode_rewards)),
            "std_return": float(np.std(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_max_x_position": float(np.mean(episode_max_positions)),
            "mean_max_score": float(np.mean(episode_scores))
        }
        
        # Add velocity and success rate from aggregator
        agg_summary = agg.get_summary()
        if "mean_velocity" in agg_summary:
            eval_summary["mean_velocity"] = agg_summary["mean_velocity"]
        if "success_rate" in agg_summary:
            eval_summary["success_rate"] = agg_summary["success_rate"]
        
        # Log evaluation metrics to TensorBoard
        logger.log_metrics(step=0, metrics=eval_summary, group="Evaluation")
        
        # Append eval metrics to training results
        results.eval_mean_return = eval_summary["mean_return"]
        results.eval_std_return = eval_summary["std_return"]
        results.eval_num_episodes = len(episode_rewards)
        
        # Append Mario-specific metrics
        results.extra_metrics["eval_mean_max_x_position"] = eval_summary["mean_max_x_position"]
        results.extra_metrics["eval_mean_max_score"] = eval_summary["mean_max_score"]
        if "mean_velocity" in eval_summary:
            results.extra_metrics["eval_mean_velocity"] = eval_summary["mean_velocity"]
        if "success_rate" in eval_summary:
            results.extra_metrics["eval_success_rate"] = eval_summary["success_rate"]
        
        # Save updated results
        try:
            results.to_json(str(results_path))
            print(f"Evaluation results appended to {results_path}")
            print(f"  Episodes: {len(episode_rewards)} | "
                  f"Reward: {eval_summary['mean_return']:.2f}±{eval_summary['std_return']:.2f} | "
                  f"Length: {eval_summary['mean_episode_length']:.1f} | "
                  f"Max X Pos: {eval_summary['mean_max_x_position']:.0f} | "
                  f"Max Score: {eval_summary['mean_max_score']:.0f} | "
                  f"Velocity: {eval_summary.get('mean_velocity', 0.0):.2f} | "
                  f"Success Rate: {eval_summary.get('success_rate', 0.0):.1%}")
        except Exception as e:
            print(f"Error saving results.json: {e}")
        finally:
            logger.close()

    