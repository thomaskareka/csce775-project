"""
Efficient metrics aggregation with batching to minimize logging overhead.
Buffers episode metrics and periodically flushes to prevent I/O bottlenecks.
Usage:
    # Initialize with batch size for flushing
    agg = MetricsAggregator(batch_size=10)
    
    # Add episodes as they complete during training
    for episode_idx in range(num_episodes):
        ep_metrics = EpisodeMetrics(episode_idx=episode_idx, total_reward=150, episode_length=100)
        agg.add_episode(ep_metrics)  # Flushes automatically when buffer reaches batch_size
    
    # Get summary statistics at any time
    summary = agg.get_summary()
    # Returns: {"mean_return": 150.0, "std_return": 20.0, "mean_episode_length": 100.0, "num_episodes": 100}
    
    # Manually flush remaining buffered episodes if needed
    remaining = agg.flush()"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from utils.metrics import EpisodeMetrics


@dataclass
class MetricsBuffer:
    """Buffer for batched metric logging."""
    episode_indices: List[int] = field(default_factory=list)
    total_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    max_x_positions: List[Optional[float]] = field(default_factory=list)
    final_epsilons: List[Optional[float]] = field(default_factory=list)
    extra_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    def add(self, episode: EpisodeMetrics) -> None:
        """Add episode metrics to buffer."""
        self.episode_indices.append(episode.episode_idx)
        self.total_rewards.append(episode.total_reward)
        self.episode_lengths.append(episode.episode_length)
        self.max_x_positions.append(episode.max_x_position)
        self.final_epsilons.append(episode.final_epsilon)
        self.extra_metrics.append(episode.extra_metrics)
    
    def is_full(self, batch_size: int) -> bool:
        """Check if buffer has reached batch size."""
        return len(self.episode_indices) >= batch_size
    
    def flush(self) -> List[EpisodeMetrics]:
        """Retrieve all buffered episodes and clear buffer."""
        episodes = [
            EpisodeMetrics(
                episode_idx=idx,
                total_reward=reward,
                episode_length=length,
                max_x_position=max_x,
                final_epsilon=epsilon,
                extra_metrics=extra
            )
            for idx, reward, length, max_x, epsilon, extra in zip(
                self.episode_indices,
                self.total_rewards,
                self.episode_lengths,
                self.max_x_positions,
                self.final_epsilons,
                self.extra_metrics
            )
        ]
        self.clear()
        return episodes
    
    def clear(self) -> None:
        """Clear all buffered data."""
        self.episode_indices.clear()
        self.total_rewards.clear()
        self.episode_lengths.clear()
        self.max_x_positions.clear()
        self.final_epsilons.clear()
        self.extra_metrics.clear()


class MetricsAggregator:
    """
    Efficiently aggregates training metrics with batched logging.
    
    Buffers episode metrics to reduce I/O overhead. Computes running statistics
    without storing all episodes in memory.
    """
    
    def __init__(self, batch_size: int = 10):
        """
        Initialize aggregator.
        
        Args:
            batch_size: Number of episodes to buffer before flushing
        """
        self.batch_size = batch_size
        self.buffer = MetricsBuffer()
        
        # Running statistics (computed incrementally)
        self.num_episodes = 0
        self.sum_rewards = 0.0
        self.sum_sq_rewards = 0.0
        self.sum_lengths = 0.0
        
        # Mario-specific metrics
        self.sum_max_x_positions = 0.0
        self.sum_max_scores = 0.0
        self.sum_velocities = 0.0
        self.num_levels_completed = 0
        
        # For final summary
        self.all_rewards: List[float] = []
        self.all_lengths: List[int] = []
        self.all_max_x_positions: List[float] = []
        self.all_max_scores: List[float] = []
        self.all_velocities: List[float] = []
        self.all_level_completed: List[bool] = []
    
    def add_episode(self, episode: EpisodeMetrics) -> None:
        """
        Add episode metrics to aggregator. May trigger flush if buffer is full.
        
        Args:
            episode: EpisodeMetrics object to track
        """
        self.buffer.add(episode)
        
        # Update running statistics
        self.num_episodes += 1
        self.sum_rewards += episode.total_reward
        self.sum_sq_rewards += episode.total_reward ** 2
        self.sum_lengths += episode.episode_length
        
        # Track Mario metrics
        if episode.max_x_position is not None:
            self.sum_max_x_positions += episode.max_x_position
            self.all_max_x_positions.append(episode.max_x_position)
        
        if episode.max_score is not None:
            self.sum_max_scores += episode.max_score
            self.all_max_scores.append(episode.max_score)
        
        if episode.velocity is not None:
            self.sum_velocities += episode.velocity
            self.all_velocities.append(episode.velocity)
        
        if episode.level_completed is not None:
            self.all_level_completed.append(episode.level_completed)
            if episode.level_completed:
                self.num_levels_completed += 1
        
        # Store for final stats
        self.all_rewards.append(episode.total_reward)
        self.all_lengths.append(episode.episode_length)
        
        if self.buffer.is_full(self.batch_size):
            self.flush()
    
    def flush(self) -> List[EpisodeMetrics]:
        """
        Retrieve and clear buffered episodes (called internally or manually).
        
        Returns:
            List of buffered EpisodeMetrics
        """
        return self.buffer.flush()
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all episodes seen so far.
        
        Returns:
            Dictionary with mean_return, std_return, mean_episode_length, 
            and Mario-specific metrics (mean_max_x_position, mean_max_score)
        """
        if self.num_episodes == 0:
            return {
                "mean_return": 0.0,
                "std_return": 0.0,
                "mean_episode_length": 0.0,
                "num_episodes": 0
            }
        
        mean_reward = self.sum_rewards / self.num_episodes
        variance = (self.sum_sq_rewards / self.num_episodes) - (mean_reward ** 2)
        std_reward = np.sqrt(max(variance, 0.0))  # Clamp to avoid numerical issues
        mean_length = self.sum_lengths / self.num_episodes
        
        summary = {
            "mean_return": float(mean_reward),
            "std_return": float(std_reward),
            "mean_episode_length": float(mean_length),
            "num_episodes": self.num_episodes
        }
        
        # Include Mario metrics if available
        if self.all_max_x_positions:
            summary["mean_max_x_position"] = float(np.mean(self.all_max_x_positions))
        
        if self.all_max_scores:
            summary["mean_max_score"] = float(np.mean(self.all_max_scores))
        
        if self.all_velocities:
            summary["mean_velocity"] = float(np.mean(self.all_velocities))
        
        if self.all_level_completed:
            success_rate = self.num_levels_completed / len(self.all_level_completed)
            summary["success_rate"] = float(success_rate)
        
        return summary
