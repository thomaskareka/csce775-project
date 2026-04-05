"""
Metrics dataclasses for structured, type-safe tracking of training and evaluation results.
Provides clean JSON serialization for experiment result storage.
Usage:
    # Track per-step metrics during training
    step_metrics = TrainingMetrics(step=100, loss=0.5, epsilon=0.1, reward=50)
    
    # Track per-episode results
    ep_metrics = EpisodeMetrics(episode_idx=0, total_reward=200, episode_length=100)
    
    # Aggregate final results for persistence
    final_results = ExperimentResults(
        experiment_name="dqn_exp_1",
        algorithm="dqn",
        model_type="simple_linear",
        seed=42,
        timestamp="2026-04-02 12:00:00",
        total_steps=10000,
        final_loss=0.1,
        num_episodes=50,
        mean_return=150.5,
        std_return=20.0,
        mean_episode_length=100.0
    )
    # Save to JSON
    final_results.to_json(Path("experiments/dqn_exp_1/results.json"))"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path


@dataclass
class TrainingMetrics:
    """Per-step metrics collected during training."""
    step: int
    loss: float
    epsilon: float
    reward: float


@dataclass
class EpisodeMetrics:
    """Per-episode summary metrics collected during training."""
    total_reward: float = 0.0
    episode_length: int = 0
    max_x_position: int = 0
    score: int = 0
    coins: int = 0

    def update_from_info(self, info, reward):
        x_pos = ((info.get('xscrollHi', 0) << 8) | info.get('xscrollLo', 0))
        self.max_x_position = max(self.max_x_position, x_pos)
        self.score = info.get('score', self.score)
        self.coins = info.get('coins', self.coins)
        self.total_reward += reward
        self.episode_length += 1
    
    def __str__(self):
        return f"Reward: {self.total_reward:.2f}, Length: {self.episode_length}, Max X: {self.max_x_position}, Score: {self.score}, Coins: {self.coins}"




@dataclass
class ExperimentResults:
    """Final experiment summary for JSON persistence."""
    experiment_name: str
    algorithm: str
    model_type: str
    seed: int
    timestamp: str
    
    # Training metrics
    total_steps: int
    final_loss: float
    num_episodes: int
    mean_return: float
    std_return: float
    mean_episode_length: float
    
    # Evaluation metrics (optional)
    eval_mean_return: Optional[float] = None
    eval_std_return: Optional[float] = None
    eval_num_episodes: Optional[int] = None
    
    # Extra metrics for ablation studies
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, filepath: Path) -> None:
        """Write results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'ExperimentResults':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
