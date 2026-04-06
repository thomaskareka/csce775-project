"""
Logger for experiment tracking with TensorBoard and JSON persistence.
Provides context manager support and batched episode metric logging.
Usage:
    # Initialize with config and experiment directory
    logger = Logger(config, exp_dir="experiments/my_exp")
    
    # Log scalar metrics during training
    logger.log_metrics(step=100, metrics={"loss": 0.5, "reward": 150})
    
    # Log episode-level metrics
    logger.log_episode_metric(episode_idx=0, metrics={"total_reward": 200, "length": 100})
    
    # Use as context manager to auto-close
    with Logger(config) as logger:
        logger.log_metrics(step=1, metrics={"loss": 0.1})
    # Writer automatically closed on exit
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import ExperimentResults


class Logger:
    """Manages TensorBoard logging and experiment result persistence."""
    
    def __init__(self, config: Dict[str, Any], exp_dir: Optional[Path] = None):
        """
        Initialize logger with experiment directory.
        
        Args:
            config: Configuration dictionary with experiment metadata
            exp_dir: Path to experiment directory (optional, can be set later)
        """
        self.exp_name = config.get('experiment_name', 'default_exp')
        
        # Support both new exp_dir parameter and legacy log_dir from config
        if exp_dir is not None:
            self.exp_dir = Path(exp_dir)
        else:
            self.exp_dir = Path(config.get('log_dir', f"experiments/{self.exp_name}"))
        
        self.log_dir = str(self.exp_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self._is_closed = False

    def log_metrics(self, step: int, metrics: Dict[str, float], group: str = "Train") -> None:
        """
        Log scalar metrics to TensorBoard.
        
        Args:
            step: Training step number
            metrics: Dictionary of metric_name -> value
            group: Prefix group for metrics (e.g., "Train", "Eval")
        """
        if self._is_closed:
            return
        
        for key, value in metrics.items():
            self.writer.add_scalar(f"{group}/{key}", value, step)

    def log_episode_metric(self, episode_idx: int, metrics: Dict[str, float], group: str = "Episodes") -> None:
        """
        Log episode-level metric to TensorBoard.
        
        Args:
            episode_idx: Episode number
            metrics: Dictionary of metric_name -> value
            group: Prefix group for metrics
        """
        if self._is_closed:
            return
        
        for key, value in metrics.items():
            self.writer.add_scalar(f"{group}/{key}", value, episode_idx)

    def log_hyperparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        Link hyperparameters to final metrics in TensorBoard hparams tab.
        
        Args:
            hparams: Hyperparameter dictionary
            metrics: Final metric values
        """
        if self._is_closed:
            return
        
        self.writer.add_hparams(hparams, metrics)

    def finalize_results(self, results) -> None:
        """
        Persist final experiment results to JSON file.
        
        Args:
            results: ExperimentResults object containing summary statistics
        """
        if self._is_closed:
            return
        
        results_path = self.exp_dir / "results.json"
        if not isinstance(results, dict):
            results = results.to_dict()
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    def close(self) -> None:
        """Close TensorBoard writer and cleanup resources."""
        if not self._is_closed:
            self.writer.close()
            self._is_closed = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit—ensure logger is closed."""
        self.close()
        return False