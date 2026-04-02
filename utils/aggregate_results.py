"""
On-demand aggregation utility to generate analysis summaries from experiment results.json files.
Provides filtering, sorting, and CSV export capabilities for ablation studies.
Usage (CLI):
    python aggregate_results.py                                    # Summarize all results
    python aggregate_results.py --algorithm dqn                    # Filter by algorithm
    python aggregate_results.py --model simple_linear              # Filter by model
    python aggregate_results.py --sort-by mean_return --reverse    # Sort by column
    python aggregate_results.py --output results.csv               # Export to CSV
    python aggregate_results.py --algorithm dqn --output dqn.csv   # Chain filters and export

Usage (Python):
    agg = ResultsAggregator(Path("experiments"))
    
    # Filter and chain operations
    dqn_results = agg.filter_by_algorithm("dqn")
    dqn_linear = dqn_results.filter_by_model("simple_linear")
    sorted_results = dqn_linear.sort_by("mean_return", reverse=True)
    
    # Export or print results
    sorted_results.to_csv(Path("analysis/dqn_linear_results.csv"))
    sorted_results.print_table()"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys


class ResultsAggregator:
    """Aggregates results.json files from all experiments."""
    
    def __init__(self, experiments_dir: Path = Path("experiments")):
        """
        Initialize aggregator.
        
        Args:
            experiments_dir: Root experiments directory
        """
        self.experiments_dir = Path(experiments_dir)
        self.results: List[Dict[str, Any]] = []
        self._load_all_results()
    
    def _load_all_results(self) -> None:
        """Load all results.json files from experiment directories."""
        if not self.experiments_dir.exists():
            return
        
        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        result = json.load(f)
                        result['experiment_name'] = exp_dir.name
                        # Flatten extra_metrics into top level, then remove the dict
                        if 'extra_metrics' in result and isinstance(result['extra_metrics'], dict):
                            result.update(result['extra_metrics'])
                            del result['extra_metrics']
                        self.results.append(result)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {results_file}", file=sys.stderr)
    
    def filter_by_algorithm(self, algorithm: str) -> 'ResultsAggregator':
        """Filter results by algorithm name."""
        agg = ResultsAggregator.__new__(ResultsAggregator)
        agg.experiments_dir = self.experiments_dir
        agg.results = [r for r in self.results if r.get('algorithm') == algorithm]
        return agg
    
    def filter_by_model(self, model_type: str) -> 'ResultsAggregator':
        """Filter results by model type."""
        agg = ResultsAggregator.__new__(ResultsAggregator)
        agg.experiments_dir = self.experiments_dir
        agg.results = [r for r in self.results if r.get('model_type') == model_type]
        return agg
    
    def sort_by(self, key: str, reverse: bool = True) -> 'ResultsAggregator':
        """Sort results by a given key."""
        agg = ResultsAggregator.__new__(ResultsAggregator)
        agg.experiments_dir = self.experiments_dir
        agg.results = sorted(self.results, key=lambda x: x.get(key, 0), reverse=reverse)
        return agg
    
    def to_csv(self, filepath: Path) -> None:
        """Export results to CSV file."""
        if not self.results:
            print("No results to export.", file=sys.stderr)
            return
        
        import csv
        
        # Determine all possible keys
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
        
        # Remove extra_metrics from export since it's flattened into columns
        all_keys.discard('extra_metrics')
        
        # Define priority columns (appear first)
        priority_cols = [
            'experiment_name', 'algorithm', 'model_type', 'seed', 'timestamp',
            'mean_return', 'std_return', 'mean_episode_length',
            'mean_max_x_position', 'mean_max_score', 'mean_velocity', 'success_rate',
            'eval_mean_return', 'eval_std_return',
            'total_steps', 'final_loss', 'num_episodes'
        ]
        
        # Order: priority columns first, then others alphabetically
        cols = [c for c in priority_cols if c in all_keys]
        cols.extend(sorted(all_keys - set(cols)))
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cols, restval='')
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Results exported to {filepath}")
    
    def print_table(self) -> None:
        """Print results as formatted table to console."""
        if not self.results:
            print("No results to display.")
            return
        
        # Get all keys and determine priority columns
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
        
        priority_cols = [
            'experiment_name', 'algorithm', 'model_type', 'seed',
            'mean_return', 'std_return', 'mean_episode_length',
            'mean_max_x_position', 'mean_max_score', 'mean_velocity', 'success_rate',
            'eval_mean_return', 'eval_std_return'
        ]
        
        # Use priority columns that exist, then others
        cols = [c for c in priority_cols if c in all_keys]
        cols.extend(sorted(all_keys - set(cols)))
        
        # Compute column widths
        widths = {col: len(col) for col in cols}
        for result in self.results:
            for col in cols:
                val = str(result.get(col, ''))
                widths[col] = max(widths[col], len(val))
        
        # Print header
        header = " | ".join(col.ljust(widths[col]) for col in cols)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for result in self.results:
            row = " | ".join(str(result.get(col, '')).ljust(widths[col]) for col in cols)
            print(row)
        
        print(f"\nTotal experiments: {len(self.results)}")


def main():
    """CLI interface for results aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from results.json files"
    )
    parser.add_argument('--algorithm', type=str, help='Filter by algorithm name')
    parser.add_argument('--model', type=str, help='Filter by model type')
    parser.add_argument('--sort-by', type=str, default='mean_return', help='Sort by column (default: mean_return)')
    parser.add_argument('--reverse', action='store_true', default=True, help='Sort in reverse order')
    parser.add_argument('--output', type=str, help='Export to CSV file')
    parser.add_argument('--experiments-dir', type=str, default='experiments', help='Experiments directory')
    
    args = parser.parse_args()
    
    agg = ResultsAggregator(experiments_dir=Path(args.experiments_dir))
    
    if args.algorithm:
        agg = agg.filter_by_algorithm(args.algorithm)
    
    if args.model:
        agg = agg.filter_by_model(args.model)
    
    agg = agg.sort_by(args.sort_by, reverse=args.reverse)
    
    if args.output:
        agg.to_csv(Path(args.output))
    else:
        agg.print_table()


if __name__ == '__main__':
    main()
