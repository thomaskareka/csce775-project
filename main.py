import argparse
from pathlib import Path

from config import load_config
from runners.train_runner import TrainRunner
from runners.eval_runner import *

def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--envs", type=int, default=1)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--checkpoint", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == "train":
        print("Training")
        config = load_config(args.config)
        runner = TrainRunner(config)
        runner.train()
    elif args.mode == "eval":
        print("Evaluating model")
        checkpoint_path = args.checkpoint
        # If a directory is provided, look for final.pt
        if Path(checkpoint_path).is_dir():
            checkpoint_path = str(Path(checkpoint_path) / "final.pt")
    
        runner = EvalRunner.load_checkpoint(checkpoint_path, args.envs)
        exp_dir = Path(args.checkpoint) if Path(args.checkpoint).is_dir() else Path(args.checkpoint).parent

        result = runner.evaluate(args.episodes, args.envs, directory=exp_dir)
        
        # runner.append_eval_results(exp_dir)

    elif args.mode == "resume":
        print("resuming form checkpoint")
        checkpoint_path = args.checkpoint
        # If a directory is provided, look for final.pt
        if Path(checkpoint_path).is_dir():
            checkpoint_path = str(Path(checkpoint_path) / "final.pt")
        runner = TrainRunner.load_checkpoint(checkpoint_path)
        runner.train()
    else:
        raise ValueError(f"unknown mode {args.mode}")

if __name__ == "__main__":
    main()