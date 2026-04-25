import argparse
from pathlib import Path

from config import load_config
from runners.train_runner import TrainRunner
from runners.eval_runner import *
from utils.get_checkpoints import get_checkpoints, get_configs

def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--batch", action="store_true", help="train all configs in the directory")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="path to file evaluates single checkpoint, directory evaluates all checkpoints within")
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--envs", type=int, default=1)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--checkpoint", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == "train":
        print("Training")
        if args.batch:
            config_files = get_configs(args.config)
            print(f"found {len(config_files)} config files")
            for file in config_files:
                print(f"training with config {file}")
                config = load_config(file)
                runner = TrainRunner(config)
                runner.train()
                del runner
        else:
            config = load_config(args.config)
            runner = TrainRunner(config)
            runner.train()
    elif args.mode == "eval":
        print("Evaluating model")
        checkpoint_path = args.checkpoint

        if Path(checkpoint_path).is_dir():
            # checkpoint_path = str(Path(checkpoint_path) / "final.pt")
            checkpoints = get_checkpoints(checkpoint_path)
            print(f"Found {len(checkpoints)} checkpoints")
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