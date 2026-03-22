import argparse

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
        runner = EvalRunner.load_checkpoint(args.checkpoint)
        result = runner.evaluate(args.episodes)
    elif args.mode == "resume":
        print("resuming form checkpoint")
        runner = TrainRunner.load_checkpoint(args.checkpoint)
        runner.train()
    else:
        raise ValueError(f"unknown mode {args.mode}")

if __name__ == "__main__":
    main()