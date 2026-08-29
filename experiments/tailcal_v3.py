import argparse
from pathlib import Path

from experiments.main_experiments import run_dataset
from src.models import SPECS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-examples", type=int, default=200)
    args = parser.parse_args()
    config = argparse.Namespace(**vars(args), seeds=[args.seed], tail_alpha=0.20, tailcal_version="v3", lambda_value=1.0)
    for name in args.datasets:
        run_dataset(name, config)


if __name__ == "__main__":
    main()
