import argparse

from experiments.main_experiments import run_dataset
from src.models import SPECS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-value", type=float, required=True, help="Low-data client weight multiplier; 1.0 is FedAvg.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-examples", type=int, default=200)
    args = parser.parse_args()
    config = argparse.Namespace(**vars(args), tail_alpha=0.20, tailcal_version="v4")
    for name in args.datasets:
        run_dataset(name, config)


if __name__ == "__main__":
    main()
