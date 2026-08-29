import argparse
from pathlib import Path

import pandas as pd

from experiments.main_experiments import run_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30])
    args = parser.parse_args()
    rows = []
    for alpha in args.alphas:
        config = argparse.Namespace(**vars(args), datasets=["CIFAR10", "CIFAR100", "FEMNIST"], tail_alpha=alpha, eval_examples=200)
        rows.append(pd.concat([run_dataset(name, config) for name in config.datasets], ignore_index=True))
    pd.concat(rows, ignore_index=True).to_csv(Path(args.output_dir) / "tail_fraction_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
