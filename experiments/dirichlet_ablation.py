from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.main_experiments import run_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.30, 0.50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=150)
    args = parser.parse_args()
    rows = []
    for alpha in args.alphas:
        result = run_dataset("CIFAR100", argparse.Namespace(**vars(args), datasets=["CIFAR100"], dirichlet_alpha=alpha, tail_alpha=0.20, eval_examples=200))
        result["dirichlet_alpha"] = alpha
        rows.append(result)
    pd.concat(rows, ignore_index=True).to_csv(Path(args.output_dir) / "cifar100_dirichlet_ablation.csv", index=False)


if __name__ == "__main__":
    main()
