import argparse
from pathlib import Path

from experiments.tailcal_common import run_intervention
from src.models import SPECS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--tail-alpha", type=float, default=0.20)
    parser.add_argument("--phase1-rounds", type=int, default=None)
    parser.add_argument("--phase2-rounds", type=int, default=50)
    parser.add_argument("--learning-rate-scale", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    args = parser.parse_args()
    args.lambda_value = 0.0
    run_intervention(args, "v3")


if __name__ == "__main__":
    main()
