from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gamma_sweep_one_run(group, gammas, tail_alpha=0.20):
    group = group.sort_values("client_id")
    e = group["ece"].to_numpy(float)
    n = group["n_train"].to_numpy(float)
    if len(e) != 30 or np.isnan(e).any() or np.any(n <= 0):
        raise ValueError("Each gamma group must contain 30 valid clients.")
    count = max(1, int(np.ceil(tail_alpha * len(e))))
    tail = float(np.sort(e)[-count:].mean())
    rows = []
    for gamma in gammas:
        raw = np.power(n, gamma)
        weights = raw / raw.sum()
        aggregate = float(weights @ e)
        rows.append({"gamma": float(gamma), "uwECE": float(e.mean()), "M_gamma": float(e.mean() / aggregate), "R_gamma": float(1.0 - aggregate / e.mean()), "aggregate_ECE_gamma": aggregate, "TailECE_fixed": tail, "TailTrap_gamma": tail / aggregate, "min_weight_gamma": float(weights.min()), "max_weight_gamma": float(weights.max()), "weight_ratio_max_min": float(weights.max() / weights.min())})
    return pd.DataFrame(rows)


def run(input_dir, output_dir, gammas=None, tail_alpha=0.20):
    gammas = np.linspace(0, 1, 21) if gammas is None else np.asarray(gammas, float)
    rows = []
    for path in sorted(Path(input_dir).rglob("*_per_client.csv")):
        dataset = path.stem.replace("_per_client", "").upper()
        frame = pd.read_csv(path)
        if not {"seed", "client_id", "n_train", "ece"}.issubset(frame.columns):
            continue
        if "method" in frame.columns:
            frame = frame[frame["method"] == "Raw"]
        if "estimator" in frame.columns:
            frame = frame[frame["estimator"] == "EW15"]
        if "eval_mode" in frame.columns and "large_client_eval" in set(frame["eval_mode"]):
            frame = frame[frame["eval_mode"] == "large_client_eval"]
        for seed, group in frame.groupby("seed"):
            values = gamma_sweep_one_run(group, gammas, tail_alpha)
            values.insert(0, "seed", int(seed))
            values.insert(0, "dataset", dataset)
            rows.append(values)
    if not rows:
        raise FileNotFoundError("No compatible per-client CSV files were found.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output_dir / "GAMMA_SWEEP_PER_SEED.csv", index=False)
    result.groupby(["dataset", "gamma"])[["uwECE", "M_gamma", "R_gamma", "aggregate_ECE_gamma", "TailTrap_gamma"]].agg(["mean", "std"]).to_csv(output_dir / "GAMMA_SWEEP_5SEED_SUMMARY.csv")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tail-alpha", type=float, default=0.20)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, tail_alpha=args.tail_alpha)
