from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def group_masks(client_type):
    values = pd.Series(client_type).astype(str).str.lower().str.strip().to_numpy()
    low = np.isin(values, ["tail", "low-data", "low_data", "low"])
    high = np.isin(values, ["majority", "high-data", "high_data", "high"])
    if (~(low | high)).any():
        raise ValueError(f"Unknown client_type values: {sorted(set(values[~(low | high)]))}")
    return low, high


def decomposition_from_group(group, tail_alpha=0.20, expected_clients=30):
    group = group.sort_values("client_id").reset_index(drop=True)
    e = group["ece"].to_numpy(float)
    n_train = group["n_train"].to_numpy(float)
    saved = group["weight"].to_numpy(float)
    ids = group["client_id"].to_numpy(int)
    types = group["client_type"].astype(str).to_numpy()
    if len(e) != expected_clients or np.isnan(e).any() or np.any(n_train <= 0):
        raise ValueError("Each group must contain 30 clients with finite ECE and positive n_train.")
    weights = n_train / n_train.sum()
    weight_error = float(np.max(np.abs(weights - saved / saved.sum())))
    uniform_ece = float(e.mean())
    weighted_ece = float(weights @ e)
    count = max(1, int(math.ceil(tail_alpha * len(e))))
    tail_ids = np.argsort(e)[::-1][:count]
    tail_ece = float(e[tail_ids].mean())
    low, high = group_masks(types)
    worst = int(np.argmax(e))
    range_ece = float(e.max() - e.min())
    bound_cw = (1 - weights[worst]) * range_ece
    bound_uw = (1 - 1 / len(e)) * range_ece
    covariance = float(np.mean((weights - 1 / len(e)) * (e - uniform_ece)))
    return {"K": len(e), "n_tail_eval": count, "cwECE": weighted_ece, "uwECE": uniform_ece, "TailECE": tail_ece, "TailTrap": tail_ece / weighted_ece, "A_tail_heterogeneity": tail_ece / uniform_ece, "M_sample_weight_masking": uniform_ece / weighted_ece, "R_weight_reduction": 1.0 - weighted_ece / uniform_ece, "A_times_M": tail_ece / weighted_ece, "Cov_u_w_ECE": covariance, "K_Cov_u_w_ECE": len(e) * covariance, "cwECE_minus_uwECE": weighted_ece - uniform_ece, "Pearson_r_weight_ECE": float(np.corrcoef(weights, e)[0, 1]) if np.std(weights) and np.std(e) else np.nan, "W_low": float(weights[low].sum()), "W_high": float(weights[high].sum()), "LowData_mean_ECE": float(e[low].mean()), "HighData_mean_ECE": float(e[high].mean()), "LowData_mean_ntrain": float(n_train[low].mean()), "HighData_mean_ntrain": float(n_train[high].mean()), "n_low_data_clients": int(low.sum()), "n_high_data_clients": int(high.sum()), "ECE_tail_low_data_count": int(low[tail_ids].sum()), "ECE_max": float(e.max()), "ECE_min": float(e.min()), "worst_client_id": int(ids[worst]), "worst_client_type": str(types[worst]), "worst_client_weight": float(weights[worst]), "Prop2_gap_cw": float(e[worst] - weighted_ece), "Prop2_bound_cw": float(bound_cw), "Prop2_tightness_cw": float((e[worst] - weighted_ece) / bound_cw) if bound_cw else np.nan, "Prop2_gap_uw": float(e[worst] - uniform_ece), "Prop2_bound_uw": float(bound_uw), "Prop2_tightness_uw": float((e[worst] - uniform_ece) / bound_uw) if bound_uw else np.nan, "max_saved_vs_recomputed_weight_diff": weight_error, "Prop3_identity_error": weighted_ece - uniform_ece - len(e) * covariance, "TailTrap_factorization_error": tail_ece / weighted_ece - (tail_ece / uniform_ece) * (uniform_ece / weighted_ece)}


def run(input_dir, output_dir, tail_alpha=0.20):
    rows = []
    for path in sorted(Path(input_dir).rglob("*_per_client.csv")):
        dataset = path.stem.replace("_per_client", "").upper()
        frame = pd.read_csv(path)
        required = {"seed", "client_id", "client_type", "n_train", "weight", "ece"}
        if not required.issubset(frame.columns):
            continue
        if "method" in frame.columns:
            frame = frame[frame["method"] == "Raw"]
        if "estimator" in frame.columns:
            frame = frame[frame["estimator"] == "EW15"]
        if "eval_mode" in frame.columns and "large_client_eval" in set(frame["eval_mode"]):
            frame = frame[frame["eval_mode"] == "large_client_eval"]
        for seed, group in frame.groupby("seed"):
            row = decomposition_from_group(group, tail_alpha)
            row.update({"dataset": dataset, "seed": int(seed)})
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No compatible per-client CSV files were found.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["dataset", "seed"])
    result.to_csv(output_dir / "PHASE1_DECOMPOSITION_PER_SEED.csv", index=False)
    result.groupby("dataset").agg(["mean", "std", "min", "max"]).to_csv(output_dir / "PHASE1_DECOMPOSITION_5SEED_SUMMARY.csv")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tail-alpha", type=float, default=0.20)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.tail_alpha)
