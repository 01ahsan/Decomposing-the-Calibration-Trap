from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import ece_equal_width

DATASET_OFFSET = {"CIFAR10": 0, "CIFAR100": 10_000, "FEMNIST": 20_000}
PAPER_SEEDS = {42, 43, 44}


def probabilities(logits, temperature=1.0):
    values = np.asarray(logits, dtype=float) / max(float(temperature), 1e-8)
    values = values - values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def raw_ece(logits, labels):
    return ece_equal_width(np.asarray(labels, dtype=int), probabilities(logits, 1.0), 15)


def metrics_selected_tail(eces, weights, selected_tail):
    eces = np.asarray(eces, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    cw = float(weights @ eces)
    uw = float(eces.mean())
    tail = float(eces[selected_tail].mean())
    return {
        "cwECE": cw,
        "uwECE": uw,
        "TailECE": tail,
        "A": tail / uw if uw else np.nan,
        "M": uw / cw if cw else np.nan,
        "TailTrap": tail / cw if cw else np.nan,
    }


def crossfit_cache(path: Path, repeats=500, half_size=100, tail_alpha=0.20, seed=20260928):
    cached = np.load(path, allow_pickle=True)
    try:
        logits = cached["logits"]
        labels = cached["labels"]
        weights = np.asarray(cached["weights"], dtype=float)
        weights /= weights.sum()
        if len(logits) != 30:
            raise ValueError(f"{path.name} must contain 30 clients.")
        if any(len(x) < half_size * 2 for x in labels):
            raise ValueError(f"{path.name} must contain at least {half_size * 2} examples per client.")

        full_ece = np.asarray([raw_ece(client_logits, client_labels) for client_logits, client_labels in zip(logits, labels)])
        full_cw = float(weights @ full_ece)
        full_uw = float(full_ece.mean())
        m_200 = full_uw / full_cw if full_cw else np.nan

        rng = np.random.RandomState(seed)
        n_tail = max(1, int(np.ceil(tail_alpha * len(logits))))
        rows, stability = [], []
        for repeat in range(repeats):
            ece_a = np.empty(len(logits), dtype=float)
            ece_b = np.empty(len(logits), dtype=float)
            for k, (client_logits, client_labels) in enumerate(zip(logits, labels)):
                order = rng.permutation(len(client_labels))
                a = order[:half_size]
                b = order[half_size:half_size * 2]
                ece_a[k] = raw_ece(client_logits[a], client_labels[a])
                ece_b[k] = raw_ece(client_logits[b], client_labels[b])

            tail_a = np.argsort(ece_a)[::-1][:n_tail]
            tail_b = np.argsort(ece_b)[::-1][:n_tail]
            naive_a = metrics_selected_tail(ece_a, weights, tail_a)
            naive_b = metrics_selected_tail(ece_b, weights, tail_b)
            cross_ab = metrics_selected_tail(ece_b, weights, tail_a)
            cross_ba = metrics_selected_tail(ece_a, weights, tail_b)
            stability.append({
                "repeat": repeat,
                "tail_overlap_fraction": len(set(tail_a) & set(tail_b)) / n_tail,
            })
            rows.append({
                "repeat": repeat,
                "naive_A_TailTrap": naive_a["TailTrap"],
                "naive_B_TailTrap": naive_b["TailTrap"],
                "cross_A_to_B_TailTrap": cross_ab["TailTrap"],
                "cross_B_to_A_TailTrap": cross_ba["TailTrap"],
                "cross_A_to_B_A": cross_ab["A"],
                "cross_B_to_A_A": cross_ba["A"],
                "M_A": cross_ab["M"],
                "M_B": cross_ba["M"],
            })
        result = pd.DataFrame(rows)
        result.attrs["stability"] = pd.DataFrame(stability)
        result.attrs["M_200"] = m_200
        return result
    finally:
        cached.close()


def run(input_dir, output_dir, repeats=500, half_size=100, tail_alpha=0.20):
    cache_root = Path(input_dir) / "prediction_cache"
    paths = sorted(cache_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError("No prediction_cache/*.npz files found. Run main_experiments.py first.")
    per_seed, stability = [], []
    for path in paths:
        match = re.search(r"(.+)_seed(\d+)$", path.stem)
        if match is None:
            continue
        dataset, seed = match.group(1).upper(), int(match.group(2))
        if seed not in PAPER_SEEDS:
            continue
        if dataset not in DATASET_OFFSET:
            raise ValueError(f"Unknown dataset key in cache filename: {dataset}")
        split_seed = 3_000_000 + DATASET_OFFSET[dataset] + seed
        result = crossfit_cache(path, repeats, half_size, tail_alpha, split_seed)
        stable = result.attrs["stability"]
        same_half = np.mean([result["naive_A_TailTrap"].mean(), result["naive_B_TailTrap"].mean()])
        crossfit_tt = np.mean([result["cross_A_to_B_TailTrap"].mean(), result["cross_B_to_A_TailTrap"].mean()])
        crossfit_a = np.mean([result["cross_A_to_B_A"].mean(), result["cross_B_to_A_A"].mean()])
        summary = {
            "dataset": dataset,
            "seed": seed,
            "repeats": repeats,
            "same_half_TailTrap_mean": float(same_half),
            "crossfit_A": float(crossfit_a),
            "M_100": float(np.mean([result["M_A"], result["M_B"]])),
            "M_200": float(result.attrs["M_200"]),
            "crossfit_TailTrap": float(crossfit_tt),
            "tail_overlap_mean": float(stable["tail_overlap_fraction"].mean()),
            "tail_overlap_std": float(stable["tail_overlap_fraction"].std(ddof=1)),
        }
        for direction in ["A_to_B", "B_to_A"]:
            for key in ["TailTrap", "A"]:
                summary[f"crossfit_{direction}_{key}"] = float(result[f"cross_{direction}_{key}"].mean())
        summary["crossfit_M_A"] = float(result["M_A"].mean())
        summary["crossfit_M_B"] = float(result["M_B"].mean())
        per_seed.append(summary)
        stable = stable.copy()
        stable.insert(0, "dataset", dataset)
        stable.insert(1, "seed", seed)
        stability.append(stable)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_df = pd.DataFrame(per_seed)
    stability_df = pd.concat(stability, ignore_index=True) if stability else pd.DataFrame()
    per_seed_df.to_csv(output_dir / "FINAL_CROSSFIT_M100_PER_SEED.csv", index=False)
    stability_df.to_csv(output_dir / "PHASE2_FINAL_TAIL_STABILITY.csv", index=False)
    return per_seed_df, stability_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--half-size", type=int, default=100)
    parser.add_argument("--tail-alpha", type=float, default=0.20)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.repeats, args.half_size, args.tail_alpha)