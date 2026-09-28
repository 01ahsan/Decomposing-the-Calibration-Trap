from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.calibration import ESTIMATORS
from src.data import load_dataset
from src.evaluation import evaluate_protocol, logits_for_indices, temperatures_from_calibration
from src.interventions import train_tailcal_phase
from src.models import SPECS


def high_data_mean(clients, meta):
    values = [row["ece"] for row, item in zip(clients, meta) if item["type"] == "high-data"]
    return float(sum(values) / len(values)) if values else float("nan")


def run_intervention(args, version):
    device = torch.device(args.device)
    output = Path(args.output_dir)
    rows = []
    for name in args.datasets:
        spec = SPECS[name]
        train_x, train_y, test_x, test_y, train_labels, test_labels = load_dataset(spec, args.data_root, device)
        for seed in args.seeds:
            if version == "v4":
                lambda_tag = f"{float(args.lambda_value):g}".replace(".", "p")
                checkpoint = output / "checkpoints" / f"{name.lower()}_v4_lambda{lambda_tag}_seed{seed}.pt"
            else:
                checkpoint = output / "checkpoints" / f"{name.lower()}_v3_seed{seed}.pt"
            model, baseline_model, _, client_val, calibration, meta = train_tailcal_phase(
                spec, seed, train_x, train_y, train_labels, device,
                version=version, phase1_rounds=args.phase1_rounds,
                phase2_rounds=args.phase2_rounds,
                learning_rate_scale=args.learning_rate_scale,
                label_smoothing=args.label_smoothing,
                lambda_value=args.lambda_value, checkpoint=checkpoint,
            )
            baseline_records = logits_for_indices(baseline_model, train_x, train_y, client_val, device)
            final_records = logits_for_indices(model, train_x, train_y, client_val, device)
            baseline_aggregate, baseline_clients = evaluate_protocol(baseline_records, meta, 1.0, "EW15", args.tail_alpha)
            baseline_high = high_data_mean(baseline_clients, meta)
            temperatures = temperatures_from_calibration(model, train_x, train_y, calibration, client_val, device, meta)
            methods = {"Raw": 1.0, "GlobalTS": temperatures[0], "LocalTS": temperatures[1], "FedTS": temperatures[2]}
            for estimator in ESTIMATORS:
                for method, temperature in methods.items():
                    aggregate, clients = evaluate_protocol(final_records, meta, temperature, estimator, args.tail_alpha)
                    final_high = high_data_mean(clients, meta)
                    baseline_tail = baseline_aggregate["TailECE"]
                    final_tail = aggregate["TailECE"]
                    rows.append({
                        "dataset": name,
                        "seed": seed,
                        "method": method,
                        "estimator": estimator,
                        "eval_mode": "small_local_val",
                        "phase1_rounds": args.phase1_rounds,
                        "phase2_rounds": args.phase2_rounds,
                        "lambda_value": args.lambda_value if version == "v4" else 0.0,
                        "baseline_phase1_TailECE": baseline_tail,
                        "baseline_high_data_ECE": baseline_high,
                        "final_TailECE": final_tail,
                        "final_high_data_ECE": final_high,
                        "relative_TailECE_reduction": (baseline_tail - final_tail) / baseline_tail if baseline_tail else float("nan"),
                        "TailECE_reduction_pct": (100.0 * (baseline_tail - final_tail) / baseline_tail if baseline_tail else float("nan")),
                        "absolute_high_data_ECE_drift": (abs(final_high - baseline_high) / baseline_high if baseline_high else float("nan")),
                        "absolute_high_data_ECE_drift_pct": (100.0 * abs(final_high - baseline_high) / baseline_high if baseline_high else float("nan")),
                        **aggregate,
                    })
    result = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    if version == "v4":
        lambda_tag = f"{float(args.lambda_value):g}".replace(".", "p")
        filename = f"tailcal_v4_summary_lambda{lambda_tag}.csv"
    else:
        filename = "tailcal_v3_summary.csv"
    result.to_csv(output / filename, index=False)
    return result