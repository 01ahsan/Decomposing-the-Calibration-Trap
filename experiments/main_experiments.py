from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.calibration import ESTIMATORS
from src.data import client_eval_indices, load_dataset
from src.evaluation import evaluate_protocol, logits_for_indices, tailcal_temperatures, temperatures_from_calibration
from src.federated import train_model
from src.models import SPECS, build_model


def run_dataset(name, args):
    spec = SPECS[name]
    device = torch.device(args.device)
    train_x, train_y, test_x, test_y, train_labels, test_labels = load_dataset(spec, args.data_root, device)
    rows = []
    client_rows = []
    output = Path(args.output_dir)
    for seed in args.seeds:
        suffix = f"_tailcal_gamma{args.lambda_value:g}" if getattr(args, "tailcal_version", None) == "v4" else ""
        checkpoint = output / "checkpoints" / f"{name.lower()}_seed{seed}{suffix}.pt"
        if checkpoint.exists():
            model = build_model(spec, device)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            _, _, client_val, calibration, meta = train_model(spec, seed, train_x, train_y, train_labels, device, alpha=getattr(args, "dirichlet_alpha", None), rounds=0)
        else:
            model, _, client_val, calibration, meta = train_model(spec, seed, train_x, train_y, train_labels, device, alpha=getattr(args, "dirichlet_alpha", None), checkpoint=checkpoint, tailcal_gamma=getattr(args, "lambda_value", None) if getattr(args, "tailcal_version", None) == "v4" else None)
        client_eval, rates = client_eval_indices(test_labels, meta, spec, seed, args.eval_examples)
        temperatures = temperatures_from_calibration(model, train_x, train_y, calibration, client_val, device, meta)
        method_temperatures = {"Raw": 1.0, "GlobalTS": temperatures[0], "LocalTS": temperatures[1], "FedTS": temperatures[2]}
        if getattr(args, "tailcal_version", None):
            method_temperatures = tailcal_temperatures(model, train_x, train_y, calibration, client_val, meta, device, args.tailcal_version, getattr(args, "lambda_value", 1.0))
        protocols = [("small_local_val", logits_for_indices(model, train_x, train_y, client_val, device), [0.0] * len(meta)), ("large_client_eval", logits_for_indices(model, test_x, test_y, client_eval, device), rates)]
        for protocol, records, replacement in protocols:
            for estimator in ESTIMATORS:
                for method, temperature in method_temperatures.items():
                    aggregate, clients = evaluate_protocol(records, meta, temperature, estimator, args.tail_alpha)
                    rows.append({"dataset": name, "seed": seed, "eval_mode": protocol, "estimator": estimator, "method": method, "tail_alpha": args.tail_alpha, **aggregate})
                    client_rows.extend({"dataset": name, "seed": seed, "eval_mode": protocol, "estimator": estimator, "method": method, "eval_replacement_rate": replacement[i], **client} for i, client in enumerate(clients))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / f"{name.lower()}_summary.csv", index=False)
    pd.DataFrame(client_rows).to_csv(output / f"{name.lower()}_per_client.csv", index=False)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--tail-alpha", type=float, default=0.20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summaries = [run_dataset(name, arguments) for name in arguments.datasets]
    pd.concat(summaries, ignore_index=True).to_csv(Path(arguments.output_dir) / "all_datasets_summary.csv", index=False)
