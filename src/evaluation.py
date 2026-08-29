from __future__ import annotations

import numpy as np
import torch

from .calibration import ESTIMATORS, aggregate_metrics, fit_temperature, proposition2_stats


@torch.no_grad()
def logits_for_indices(model, x, y, client_indices, device, batch_size=1024):
    model.eval()
    records = []
    for ids in client_indices:
        ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        logits = []
        for start in range(0, len(ids), batch_size):
            logits.append(model(x[ids[start:start + batch_size]]).cpu())
        records.append((torch.cat(logits).numpy(), y[ids].cpu().numpy()))
    return records


def calibrated_probabilities(logits, temperature=1.0):
    z = logits / max(float(temperature), 1e-6)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def evaluate_clients(records, meta, temperatures=1.0, estimator="EW15"):
    values = []
    for index, (logits, labels) in enumerate(records):
        temperature = temperatures[index] if isinstance(temperatures, dict) else temperatures
        probabilities = calibrated_probabilities(logits, temperature)
        ece = ESTIMATORS[estimator](labels, probabilities)
        confidence = probabilities.max(1)
        wrong = probabilities.argmax(1) != labels
        values.append({"client_id": meta[index]["id"], "client_type": meta[index]["type"], "weight": meta[index]["weight"], "n_train": meta[index]["n_train"], "n_eval": len(labels), "ece": ece, "accuracy": float((probabilities.argmax(1) == labels).mean()), "overconf_wrong": float(confidence[wrong].mean()) if wrong.any() else 0.0})
    return values


def evaluate_protocol(records, meta, temperatures=1.0, estimator="EW15", tail_alpha=0.20):
    clients = evaluate_clients(records, meta, temperatures, estimator)
    aggregate = aggregate_metrics([row["ece"] for row in clients], [row["weight"] for row in clients], tail_alpha)
    aggregate.update(proposition2_stats([row["ece"] for row in clients], [row["weight"] for row in clients]))
    return aggregate, clients


def temperatures_from_calibration(model, x, y, calibration_indices, client_val_indices, device, meta=None):
    calibration = logits_for_indices(model, x, y, [calibration_indices], device)[0]
    global_temperature = fit_temperature(*calibration)
    local_temperatures = {}
    for index, ids in enumerate(client_val_indices):
        labels = y[torch.as_tensor(ids, dtype=torch.long, device=device)].cpu().numpy()
        if len(ids) < 10 or len(np.unique(labels)) < 2:
            local_temperatures[index] = 1.0
            continue
        local_records = logits_for_indices(model, x, y, [ids], device)[0]
        local_temperatures[index] = fit_temperature(*local_records)
    weights = np.asarray([item["weight"] for item in meta], dtype=float) if meta is not None else np.ones(len(local_values), dtype=float)
    local_values = np.asarray([local_temperatures[i] for i in range(len(client_val_indices))], dtype=float)
    fed_temperature = float(np.dot(weights / weights.sum(), local_values))
    return global_temperature, local_temperatures, fed_temperature


def tailcal_temperatures(model, x, y, calibration_indices, client_val_indices, meta, device, version="v3", lambda_value=1.0):
    global_temperature, local_temperatures, fed_temperature = temperatures_from_calibration(model, x, y, calibration_indices, client_val_indices, device, meta)
    if version == "v3":
        validation_records = logits_for_indices(model, x, y, client_val_indices, device)
        raw_ece = np.asarray([ESTIMATORS["EW15"](labels, calibrated_probabilities(logits)) for logits, labels in validation_records])
        adjusted = float(np.dot(raw_ece / max(raw_ece.sum(), 1e-12), np.asarray([local_temperatures[i] for i in range(len(meta))])))
    else:
        adjusted = global_temperature
    return {"Raw": 1.0, "GlobalTS": global_temperature, "LocalTS": local_temperatures, "FedTS": fed_temperature, "TailCal": adjusted}
