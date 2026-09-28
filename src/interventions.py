from __future__ import annotations

import copy

import torch

from .federated import fedavg, get_parameters, local_update, seed_everything, set_parameters, train_model
from .models import DatasetSpec, build_model


def train_tailcal_phase(spec: DatasetSpec, seed, train_x, train_y, train_labels, device, version="v3", phase1_rounds=None, phase2_rounds=50, learning_rate_scale=0.1, label_smoothing=0.1, lambda_value=1.0, checkpoint=None):
    seed_everything(seed)
    phase1, client_train, client_val, calibration, meta = train_model(spec, seed, train_x, train_y, train_labels, device, rounds=phase1_rounds, checkpoint=None)
    baseline_model = copy.deepcopy(phase1).to(device).eval()
    reference = copy.deepcopy(phase1).to(device).eval()
    calibration_ids = torch.as_tensor(calibration, dtype=torch.long, device=device)
    calibration_x, calibration_y = train_x[calibration_ids], train_y[calibration_ids]
    client_model = build_model(spec, device)
    parameters = get_parameters(phase1)
    for _ in range(phase2_rounds):
        updates, weights = [], []
        for ids, item in zip(client_train, meta):
            if len(ids) == 0 or item["type"] != "low-data":
                continue
            set_parameters(client_model, parameters, device)
            update = local_update(client_model, ids, train_x, train_y, spec, device, label_smoothing=label_smoothing, reference=reference if version == "v4" else None, distill_weight=lambda_value if version == "v4" else 0.0, distill_x=calibration_x if version == "v4" else None, distill_y=calibration_y if version == "v4" else None, lr_scale=learning_rate_scale)
            updates.append(update)
            weights.append(item["n_train"])
        if updates:
            parameters = fedavg(updates, weights)
    set_parameters(phase1, parameters, device)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(phase1.state_dict(), checkpoint)
    return phase1, baseline_model, client_train, client_val, calibration, meta
