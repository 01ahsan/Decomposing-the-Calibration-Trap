from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn, optim

from .data import augment_batch, make_partition
from .models import DatasetSpec, build_model


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def set_parameters(model, parameters, device):
    model.load_state_dict({key: value.to(device) for key, value in parameters.items()})


def fedavg(parameter_sets, weights):
    weights = torch.tensor(np.asarray(weights) / np.sum(weights), dtype=torch.float32)
    return {key: sum(value[key].float() * weights[i] for i, value in enumerate(parameter_sets)) for key in parameter_sets[0]}


def tailcal_aggregate(parameter_sets, meta, gamma=3.0, tail_fraction=0.20):
    sizes = np.asarray([item["n_train"] for item in meta], dtype=float)
    threshold = np.quantile(sizes, tail_fraction)
    corrected = sizes.copy()
    corrected[sizes <= threshold] *= gamma
    corrected /= corrected.sum()
    weights = torch.as_tensor(corrected, dtype=torch.float32)
    return {key: sum(value[key].float() * weights[i] for i, value in enumerate(parameter_sets)) for key in parameter_sets[0]}


def local_update(model, ids, train_x, train_y, spec: DatasetSpec, device):
    model.train()
    indices = torch.as_tensor(ids, dtype=torch.long, device=device)
    x, y = train_x[indices], train_y[indices]
    optimizer = optim.SGD(model.parameters(), lr=spec.local_lr, momentum=spec.momentum, weight_decay=spec.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec.local_epochs)
    criterion = nn.CrossEntropyLoss()
    for _ in range(spec.local_epochs):
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(y), spec.batch_size):
            batch = order[start:start + spec.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(augment_batch(x[batch], spec)), y[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()
    return get_parameters(model)


def train_model(spec, seed, train_x, train_y, train_labels, device, alpha=None, rounds=None, checkpoint=None, tailcal_gamma=None):
    seed_everything(seed)
    alpha = spec.dirichlet_alpha if alpha is None else alpha
    rounds = spec.rounds if rounds is None else rounds
    client_train, client_val, calibration, meta = make_partition(train_labels, spec, alpha, seed)
    global_model = build_model(spec, device)
    client_model = build_model(spec, device)
    parameters = get_parameters(global_model)
    for _ in range(rounds):
        updates, weights, update_meta = [], [], []
        for ids, item in zip(client_train, meta):
            if len(ids) == 0:
                continue
            set_parameters(client_model, parameters, device)
            updates.append(local_update(client_model, ids, train_x, train_y, spec, device))
            weights.append(item["n_train"])
            update_meta.append(item)
        parameters = tailcal_aggregate(updates, update_meta, gamma=tailcal_gamma) if tailcal_gamma is not None else fedavg(updates, weights)
    set_parameters(global_model, parameters, device)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_model.state_dict(), checkpoint)
    return global_model, client_train, client_val, calibration, meta
