from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST

from .models import DatasetSpec


def load_dataset(spec: DatasetSpec, root: str | Path, device: torch.device):
    root = Path(root)
    if spec.name == "CIFAR10":
        transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_ds = CIFAR10(root, True, download=True, transform=transform)
        test_ds = CIFAR10(root, False, download=True, transform=transform)
    elif spec.name == "CIFAR100":
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        train_ds = CIFAR100(root, True, download=True, transform=transform)
        test_ds = CIFAR100(root, False, download=True, transform=transform)
    else:
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train_ds = EMNIST(root, split="byclass", train=True, download=True, transform=transform)
        test_ds = EMNIST(root, split="byclass", train=False, download=True, transform=transform)

    def preload(dataset):
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)
        xs, ys = zip(*[(x, y) for x, y in loader])
        return torch.cat(xs).to(device), torch.cat(ys).to(device)

    train_x, train_y = preload(train_ds)
    test_x, test_y = preload(test_ds)
    return train_x, train_y, test_x, test_y, np.asarray(train_ds.targets), np.asarray(test_ds.targets)


def augment_batch(x: torch.Tensor, spec: DatasetSpec):
    if spec.name.startswith("CIFAR"):
        mask = torch.rand(x.shape[0], device=x.device) > 0.5
        x = x.clone()
        x[mask] = torch.flip(x[mask], dims=[3])
        x = torch.nn.functional.pad(x, [4] * 4, mode="reflect")
        top = torch.randint(0, 9, ()).item()
        left = torch.randint(0, 9, ()).item()
        return x[:, :, top:top + 32, left:left + 32]
    x = torch.nn.functional.pad(x, [2] * 4)
    top = torch.randint(0, 5, ()).item()
    left = torch.randint(0, 5, ()).item()
    return x[:, :, top:top + 28, left:left + 28]


def make_partition(labels, spec: DatasetSpec, alpha: float, seed: int):
    rng = np.random.RandomState(seed)
    n_clients = spec.high_data_clients + spec.low_data_clients
    pools = {}
    calibration = []
    per_class = spec.global_calibration_samples // spec.num_classes
    for cls in range(spec.num_classes):
        ids = np.where(labels == cls)[0].copy()
        rng.shuffle(ids)
        calibration.extend(ids[:per_class])
        pools[cls] = list(ids[per_class:])
    sizes = np.concatenate([rng.randint(*spec.high_data_range, spec.high_data_clients), rng.randint(*spec.low_data_range, spec.low_data_clients)])
    proportions = rng.dirichlet([alpha] * spec.num_classes, size=n_clients)
    train_ids, val_ids, meta = [], [], []
    for client_id, size in enumerate(sizes):
        counts = (proportions[client_id] * int(size)).astype(int)
        counts[np.argmax(proportions[client_id])] += int(size) - counts.sum()
        ids = []
        for cls, count in enumerate(np.maximum(counts, 0)):
            ids.extend(pools[cls][:count])
            pools[cls] = pools[cls][count:]
        ids = np.asarray(ids, dtype=np.int64)
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * spec.local_validation_fraction))
        val, train = ids[:n_val], ids[n_val:]
        hist = np.bincount(labels[ids], minlength=spec.num_classes).astype(float) if len(ids) else np.ones(spec.num_classes)
        hist /= hist.sum()
        val_ids.append(val)
        train_ids.append(train)
        meta.append({"id": client_id, "type": "high-data" if client_id < spec.high_data_clients else "low-data", "n_train": len(train), "n_val": len(val), "label_prop_realized": hist})
    total = sum(item["n_train"] for item in meta)
    for item in meta:
        item["weight"] = item["n_train"] / max(total, 1)
    return train_ids, val_ids, np.asarray(calibration), meta


def client_eval_indices(test_labels, meta, spec: DatasetSpec, seed: int, n_per_client: int = 200):
    rng = np.random.RandomState(seed + 100000)
    by_class = {c: np.where(test_labels == c)[0] for c in range(spec.num_classes)}
    result, replacement_rates = [], []
    for item in meta:
        counts = rng.multinomial(n_per_client, item["label_prop_realized"])
        ids, replaced = [], 0
        for cls, count in enumerate(counts):
            if count:
                replace = count > len(by_class[cls])
                ids.extend(rng.choice(by_class[cls], count, replace=replace).tolist())
                replaced += count if replace else 0
        rng.shuffle(ids)
        result.append(np.asarray(ids, dtype=np.int64))
        replacement_rates.append(replaced / max(len(ids), 1))
    return result, replacement_rates
