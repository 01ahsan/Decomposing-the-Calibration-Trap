from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    high_data_clients: int
    low_data_clients: int
    high_data_range: tuple[int, int]
    low_data_range: tuple[int, int]
    dirichlet_alpha: float
    rounds: int
    local_epochs: int
    local_lr: float
    batch_size: int
    global_calibration_samples: int
    local_validation_fraction: float = 0.20
    momentum: float = 0.9
    weight_decay: float = 1e-4


SPECS = {
    "CIFAR10": DatasetSpec("CIFAR10", 10, 20, 10, (600, 1000), (15, 35), 0.5, 300, 2, 0.05, 256, 500),
    "CIFAR100": DatasetSpec("CIFAR100", 100, 20, 10, (800, 1500), (30, 60), 0.5, 300, 2, 0.05, 256, 1000),
    "FEMNIST": DatasetSpec("FEMNIST", 62, 20, 10, (400, 800), (15, 35), 0.5, 200, 2, 0.05, 256, 620),
}


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(8, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(8, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.GroupNorm(8, planes),
            )

    def forward(self, x):
        y = torch.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return torch.relu(y + self.shortcut(x))


class CIFARResNet18GN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.norm1 = nn.GroupNorm(8, 64)
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _layer(self, planes: int, blocks: int, stride: int):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(blocks - 1):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1))


class SmallCNNGN(nn.Module):
    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.GroupNorm(4, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.GroupNorm(4, 64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.GroupNorm(8, 128), nn.ReLU(), nn.AdaptiveAvgPool2d(3),
        )
        self.classifier = nn.Sequential(nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def build_model(spec: DatasetSpec, device: torch.device):
    model = CIFARResNet18GN(spec.num_classes) if spec.name.startswith("CIFAR") else SmallCNNGN(spec.num_classes)
    return model.to(device)
