from __future__ import annotations

import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    def __init__(self, n_classes: int = 3, base_ch: int = 32, in_ch: int = 1):
        super().__init__()
        c = base_ch
        self.features_net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, stride=1, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1), nn.BatchNorm2d(4 * c), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_dim = 4 * c
        self.head = nn.Linear(4 * c, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features_net(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
