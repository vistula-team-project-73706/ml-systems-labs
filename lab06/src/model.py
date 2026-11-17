"""
Lab 06: Model

The CNN from Lab 03. It had the best test accuracy there (0.8954) with the
fewest parameters (20,490).
"""

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


def build_model(cfg):
    if cfg["model"]["name"] != "cnn":
        raise ValueError(f"Unknown model {cfg['model']['name']!r}")
    return CNN()
