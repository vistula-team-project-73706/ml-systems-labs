"""
Lab 07: Models

TeacherCNN is the oversized baseline (Task 1, about 1M parameters).
StudentCNN is the small model for distillation (Task 4, about 100k).

The layers are named (conv1, conv2, conv3, fc1, fc2) so that structured
pruning can rebuild a smaller TeacherCNN with fewer filters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherCNN(nn.Module):
    """Three conv layers and a large fully connected layer.

    28x28 -> conv1 -> pool -> 14x14 -> conv2 -> pool -> 7x7 -> conv3 -> fc1 -> fc2.
    With the default sizes it has 997,434 parameters. Most of them (90%)
    are in fc1, which sees 128 * 7 * 7 = 6,272 inputs.
    """

    def __init__(self, c1=32, c2=64, c3=128, hidden=144, dropout=0.3):
        super().__init__()
        self.channels = (c1, c2, c3)
        self.hidden = hidden
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.fc1 = nn.Linear(c3 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class StudentCNN(nn.Module):
    """Small CNN: 105,866 parameters, about 9.4x fewer than the teacher."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def prunable_layers(model):
    """(name, module) for every conv and linear layer, in forward order."""

    return [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def build_teacher_from_state(state):
    """Rebuild a TeacherCNN (possibly with fewer filters) from a saved checkpoint."""

    model = TeacherCNN(*state["channels"], hidden=state["hidden"])
    model.load_state_dict(state["model"])
    return model


def teacher_state(model):
    return {"channels": model.channels, "hidden": model.hidden, "model": model.state_dict()}
