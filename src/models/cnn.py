# src/models/cnn.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Input: (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # -> (N,32,26,26)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1) # -> (N,64,24,24)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # after 2x conv + 2x2 maxpool
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)  # -> (N,64,12,12)
        x = self.dropout(x)
        x = torch.flatten(x, 1)            # -> (N, 64*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x