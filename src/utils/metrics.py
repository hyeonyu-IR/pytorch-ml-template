# src/utils/metrics.py
from __future__ import annotations
import torch

@torch.inference_mode()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy for classification.
    """
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()