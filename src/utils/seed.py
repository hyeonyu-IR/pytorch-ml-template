# src/utils/seed.py
from __future__ import annotations
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set PRNG seeds across Python, NumPy, and PyTorch.
    Optionally enable deterministic algorithms in PyTorch for reproducibility.

    Note: Deterministic mode may reduce performance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Deterministic algorithms (where possible)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Disable cuDNN autotuner for determinism
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True