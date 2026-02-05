# src/config.py
from dataclasses import dataclass, field
import torch
import os

def get_default_device() -> str:
    """
    Prefer CUDA if available, then Apple MPS, else CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    # MPS: Apple Silicon
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class Config:
    # Data
    data_dir: str = field(default_factory=lambda: os.path.join("data"))
    batch_size: int = 64
    num_workers: int = 4

    # Training (will be used in later steps)
    num_epochs: int = 5
    learning_rate: float = 1e-3

    # Reproducibility
    seed: int = 42

    # Device
    device: str = field(default_factory=get_default_device)

    # DataLoader performance hints (set at runtime based on device/OS)
    pin_memory: bool = False
    persistent_workers: bool = False

    def finalize(self):
        """
        Compute derived settings based on environment.
        Call this once after instantiating the config.
        """
        # Pin memory only helps with CUDA; it’s harmless to leave False otherwise.
        self.pin_memory = (self.device == "cuda")
        # persistent_workers only valid if num_workers > 0
        self.persistent_workers = (self.num_workers > 0)
        return self