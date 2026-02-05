# src/datasets/mnist.py
from __future__ import annotations
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ..config import Config

# Standard MNIST normalization stats
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

def get_mnist_transforms() -> transforms.Compose:
    """
    Returns torchvision transforms for MNIST.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation (test) dataloaders for MNIST.
    """
    tfm = get_mnist_transforms()

    train_ds = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=tfm,
    )
    val_ds = datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=tfm,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
    )

    return train_loader, val_loader