# src/training/engine.py
from __future__ import annotations
from typing import Dict
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ..utils.metrics import accuracy

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Train {epoch:02d}/{num_epochs}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(logits.detach(), targets)
        running_loss += loss.item()
        running_acc += batch_acc
        n_batches += 1

        pbar.set_postfix({"loss": f"{running_loss/n_batches:.4f}", "acc": f"{(running_acc/n_batches)*100:.2f}%"})

    return {"loss": running_loss / n_batches, "acc": running_acc / n_batches}

@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Val   {epoch:02d}/{num_epochs}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_acc = accuracy(logits, targets)
        running_loss += loss.item()
        running_acc += batch_acc
        n_batches += 1

        pbar.set_postfix({"val_loss": f"{running_loss/n_batches:.4f}", "val_acc": f"{(running_acc/n_batches)*100:.2f}%"})

    return {"val_loss": running_loss / n_batches, "val_acc": running_acc / n_batches}