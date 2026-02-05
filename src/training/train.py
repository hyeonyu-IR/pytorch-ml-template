# src/training/train.py
from __future__ import annotations
from typing import Dict
import os
import torch
from torch.utils.data import DataLoader
from .engine import train_one_epoch, validate_one_epoch
from ..utils.logging import CSVLogger

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    cfg,
    run_dir: str,
) -> Dict[str, object]:
    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    model.to(device)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "mnist_cnn_best.pt")

    csv_logger = CSVLogger(os.path.join(run_dir, "logs", "metrics.csv"))

    for epoch in range(1, cfg.num_epochs + 1):
        # Train & Validate
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, cfg.num_epochs)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device, epoch, cfg.num_epochs)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])

        lr = optimizer.param_groups[0].get("lr", cfg.learning_rate)
        print(
            f"Epoch {epoch:02d}/{cfg.num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']*100:.2f}% | "
            f"val_loss={val_metrics['val_loss']:.4f} val_acc={val_metrics['val_acc']*100:.2f}% | lr={lr:g}"
        )

        # CSV log
        csv_logger.log({
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_acc": round(val_metrics["val_acc"], 6),
            "lr": lr,
        })

        # Save best model
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            torch.save(
                {"model_state": model.state_dict(), "best_val_acc": best_val_acc, "config": vars(cfg)},
                ckpt_path,
            )

    csv_logger.close()
    print(f"\nBest val_acc: {best_val_acc*100:.2f}% | checkpoint: {ckpt_path}")
    return {
        "best_val_acc": best_val_acc,
        "history": history,
        "best_ckpt_path": ckpt_path,
        "last_train_loss": history["train_loss"][-1],
        "last_train_acc": history["train_acc"][-1],
        "last_val_loss": history["val_loss"][-1],
        "last_val_acc": history["val_acc"][-1],
    }