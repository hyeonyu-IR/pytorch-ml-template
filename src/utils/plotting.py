# src/utils/plotting.py
from __future__ import annotations
import os
import matplotlib.pyplot as plt

def plot_training_curves(history: dict, out_path: str) -> str:
    """
    history: {
      'train_loss': [...], 'train_acc': [...], 'val_loss': [...], 'val_acc': [...]
    }
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=140)

    # Loss
    axs[0].plot(epochs, history["train_loss"], label="Train", marker="o")
    axs[0].plot(epochs, history["val_loss"], label="Val", marker="o")
    axs[0].set_title("Loss")  # localized title example
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Accuracy
    axs[1].plot(epochs, [a*100 for a in history["train_acc"]], label="Train", marker="o")
    axs[1].plot(epochs, [a*100 for a in history["val_acc"]], label="Val", marker="o")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path