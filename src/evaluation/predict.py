# src/evaluation/predict.py
from __future__ import annotations
import os
import random
from typing import Optional, Sequence
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def predict_random_samples(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    n_samples: int = 16,
    seed: Optional[int] = None,
    out_path: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
) -> str:
    """
    Draws n_samples random examples from dataset, runs model, and saves a grid plot
    with predicted vs. ground truth labels. Returns path to saved image.
    """
    if seed is not None:
        random.seed(seed)

    idxs = random.sample(range(len(dataset)), k=n_samples)
    model.eval()

    images = []
    labels = []
    preds = []

    with torch.no_grad():
        for i in idxs:
            img, label = dataset[i]  # img: tensor CxHxW (already normalized)
            images.append(img)
            labels.append(label)
        batch = torch.stack(images, dim=0).to(device)
        logits = model(batch)
        pred = logits.argmax(dim=1).cpu().tolist()
        preds.extend(pred)

    # Plot
    cols = int(min(8, n_samples))
    rows = int((n_samples + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 2.0*rows), dpi=150)
    axes = axes.flatten() if isinstance(axes, (list, tuple)) or hasattr(axes, "flatten") else [axes]

    for ax, img_t, y, yhat in zip(axes, images, labels, preds):
        ax.imshow(img_t.squeeze(0), cmap="gray")
        correct = (yhat == y)
        pred_txt = f"{yhat}" if class_names is None else f"{class_names[yhat]}"
        gt_txt = f"{y}" if class_names is None else f"{class_names[y]}"
        title = f"pred={pred_txt} {'✓' if correct else '✗'}\n(gt={gt_txt})"
        ax.set_title(title, color=("green" if correct else "red"), fontsize=9)
        ax.axis("off")

    # If there are unused axes, hide them
    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    if out_path is None:
        out_path = "reports/predictions_grid.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path