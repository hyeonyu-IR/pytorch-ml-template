# scripts/run_training.py
import sys
from pathlib import Path

# --- Local import convenience for development ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ------------------------------------------------

import argparse
import torch
from src.config import Config
from src.utils.seed import set_seed
from src.datasets.mnist import get_dataloaders
from src.models.cnn import MNISTCNN
from src.training.train import train
from src.utils.logging import make_run_dir, save_config_json
from src.utils.plotting import plot_training_curves
from src.evaluation.predict import predict_random_samples
from src.reports.pdf_report import build_mnist_report

def build_argparser():
    p = argparse.ArgumentParser(description="MNIST Training (PyTorch)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Override device (cuda|mps|cpu)")
    p.add_argument("--run-name", type=str, default=None, help="Optional name for run dir")
    p.add_argument("--samples", type=int, default=16, help="Number of random prediction samples to visualize")
    return p

def main():
    args = build_argparser().parse_args()

    # Create config
    cfg = Config(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        device=(args.device if args.device else None) or Config().device,  # optional override
    ).finalize()

    # Reproducibility
    set_seed(cfg.seed, deterministic=True)

    # Run directory
    run_dir = make_run_dir(base_dir="runs", run_name=args.run_name)
    save_config_json(cfg, Path(run_dir) / "config.json")

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Model/optim/loss
    model = MNISTCNN()
    device = torch.device(cfg.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(f"Using device: {device.type} | Run dir: {run_dir}")
    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        cfg=cfg,
        run_dir=run_dir,
    )

    # Plot training curves
    curves_path = Path(run_dir) / "reports" / "training_curves.png"
    plot_training_curves(summary["history"], str(curves_path))
    print(f"Saved training curves → {curves_path}")

    # Reload best checkpoint before prediction demo
    best_ckpt = summary["best_ckpt_path"]
    state = torch.load(
        best_ckpt,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    # Random predictions grid using the test dataset (val_loader.dataset)
    grid_path = Path(run_dir) / "reports" / "predictions_grid.png"
    out_grid = predict_random_samples(
        model=model,
        dataset=val_loader.dataset,
        device=device,
        n_samples=args.samples,
        seed=cfg.seed,
        out_path=str(grid_path),
        class_names=[str(i) for i in range(10)],
    )
    print(f"Saved random prediction grid → {out_grid}")

    # Build PDF summary report
    pdf_path = build_mnist_report(
        run_dir=run_dir,
        summary=summary,
        cfg=cfg,
        curves_path=str(curves_path),
        grid_path=str(grid_path),
    )
    print(f"Saved PDF summary report → {pdf_path}")

    # Print final summary
    print("\nTraining summary:", {
        "best_val_acc": round(float(summary["best_val_acc"]), 4),
        "last_train_loss": round(float(summary["last_train_loss"]), 4),
        "last_train_acc": round(float(summary["last_train_acc"]), 4),
        "last_val_loss": round(float(summary["last_val_loss"]), 4),
        "last_val_acc": round(float(summary["last_val_acc"]), 4),
    })
    print(f"CSV metrics → {Path(run_dir) / 'logs' / 'metrics.csv'}")

if __name__ == "__main__":
    main()
