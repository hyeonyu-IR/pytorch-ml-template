# src/utils/logging.py
from __future__ import annotations
import os
import csv
import json
from datetime import datetime
from typing import Dict, Optional

def make_run_dir(base_dir: str = "runs", run_name: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or f"mnist-{ts}"
    run_dir = os.path.join(base_dir, name)
    subdirs = ["checkpoints", "reports", "logs"]
    for sd in [run_dir, *[os.path.join(run_dir, s) for s in subdirs]]:
        os.makedirs(sd, exist_ok=True)
    return run_dir

def save_config_json(cfg_obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg_obj if isinstance(cfg_obj, dict) else vars(cfg_obj), f, indent=2)

class CSVLogger:
    """
    Simple CSV logger that writes one row per epoch.
    """
    def __init__(self, filepath: str, fieldnames: Optional[list] = None):
        self.filepath = filepath
        self.fieldnames = fieldnames or ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        self._init_file()

    def _init_file(self):
        file_exists = os.path.exists(self.filepath)
        self._f = open(self.filepath, mode="a", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        if not file_exists:
            self._writer.writeheader()

    def log(self, row: Dict):
        self._writer.writerow({k: row.get(k) for k in self.fieldnames})
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
