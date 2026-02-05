# PyTorch Machine Learning Project Template

A **reusable, research‑grade PyTorch template repository** for training, evaluating, and reporting deep learning models.

This template was developed using **MNIST** as a reference task but is designed to scale seamlessly to **medical imaging, tabular data, and multimodal ML projects**.

---

## ✅ Key Features

- Clean, modular **project structure** (dataset / model / training / evaluation)
- Reproducible experiments (fixed seeds, saved configs)
- **tqdm** progress bars for training & validation
- Automatic **CSV logging** of metrics
- Auto‑generated **training curves** (loss & accuracy)
- Qualitative **model predictions** on held‑out data
- One‑click **PDF summary report** per experiment (ideal for research & manuscripts)
- Device‑agnostic: CPU, CUDA, or Apple Silicon (MPS)

---

## 📁 Repository Structure

```text
.
├── data/                     # Datasets (auto‑downloaded or mounted)
├── runs/                     # All experiment outputs (auto‑generated)
│   └── <run‑id>/
│       ├── checkpoints/       # Best model checkpoint
│       ├── logs/             # CSV logs
│       ├── reports/          # PNG figures + PDF summary
│       └── config.json       # Saved experiment configuration
├── src/
│   ├── datasets/             # Dataset & DataLoader definitions
│   ├── models/               # PyTorch model architectures
│   ├── training/             # Training & validation engines
│   ├── evaluation/           # Inference & qualitative evaluation utilities
│   ├── reports/              # PDF report builders
│   ├── utils/                # Logging, metrics, seeding, plotting
│   └── config.py             # Central configuration
├── scripts/
│   └── run_training.py       # Main entry point
├── tests/                    # Unit & smoke tests
├── requirements.txt
└── README.md
```

---

## 🧪 Example Use Case (MNIST Baseline)

MNIST is included as a **reference implementation** to verify pipeline correctness, reproducibility, and reporting infrastructure.

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Training

```bash
python scripts/run_training.py
```

### Common options

```bash
python scripts/run_training.py   --epochs 10   --batch-size 64   --lr 1e-3   --num-workers 4   --samples 16
```

---

## 📊 Training Outputs

Each run automatically generates:

- `metrics.csv` – epoch‑level metrics
- `training_curves.png` – loss & accuracy plots
- `predictions_grid.png` – qualitative predictions
- `summary_report.pdf` – one‑page experiment summary

All outputs are saved under `runs/<run-id>/`.

---

## 🔄 Reusing the Template

To adapt for a new ML project:

1. Add a new dataset loader in `src/datasets/`
2. Add or modify models in `src/models/`
3. Keep training, logging, and reporting unchanged

---