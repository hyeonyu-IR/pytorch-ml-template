
# PyTorch ML Template

A **reusable, research‑grade PyTorch template** for training, evaluating, and reporting machine‑learning models.

This repository provides:
- A **generic ML baseline** (e.g., MNIST) on `main`
- A **medical‑imaging base** on `medical-imaging` with multi‑label support and AUROC reporting
- Clean experiment artifacts: CSV logs, curves, qualitative grids, and **PDF summary reports**

---

## ✨ Key Features

- Modular project structure (datasets / models / training / evaluation / reports)
- Reproducible experiments (saved config, seeds)
- Device‑agnostic (CPU, CUDA, Apple Silicon MPS)
- `tqdm` progress bars
- CSV logging (epoch‑level metrics)
- Auto‑generated plots and **PDF reports**
- **Medical‑imaging support**: multi‑label tasks, AUROC (macro/micro + per‑class)
- GitHub Actions CI
- Template repository ("Use this template")

---

## 📁 Repository Structure

```
.
├── data/                     # Datasets (auto‑downloaded or mounted)
├── runs/                     # Experiment outputs (auto‑generated)
│   └── <task>-YYYYMMDD-HHMMSS/
│       ├── checkpoints/
│       ├── logs/             # metrics.csv
│       └── reports/          # curves, grids, summary_report.pdf
├── src/
│   ├── datasets/
│   │   ├── mnist.py
│   │   ├── base_image_dataset.py
│   │   ├── csv_multilabel_dataset.py
│   │   └── medical_demo.py
│   ├── models/
│   │   ├── cnn.py
│   │   └── cxr_resnet.py
│   ├── training/
│   │   ├── engine.py
│   │   └── train.py
│   ├── evaluation/
│   │   └── predict.py
│   ├── reports/
│   │   └── pdf_report.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── imaging_metrics.py
│   │   ├── logging.py
│   │   └── plotting.py
│   └── config.py
├── scripts/
│   ├── run_training.py
│   └── generate_synthetic_cxr.py
├── tests/
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### MNIST (single‑label classification)

```bash
python scripts/run_training.py --task mnist --epochs 5 --batch-size 64
```

Outputs are written to `runs/mnist-YYYYMMDD-HHMMSS/`.

### Medical Imaging Demo (multi‑label, synthetic CXR)

Generate a small synthetic dataset (no real patient data):

```bash
python scripts/generate_synthetic_cxr.py --out-dir data/demo_cxr
```

Train a multi‑label model with AUROC reporting:

```bash
python scripts/run_training.py --task demo_cxr --epochs 20 --batch-size 8
```

Outputs are written to `runs/demo_cxr-YYYYMMDD-HHMMSS/`.

---

## 📊 Metrics & Reports

### CSV Logging

Each run creates `logs/metrics.csv` with epoch‑level metrics.

- **Single‑label tasks** (e.g., MNIST): loss, accuracy
- **Multi‑label imaging tasks**: loss, **AUROC (macro/micro)**

### Plots

- `reports/training_curves.png` – loss (and accuracy/AUROC where applicable)
- `reports/predictions_grid.png` – qualitative predictions vs ground truth

### PDF Summary Report

Each run auto‑generates:

- `reports/summary_report.pdf`

For **medical‑imaging** tasks, the PDF includes:
- Macro and micro AUROC
- Per‑class AUROC table
- Training curves
- Qualitative prediction grid

---

## 🏥 Medical‑Imaging Base (`medical-imaging` branch)

The `medical-imaging` branch extends the template with:

- Image‑based dataset abstractions
- Multi‑label training with `BCEWithLogitsLoss`
- AUROC (macro/micro + per‑class) computation and logging
- Imaging‑specific PDF reports

This branch is intended as the **base for real projects**, such as:

- Chest X‑ray disease classification (NIH, CheXpert, MIMIC‑CXR)
- CT/MR slice‑based classification
- Body composition and sarcopenia research

Create a project branch from it:

```bash
git checkout medical-imaging
git checkout -b cxr-classification
```

---

## ✅ CI & Quality

- GitHub Actions runs tests on push/PR
- Branch protection recommended for `main`
- Safe checkpoint loading (`weights_only=True`)

---

## 📜 License & Citation

This repository is intended for **research and educational use**.

If you use this template in academic work, please cite:

> Yu H, et al. *A reproducible PyTorch training and reporting framework for machine‑learning research.*

---

## 🚀 Next Steps

- Plug in a real CXR dataset (NIH/CheXpert CSV schema)
- Add class prevalence & `pos_weight` for imbalance
- Select best checkpoints by AUROC for imaging tasks
- Extend reports with PR‑AUC and calibration

This repo is designed to be **forked, reused, and extended** for future ML projects.
