# Project Instructions: Predictive Maintenance in Metro Trains Using ML & DL

**Course / Project**: AST Project Work — Predictive Maintenance in Metro Trains Using Machine Learning and Deep Learning Techniques  
**Author**: Moorthy Chetan (Reg. No: 24BIT0246)  
**Institution**: Vellore Institute of Technology (VIT)  

---

## 1. Project Objective & Scope

The objective is to investigate and compare predictive maintenance algorithms on real-world Air Production Unit (APU) compressor telemetry from Metro do Porto trains across two distinct benchmark datasets:
- **Dataset 1**: MetroPT / MetroPT-3 (Air Compressor Telemetry)
- **Dataset 2**: MetroPT-2 (Zenodo benchmark telemetry)

### Core Experimental Matrix (6 Experiments)
Evaluate three candidate time-series/deep learning architectures independently on both datasets:
1. **LSTM** (Long Short-Term Memory)
2. **GRU** (Gated Recurrent Unit)
3. **TCN** (Temporal Convolutional Network)

Comparative evaluation will measure:
- Primary Metrics: **PR-AUC (Precision-Recall Area Under Curve)**, **Recall**, **F1-Score**
- Secondary Metrics: **Precision**, **Confusion Matrices**, **Training / Inference Time**

---

## 2. Research Integrity & Methodological Principles

To ensure academic rigor, reproducibility, and publication-grade integrity:
1. **No Data Fabrication**: Never fabricate observations, synthetic rows, failure events, class counts, or model metrics. All figures, tables, and samples must be extracted from genuine data.
2. **Defensible, Documented Event Labeling**: Raw telemetry does not contain supervised target labels. Ground truth failure intervals and early warning horizons must strictly adhere to official maintenance reports (Nature *Scientific Data* 2022, Veloso et al.; Jakobs et al. 2026; Zenodo records).
3. **Strict Chronological Splitting**: Time-series splits (Train / Validation / Test) must be chronological to prevent look-ahead bias and temporal leakage.
4. **Isolated Preprocessing**: Scalers (e.g., `StandardScaler`, `MinMaxScaler`), encoders, and feature selectors must be fitted strictly on the chronological training split and applied to validation/test splits.
5. **Class Imbalance Realism**: Failures and warning windows are rare events (< 1% to 3% prevalence). Training splits may employ defensible balancing strategies (e.g., class-weighted loss, focal loss, or targeted window sampling), but **validation and test splits must retain their natural class distribution** to reflect real-world operating conditions.
6. **No Inter-Dataset Leakage**: MetroPT-3 and MetroPT-2 differ in sampling resolution (integer seconds vs milliseconds), timestamp ranges, column ordering, and specific failure modes. They must be preprocessed, windowed, and modeled separately.

---

## 3. Directory Layout

```text
MetroTrain-Predictive-Maintenance/
├── PROJECT_INSTRUCTIONS.md
├── README.md
├── data/
│   ├── raw/
│   │   ├── metropt3/
│   │   │   └── dataset_train.csv          # 10.77M rows, Jan-Jun 2022
│   │   └── metropt2/
│   │       └── MetroPT2.csv               # 7.12M rows, Apr-Jul 2022
│   └── processed/                         # Processed/downsampled parquet/npz files
├── notebooks/
│   ├── 01_dataset_inspection.ipynb        # EDA, sampling rates, genuine sample extraction
│   ├── 02_preprocessing.ipynb             # Cleaning, chronological split, labeling, scaling
│   └── 03_model_comparison.ipynb          # LSTM vs GRU vs TCN benchmark & PR curves
├── src/
│   ├── data_inspection.py                 # Statistical inspection and sample extraction
│   ├── preprocessing.py                   # Missing value handling, scaling, feature engineering
│   ├── labeling.py                        # Ground-truth failure interval and early warning labeling
│   ├── windowing.py                       # Memory-efficient sliding sequence generation
│   ├── train.py                           # Training loops for LSTM, GRU, TCN with seeds & logging
│   └── evaluate.py                        # PR-AUC, F1, Recall, confusion matrices, figures/tables
├── configs/
│   ├── metropt3_config.yaml               # Experiment hyperparameters for Dataset 1
│   └── metropt2_config.yaml               # Experiment hyperparameters for Dataset 2
├── tests/
│   └── test_pipeline.py                   # Automated tests for data loading, labeling, shapes
├── reports/
│   ├── figures/                           # Class distribution, PR curves, confusion matrices
│   ├── tables/                            # Dataset sizes, imbalance ratios, performance tables
│   ├── samples/                           # Genuine 2-record CSV samples per class per dataset
│   └── methodology/                       # Methodology write-up and Mermaid flowchart
├── models/                                # PyTorch checkpoint weights (.pt)
├── results/                               # Metrics JSON, classification reports, logs
├── requirements.txt
└── .gitignore
```

---

## 4. Hardware Constraints & Memory Management

**Local Hardware Specifications**:
- **CPU**: 8 physical cores / 16 threads
- **System RAM**: ~13.3 GB total (~7.4 GB available)
- **GPU**: None (CPU-only execution via `torch.device('cpu')`)

**Engineering Directives**:
1. Avoid loading entire multi-gigabyte raw CSVs repeatedly into memory.
2. Store processed splits in binary formats (e.g., Parquet with snappy compression or compressed NumPy `.npz` / memory-mapped `.npy`).
3. For deep sequential models on CPU, downsample or aggregate high-frequency 1 Hz / sub-second telemetry into regularized 10-second or 1-minute steps (matching the official UCI 0.1 Hz sampling specification), or use sequence strides to prevent combinatorial explosion.
4. Use PyTorch `DataLoader` with batch sizes appropriate for CPU cache (e.g., `batch_size = 64` or `128`) and multi-threaded data workers.

---

## 5. Execution Workflow

### Step 1: Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Dataset Inspection & Sample Extraction
```bash
python3 src/data_inspection.py
```
Outputs:
- Generates summary statistics and missing value checks.
- Exports genuine 2-record CSV samples per class to `reports/samples/`.

### Step 3: Labeling & Preprocessing
```bash
python3 src/preprocessing.py --dataset metropt3
python3 src/preprocessing.py --dataset metropt2
```
Outputs:
- Applies ground truth failure interval labeling and early warning horizon (2 hours).
- Performs chronological 70/15/15 train/val/test splits.
- Fits scalers strictly on training splits.
- Exports imbalanced and balanced distribution tables to `reports/tables/` and figures to `reports/figures/`.

### Step 4: Model Training (6 Experiments)
```bash
python3 src/train.py --dataset metropt3 --model lstm
python3 src/train.py --dataset metropt3 --model gru
python3 src/train.py --dataset metropt3 --model tcn

python3 src/train.py --dataset metropt2 --model lstm
python3 src/train.py --dataset metropt2 --model gru
python3 src/train.py --dataset metropt2 --model tcn
```

### Step 5: Evaluation & Report Generation
```bash
python3 src/evaluate.py
```
Outputs:
- Computes PR-AUC, Recall, F1, Precision, and training times.
- Plots comparative Precision-Recall curves and confusion matrices.
- Assembles final experimental comparison tables.
