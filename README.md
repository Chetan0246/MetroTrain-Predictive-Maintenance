# Predictive Maintenance in Metro Trains Using Machine Learning and Deep Learning

**Course**: AST Project Work  
**Student Name**: Moorthy Chetan  
**Registration Number**: 24BIT0246  
**Institution**: Vellore Institute of Technology (VIT)  

---

## 1. Project Overview

This project investigates and benchmarks data-driven predictive maintenance on Air Production Units (APU) of metro train compressors operated by Metro do Porto, Portugal. 

We benchmark three deep sequential architectures:
1. **LSTM** (Long Short-Term Memory)
2. **GRU** (Gated Recurrent Unit)
3. **TCN** (Temporal Convolutional Network)

Across two real-world operational datasets:
- **Dataset 1: MetroPT / MetroPT-3** (UCI Repository / Zenodo 6854240; 10,773,588 records, Jan–Jun 2022)
- **Dataset 2: MetroPT-2** (Zenodo Record 7766691; 7,116,940 records, Apr–Jul 2022)

---

## 2. Directory Structure

```text
MetroTrain-Predictive-Maintenance/
├── PROJECT_INSTRUCTIONS.md        # Detailed academic and technical guidelines
├── README.md                      # Setup, commands, and project summary
├── data/
│   ├── raw/
│   │   ├── metropt3/
│   │   │   └── dataset_train.csv  # 1.6 GB, 10.77M records (Jan 1, 2022 - Jun 2, 2022)
│   │   └── metropt2/
│   │       └── MetroPT2.csv       # 1.2 GB, 7.12M records (Apr 28, 2022 - Jul 28, 2022)
│   └── processed/                 # Processed parquet/npz splits
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── data_inspection.py         # EDA and sample extraction
│   ├── preprocessing.py           # Cleaning, scaling, and chronological splitting
│   ├── labeling.py                # Failure event and early-warning horizon labeling
│   ├── windowing.py               # Memory-efficient sliding sequence generation
│   ├── train.py                   # PyTorch training loops for LSTM, GRU, and TCN
│   └── evaluate.py                # Evaluation metrics, PR-AUC, and visualization
├── configs/                       # YAML configurations for experiments
├── tests/                         # Unit tests for data loaders and pipelines
├── reports/
│   ├── figures/                   # Imbalance plots, PR curves, confusion matrices
│   ├── tables/                    # Per-class counts, distribution, performance tables
│   ├── samples/                   # Genuine 2-record CSV samples per class per dataset
│   └── methodology/               # Methodology write-up and Mermaid flowchart
├── models/                        # Trained model checkpoints (.pt)
├── results/                       # Performance metrics and logs
├── requirements.txt
└── .gitignore
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.10+
- Linux (x86_64)

### Installation
```bash
# Clone the repository and navigate to root
cd MetroTrain-Predictive-Maintenance

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Pipeline Execution Commands

### 1. Data Inspection & Sample Extraction
```bash
python3 src/data_inspection.py
```
- Validates data integrity, schemas, and timestamps.
- Exports genuine 2-sample records per class for both datasets to `reports/samples/`.

### 2. Preprocessing & Chronological Splitting
```bash
python3 src/preprocessing.py --dataset metropt3
python3 src/preprocessing.py --dataset metropt2
```
- Applies ground truth failure interval labeling.
- Performs strict chronological train/val/test splits (70/15/15).
- Fits scalers strictly on training data.
- Exports distribution figures to `reports/figures/` and tables to `reports/tables/`.

### 3. Model Training (6 Experiments)
```bash
# MetroPT-3 (Dataset 1)
python3 src/train.py --dataset metropt3 --model lstm
python3 src/train.py --dataset metropt3 --model gru
python3 src/train.py --dataset metropt3 --model tcn

# MetroPT-2 (Dataset 2)
python3 src/train.py --dataset metropt2 --model lstm
python3 src/train.py --dataset metropt2 --model gru
python3 src/train.py --dataset metropt2 --model tcn
```

### 4. Evaluation & Reporting
```bash
python3 src/evaluate.py
```
- Computes PR-AUC, Recall, F1, Precision, and training times across all 6 runs.
- Saves comparative Precision-Recall curves to `reports/figures/pr_curves_comparison.png`.
- Assembles comprehensive professor review tables in `reports/tables/`.

---

## 5. Official References & Dataset Sources
1. **MetroPT Dataset**: Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2022). *The MetroPT dataset for predictive maintenance*. Scientific Data, 9(1), 764. [DOI: 10.1038/s41597-022-01877-3](https://doi.org/10.1038/s41597-022-01877-3).
2. **MetroPT2 Benchmark**: Zenodo record 7766691. [DOI: 10.5281/zenodo.7766691](https://doi.org/10.5281/zenodo.7766691).
3. **Failure Analysis & Rule Learning**: Jakobs, M., Veloso, B., & Gama, J. (2026). *Interpretable rules for online failure prediction: a case study on metro do porto datasets*. International Journal of Data Science and Analytics, 22:59. [DOI: 10.1007/s41060-026-01039-3](https://doi.org/10.1007/s41060-026-01039-3).
