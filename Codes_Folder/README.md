# (3) Codes Folder

**Course**: AI / Predictive Maintenance Project Work  
**Student**: Moorthy Chetan (`24BIT0246`)  
**Institution**: School of Information Technology and Engineering (SITE), VIT Vellore  

---

This directory fulfills the submission requirement for **(3) Codes folder** and contains **all Python source codes (`.py`)**, Jupyter notebooks (`.ipynb`), model architectures, training routines, and evaluation scripts used throughout the project.

---

## 1. Directory Structure

```
Codes_Folder/
├── models.py                          # All 10 Deep Learning Model Architectures (PyTorch)
├── train.py                           # Unified training routine with early stopping & AdamW
├── benchmark_all_10.py                # End-to-end benchmarking orchestrator for all 10 models
├── preprocessing.py                   # Zero-leakage chronological split & scaling pipeline
├── windowing.py                       # 10-minute sliding sequence window generator (60x16)
├── labeling.py                        # Failure interval & 2-hour pre-failure early warning labeling
├── class_distribution_and_smote.py    # Class imbalance quantification & SMOTE benchmark
├── dual_evaluation.py                 # Task 1 vs Task 2 evaluation & threshold calibration
├── evaluate.py                        # Precision, Recall, F1, PR-AUC, and latency evaluation
├── data_inspection.py                 # Telemetry integrity & thermodynamic distribution verification
├── generate_flowchart.py              # Publication-quality pipeline flowchart generator
├── generate_review3_document.py       # Automated academic review report & PDF compiler
├── notebooks/                         # Executable Jupyter and Google Colab Notebooks
│   ├── MetroTrain_Predictive_Maintenance_Complete_Pipeline.ipynb  # Self-contained Colab notebook
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_comparison.ipynb
└── requirements.txt                   # Complete Python dependencies
```

---

## 2. Description of Core Code Modules

### 1. `models.py` (All 10 Deep Learning Architectures)
Defines 10 PyTorch neural architectures across four sequence modeling paradigms:
1. **LSTM**: 2-layer stacked LSTM with cell-state gating for long-range thermodynamic cycles.
2. **GRU**: 2-layer Gated Recurrent Unit offering rapid convergence and low parameter footprint (42,817 params).
3. **BiLSTM**: Bidirectional LSTM capturing both forward and reverse sequence trends.
4. **CNN-LSTM**: 1D temporal convolution for local feature extraction feeding into an LSTM backbone.
5. **CNN-BiLSTM with Attention**: 1D CNN + Bidirectional LSTM equipped with Additive Bahdanau Attention.
6. **Temporal Convolutional Network (TCN)**: Dilated causal residual convolutions ($d \in \{1, 2, 4\}$) with receptive field $R=15$.
7. **Transformer Encoder**: Multi-head self-attention (4 heads, 2 layers) with sinusoidal positional encodings.
8. **Temporal Fusion Transformer (TFT)**: Multi-head attention combined with Variable Selection Networks and Gated Residual Networks (GRN).
9. **LSTM Autoencoder**: Unsupervised sequence-to-sequence reconstruction using MSE loss.
10. **Variational Autoencoder (VAE)**: Unsupervised generative latent space encoder-decoder optimized via ELBO (Reconstruction + KL-Divergence).
*Bonus*: **CNN-Transformer** (Multi-stage convolutional sequence transformer).

### 2. `train.py`
- Implements unified model training using `AdamW` optimizer, cosine/plateau learning rate scheduling, and gradient norm clipping ($1.0$).
- Supports both supervised BCE with Logits loss and unsupervised Autoencoder MSE reconstruction loss.
- Implements validation early stopping with model checkpoint persistence.

### 3. `benchmark_all_10.py`
- Benchmarking engine that iterates through all 10 architectures sequentially.
- Trains each architecture on Google Colab / GPU, records parameter count, training runtime, inference latency per sequence (in milliseconds), PR-AUC, Recall, Precision, and F1-score.
- Exports ground truth benchmark JSONs: `all_10_algorithms_metropt2.json` and `all_10_algorithms_metropt3.json`.

### 4. `preprocessing.py`
- Implements strict zero-leakage chronological partitioning:
  - **MetroPT-2**: 65% Train, 15% Validation, 20% Held-Out Test.
  - **MetroPT-3**: 70% Train, 15% Validation, 15% Held-Out Test.
- Computes `StandardScaler` / `MinMaxScaler` parameters strictly on the training partition and applies them outward to validation and test partitions, preventing forward lookahead leakage.

### 5. `windowing.py`
- Generates 10-minute sliding sequence windows ($T=60$ timesteps at 0.1 Hz, 16 features).
- Produces tensor batches of shape `(Batch_Size, 60, 16)`.
- Implements 1:1 balanced temporal undersampling on the training partition to stabilize gradient updates.

### 6. `dual_evaluation.py`
- Implements the Dual-Evaluation testing methodology:
  - **Task 1: Active Fault State Monitoring**: Evaluates detection across the 2-hour pre-failure window plus ongoing failure outage.
  - **Task 2: Anticipatory Pre-Failure Warning**: Evaluates pure prognostic forecasting strictly during the 2-hour pre-failure window (all active failure windows excluded).
- Evaluates unsupervised validation-only quantile threshold calibration ($q_{90}, q_{95}, q_{98}$) to eliminate out-of-distribution recall collapse on MetroPT-3 without test-label leakage.

---

## 3. How to Run the Codes

### Step 1: Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Dataset Inspection & Preprocessing
```bash
python data_inspection.py
python preprocessing.py
```

### Step 3: Train & Benchmark All 10 Algorithms
```bash
python benchmark_all_10.py --dataset metropt2
python benchmark_all_10.py --dataset metropt3
```

### Step 4: Run Dual-Evaluation & Threshold Calibration
```bash
python dual_evaluation.py
```

### Step 5: Run in Google Colab (One-Click)
Open `notebooks/MetroTrain_Predictive_Maintenance_Complete_Pipeline.ipynb` in Google Colab, select GPU runtime (T4), and run all cells.
