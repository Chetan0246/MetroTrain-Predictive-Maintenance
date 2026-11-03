# Task 1 Benchmark: Active Fault Detection & State Monitoring

> [!NOTE]
> **Task 1 Scope**: The positive class ($y=1$) includes **both the 2-hour pre-failure early warning window and the full duration of active fault presence**. This evaluates the system's ability to recognize that an APU fault state exists and maintain continuous alarm persistence.

---

## 1. MetroPT-2 Active Fault Detection (3,767 Positive Windows / 23,714 Total Windows)

| Rank | Model Architecture | Parameters | Train Time | Latency (ms) | Recall ($\tau=0.5$) | Precision ($\tau=0.5$) | F1-Score ($\tau=0.5$) | PR-AUC (Primary) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **10. Variational Autoencoder (VAE)** | 49,137 | 0.90s | 0.09 | 1.0000 | 0.3310 | 0.4974 | **0.9833** |
| 2 | **9. LSTM Autoencoder** | 60,625 | 0.33s | 0.04 | 1.0000 | 0.1589 | 0.2742 | **0.9833** |
| 3 | **8. Temporal Fusion Transformer (TFT)** | 172,753 | 4.27s | 0.31 | 0.9995 | 0.9674 | 0.9832 | **0.9817** |
| 4 | **5. CNN-BiLSTM with Attention** | 64,290 | 0.38s | 0.05 | 1.0000 | 0.9666 | 0.9830 | **0.9787** |
| 5 | **4. CNN-LSTM (1D Conv + LSTM)** | 28,833 | 0.21s | 0.02 | 1.0000 | 0.9684 | 0.9839 | **0.9645** |
| 6 | **1. LSTM (Long Short-Term Memory)** | 56,385 | 0.50s | 0.02 | 1.0000 | 0.9686 | 0.9841 | **0.9633** |
| 7 | **3. BiLSTM (Bidirectional LSTM)** | 145,473 | 0.93s | 0.06 | 0.9981 | 0.9671 | 0.9824 | **0.9621** |
| 8 | **6. Temporal Convolutional Network (TCN)** | 112,353 | 1.13s | 0.08 | 0.9992 | 0.9676 | 0.9832 | **0.9599** |
| 9 | **2. GRU (Gated Recurrent Unit)** | 42,817 | 0.86s | 0.07 | 1.0000 | 0.9689 | 0.9842 | **0.9560** |
| 10 | **CNN-Transformer (Custom Stage-by-Stage)** | 72,321 | 2.41s | 0.08 | 1.0000 | 0.9669 | 0.9832 | **0.9536** |
| 11 | **7. Transformer Encoder** | 70,145 | 2.14s | 0.09 | 0.9989 | 0.9676 | 0.9830 | **0.9475** |

---

## 2. MetroPT-3 Active Fault Detection (3,378 Positive Windows / 26,925 Total Windows)

| Rank | Model Architecture | Parameters | Train Time | Latency (ms) | Recall ($\tau=0.5$) | Precision ($\tau=0.5$) | F1-Score ($\tau=0.5$) | PR-AUC (Primary) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **7. Transformer Encoder** | 70,145 | 3.51s | 0.13 | 0.0000 | 0.0000 | 0.0000 | **0.1714** |
| 2 | **3. BiLSTM (Bidirectional LSTM)** | 145,473 | 2.27s | 0.12 | 0.0000 | 0.0000 | 0.0000 | **0.1398** |
| 3 | **8. Temporal Fusion Transformer (TFT)** | 172,753 | 7.06s | 0.49 | 0.0000 | 0.0000 | 0.0000 | **0.1381** |
| 4 | **6. Temporal Convolutional Network (TCN)** | 112,353 | 1.79s | 0.12 | 0.0000 | 0.0000 | 0.0000 | **0.1378** |
| 5 | **9. LSTM Autoencoder** | 60,625 | 0.82s | 0.06 | 1.0000 | 0.1255 | 0.2229 | **0.1328** |
| 6 | **5. CNN-BiLSTM with Attention** | 64,290 | 1.02s | 0.07 | 0.0000 | 0.0000 | 0.0000 | **0.1217** |
| 7 | **10. Variational Autoencoder (VAE)** | 49,137 | 1.41s | 0.10 | 0.7380 | 0.1235 | 0.2117 | **0.1198** |
| 8 | **2. GRU (Gated Recurrent Unit)** | 42,817 | 1.45s | 0.09 | 0.0000 | 0.0000 | 0.0000 | **0.1182** |
| 9 | **4. CNN-LSTM (1D Conv + LSTM)** | 28,833 | 0.40s | 0.03 | 0.0000 | 0.0000 | 0.0000 | **0.1164** |
| 10 | **1. LSTM (Long Short-Term Memory)** | 56,385 | 1.00s | 0.05 | 0.0000 | 0.0000 | 0.0000 | **0.1155** |

*Note on Uncalibrated Default Thresholds ($\tau=0.50$)*: Supervised models on MetroPT-3 suffer from cross-dataset transfer recall collapse under fixed $\tau=0.50$. As demonstrated in Table 8, validation threshold calibration ($q_{90}$) restores recall to 35.02% without test label leakage.
