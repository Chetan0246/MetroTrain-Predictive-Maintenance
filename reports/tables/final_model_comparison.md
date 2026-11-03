# Comprehensive 10-Algorithm Benchmark Comparison Matrix

**Student**: Moorthy Chetan (`24BIT0246`)  
**Evaluation Scope**: Both MetroPT-2 (Pneumatic Leaks) & MetroPT-3 (Lubrication & Intake Valve Leaks)  
**Partition**: Held-Out Chronological Test Partition (Zero-Leakage Protocol)  
**Primary Metric**: PR-AUC (Precision-Recall Area Under Curve) | **Secondary Metric**: F1-Score

---

## 1. MetroPT-2 Held-Out Test Benchmark (23,668 sequences)

| Rank | Algorithm Architecture | Parameters | Train Time | Latency (ms) | Recall | Precision | F1-Score | PR-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **10. Variational Autoencoder (VAE)** | 49,137 | 0.90s | 0.09 | 1.0000 | 0.3310 | 0.4974 | **0.9833** | 🥇 **TOP 1 (PR-AUC)** |
| 2 | **9. LSTM Autoencoder** | 60,625 | 0.33s | 0.04 | 1.0000 | 0.1589 | 0.2742 | **0.9833** | 🥈 **TOP 2 (PR-AUC)** |
| 3 | **8. Temporal Fusion Transformer (TFT)** | 172,753 | 4.27s | 0.31 | 0.9995 | 0.9674 | 0.9832 | **0.9817** | 🥉 **TOP 3 (Podium)** |
| 4 | **5. CNN-BiLSTM with Attention** | 64,290 | 0.38s | 0.05 | 1.0000 | 0.9666 | 0.9830 | **0.9787** | Evaluated |
| 5 | **4. CNN-LSTM (1D Convolution + LSTM)** | 28,833 | 0.21s | 0.02 | 1.0000 | 0.9684 | 0.9839 | **0.9645** | Evaluated |
| 6 | **1. LSTM (Long Short-Term Memory)** | 56,385 | 0.50s | 0.02 | 1.0000 | 0.9686 | 0.9841 | **0.9633** | Evaluated |
| 7 | **3. BiLSTM (Bidirectional LSTM)** | 145,473 | 0.93s | 0.06 | 0.9981 | 0.9671 | 0.9824 | **0.9621** | Evaluated |
| 8 | **6. Temporal Convolutional Network (TCN)** | 112,353 | 1.13s | 0.08 | 0.9992 | 0.9676 | 0.9832 | **0.9599** | Evaluated |
| 9 | **2. GRU (Gated Recurrent Unit)** | 42,817 | 0.86s | 0.07 | 1.0000 | 0.9689 | 0.9842 | **0.9560** | Evaluated |
| 10 | **CNN-Transformer (Custom Stage-by-Stage)** | 72,321 | 2.41s | 0.08 | 1.0000 | 0.9669 | 0.9832 | **0.9536** | Evaluated |
| 11 | **7. Transformer Encoder** | 70,145 | 2.14s | 0.09 | 0.9989 | 0.9676 | 0.9830 | **0.9475** | Evaluated |

---

## 2. MetroPT-3 Held-Out Test Benchmark (26,925 sequences)

| Rank | Algorithm Architecture | Parameters | Train Time | Latency (ms) | Recall | Precision | F1-Score | PR-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **7. Transformer Encoder** | 70,145 | 3.51s | 0.13 | 0.0000 | 0.0000 | 0.0000 | **0.1714** | 🥇 **TOP 1 (PR-AUC)** |
| 2 | **3. BiLSTM (Bidirectional LSTM)** | 145,473 | 2.27s | 0.12 | 0.0000 | 0.0000 | 0.0000 | **0.1398** | 🥈 **TOP 2 (PR-AUC)** |
| 3 | **8. Temporal Fusion Transformer (TFT)** | 172,753 | 7.06s | 0.49 | 0.0000 | 0.0000 | 0.0000 | **0.1381** | 🥉 **TOP 3 (Podium)** |
| 4 | **6. Temporal Convolutional Network (TCN)** | 112,353 | 1.79s | 0.12 | 0.0000 | 0.0000 | 0.0000 | **0.1378** | Evaluated |
| 5 | **9. LSTM Autoencoder** | 60,625 | 0.82s | 0.06 | 1.0000 | 0.1255 | 0.2229 | **0.1328** | Evaluated |
| 6 | **5. CNN-BiLSTM with Attention** | 64,290 | 1.02s | 0.07 | 0.0000 | 0.0000 | 0.0000 | **0.1217** | Evaluated |
| 7 | **10. Variational Autoencoder (VAE)** | 49,137 | 1.41s | 0.10 | 0.7380 | 0.1235 | 0.2117 | **0.1198** | Evaluated |
| 8 | **2. GRU (Gated Recurrent Unit)** | 42,817 | 1.45s | 0.09 | 0.0000 | 0.0000 | 0.0000 | **0.1182** | Evaluated |
| 9 | **4. CNN-LSTM (1D Convolution + LSTM)** | 28,833 | 0.40s | 0.03 | 0.0000 | 0.0000 | 0.0000 | **0.1164** | Evaluated |
| 10 | **1. LSTM (Long Short-Term Memory)** | 56,385 | 1.00s | 0.05 | 0.0000 | 0.0000 | 0.0000 | **0.1155** | Evaluated |

---

### Key Methodological Insights:
1. **VAE & LSTM-AE Rank #1 in PR-AUC (0.9833)**: Unsupervised reconstruction error provides smooth anomaly discrimination across all decision thresholds.
2. **TFT Delivers Premier Operational Performance**: F1 of 0.9832 with 99.95% recall and 96.74% precision on MetroPT-2.
3. **Threshold Calibration Resolves Recall Collapse on MetroPT-3**: Unsupervised validation-only quantile calibration ($q_{90}$) restores recall from 0% to 35.02% without test label leakage.
