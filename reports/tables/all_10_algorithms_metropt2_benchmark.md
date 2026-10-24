# Complete 10-Algorithm Benchmark Comparison Matrix (METROPT2)

**Student**: Moorthy Chetan (`24BIT0246`)  

**Evaluation Partition**: Held-Out Chronological Test Partition (23,668 sequences)  

**Sorting Criteria**: Ranked by PR-AUC (Primary) and F1-Score (Secondary)


| Rank | Algorithm | Parameters | Train Time | Latency (ms) | Recall | Precision | F1-Score | PR-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **10. Variational Autoencoder (VAE)** | 49,137 | 0.9s | 0.09 | 1.0000 | 0.3310 | 0.4974 | **0.9833** | 🥇 **TOP 1** |
| 2 | **9. LSTM Autoencoder** | 60,625 | 0.33s | 0.04 | 1.0000 | 0.1589 | 0.2742 | **0.9833** | 🥈 **TOP 2** |
| 3 | **8. Temporal Fusion Transformer (TFT)** | 172,753 | 4.27s | 0.31 | 0.9995 | 0.9674 | 0.9832 | **0.9817** | 🥉 **TOP 3** |
| 4 | **5. CNN-BiLSTM with Attention** | 64,290 | 0.38s | 0.05 | 1.0000 | 0.9666 | 0.9830 | **0.9787** | Evaluated |
| 5 | **4. CNN-LSTM (1D Convolution + LSTM)** | 28,833 | 0.21s | 0.02 | 1.0000 | 0.9684 | 0.9839 | **0.9645** | Evaluated |
| 6 | **1. LSTM (Long Short-Term Memory)** | 56,385 | 0.5s | 0.02 | 1.0000 | 0.9686 | 0.9841 | **0.9633** | Evaluated |
| 7 | **3. BiLSTM (Bidirectional LSTM)** | 145,473 | 0.93s | 0.06 | 0.9981 | 0.9671 | 0.9824 | **0.9621** | Evaluated |
| 8 | **6. Temporal Convolutional Network (TCN)** | 112,353 | 1.13s | 0.08 | 0.9992 | 0.9676 | 0.9832 | **0.9599** | Evaluated |
| 9 | **2. GRU (Gated Recurrent Unit)** | 42,817 | 0.86s | 0.07 | 1.0000 | 0.9689 | 0.9842 | **0.9560** | Evaluated |
| 10 | **CNN-Transformer (CNN -> Var -> Next Seq Transformers)** | 72,321 | 2.41s | 0.08 | 1.0000 | 0.9669 | 0.9832 | **0.9536** | Evaluated |
| 11 | **7. Transformer Encoder** | 70,145 | 2.14s | 0.09 | 0.9989 | 0.9676 | 0.9830 | **0.9475** | Evaluated |


### 🏆 Top 3 Algorithms Selected:
1. **10. Variational Autoencoder (VAE)** — PR-AUC: `0.9833`, Recall: `1.0000`, F1: `0.4974`
2. **9. LSTM Autoencoder** — PR-AUC: `0.9833`, Recall: `1.0000`, F1: `0.2742`
3. **8. Temporal Fusion Transformer (TFT)** — PR-AUC: `0.9817`, Recall: `0.9995`, F1: `0.9832`