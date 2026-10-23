# Complete 10-Algorithm Benchmark Comparison Matrix (METROPT3)

**Student**: Moorthy Chetan (`24BIT0246`)  

**Evaluation Partition**: Held-Out Chronological Test Partition (23,668 sequences)  

**Sorting Criteria**: Ranked by PR-AUC (Primary) and F1-Score (Secondary)


| Rank | Algorithm | Parameters | Train Time | Latency (ms) | Recall | Precision | F1-Score | PR-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **7. Transformer Encoder** | 70,145 | 3.51s | 0.13 | 0.0000 | 0.0000 | 0.0000 | **0.1714** | 🥇 **TOP 1** |
| 2 | **3. BiLSTM (Bidirectional LSTM)** | 145,473 | 2.27s | 0.12 | 0.0000 | 0.0000 | 0.0000 | **0.1398** | 🥈 **TOP 2** |
| 3 | **8. Temporal Fusion Transformer (TFT)** | 172,753 | 7.06s | 0.49 | 0.0000 | 0.0000 | 0.0000 | **0.1381** | 🥉 **TOP 3** |
| 4 | **6. Temporal Convolutional Network (TCN)** | 112,353 | 1.79s | 0.12 | 0.0000 | 0.0000 | 0.0000 | **0.1378** | Evaluated |
| 5 | **9. LSTM Autoencoder** | 60,625 | 0.82s | 0.06 | 1.0000 | 0.1255 | 0.2229 | **0.1328** | Evaluated |
| 6 | **5. CNN-BiLSTM with Attention** | 64,290 | 1.02s | 0.07 | 0.0000 | 0.0000 | 0.0000 | **0.1217** | Evaluated |
| 7 | **10. Variational Autoencoder (VAE)** | 49,137 | 1.41s | 0.10 | 0.7380 | 0.1235 | 0.2117 | **0.1198** | Evaluated |
| 8 | **2. GRU (Gated Recurrent Unit)** | 42,817 | 1.45s | 0.09 | 0.0000 | 0.0000 | 0.0000 | **0.1182** | Evaluated |
| 9 | **4. CNN-LSTM (1D Convolution + LSTM)** | 28,833 | 0.4s | 0.03 | 0.0000 | 0.0000 | 0.0000 | **0.1164** | Evaluated |
| 10 | **1. LSTM (Long Short-Term Memory)** | 56,385 | 1.0s | 0.05 | 0.0000 | 0.0000 | 0.0000 | **0.1155** | Evaluated |


### 🏆 Top 3 Algorithms Selected:
1. **7. Transformer Encoder** — PR-AUC: `0.1714`, Recall: `0.0000`, F1: `0.0000`
2. **3. BiLSTM (Bidirectional LSTM)** — PR-AUC: `0.1398`, Recall: `0.0000`, F1: `0.0000`
3. **8. Temporal Fusion Transformer (TFT)** — PR-AUC: `0.1381`, Recall: `0.0000`, F1: `0.0000`