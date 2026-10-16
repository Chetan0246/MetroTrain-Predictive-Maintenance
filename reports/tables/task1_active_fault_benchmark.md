# Task 1 Benchmark: Active Fault Detection & State Monitoring

> [!NOTE]
> **Task 1 Scope**: The positive class ($y=1$) includes **both the 2-hour pre-failure early warning window and the full duration of active fault presence**. This evaluates the system's ability to recognize that an APU fault state exists and maintain continuous alarm persistence.

| Dataset | Model Architecture | PR-AUC (Primary) | Recall ($	au=0.5$) | F1-Score ($	au=0.5$) | Precision ($	au=0.5$) | Training Runtime | Benchmark Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetroPT-3 (APU)** | **TCN** | **0.1939** | 0.0151 | 0.0291 | 0.4146 | 21.6s | 🥇 Rank 1 |
| **MetroPT-3 (APU)** | **GRU** | **0.1830** | 0.0000 | 0.0000 | 0.0000 | 21.2s | 🥈 Rank 2 |
| **MetroPT-3 (APU)** | **LSTM** | **0.1783** | 0.0000 | 0.0000 | 0.0000 | 8.8s | 🥉 Rank 3 |
| **MetroPT-2 (APU)** | **GRU** | **0.9673** | 1.0000 | 0.9841 | 0.9686 | 13.8s | 🥇 Rank 1 |
| **MetroPT-2 (APU)** | **LSTM** | **0.9669** | 1.0000 | 0.9838 | 0.9681 | 5.8s | 🥈 Rank 2 |
| **MetroPT-2 (APU)** | **TCN** | **0.9543** | 0.9997 | 0.9830 | 0.9669 | 15.2s | 🥉 Rank 3 |

*Test Set Prevalences*: MetroPT-3 Test: 12.54% positive (3,378 / 26,925 windows). MetroPT-2 Test: 15.88% positive (3,767 / 23,714 windows).
