# Validation-Calibrated Operating Thresholds for MetroPT-3 (Out-of-Distribution Transfer)

> [!NOTE]
> **Calibration Procedure**: Operating thresholds were selected strictly using **unsupervised quantiles on the healthy validation set** ($q_{90}, q_{95}, q_{98}$) without observing test labels. They were evaluated once on the held-out test set to demonstrate how threshold calibration resolves the recall collapse at $\tau=0.5$.

| Model | Operating Point | Validation Threshold ($	au$) | Test Precision | Test Recall | Test F1-Score | Predicted Alarm Windows |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | Val-$q_{90}$ (10% FPR budget) | $	au = 0.2467$ | 17.11% | **31.35%** | **0.2214** | 6,189 |
| **LSTM** | Val-$q_{95}$ (5% FPR budget)  | $	au = 0.2777$ | 17.70% | **20.16%** | **0.1885** | 3,848 |
| **LSTM** | Val-$q_{98}$ (2% FPR budget)  | $	au = 0.3075$ | 19.94% | **12.91%** | **0.1567** | 2,187 |
| **LSTM** | Default Fixed Boundary        | $	au = 0.5000$ | 0.00%  | 0.00%      | 0.0000     | 0 (Recall Collapse) |
| **GRU**  | Val-$q_{90}$ (10% FPR budget) | $	au = 0.0677$ | 18.29% | **35.02%** | **0.2403** | 6,469 |
| **GRU**  | Val-$q_{95}$ (5% FPR budget)  | $	au = 0.0883$ | 19.93% | **26.20%** | **0.2264** | 4,440 |
| **GRU**  | Val-$q_{98}$ (2% FPR budget)  | $	au = 0.1587$ | 22.78% | **16.84%** | **0.1937** | 2,498 |
| **GRU**  | Default Fixed Boundary        | $	au = 0.5000$ | 0.00%  | 0.00%      | 0.0000     | 29 (All False Positives) |
| **TCN**  | Val-$q_{90}$ (10% FPR budget) | $	au = 0.0511$ | 17.71% | **34.04%** | **0.2330** | 6,493 |
| **TCN**  | Val-$q_{95}$ (5% FPR budget)  | $	au = 0.0820$ | 20.52% | **24.93%** | **0.2251** | 4,104 |
| **TCN**  | Val-$q_{98}$ (2% FPR budget)  | $	au = 0.1386$ | 24.74% | **16.70%** | **0.1994** | 2,280 |
| **TCN**  | Default Fixed Boundary        | $	au = 0.5000$ | **41.46%** | 1.51%  | 0.0291     | 123 |
