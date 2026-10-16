# Validation-Calibrated Operating Thresholds for MetroPT-3 (Out-of-Distribution Transfer)

> [!NOTE]
> **Calibration Procedure**: Operating thresholds were chosen strictly using **unsupervised quantiles on the healthy validation set** ($q_{90}, q_{95}, q_{98}$) without observing test labels. They were evaluated once on the held-out test set to demonstrate how threshold calibration resolves the recall collapse observed at the arbitrary default boundary $\tau=0.50$.

| Model | Operating Point | Validation Threshold ($\tau$) | Target Validation FPR | Measured Test FPR | Test Precision | Test Recall | Test F1-Score | Predicted Alarm Windows |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | Val-$q_{90}$ | $\tau = 0.2467$ | 10.0% | **21.79%** (5,130 / 23,547) | 17.11% | **31.35%** | **0.2214** | 6,189 |
| **LSTM** | Val-$q_{95}$ | $\tau = 0.2777$ | 5.0%  | **13.45%** (3,167 / 23,547) | 17.70% | **20.16%** | **0.1885** | 3,848 |
| **LSTM** | Val-$q_{98}$ | $\tau = 0.3075$ | 2.0%  | **7.44%**  (1,751 / 23,547) | 19.94% | **12.91%** | **0.1567** | 2,187 |
| **LSTM** | Default Boundary | $\tau = 0.5000$ | N/A   | **0.00%**  (0 / 23,547)     | 0.00%  | 0.00%      | 0.0000     | 0 *(Recall Collapse)* |
| **GRU**  | Val-$q_{90}$ | $\tau = 0.0677$ | 10.0% | **22.45%** (5,286 / 23,547) | 18.29% | **35.02%** | **0.2403** | 6,469 |
| **GRU**  | Val-$q_{95}$ | $\tau = 0.0883$ | 5.0%  | **15.10%** (3,555 / 23,547) | 19.93% | **26.20%** | **0.2264** | 4,440 |
| **GRU**  | Val-$q_{98}$ | $\tau = 0.1587$ | 2.0%  | **8.19%**  (1,929 / 23,547) | 22.78% | **16.84%** | **0.1937** | 2,498 |
| **GRU**  | Default Boundary | $\tau = 0.5000$ | N/A   | **0.12%**  (29 / 23,547)    | 0.00%  | 0.00%      | 0.0000     | 29 *(All False Positives)* |
| **TCN**  | Val-$q_{90}$ | $\tau = 0.0511$ | 10.0% | **22.69%** (5,343 / 23,547) | 17.71% | **34.04%** | **0.2330** | 6,493 |
| **TCN**  | Val-$q_{95}$ | $\tau = 0.0820$ | 5.0%  | **13.85%** (3,262 / 23,547) | 20.52% | **24.93%** | **0.2251** | 4,104 |
| **TCN**  | Val-$q_{98}$ | $\tau = 0.1386$ | 2.0%  | **7.29%**  (1,716 / 23,547) | 24.74% | **16.70%** | **0.1994** | 2,280 |
| **TCN**  | Default Boundary | $\tau = 0.5000$ | N/A   | **0.31%**  (72 / 23,547)    | **41.46%** | 1.51%  | 0.0291     | 123 |

### Distinguishing Target Validation FPR from Measured Test FPR:
- On the validation set (recorded in April), the target FPRs (10%, 5%, 2%) define the quantile cutoff $\tau$ on baseline telemetry.
- On the test set (recorded in late May and early June), the measured FPR at these thresholds increases to **7.3%–8.2%** at the $q_{98}$ point and **13.5%–15.1%** at the $q_{95}$ point.
- **Physical Reason**: Seasonal transition into warmer weather in late May elevated ambient temperatures and normal compressor operating baselines relative to April, demonstrating the operational necessity of continuous or adaptive threshold calibration over extended operating periods.
