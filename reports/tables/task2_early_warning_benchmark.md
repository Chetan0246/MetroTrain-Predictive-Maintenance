# Task 2 Benchmark: Anticipatory Pre-Failure Warning (Pure 2-Hour Lead Time)

> [!IMPORTANT]
> **Task 2 Scope**: The positive class ($y=1$) consists **strictly of the documented 2-hour pre-failure window** ($[T_{\text{failure}} - 2\,\text{hours}, T_{\text{failure}}]$). **All observations during the active failure outage are explicitly excluded** from the test evaluation. This measures pure prognostic forecasting before fault onset.

| Dataset | Model Architecture | PR-AUC (Primary) | Recall ($\tau=0.5$) | Precision ($\tau=0.5$) | F1-Score ($\tau=0.5$) | Test False-Positive Rate (FPR) | Test Pre-Failure Prevalence | Excluded Active Fault Windows | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetroPT-2** | **GRU**  | **0.4335** | **1.0000** | 49.59% | **0.6630** | **0.61%** (122 / 19,947) | 0.60% (120 / 20,067) | 3,647 windows | 🥇 Rank 1 |
| **MetroPT-2** | **LSTM** | **0.4026** | **1.0000** | 49.18% | **0.6593** | **0.62%** (124 / 19,947) | 0.60% (120 / 20,067) | 3,647 windows | 🥈 Rank 2 |
| **MetroPT-2** | **TCN**  | **0.3644** | 0.9917     | 47.98% | **0.6467** | **0.65%** (129 / 19,947) | 0.60% (120 / 20,067) | 3,647 windows | 🥉 Rank 3 |
| **MetroPT-3** | **GRU**  | **0.0207** | 0.0000     | 0.00%  | 0.0000     | **0.12%** (29 / 23,547)   | 0.51% (120 / 23,667) | 3,258 windows | 🥇 Rank 1 |
| **MetroPT-3** | **TCN**  | **0.0177** | 0.0000     | 0.00%  | 0.0000     | **0.31%** (72 / 23,547)   | 0.51% (120 / 23,667) | 3,258 windows | 🥈 Rank 2 |
| **MetroPT-3** | **LSTM** | **0.0135** | 0.0000     | 0.00%  | 0.0000     | **0.00%** (0 / 23,547)    | 0.51% (120 / 23,667) | 3,258 windows | 🥉 Rank 3 |

### Methodological & Mathematical Analysis:

#### 1. Mathematical Explanation of Observed ~49.6% Precision in MetroPT-2
The observed precision of ~49.6% is not an arbitrary limitation, but a direct consequence of the **base-rate effect under extreme class imbalance**:
- In the MetroPT-2 Task 2 test set, positive warning windows are extremely rare ($N_{\text{pos}} = 120$ out of $N_{\text{total}} = 20,067$, an empirical prevalence of **0.60%**).
- At $\tau = 0.50$, the GRU model achieved perfect sensitivity, identifying all $TP = 120$ true warning windows ($100\%$ Recall).
- Concurrently, the model generated only $FP = 122$ false alarms across the 19,947 normal operating windows, achieving an exceptionally low test False-Positive Rate of:
  $$\text{FPR} = \frac{FP}{TN + FP} = \frac{122}{19,825 + 122} = \frac{122}{19,947} \approx 0.61\%$$
- However, because normal windows outnumber pre-failure windows by approximately **166 to 1**, even a minor false-positive rate of $0.61\%$ produces an absolute count of false alarms ($FP = 122$) comparable to the entire pool of true events ($TP = 120$):
  $$\text{Precision} = \frac{TP}{TP + FP} = \frac{120}{120 + 122} = \frac{120}{242} \approx 49.59\%$$
- This demonstrates that while the model successfully captured the pre-failure signal with high sensitivity, operational deployment must account for the high ratio of healthy background operation when interpreting alerts.

#### 2. Cross-Fault Generalization Barrier in MetroPT-3
- In MetroPT-3, models trained on pneumatic Air Leaks achieved low PR-AUC (~0.014–0.021) when evaluating purely on the 2-hour pre-failure window of the unseen Oil Leak.
- The physical precursors of the lubrication failure did not exhibit the pressure drops characteristic of pneumatic air leaks during the 2 hours preceding failure onset.
