# Academic Project Defense & Viva Voce Q&A Preparation Guide

**Project Title**: Predictive Maintenance in Metro Trains Using Machine Learning and Deep Learning Techniques  
**Student Name**: Moorthy Chetan  
**Registration Number**: `24BIT0246`  
**Institution**: Vellore Institute of Technology (VIT)  
**Academic Year**: 2026–2027  
**GitHub Repository**: [https://github.com/Chetan0246/MetroTrain-Predictive-Maintenance](https://github.com/Chetan0246/MetroTrain-Predictive-Maintenance)

---

## Part 1: How to Address Submitted PDF Inconsistencies (If Spotted by Panel)

If an evaluator spots any of the three minor discrepancies in the submitted PDF, remain calm and professional. Answer transparently using the speaking points below:

### Question 1: MetroPT-3 Test Window Counts ($23,551 / 3,374$ vs. $23,547 / 3,378$)
* **Panel Question**: *"In Table 2 of Section 5, you listed 23,551 normal windows and 3,374 positive windows. But in Table 3, Section 10, and your confusion matrices, you report 23,547 normal and 3,378 positive windows. Why is there a 4-window difference?"*
* **Verbatim Response**:
  > *"Yes, professor. The total test sequence count of 26,925 is identical across all tables. The 23,551 figure in Table 2 was an initial count prior to a 4-sequence boundary stride alignment around the failure onset timestamp. The verified ground-truth numbers matching our confusion matrices, evaluation scripts, and calibration benchmarks are **23,547 normal windows and 3,378 positive windows** ($23,547 + 3,378 = 26,925$). We have synchronized this across our latest GitHub repository commit."*

### Question 2: MetroPT-2 Page 13 Narrative Typo ($TN = 18,725, FP = 1,222$)
* **Panel Question**: *"Your text takeaway on Page 13 states TN = 18,725 and FP = 1,222, but the confusion matrix graphic directly above it shows TN = 19,825 and FP = 122. Which one is correct?"*
* **Verbatim Response**:
  > *"The confusion matrix graphic and Task 2 benchmark table show the verified empirical outputs: **$TN = 19,825$ and $FP = 122$**, which correctly sum to our 19,947 actual negative test windows ($19,825 + 122 = 19,947$) and yield our documented test False Positive Rate of $0.61\%$. The text takeaway bullet had a typographical digit transposition ($18,725$ instead of $19,825$), but the actual matrix and tables reflect the verified calculations."*

### Question 3: MetroPT-3 Task 1 Positive % ($12.54\%$ vs. $12.55\%$)
* **Panel Question**: *"In some sections you state positive prevalence is 12.54%, while elsewhere you state 12.55%. Is this a discrepancy?"*
* **Verbatim Response**:
  > *"Both figures are mathematically exact representations at different levels of data aggregation:  
  > 1. **$12.54\%$** represents the raw 10-second telemetry timestep prevalence ($\frac{20,268\text{ positive timesteps}}{161,604\text{ total test timesteps}} = 12.5417\%$).  
  > 2. **$12.55\%$** represents the sequence-windowed dataset prevalence after constructing 10-minute sliding sequences of length $W=60$ with stride $S=6$ ($\frac{3,378\text{ positive windows}}{26,925\text{ total test windows}} = 12.54596\%$).  
  > Explaining this difference highlights the physical transition from discrete sampling to sequence windowing."*

---

## Part 2: Core Machine Learning & Methodology Defense Questions

### Question 4: "Why did you implement a Dual-Evaluation Framework? What is the difference between Task 1 and Task 2?"
* **Why Professors Ask**: In published literature, researchers frequently combine ongoing breakdown hours into the positive class and claim 98%+ precision for 'early failure prediction.'
* **Verbatim Response**:
  > *"In real railway maintenance, detecting an ongoing outage is an **active state monitoring task (Task 1)**, whereas detecting the subtle precursors 2 hours prior to breakdown is a **pure prognostic early-warning task (Task 2)**.  
  > In Task 1, positive windows include both the 2-hour warning window and the active breakdown outage ($N=3,378$ in MetroPT-3; $N=3,767$ in MetroPT-2). In Task 2, all observations during active repair outages are strictly excluded from the test target ($N=120$ warning windows, prevalence $0.60\%$). By separating these two tasks, our framework prevents artificial metric inflation and demonstrates true predictive utility."*

### Question 5: "In MetroPT-2 Task 2, your recall is 100%, but precision is only ~49.6%. Did the model fail?"
* **Why Professors Ask**: Tests whether you understand the mathematics of class imbalance and the base-rate effect.
* **Verbatim Response**:
  > *"No, professor. This is a direct mathematical consequence of the **Base-Rate Fallacy** under severe class imbalance:  
  > 1. In Task 2, pre-failure warning windows comprise only **$0.60\%$** of the held-out test set ($N_{\text{pos}} = 120$ vs. $N_{\text{neg}} = 19,947$, a 166:1 normal-to-fault ratio).  
  > 2. GRU achieved an exceptionally low test False Positive Rate of **$0.61\%$** across $19,947$ healthy windows ($FP = 122$).  
  > 3. Because normal observations outnumber warning windows by 166 to 1, a sub-1% false positive rate naturally produces an absolute count of false alarms ($122$) comparable to true events ($120$):  
  > $$\text{Precision} = \frac{TP}{TP + FP} = \frac{120}{120 + 122} \approx 49.59\%$$  
  > In real train fleet dispatch, capturing 100% of impending breakdowns with only ~2 false alarms per operating day is highly operational."*

### Question 6: "Why did MetroPT-3 suffer a complete recall collapse at threshold 0.50, and how did you calibrate it?"
* **Why Professors Ask**: Tests your understanding of distribution shift and leak-free threshold calibration.
* **Verbatim Response**:
  > *"In MetroPT-3, the training partition contained pneumatic **Air Leak** failures (manifesting as acute pressure drops in $TP2$ and $TP3$), whereas the held-out test partition contained an unobserved lubrication **Oil Leak** (manifesting as thermal rise in $Oil\_temperature$). Because the model had never seen a thermal breakdown during training, its predicted failure probabilities were muted between $0.10$ and $0.35$.  
  > At the arbitrary default threshold of $\tau = 0.50$, the model generated zero positive predictions. Rather than tuning thresholds on test labels (which would constitute data leakage), we calibrated operating thresholds using **unsupervised quantiles ($q_{90}, q_{95}, q_{98}$) strictly on healthy validation data**. Operating at the validation $q_{90}$ threshold ($\tau \approx 0.068$ for GRU) successfully recovered **$35.0\%$ test recall** on the unseen failure mode."*

### Question 7: "Why did you choose PR-AUC over ROC-AUC and Accuracy?"
* **Why Professors Ask**: Demonstrates understanding of evaluation metrics under extreme class imbalance.
* **Verbatim Response**:
  > *"Under severe class imbalance ($>96\%$ normal data), **Accuracy is deceptive**: a naive dummy classifier predicting 'Healthy' 100% of the time achieves **$99.4\%$ accuracy** while missing every failure.  
  > **ROC-AUC is artificially inflated** because the huge true-negative count ($TN \approx 20,000$) in the false positive rate denominator ($FPR = \frac{FP}{FP+TN}$) suppresses hundreds of false alarms. In contrast, **PR-AUC calculates Area Under the Precision-Recall curve**, focusing exclusively on the minority positive class and penalizing false alarms directly."*

### Question 8: "How did you ensure zero data leakage across your preprocessing and windowing pipeline?"
* **Why Professors Ask**: Lookahead leakage is the #1 reason time-series ML models fail when deployed.
* **Verbatim Response**:
  > *"We enforced three strict engineering safeguards:  
  > 1. **Strict Chronological Splitting**: We partitioned telemetry chronologically by date (Train: Jan–Apr, Val: Apr–May, Test: May–Jun). We strictly avoided random $k$-fold or shuffled splitting, which would allow models to peek into future timestamps.  
  > 2. **Leak-Free Scaling**: The StandardScaler mean and standard deviation parameters were fitted exclusively on the training partition and applied without modification to validation and test sets.  
  > 3. **Sequence Window Isolation**: Sequences ($W=60$, Stride $S=6$) were indexed within partition boundaries to guarantee zero window overlap across split points."*

---

## Part 3: Deep Learning Architectures (LSTM vs. GRU vs. TCN)

| Architecture | Model Configuration | MetroPT-2 (In-Distribution) | MetroPT-3 (Distribution Shift) | Engineering Strengths |
| :--- | :--- | :---: | :---: | :--- |
| **GRU** | 2 layers, hidden 64, dropout 0.2 | **Rank 1** (PR-AUC 0.9673 Task 1; 0.4335 Task 2) | Rank 2 (PR-AUC 0.1830) | Faster convergence, fewer parameters than LSTM, optimal gating for cyclic pressure dynamics. |
| **LSTM** | 2 layers, hidden 64, dropout 0.2 | **Rank 2** (PR-AUC 0.9669 Task 1; 0.4026 Task 2) | Rank 3 (PR-AUC 0.1783) | Effective cell state memory, slightly higher computational overhead. |
| **TCN** | Dilated causal residual convs ($d \in \{1, 2, 4\}$, $k=3$) | Rank 3 (PR-AUC 0.9543 Task 1; 0.3644 Task 2) | **Rank 1** (PR-AUC 0.1939 Task 1) | **Only model with positive recall at $\tau=0.50$ on unseen oil leak**. Exponential receptive field without recurrent vanishing gradients. |

### Key Talking Point on TCN:
> *"TCN's dilated causal residual convolutions allow the network's receptive field to expand exponentially with depth without losing temporal ordering or suffering from recurrent gradient degradation. Under out-of-distribution transfer (MetroPT-3), TCN captured multi-scale thermodynamic anomalies that recurrent baselines missed."*

---

## Part 4: Physical Domain Knowledge of the Metro Train APU

* **APU (Air Production Unit)**: Supplies compressed air ($8.5$–$10\text{ bar}$) to pneumatic braking assemblies, secondary air suspension bellows, pantograph actuators, and passenger door mechanisms.
* **Compressor Duty Cycle**: The intake valve (`COMP`) switches between pumping loaded ($0$) and offloaded unpressurized ($1$) states governed by the `MPG` pressure switch ($< 8.2\text{ bar}$).
* **Air Leak Signature**: Characterized by rapid pressure drops (`TP2`, `TP3`), increased motor current (`Motor_current`), and abnormal compressor cycling duty.
* **Oil Leak Signature**: Characterized by thermal degradation (`Oil_temperature` rising abnormally) and eventual float switch activation (`Oil_level`).

---

## Part 5: Rapid Defense Fact Sheet

```
Student:        Moorthy Chetan (24BIT0246) — VIT
Fleet 1:        MetroPT-3 (UCI #791) — 10,773,588 rows (10s resampled: 1,077,359 timesteps)
Fleet 2:        MetroPT-2 (Zenodo)   — 7,116,940 rows  (10s resampled: 711,694 timesteps)
Images:         0 (Pure multivariate sensor telemetry)
Features:       16 APU telemetry channels (continuous thermodynamic + binary switches)
Window Size:    W = 60 timesteps (10 minutes of operational telemetry)
Stride:         S = 6 timesteps (1-minute sliding progression)
Warning Lead:   2 Hours prior to documented maintenance intervention
Test Windows:   MetroPT-3: 23,547 Normal + 3,378 Positive = 26,925 Total (Task 2 Pos = 120)
                MetroPT-2: 19,947 Normal + 3,767 Positive = 23,714 Total (Task 2 Pos = 120)
Test FPR:       MetroPT-2 GRU: 122 / 19,947 = 0.61%
                MetroPT-2 LSTM: 124 / 19,947 = 0.62%
                MetroPT-2 TCN: 129 / 19,947 = 0.65%
Test Suite:     pytest (3 passed in 1.3s)
Repo:           https://github.com/Chetan0246/MetroTrain-Predictive-Maintenance
```
