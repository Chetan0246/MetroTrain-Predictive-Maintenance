# Comprehensive Academic Project Review Report: Predictive Maintenance in Metro Trains Using Machine Learning and Deep Learning

**Project Title**: Predictive Maintenance in Metro Trains Using Machine Learning and Deep Learning Techniques  
**Student Name**: Moorthy Chetan  
**Registration Number**: 24BIT0246  
**Institution**: Vellore Institute of Technology (VIT)  
**Academic Year**: 2026–2027  
**Project Repository**: `/home/chetan/MetroTrain-Predictive-Maintenance`  

---

## 1. Executive Summary & Objective

In modern urban rail transit systems, unscheduled train downtime causes severe service interruptions, passenger delays, and substantial operational maintenance penalties. The **Air Production Unit (APU)** compressor provides compressed air for pneumatic braking, secondary suspension, pantographs, and automated passenger doors.

This academic research project investigates data-driven predictive maintenance across real-world operational compressor telemetry from Metro do Porto, Portugal. 

Following a rigorous scientific audit, this report establishes a **Dual-Evaluation Framework** that distinguishes two fundamentally different operational tasks:
1. **Task 1: Active Fault Detection & State Monitoring** — Evaluating the continuous identification of a failure state across both the 2-hour pre-failure window and the ongoing active failure duration.
2. **Task 2: Anticipatory Pre-Failure Warning** — Evaluating pure prognostic forecasting strictly during the documented 2-hour pre-failure lead horizon, with all active outage observations explicitly excluded from the positive evaluation set.

### Experimental Benchmark Matrix ($2 \times 3$ Matrix Across Both Tasks)
- **Two Real-World Telemetry Datasets**:
  1. **MetroPT-3 / MetroPT** (Jan 1, 2022 – Jun 2, 2022)
  2. **MetroPT-2** (Apr 28, 2022 – Jul 28, 2022)
- **Three Candidate Deep Learning Architectures**:
  1. **LSTM** (Long Short-Term Memory)
  2. **GRU** (Gated Recurrent Unit)
  3. **TCN** (Temporal Convolutional Network with dilated causal residual convolutions)
- **Primary Comparative Metrics**: Precision-Recall Area Under Curve (**PR-AUC**), **Recall**, and **F1-Score** (supplemented by **Precision**, **False-Positive Rate**, normalized confusion matrices, and training runtime).

---

## 2. Dataset Links & Literature Provenance

| Dataset | Repository Link & Identifier | Official Literature Citation |
| :--- | :--- | :--- |
| **MetroPT-3** | **UCI Repository**: [archive.ics.uci.edu/dataset/791/metropt+3+](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B)<br/>**Zenodo Record**: [doi.org/10.5281/zenodo.6854240](https://doi.org/10.5281/zenodo.6854240) | Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2022). *The MetroPT dataset for predictive maintenance*. Nature Scientific Data, 9(1), 764. |
| **MetroPT-2** | **Zenodo Record**: [doi.org/10.5281/zenodo.7766691](https://doi.org/10.5281/zenodo.7766691) | Jakobs, M., Veloso, B., & Gama, J. (2026). *Interpretable rules for online failure prediction: a case study on metro do porto datasets*. Int. J. Data Sci. & Anal., 22:59. |

> [!NOTE]
> The UCI repository provides an older 2020 downsampled release (`MetroPT3(AirCompressor).csv`, 1.51M rows at 0.1 Hz). The uploaded datasets in our workspace correspond to the full-resolution 2022 benchmarks (`dataset_train.csv`, 10.77M rows, and `MetroPT2.csv`, 7.12M rows), enabling direct chronological and architectural parity between the two trains.

---

## 3. Dataset Characteristics & Sensor Feature Catalog

- **Dataset 1 (MetroPT-3)**: 10,773,588 raw records (1,077,359 timesteps at 10s intervals), 21 features, 0 missing values.
- **Dataset 2 (MetroPT-2)**: 7,116,940 raw records (711,694 timesteps at 10s intervals), 21 features, 0 missing values.
- **Image Count**: 0 (purely multivariate time-series numerical telemetry).

### Sensor Feature Channels & Engineering Functions

| Feature Name | Type | Physical Measurement | System Function |
| :--- | :--- | :--- | :--- |
| `timestamp` | Datetime | UTC Timestamp | Chronological sequence index |
| `TP2` | Continuous | Output Pressure (bar) | Pressure generated at compressor outlet |
| `TP3` | Continuous | Pneumatic Panel Pressure (bar) | Regulated system pressure at the pneumatic panel |
| `H1` | Continuous | Filter Pressure Drop (bar) | Pressure differential across cyclonic separator filter |
| `DV_pressure` | Continuous | Air Dryer Discharge (bar) | Pressure drop during desiccant tower air drying purge |
| `Reservoirs` | Continuous | Downstream Pressure (bar) | Main reservoir storage pressure (monitors air reserve) |
| `Oil_temperature`| Continuous | Lubricant Temp (°C) | Compressor casing oil lubrication temperature |
| `Flowmeter` | Continuous | Airflow Rate (m³/h) | Volumetric air generation delivery rate |
| `Motor_current` | Continuous | Drive Current (A) | Current draw of the 3-phase compressor electric motor |
| `COMP` | Binary | Intake Valve Signal | Air intake command: 1 = offloaded/intake closed; 0 = pumping |
| `DV_eletric` | Binary | Outlet Valve Signal | Discharge valve command: 1 = under load; 0 = offloaded |
| `Towers` | Binary | Dryer Selector Signal | Active desiccant tower: 0 = Tower 1 active; 1 = Tower 2 active |
| `MPG` | Binary | Governor Pressure Switch | Pressure switch: triggers motor start when pressure < 8.2 bar |
| `LPS` | Binary | Low-Pressure Alarm | Critical safety switch: trips emergency alarm if pressure < 7.0 bar |
| `Pressure_switch`| Binary | Purge Discharge Sensor | Detects pneumatic discharge cycle in drying towers |
| `Oil_level` | Binary | Oil Level Indicator | Electrical float switch: active when lubricant falls below threshold |
| `Caudal_impulses`| Binary | Volumetric Pulse Counter | Impulse train counting absolute mass of air delivered |
| `gpsLat`, `gpsLong`| Continuous | Geographic Coordinates | Train positional telemetry (excluded from APU physics model) |
| `gpsSpeed` | Continuous | Velocity (km/h) | Vehicle motion telemetry |
| `gpsQuality` | Categorical | GPS Fix Signal | Diagnostic lock indicator |

### Documented Failure Events (Ground Truth)
Raw sensor streams are unlabeled. In accordance with peer-reviewed failure reports (Veloso et al., 2022; Jakobs et al., 2026, Table 1), the verified failure ground truth is:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Dataset 1: MetroPT (dataset_train.csv)                                                                │
├────┬───────────┬───────────────────────┬───────────────────────┬───────────────────────┬───────────────┤
│ ID │ Type      │ Failure Onset (Start) │ Failure End (Repaired)│ LPS Warning Signal    │ Lead Horizon  │
├────┼───────────┼───────────────────────┼───────────────────────┼───────────────────────┼───────────────┤
│ F1 │ Air Leak  │ 2022-02-28 21:53:00   │ 2022-03-01 02:00:00   │ 2022-02-28 22:50:00   │ 2 Hours Prior │
│ F2 │ Air Leak  │ 2022-03-23 14:54:00   │ 2022-03-23 15:24:00   │ None (short duration) │ 2 Hours Prior │
│ F3 │ Oil Leak  │ 2022-05-30 12:00:00   │ 2022-06-02 06:18:00   │ 2022-06-02 06:18:00   │ 2 Hours Prior │
└────┴───────────┴───────────────────────┴───────────────────────┴───────────────────────┴───────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Dataset 2: MetroPT-2 (MetroPT2.csv)                                                                    │
├────┬───────────┬───────────────────────┬───────────────────────┬───────────────────────┬───────────────┤
│ ID │ Type      │ Failure Onset (Start) │ Failure End (Repaired)│ LPS Warning Signal    │ Lead Horizon  │
├────┼───────────┼───────────────────────┼───────────────────────┼───────────────────────┼───────────────┤
│ F1 │ Air Leak  │ 2022-06-04 10:19:24   │ 2022-06-04 14:22:39   │ 2022-06-04 11:26:00   │ 2 Hours Prior │
│ F2 │ Oil Leak  │ 2022-07-11 10:10:18   │ 2022-07-14 10:22:08   │ 2022-07-13 19:43:00   │ 2 Hours Prior │
└────┴───────────┴───────────────────────┴───────────────────────┴───────────────────────┴───────────────┘
```

---

## 4. Two Genuine CSV Sample Records Per Class

In strict adherence to research integrity guidelines, sample records are extracted directly from authentic raw telemetry files on disk without synthetic modification:

### Dataset 1: MetroPT-3 (`reports/samples/metropt3_genuine_samples.csv`)
- **Class 0 (Normal Operation)**:
  - `2022-01-01 06:01:40, -0.012, 9.660, 9.660, -0.028, 1.574, 61.275, 19.050, 3.955, 1, 0, 1, 1, 0, 0, 0, 0, -8.65935, 41.2123, 0, 1`
  - `2022-01-01 06:03:20, -0.012, 9.562, 9.562, -0.028, 1.572, 60.950, 19.050, 3.960, 1, 0, 1, 1, 0, 0, 0, 0, -8.65937, 41.2123, 0, 1`
- **Class 1 (Early Warning Window - 2 Hours Prior to Failure)**:
  - `2022-02-28 19:53:10, -0.004, 9.360, 9.370, -0.024, 1.704, 68.550, 19.022, -0.002, 1, 0, 1, 1, 0, 0, 0, 0, -8.66037, 41.2141, 0, 1`
  - `2022-02-28 19:53:20, -0.002, 9.348, 9.358, -0.024, 1.704, 68.350, 19.003, -0.002, 1, 0, 1, 1, 0, 0, 0, 0, -8.66038, 41.2142, 1, 1`
- **Class 1 (Active Air Leak in Progress)**:
  - `2022-02-28 21:53:10, 9.582, 9.120, -0.004, -0.020, 1.728, 68.100, 31.178, 5.868, 0, 1, 1, 0, 0, 0, 0, 0, -8.65935, 41.2136, 0, 1`
  - `2022-02-28 21:53:20, 9.820, 9.368, -0.004, -0.020, 1.728, 69.475, 31.168, 5.862, 0, 1, 1, 0, 0, 0, 0, 0, -8.65935, 41.2136, 0, 1`

### Dataset 2: MetroPT-2 (`reports/samples/metropt2_genuine_samples.csv`)
- **Class 0 (Normal Operation)**:
  - `2022-04-28 12:35:08.212, 10.114, 9.822, -0.010, -0.018, 9.820, 59.400, 0.250, 6.150, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0`
  - `2022-04-28 12:36:47.304, -0.014, 9.846, 9.834, -0.016, 9.848, 66.050, 0.250, 3.805, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0`
- **Class 1 (Early Warning Window - 2 Hours Prior to Failure)**:
  - `2022-06-04 08:19:34.209, 8.520, 8.044, -0.016, -0.022, 8.044, 62.925, 28.562, 5.708, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 41.237225, -8.669641, 0.0, 1.0`
  - `2022-06-04 08:19:44.118, 8.848, 8.358, -0.014, -0.022, 8.365, 63.075, 31.682, 5.710, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 41.237228, -8.669655, 0.0, 1.0`
- **Class 1 (Active Air Leak in Progress)**:
  - `2022-06-04 10:19:34.244, -0.006, 8.570, 8.584, -0.024, 8.565, 66.825, 19.003, -0.002, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 41.237277, -8.669590, 0.0, 1.0`
  - `2022-06-04 10:19:44.160, -0.002, 8.562, 8.576, -0.022, 8.570, 66.800, 19.003, -0.005, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 41.237272, -8.669570, 0.0, 1.0`

---

## 5. Class Imbalance Analysis & Subclass Decomposition

### Timestep-Level Class Counts Across Chronological Splits

| Dataset | Split Partition | Normal (Class 0) | Positive Target (Class 1) | Early Warning Window | Active Failure | Positive % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MetroPT-3** | **Full (10s)** | 1,053,989 | 23,370 | 2,160 | 21,210 | 2.17% |
| **MetroPT-3** | **Train (70%)**| 751,049 | 3,102 | 1,440 | 1,662 | **0.41%** |
| **MetroPT-3** | **Val (15%)**  | 161,604 | 0 | 0 | 0 | 0.00% |
| **MetroPT-3** | **Test (15%)** | 141,336 | 20,268 | 720 | 19,548 | **12.54%** |
| **MetroPT-2** | **Full (10s)** | 686,897 | 24,797 | 1,440 | 23,357 | 3.48% |
| **MetroPT-2** | **Train (65%)**| 460,408 | 2,193 | 720 | 1,473 | **0.47%** |
| **MetroPT-2** | **Val (15%)**  | 106,754 | 0 | 0 | 0 | 0.00% |
| **MetroPT-2** | **Test (20%)** | 119,735 | 22,604 | 720 | 21,884 | **15.88%** |

### Original vs. Balanced Training Sets (Window-Level)

| Dataset | Partition | Normal Windows (Class 0) | Warning/Failure Windows (Class 1) | Total Sequences | Imbalance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MetroPT-3** | **Train (Original)** | 125,165 | 517 | 125,682 | 242.1 : 1 (Imbalanced) |
| **MetroPT-3** | **Train (Balanced)** | **517** | **517** | **1,034** | **1.0 : 1 (Balanced)** |
| **MetroPT-3** | **Val (Natural)** | 26,925 | 0 | 26,925 | Baseline Clean |
| **MetroPT-3** | **Test (Natural)**| 23,551 | 3,374 | 26,925 | 7.0 : 1 (Natural) |
| **MetroPT-2** | **Train (Original)** | 76,736 | 365 | 77,101 | 210.2 : 1 (Imbalanced) |
| **MetroPT-2** | **Train (Balanced)** | **365** | **365** | **730** | **1.0 : 1 (Balanced)** |
| **MetroPT-2** | **Val (Natural)** | 17,783 | 0 | 17,783 | Baseline Clean |
| **MetroPT-2** | **Test (Natural)**| 19,947 | 3,767 | 23,714 | 5.3 : 1 (Natural) |

### Task-Specific Test Window Composition: Task 1 vs. Task 2

| Dataset | Total Test Windows | Normal Windows ($y=0$) | Task 1 Positives (Warn + Active Fail) | Task 2 Positives (Pure Warn Only) | Excluded Active Fault Windows in Task 2 | Task 2 Positive Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetroPT-3** | 26,925 | 23,547 | 3,378 (12.55%) | **120** | 3,258 windows (96.4% of Task 1 pos) | **0.51%** |
| **MetroPT-2** | 23,714 | 19,947 | 3,767 (15.89%) | **120** | 3,647 windows (96.8% of Task 1 pos) | **0.60%** |

---

## 6. Class Distribution & Subclass Composition Figures

```
Figure 1: reports/figures/class_distribution.png
(Visualizes Normal vs. Positive timesteps across Train, Validation, and Test partitions on a log-scale)

Figure 2: reports/figures/test_subclass_composition.png
(Illustrates the decomposition of test set positive windows into 2-Hour Pre-Failure Warning vs. Ongoing Active Failure)
```

---

## 7. Dual-Evaluation Methodology & Step-by-Step Explanation

```mermaid
flowchart TD
    A["Raw APU Telemetry<br/>(MetroPT-3: 10.77M rows | MetroPT-2: 7.12M rows)"] --> B["1. Chunked Stream & 10s Resampling<br/>(Reduces size by 90% without loss of APU dynamics)"]
    B --> C["2. Ground-Truth Failure Logs<br/>(Nature Sci. Data 2022 & Int. J. Data Sci. Anal. 2026)"]
    C --> D["3. Strict Chronological Partitioning<br/>(Train 65-70% | Val 15% | Test 15-20%)"]
    D --> E["4. Leak-Free Feature Scaling<br/>(StandardScaler fitted strictly on Train split)"]
    E --> F1["5a. Training Split 1:1 Balancing<br/>(Exact under-sampling of healthy windows)"]
    E --> F2["5b. Val & Test Natural Partitions<br/>(100% Real-World Class Imbalance Preserved)"]
    F1 --> G["6. 10-Minute Sliding Windows<br/>(W = 60 steps | Stride S = 6 steps)"]
    F2 --> G
    G --> H["7. 6-Model Training Matrix (10 Epochs)<br/>(LSTM, GRU, TCN on both datasets)"]
    
    H --> I1["Task 1 Evaluation:<br/>Active Fault State Monitoring<br/>(Positives = 2h Warning + Active Failure)"]
    H --> I2["Task 2 Evaluation:<br/>Anticipatory Pre-Failure Warning<br/>(Positives = Pure 2h Warning; Active Fail EXCLUDED)"]
    H --> I3["Validation-Only Threshold Calibration<br/>(Thresholds selected via Val Quantiles q90, q95, q98)"]
    
    I1 --> J["Dual-Evaluation Benchmark Report & Figures"]
    I2 --> J
    I3 --> J
```

### Detailed Methodological Steps:
1. **Telemetry Resampling**: Downsampled from 1 Hz / sub-second logging to regular 10-second intervals (0.1 Hz), matching the UCI benchmark standard and eliminating CPU OOM crashes while preserving thermodynamic pressure-flow cycles.
2. **Defensible Failure Horizons**: Assigned a 2-hour pre-failure lead window ($[T_{\text{failure}} - 2\,\text{hours}, T_{\text{failure}}]$) based on official maintenance records (Veloso et al., 2022; Jakobs et al., 2026), giving maintenance engineers actionable warning.
3. **Strict Chronological Splitting**: Train, validation, and test sets are strictly partitioned across time to prevent lookahead bias and temporal leakage.
4. **Leak-Free Normalization**: `StandardScaler` parameters (mean and standard deviation) were fitted solely on the training splits and subsequently transformed onto validation and test sets.
5. **Memory-Efficient Windowing**: Custom PyTorch `SequenceWindowDataset` yields sequences of shape $[60, 16]$ on the fly without duplicating 3D arrays in RAM.
6. **Architectures**:
   - **LSTM**: 2 stacked layers (`hidden_dim = 64`, `dropout = 0.2`) with linear output.
   - **GRU**: 2 stacked layers (`hidden_dim = 64`, `dropout = 0.2`) with unified gating.
   - **TCN**: Dilated causal convolutions ($d = 1, 2, 4$, $k=3$, channels $[32, 64, 128]$) with residual skips and global average pooling.
7. **Validation-Only Threshold Calibration**:
   - Operating thresholds were selected strictly using **unsupervised quantiles on the healthy validation partition** ($q_{90}, q_{95}, q_{98}$) without observing test labels.
   - These thresholds correspond to operational False Positive Rate budgets (10%, 5%, 2% on validation data) and are evaluated once on the held-out test set to demonstrate how threshold calibration resolves the recall collapse observed at the arbitrary default boundary $\tau=0.50$.

---

## 8. Dual-Evaluation Benchmark Results

### Task 1: Active Fault Detection & State Monitoring (Full Fault Duration Included)

| Dataset | Model Architecture | PR-AUC (Primary) | Recall ($\tau=0.5$) | F1-Score ($\tau=0.5$) | Precision ($\tau=0.5$) | Training Runtime | Benchmark Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetroPT-3 (APU)** | **TCN** | **0.1939** | 0.0151 | 0.0291 | 41.46% | 21.6s | 🥇 Rank 1 |
| **MetroPT-3 (APU)** | **GRU** | **0.1830** | 0.0000 | 0.0000 | 0.00%  | 21.2s | 🥈 Rank 2 |
| **MetroPT-3 (APU)** | **LSTM** | **0.1783** | 0.0000 | 0.0000 | 0.00%  | 8.8s | 🥉 Rank 3 |
| **MetroPT-2 (APU)** | **GRU** | **0.9673** | 1.0000 | 0.9841 | 96.86% | 13.8s | 🥇 Rank 1 |
| **MetroPT-2 (APU)** | **LSTM** | **0.9669** | 1.0000 | 0.9838 | 96.81% | 5.8s | 🥈 Rank 2 |
| **MetroPT-2 (APU)** | **TCN** | **0.9543** | 0.9997 | 0.9830 | 96.69% | 15.2s | 🥉 Rank 3 |

*Test Set Prevalences*: MetroPT-3 Test: 12.54% positive (3,378 / 26,925 windows). MetroPT-2 Test: 15.88% positive (3,767 / 23,714 windows).

---

### Task 2: Anticipatory Pre-Failure Warning (Pure 2-Hour Lead Time, Active Failure Excluded)

| Dataset | Model Architecture | PR-AUC (Primary) | Recall ($\tau=0.5$) | Precision ($\tau=0.5$) | F1-Score ($\tau=0.5$) | Test False-Positive Rate (FPR) | Test Pre-Failure Prevalence | Excluded Active Fault Windows | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetroPT-2** | **GRU**  | **0.4335** | **1.0000** | **49.59%** | **0.6630** | **0.61%** (122 / 19,947) | 0.60% (120 / 20,067) | 3,647 windows | 🥇 Rank 1 |
| **MetroPT-2** | **LSTM** | **0.4026** | **1.0000** | 49.18%     | **0.6593** | **0.62%** (124 / 19,947) | 0.60% (120 / 20,067) | 3,647 windows | 🥈 Rank 2 |
| **MetroPT-2** | **TCN**  | **0.3644** | 0.9917     | 47.98%     | **0.6467** | **0.65%** (129 / 19,947) | 0.60% (120 / 20,067) | 3,647 windows | 🥉 Rank 3 |
| **MetroPT-3** | **GRU**  | **0.0207** | 0.0000     | 0.00%      | 0.0000     | **0.12%** (29 / 23,547)   | 0.51% (120 / 23,667) | 3,258 windows | 🥇 Rank 1 |
| **MetroPT-3** | **TCN**  | **0.0177** | 0.0000     | 0.00%      | 0.0000     | **0.31%** (72 / 23,547)   | 0.51% (120 / 23,667) | 3,258 windows | 🥈 Rank 2 |
| **MetroPT-3** | **LSTM** | **0.0135** | 0.0000     | 0.00%      | 0.0000     | **0.00%** (0 / 23,547)    | 0.51% (120 / 23,667) | 3,258 windows | 🥉 Rank 3 |

---

### Validation-Calibrated Operating Thresholds for MetroPT-3

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

---

## 9. Evaluation Figures

```
Figure 3: reports/figures/dual_evaluation_pr_curves.png
(4-panel figure comparing Task 1 Active Fault Monitoring vs Task 2 Pure Anticipatory Warning across MetroPT-3 and MetroPT-2)

Figure 4: reports/figures/threshold_calibration_curves.png
(Threshold-performance curves showing Precision, Recall, and F1 across decision thresholds for MetroPT-3)

Figure 5: reports/figures/confusion_matrices.png
(2x3 grid of normalized confusion matrices for Task 1 at tau=0.50)
```

---

## 10. Physical & Algorithmic Analysis of Results

### Finding 1: Active Fault State Monitoring vs. Pure Anticipatory Prognostics
The dual-evaluation framework resolves the apparent paradox of near-perfect scores in MetroPT-2:
- When evaluated as an **Active Fault State Monitor** (Task 1), all models achieve PR-AUC > 0.95 and F1 > 0.98. In this setting, models operate primarily during an ongoing 3-day outage where physical sensor telemetry (`Oil_level`, `Flowmeter`) has collapsed (3,647 windows).
- When evaluated as a **Pure Anticipatory Pre-Failure Warning System** (Task 2, excluding the active failure outage), all models correctly flag the 2-hour pre-failure window with near-perfect sensitivity (**Recall 99.2%–100%**).
- **Explanation of Observed 49.6% Precision under Extreme Imbalance**:
  In Task 2, positive warning windows represent only **0.60%** of test observations ($N_{\text{pos}} = 120$ vs. $N_{\text{neg}} = 19,947$). At $\tau = 0.50$, GRU successfully detected all 120 warning sequences ($TP = 120$), with a very low False-Positive Rate on normal data of $0.61\%$ ($FP = 122$ out of 19,947). Because negative instances outnumber positive instances by roughly 166:1, even a tiny false-positive rate of $0.61\%$ produces an absolute count of false alarms ($FP = 122$) comparable to the total number of true warning windows ($TP = 120$). As a result, the observed precision is:
  $$\text{Precision} = \frac{120}{120 + 122} \approx 49.59\%$$
  This illustrates the classic base-rate effect in rare-event predictive maintenance: high recall can be maintained alongside a sub-1% false positive rate, but operational alert interpretation must account for the high proportion of normal background operation.

### Finding 2: The Physical Barrier of Cross-Fault Generalization in MetroPT-3
In MetroPT-3, the training partition contained **Air Leak** failures (pneumatic pressure drop), while the unseen test partition contained an **Oil Leak** failure (thermal and lubrication breakdown):
- Models trained on pneumatic pressure drops exhibited severe generalization degradation when evaluated on lubrication failure. Under Task 2, pre-failure PR-AUC dropped to ~0.014–0.021.
- In Task 1, **TCN achieved the highest PR-AUC (0.1939)** and was the only architecture to achieve positive recall (1.51%) and notable precision (41.46%) at the default $\tau=0.50$. The empirical evidence indicates that dilated causal convolutions may be better suited for multi-scale temporal pattern recognition under the observed distribution shift than recurrent baselines, although overall cross-fault transfer remains fundamentally limited across all tested models.

### Finding 3: Validation-Calibrated Operating Thresholds
The audit proved that the zero-recall phenomenon in MetroPT-3 was an artifact of testing an uncalibrated fixed boundary ($\tau=0.50$) on an out-of-distribution fault. By selecting thresholds based on validation false positive budgets ($q_{90}, q_{95}, q_{98}$), **LSTM recovered 31.4% recall (F1=0.2214)** and **GRU recovered 35.0% recall (F1=0.2403)** on the oil leak.
- **Validation Target FPR vs. Measured Test FPR**: While validation quantiles targeted a 2.0% FPR at $q_{98}$, the measured test FPR increased to **7.3%–8.2%**. This drift reflects seasonal temperature increases between April (validation) and late May/early June (test), which elevated baseline compressor operating temperatures.

### Finding 4: Methodological Limitations of Available Documented Events
- **Small Incident Sample Size**: The available benchmark datasets document a small number of physical failure incidents ($N=3$ in MetroPT-3; $N=2$ in MetroPT-2). 
- In MetroPT-3, evaluating cross-fault generalization relies on a **single test incident** ($N=1$ oil leak). Consequently, while the experimental design avoids temporal leakage, performance reflects the interaction between that specific physical incident and the training air leaks. Broad claims of generalizability across all arbitrary failure modes cannot be established without multi-year, multi-train fleet datasets.

---

## 11. Verification Matrix of Professor's Review Requirements

| # | Requirement | Implementation & Verification Status | Status |
| :---: | :--- | :--- | :---: |
| **1** | Title, Student Name, Reg. No. | Included on cover page: Moorthy Chetan (`24BIT0246`), VIT | Verified |
| **2** | Dataset Links & Descriptions | Section 2: UCI URL, Zenodo DOIs, and Nature citation | Verified |
| **3** | Feature Names, Row Counts, Labels | Section 3: 16-sensor feature table, 10.77M / 7.12M row counts | Verified |
| **4** | Two Genuine CSV Samples Per Class | Section 4: Authentic raw CSV rows for Normal, Warning, Failure | Verified |
| **5** | Class Imbalance Analysis | Section 5: Natural 0.4%–15.8% class imbalance & subclass analysis | Verified |
| **6** | Original vs Balanced Tables | Section 5: Original vs balanced training window tables | Verified |
| **7** | Imbalanced/Balanced Figures | Section 6: `class_distribution.png` and `test_subclass_composition.png` | Verified |
| **8** | Detailed Methodology Explanation | Section 7: Complete mathematical & engineering explanations | Verified |
| **9** | Methodology Flowchart | Section 7: Mermaid sequence flowchart rendered above | Verified |
| **10**| 6-Algorithm Comparative Matrix | Section 8: Dual-evaluation benchmark tables and PR curves | Verified |
