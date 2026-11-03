# Dataset & Telemetry Image Files Directory

This folder contains all high-resolution figures, dataset sensor waveform visualizations, and benchmark curves for the Metro Train Predictive Maintenance project.

---

## 1. Dataset Telemetry Image Files

| Image File Name | Image Title & Scope | Dataset Source & Physical Significance |
| :--- | :--- | :--- |
| **`raw_telemetry_waveforms_metropt2.png`** | MetroPT-2 Multi-Channel Sensor Waveforms | Direct visual plot of raw dataset channels (`TP2`, `TP3`, `H1`, `Motor_current`, `Oil_temperature`, `Reservoirs`, `DV_pressure`, `COMP`, `Towers`) showing compressor pumping cycles and pressure oscillation. |
| **`raw_telemetry_waveforms_metropt3.png`** | MetroPT-3 Multi-Channel Sensor Waveforms | Direct visual plot of raw dataset channels covering lubrication temperature, float switch state (`Oil_level`), motor current, and air production dynamics across 10-second intervals. |

---

## 2. Experimental & Benchmark Image Files

| Figure File Name | Figure Title & Scope | Research & Practical Significance |
| :--- | :--- | :--- |
| **`class_distribution.png`** | Log-Scale Telemetry Class Distribution | Demonstrates the extreme natural class imbalance across Train, Validation, and Held-out Test partitions for both MetroPT-2 and MetroPT-3 at the timestep level (<0.5% failure prevalence). |
| **`test_subclass_composition.png`** | Test Set Sequence Composition | Highlights the composition of positive test windows: isolating the subtle 2-Hour Pre-Failure Early Warning Horizon (120 windows) from ongoing catastrophic failure breakdown outages (~3,258 to 3,647 windows). |
| **`methodology_flowchart.png`** | Complete 8-Step Predictive Maintenance Pipeline | Illustrates the full end-to-end architecture: from raw 1 Hz telemetry ingestion, median aggregation, chronological splitting, training-only scaling, 1:1 windowing, deep learning benchmarking, dual-evaluation testing, to validation threshold calibration. |
| **`dual_evaluation_pr_curves.png`** | Dual-Evaluation Precision-Recall Curves | Compares model performance across Task 1 (Active Monitoring) vs Task 2 (Pure Anticipatory Warning), revealing how the 166:1 class imbalance governs precision in early warning. |
| **`threshold_calibration_curves.png`** | Unsupervised Threshold Calibration Curves | Illustrates precision, recall, and false positive rate trade-offs across validation quantiles ($q_{90}$, $q_{95}$, $q_{98}$), proving how calibration recovers recall from 0.00% collapse on out-of-distribution transfer without test label leakage. |
| **`pr_curves_comparison.png`** | Comparative PR-AUC Curves Across Architectures | Visualises the Precision-Recall curves of all evaluated sequence architectures against the horizontal no-skill baseline prevalence. |
| **`confusion_matrices.png`** | Normalized Confusion Matrices | Displays true positive, false positive, true negative, and false negative counts under standard decision thresholds ($\tau = 0.50$). |

---

## 3. Technical Specifications

- **Format**: Lossless Portable Network Graphics (`.png`)
- **Resolution**: 300 DPI publication quality
- **Dimensions**: 14 × 10 inches / publication grid
- **Domain Context**: Metro do Porto Air Production Unit (APU) compressor telemetry
