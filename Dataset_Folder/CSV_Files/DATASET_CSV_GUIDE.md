# Dataset CSV Files Directory Guide

This folder contains the complete, full-scale raw telemetry datasets, continuous chronological subsets, and failure-period slices for the Metro Train Predictive Maintenance project.

---

## 1. Full Dataset CSV Files & Subsets in this Directory

| File Name | Description | Rows / Timesteps | File Size | Recommended Usage |
| :--- | :--- | :---: | :---: | :--- |
| **`MetroPT2.csv`** | **Full Complete Raw Dataset** for MetroPT-2 (Pneumatic Air Leak Failures campaign, April 12 – July 16, 2022). | **7,116,940 rows** | **1.2 GB** | Full deep learning model training, sequence window extraction, and benchmark reproduction. |
| **`MetroPT3.csv`** (alias: `dataset_train.csv`) | **Full Complete Raw Dataset** for MetroPT-3 (Lubrication Oil & Intake Failures campaign, Jan 1 – June 3, 2022). | **10,773,590 rows** | **1.6 GB** | Full cross-dataset transfer evaluation and out-of-distribution benchmark reproduction. |
| **`metropt2_10k_subset.csv`** | Continuous 10,000-row chronological slice of MetroPT-2 raw telemetry with all 21 columns. | 10,000 rows | ~1.6 MB | Fast local testing, model prototyping, and opening directly in spreadsheet tools. |
| **`metropt3_10k_subset.csv`** | Continuous 10,000-row chronological slice of MetroPT-3 raw telemetry with all 21 columns. | 10,000 rows | ~1.6 MB | Fast local testing and exploratory data analysis without waiting for multi-gigabyte loading. |
| **`metropt2_genuine_samples.csv`** | Labeled genuine records from MetroPT-2 across Normal, Pre-Failure Warning, and Active Breakdown states. | 13 rows | ~1.6 KB | Quick inspection in Excel/Notepad of physical sensor values across operating health regimes. |
| **`metropt3_genuine_samples.csv`** | Labeled genuine records from MetroPT-3 covering normal baselines, early warnings, and active fault intervals. | 13 rows | ~1.5 KB | Quick inspection of lubrication failure indicators and sump float states. |

---

## 2. Telemetry Columns Structure (21 Features)

Each CSV dataset contains 21 channels:
1. `timestamp`: ISO-8601 formatted date and time (`YYYY-MM-DD HH:MM:SS.mmm`)
2. `TP2`: Compressor delivery head pressure (bar)
3. `TP3`: Pneumatic panel input pressure (bar)
4. `H1`: Air dryer purge valve pressure (bar)
5. `DV_pressure`: Differential pressure across desiccant regeneration valve (bar)
6. `Reservoirs`: Auxiliary brake reservoir pressure (bar)
7. `Motor_current`: Phase current drawn by electric motor (Amperes)
8. `Oil_temperature`: Crankcase lubricant temperature (°C)
9. `Flow_meter`: Air delivery flow meter (if equipped)
10. `COMP`: Digital command signal (1 = motor pumping, 0 = idle unloader)
11. `DV_eletric`: Solenoid moisture expulsion switch (0/1)
12. `Towers`: Twin-tower desiccant alternating switch (0 = Tower B, 1 = Tower A)
13. `MPG`: Mechanical governor pressure switch (0/1)
14. `LPS`: Emergency low pressure safety alarm switch (0/1)
15. `Pressure_switch`: Auxiliary door/brake interlock switch (0/1)
16. `Oil_level`: Oil sump float switch (0 = Low oil alarm, 1 = Normal)
17. `Caudal_impulses`: Pulse air volume delivery counter
18. `GPS_lat`, `GPS_long`, `GPS_quality`, `GPS_speed`: Geographic positioning channels (excluded from model input to prevent route-memorization leakage).

---

## 3. Remote Download Mirrors

If downloading or mirroring the raw multi-gigabyte files from external repositories:
- **Zenodo Official Archive (CERN)**: [https://doi.org/10.5281/zenodo.6854240](https://doi.org/10.5281/zenodo.6854240)
- **UCI Machine Learning Repository**: [https://archive.ics.uci.edu/dataset/791/metropt+3+dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
- **Kaggle**: [https://www.kaggle.com/datasets/anthonytherrien/metropt-dataset-air-production-unit](https://www.kaggle.com/datasets/anthonytherrien/metropt-dataset-air-production-unit)
