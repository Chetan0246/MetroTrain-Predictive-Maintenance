# Dataset CSV Files Directory Guide

This folder contains verified CSV data samples, subsets, and failure-period telemetry slices from the MetroPT datasets.

---

## 1. Files in this Directory

| File Name | Description | Rows | File Size | Recommended Usage |
| :--- | :--- | :---: | :---: | :--- |
| **`metropt2_genuine_samples.csv`** | Labeled genuine records from MetroPT-2 representing Normal, Pre-Failure Warning, and Active Breakdown states. | 13 | ~1.6 KB | Quick inspection in Excel/Notepad of physical feature values across health regimes. |
| **`metropt3_genuine_samples.csv`** | Labeled genuine records from MetroPT-3 covering normal baselines, early warnings, and active fault intervals. | 13 | ~1.5 KB | Verifying feature differences between air-leak and lubrication failure states. |
| **`metropt2_10k_subset.csv`** | Continuous 10,000-row chronological slice of MetroPT-2 raw telemetry with all 21 columns. | 10,000 | ~1.8 MB | Fast local testing, model prototyping, and pandas exploratory analysis. |
| **`metropt3_10k_subset.csv`** | Continuous 10,000-row chronological slice of MetroPT-3 raw telemetry with all 21 columns. | 10,000 | ~1.8 MB | Testing preprocessing scripts without loading gigabyte-scale datasets. |

---

## 2. Telemetry Columns Structure

Each CSV file contains 21 raw columns:
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

## 3. Location of Full Multi-Gigabyte Raw CSV Datasets

For full large-scale deep learning model training, the complete raw CSV files are stored in the repository's data cache:
- **MetroPT-2 Full Raw CSV (1.2 GB)**: Located at `data/raw/metropt2/MetroPT2.csv` (7,116,940 rows)
- **MetroPT-3 Full Raw CSV (1.6 GB)**: Located at `data/raw/metropt3/dataset_train.csv` (10,773,590 rows)

They can also be downloaded directly from the official Zenodo link:
`https://zenodo.org/records/6854240`
