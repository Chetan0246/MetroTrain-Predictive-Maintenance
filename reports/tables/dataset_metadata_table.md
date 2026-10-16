# Dataset Metadata & Structural Summary

| Attribute | Dataset 1: MetroPT (APU Compressor) | Dataset 2: MetroPT-2 (APU Compressor) |
| :--- | :--- | :--- |
| **File Path** | `data/raw/metropt3/dataset_train.csv` | `data/raw/metropt2/MetroPT2.csv` |
| **Source Repository** | Zenodo Record 6854240 (Nature *Sci. Data* 2022) | Zenodo Record 7766691 |
| **Total Rows** | 10,773,588 | 7,116,940 |
| **Total Columns** | 21 | 21 |
| **Start Timestamp** | `2022-01-01 06:00:00` | `2022-04-28 12:33:29.120` |
| **End Timestamp** | `2022-06-02 15:49:53` | `2022-07-28 12:24:34.839` |
| **Sampling Rate** | 1.0 Hz (1 second) | $\approx$ 1.0 Hz (sub-second resolution) |
| **Missing Values** | Zero missing values across all columns | Zero missing values across all columns |
| **Identified Failures** | 3 (2 Air Leaks, 1 Oil Leak) | 2 (1 Air Leak, 1 Oil Leak) |
| **Early Warning Window** | 2 Hours (7,200 seconds prior to onset) | 2 Hours (7,200 seconds prior to onset) |

## Sensor Features & Engineering Roles

| Column Name | Type | Physical Meaning | Engineering Role |
| :--- | :--- | :--- | :--- |
| `timestamp` | Datetime | Telemetry logging timestamp | Temporal index (chronological split) |
| `TP2` | Continuous (bar) | Pressure on compressor output | Analogue sensor feature |
| `TP3` | Continuous (bar) | Pneumatic panel pressure | Analogue sensor feature |
| `H1` | Continuous (bar) | Cyclonic separator filter pressure drop | Analogue sensor feature |
| `DV_pressure` | Continuous (bar) | Air dryer discharge pressure drop | Analogue sensor feature |
| `Reservoirs` | Continuous (bar) | Downstream reservoir pressure | Analogue sensor feature |
| `Oil_temperature` | Continuous (°C) | Compressor lubrication oil temperature | Analogue sensor feature |
| `Flowmeter` | Continuous (m³/h) | Airflow volume measurement | Analogue sensor feature |
| `Motor_current` | Continuous (A) | Electric motor phase current | Analogue sensor feature |
| `COMP` | Binary (0/1) | Air intake valve electrical command | Digital signal feature |
| `DV_eletric` | Binary (0/1) | Compressor outlet valve electrical control | Digital signal feature |
| `Towers` | Binary (0/1) | Active air drying tower selector | Digital signal feature |
| `MPG` | Binary (0/1) | Compressor start under load control | Digital signal feature |
| `LPS` | Binary (0/1) | Low pressure safety switch (< 7 bar) | Diagnostic signal / Target indicator |
| `Pressure_switch` | Binary (0/1) | Air-drying tower discharge indicator | Digital signal feature |
| `Oil_level` | Binary (0/1) | Low oil level alarm indicator | Digital signal feature |
| `Caudal_impulses` | Binary (0/1) | Absolute airflow pulse counter | Digital signal feature |
| `gpsLat` / `gpsLong` | Continuous | Geographic position of train | Contextual telemetry (excluded from APU physics) |
| `gpsSpeed` | Continuous (km/h) | Train vehicle operational velocity | Contextual operational telemetry |
| `gpsQuality` | Categorical | GPS signal lock status | Diagnostic signal |
