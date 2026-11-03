# Metro Train Predictive Maintenance: Official Dataset Links & Metadata

**Course**: AI / Predictive Maintenance Project Work  
**Student**: Moorthy Chetan (Reg. No: `24BIT0246`)  
**Institution**: School of Information Technology and Engineering (SITE), VIT Vellore  

---

## 1. Official Dataset Repository Links

| Repository / Source | Direct Web Link | Identifier / DOI |
| :--- | :--- | :--- |
| **UCI Machine Learning Repository** | [https://archive.ics.uci.edu/dataset/791/metropt+3+dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) | `Dataset ID: 791 (MetroPT-3)` |
| **Zenodo Open Science (CERN)** | [https://zenodo.org/records/6854240](https://zenodo.org/records/6854240) | `DOI: 10.5281/zenodo.6854240` |
| **Kaggle Datasets** | [https://www.kaggle.com/datasets/anthonytherrien/metropt-dataset-air-production-unit](https://www.kaggle.com/datasets/anthonytherrien/metropt-dataset-air-production-unit) | `anthonytherrien/metropt-dataset` |
| **Nature Scientific Data (Publication)** | [https://doi.org/10.1038/s41597-022-01877-3](https://doi.org/10.1038/s41597-022-01877-3) | `Sci Data 9, 764 (2022)` |

---

## 2. Dataset Overview

The dataset records real-world industrial IoT telemetry from the **Air Production Unit (APU)** compressor installed onboard passenger rolling stock operating on the urban metro network of **Metro do Porto, Portugal**.

### Dataset Campaigns:
1. **MetroPT-2 (Pneumatic Air Leak Failures)**:
   - **Time Horizon**: April 12, 2022 to July 16, 2022 (~3 months)
   - **Raw Telemetry**: 7,116,940 readings (resampled to 711,694 10-second timesteps)
   - **10-Minute Windows**: 77,091 sequence windows (Tensor shape: `60 × 16`)
   - **Failure Modes**: Compressor pipe fractures, pneumatic seal degradation, desiccant air leaks.

2. **MetroPT-3 (Lubrication Oil & Intake Valve Failures)**:
   - **Time Horizon**: January 1, 2022 to June 3, 2022 (~5 months)
   - **Raw Telemetry**: 10,773,590 readings (resampled to 1,077,359 10-second timesteps)
   - **10-Minute Windows**: 125,682 sequence windows (Tensor shape: `60 × 16`)
   - **Failure Modes**: Sump lubrication oil depletion, intake valve sticking, desiccant saturation.

---

## 3. Telemetry Channel Dictionary (16 Features)

| Feature | Physical Description | Sensor Type | Physical Unit | Operational Nominal Range |
| :--- | :--- | :--- | :--- | :--- |
| `TP2` | Compressor head outlet pressure | Analog Pressure | bar | 0.0 to 10.5 bar |
| `TP3` | Pneumatic control panel intake pressure | Analog Pressure | bar | 8.0 to 10.0 bar |
| `H1` | Air dryer purge and desiccant regeneration pressure | Analog Pressure | bar | 0.0 to 10.0 bar |
| `DV_pressure` | Differential pressure across desiccant dryer valve | Analog Pressure | bar | -0.1 to 1.5 bar |
| `Reservoirs` | Auxiliary braking pressure reservoir pressure | Analog Pressure | bar | 8.5 to 10.0 bar |
| `Motor_current` | Phase current drawn by 3-phase electric drive motor | Analog Current | Amperes (A) | 0.0 to 9.5 A (Load: ~6.2 A) |
| `Oil_temperature` | Sump crankcase lubricant oil temperature | Analog Temperature | Celsius (°C) | 50.0°C to 85.0°C |
| `COMP` | TCMS compressor run cycling command | Digital Switch | Binary (0 / 1) | 0 = Idle, 1 = Pumping |
| `DV_eletric` | Discharge valve moisture solenoid actuator | Digital Switch | Binary (0 / 1) | 0 = Closed, 1 = Open |
| `Towers` | Active air dryer twin-column selector | Digital Switch | Binary (0 / 1) | 0 = Column B, 1 = Column A |
| `MPG` | Mechanical governor pressure switch | Digital Switch | Binary (0 / 1) | 0 = Cut-out, 1 = Cut-in |
| `LPS` | Emergency low-pressure safety switch | Digital Switch | Binary (0 / 1) | 0 = Critical (<7.0 bar), 1 = Normal |
| `Pressure_switch`| Door interlock auxiliary pressure switch | Digital Switch | Binary (0 / 1) | 0 = Low, 1 = Safe (>6.0 bar) |
| `Oil_level` | Crankcase oil reservoir float switch | Digital Switch | Binary (0 / 1) | 0 = Depleted, 1 = Adequate |
| `Caudal_impulses` | Cumulative volumetric air flow pulse counter | Digital Pulse | Pulses / sec | 0 to 45 pulses |

---

## 4. Documented Failure Outages

| Dataset | Failure ID | Failure Type | Failure Onset | Repair Completed | 2-Hour Pre-Warning Window |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **MetroPT-3** | F1 | Pneumatic Air Leak | 2022-02-28 21:53 | 2022-03-01 02:00 | 2022-02-28 19:53 – 21:53 |
| **MetroPT-3** | F2 | Pneumatic Air Leak | 2022-03-23 14:54 | 2022-03-23 15:24 | 2022-03-23 12:54 – 14:54 |
| **MetroPT-3** | F3 | Lubrication Oil Leak | 2022-05-30 12:00 | 2022-06-02 00:00 | 2022-05-30 10:00 – 12:00 |
| **MetroPT-2** | F1 | Pneumatic Air Leak | 2022-04-12 11:50 | 2022-04-12 23:30 | 2022-04-12 09:50 – 11:50 |
| **MetroPT-2** | F2 | Pneumatic Air Leak | 2022-06-11 10:40 | 2022-06-12 05:00 | 2022-06-11 08:40 – 10:40 |
| **MetroPT-2** | F3 | Lubrication Oil Leak | 2022-07-15 14:30 | 2022-07-15 19:00 | 2022-07-15 12:30 – 14:30 |
