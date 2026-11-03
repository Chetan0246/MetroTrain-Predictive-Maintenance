# (2) Dataset Folder

**Course**: AI / Predictive Maintenance Project Work  
**Student**: Moorthy Chetan (`24BIT0246`)  
**Institution**: School of Information Technology and Engineering (SITE), VIT Vellore  

---

This directory fulfills the submission requirement for **(2) Dataset Folder** and contains:

1. **Link of the dataset**:
   - [`DATASET_LINKS.md`](DATASET_LINKS.md): Detailed markdown document with official UCI, Zenodo DOI, and Kaggle repository URLs, sensor descriptions, and failure logs.
   - [`dataset_links.txt`](dataset_links.txt): Plain-text file with direct copy-paste links for quick access.

2. **CSV files**:
   - Located in the [`CSV_Files/`](CSV_Files/) subfolder:
     - `metropt2_genuine_samples.csv`: Labeled sample records across normal, pre-failure, and outage regimes.
     - `metropt3_genuine_samples.csv`: Labeled sample records for cross-dataset oil leak validation.
     - `metropt2_10k_subset.csv`: 10,000-row chronological telemetry slice for fast local testing.
     - `metropt3_10k_subset.csv`: 10,000-row chronological telemetry slice for out-of-distribution evaluation.
     - `DATASET_CSV_GUIDE.md`: Comprehensive description of all CSV files and channels.

3. **Image files**:
   - Located in the [`Image_Files/`](Image_Files/) subfolder:
     - `class_distribution.png`: Log-scale class distribution across splits.
     - `test_subclass_composition.png`: Early warning vs active breakdown sequence composition.
     - `methodology_flowchart.png`: Complete 8-step pipeline flowchart.
     - `dual_evaluation_pr_curves.png`: Precision-Recall curves for Task 1 vs Task 2.
     - `threshold_calibration_curves.png`: Quantile threshold calibration curves ($q_{90}, q_{95}, q_{98}$).
     - `pr_curves_comparison.png`: Comprehensive comparative PR curves.
     - `confusion_matrices.png`: Normalized confusion matrices.
     - `IMAGE_DESCRIPTIONS.md`: Technical documentation of each figure.
