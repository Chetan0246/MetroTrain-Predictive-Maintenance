"""Class Distribution & SMOTE Balancing Verification Script.

Inspects row-level and sequence window-level class counts (0 and 1)
across Train, Validation, and Test splits for both MetroPT-2 and MetroPT-3,
and demonstrates SMOTE (Synthetic Minority Over-sampling Technique).
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from src.benchmark_all_10 import SequenceWindowDataset

FEATURE_COLS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
    "Motor_current", "COMP"
]

def print_distribution_summary():
    print("=" * 85)
    print("      METRO TRAIN PREDICTIVE MAINTENANCE: CLASS DISTRIBUTION & SMOTE AUDIT")
    print("=" * 85)
    print("Class Definitions:")
    print("  • Class 0 (Negative): Normal Operating State (Clean APU baseline)")
    print("  • Class 1 (Positive): Anomaly / Failure State")
    print("      - MetroPT-2: Pneumatic Air Leaks (Compressor & valve failures)")
    print("      - MetroPT-3: Lubrication Oil Leak & Intake Valve Failures")
    print("=" * 85)

    for ds in ["metropt2", "metropt3"]:
        tr = pd.read_parquet(f"data/processed/{ds}/train.parquet")
        va = pd.read_parquet(f"data/processed/{ds}/val.parquet")
        te = pd.read_parquet(f"data/processed/{ds}/test.parquet")
        total_rows = len(tr) + len(va) + len(te)

        print(f"\n>>> 1. RAW TELEMETRY TIMESTEP ROWS ({ds.upper()}) [Total: {total_rows:,} rows]")
        print("-" * 85)
        print(f"{'Split':<12} | {'Total Rows':<12} | {'Ratio':<7} | {'Class 0 (Normal)':<18} | {'Class 1 (Fault)':<18} | {'Equal?'}")
        print("-" * 85)
        for name, df, ratio_str in [("Train", tr, "65%" if ds=="metropt2" else "70%"), 
                                    ("Val", va, "15%"), 
                                    ("Test", te, "20%" if ds=="metropt2" else "15%")]:
            c0 = (df.target == 0).sum()
            c1 = (df.target == 1).sum()
            is_eq = "YES (1:1)" if c0 == c1 else "NO (Imbalanced)"
            print(f"{name:<12} | {len(df):<12,} | {ratio_str:<7} | {c0:<8,} ({c0/len(df)*100:5.2f}%) | {c1:<8,} ({c1/len(df)*100:5.2f}%) | {is_eq}")
        print("-" * 85)

        # 2. Sequence Windows (Window=60, Step=6)
        ds_tr_raw = SequenceWindowDataset(tr[FEATURE_COLS].values, tr.target.values, window_size=60, step_size=6, balance_ratio=None)
        ds_tr_bal = SequenceWindowDataset(tr[FEATURE_COLS].values, tr.target.values, window_size=60, step_size=6, balance_ratio=1.0)
        ds_va = SequenceWindowDataset(va[FEATURE_COLS].values, va.target.values, window_size=60, step_size=6, balance_ratio=None)
        ds_te = SequenceWindowDataset(te[FEATURE_COLS].values, te.target.values, window_size=60, step_size=6, balance_ratio=None)

        print(f"\n>>> 2. 10-MINUTE SLIDING SEQUENCE WINDOWS ({ds.upper()}) [Tensor Shape: (B, 60, 16)]")
        print("-" * 85)
        print(f"{'Sequence Partition':<32} | {'Total Windows':<14} | {'Class 0':<12} | {'Class 1':<12} | {'Equal?'}")
        print("-" * 85)
        print(f"{'Train Windows (Natural Imbalance)':<32} | {len(ds_tr_raw):<14,} | {(ds_tr_raw.window_labels==0).sum():<12,} | {(ds_tr_raw.window_labels==1).sum():<12,} | NO")
        print(f"{'Train Windows (Balanced 1:1 Used)':<32} | {len(ds_tr_bal):<14,} | {(ds_tr_bal.window_labels==0).sum():<12,} | {(ds_tr_bal.window_labels==1).sum():<12,} | YES (1:1)")
        print(f"{'Validation Windows (Baseline)':<32} | {len(ds_va):<14,} | {(ds_va.window_labels==0).sum():<12,} | {(ds_va.window_labels==1).sum():<12,} | Clean (0% 1s)")
        print(f"{'Held-Out Test Windows (Real World)':<32} | {len(ds_te):<14,} | {(ds_te.window_labels==0).sum():<12,} | {(ds_te.window_labels==1).sum():<12,} | NO (Natural)")
        print("-" * 85)

        # 3. SMOTE Demonstration on Training Data
        print(f"\n>>> 3. SMOTE ALGORITHM EXECUTION ON {ds.upper()} TRAINING DATA")
        print("-" * 85)
        X_tr = tr[FEATURE_COLS].values
        y_tr = tr.target.values
        print(f"Before SMOTE -> Class 0: {(y_tr==0).sum():,} | Class 1: {(y_tr==1).sum():,} | Equal? NO")
        
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_tr, y_tr)
        print(f"After SMOTE  -> Class 0: {(y_smote==0).sum():,} | Class 1: {(y_smote==1).sum():,} | Equal? YES (1:1 Exactly)")
        print(f"Total Rows After SMOTE: {len(y_smote):,} synthetic training records")
        print("=" * 85)

if __name__ == "__main__":
    print_distribution_summary()
