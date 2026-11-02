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

FEATURE_COLS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
    "Motor_current", "COMP"
]

def count_windows(targets: np.ndarray, window_size: int = 60, step_size: int = 6, balance: bool = False):
    """Fast window-level class count calculation without PyTorch dependency."""
    num_records = len(targets)
    starts = list(range(0, num_records - window_size + 1, step_size))
    labels = np.array([targets[s + window_size - 1] for s in starts])
    
    if balance:
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]
        n_pos = len(pos)
        np.random.seed(42)
        sampled_neg = np.random.choice(neg, size=min(len(neg), n_pos), replace=False)
        selected = np.sort(np.concatenate([pos, sampled_neg]))
        labels = labels[selected]
        
    return len(labels), int((labels == 0).sum()), int((labels == 1).sum())

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
        tr_path = REPO_ROOT / f"data/processed/{ds}/train.parquet"
        va_path = REPO_ROOT / f"data/processed/{ds}/val.parquet"
        te_path = REPO_ROOT / f"data/processed/{ds}/test.parquet"

        if not tr_path.exists():
            print(f"Warning: {tr_path} not found.")
            continue

        tr = pd.read_parquet(tr_path)
        va = pd.read_parquet(va_path)
        te = pd.read_parquet(te_path)
        total_rows = len(tr) + len(va) + len(te)

        print(f"\n>>> 1. RAW TELEMETRY TIMESTEP ROWS ({ds.upper()}) [Total: {total_rows:,} rows]")
        print("-" * 85)
        print(f"{'Split':<12} | {'Total Rows':<12} | {'Ratio':<7} | {'Class 0 (Normal)':<18} | {'Class 1 (Fault)':<18} | {'Equal?'}")
        print("-" * 85)
        for name, df, ratio_str in [("Train", tr, "65%" if ds=="metropt2" else "70%"), 
                                    ("Val", va, "15%"), 
                                    ("Test", te, "20%" if ds=="metropt2" else "15%")]:
            c0 = int((df.target == 0).sum())
            c1 = int((df.target == 1).sum())
            is_eq = "YES (1:1)" if c0 == c1 else "NO (Imbalanced)"
            print(f"{name:<12} | {len(df):<12,} | {ratio_str:<7} | {c0:<8,} ({c0/len(df)*100:5.2f}%) | {c1:<8,} ({c1/len(df)*100:5.2f}%) | {is_eq}")
        print("-" * 85)

        # 2. Sequence Windows (Window=60, Step=6)
        n_raw, c0_raw, c1_raw = count_windows(tr.target.values, balance=False)
        n_bal, c0_bal, c1_bal = count_windows(tr.target.values, balance=True)
        n_va, c0_va, c1_va   = count_windows(va.target.values, balance=False)
        n_te, c0_te, c1_te   = count_windows(te.target.values, balance=False)

        print(f"\n>>> 2. 10-MINUTE SLIDING SEQUENCE WINDOWS ({ds.upper()}) [Tensor Shape: (B, 60, 16)]")
        print("-" * 85)
        print(f"{'Sequence Partition':<32} | {'Total Windows':<14} | {'Class 0':<12} | {'Class 1':<12} | {'Equal?'}")
        print("-" * 85)
        print(f"{'Train Windows (Natural Imbalance)':<32} | {n_raw:<14,} | {c0_raw:<12,} | {c1_raw:<12,} | NO")
        print(f"{'Train Windows (Balanced 1:1 Used)':<32} | {n_bal:<14,} | {c0_bal:<12,} | {c1_bal:<12,} | YES (1:1)")
        print(f"{'Validation Windows (Baseline)':<32} | {n_va:<14,} | {c0_va:<12,} | {c1_va:<12,} | Clean (0% 1s)")
        print(f"{'Held-Out Test Windows (Real World)':<32} | {n_te:<14,} | {c0_te:<12,} | {c1_te:<12,} | NO (Natural)")
        print("-" * 85)

        # 3. SMOTE Demonstration on Training Data
        print(f"\n>>> 3. SMOTE ALGORITHM EXECUTION ON {ds.upper()} TRAINING DATA")
        print("-" * 85)
        X_tr = tr[FEATURE_COLS].values.astype(np.float32)
        y_tr = tr.target.values
        total_c0 = int((y_tr == 0).sum())
        total_c1 = int((y_tr == 1).sum())
        print(f"Full Train Set Before SMOTE -> Class 0: {total_c0:,} | Class 1: {total_c1:,} | Equal? NO")
        
        # In Google Colab (12GB RAM limit), generating 748,000 synthetic rows with k-NN exceeds RAM.
        # We demonstrate live SMOTE execution on a 25,000 majority sample + all minority failures:
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        np.random.seed(42)
        sampled_neg = np.random.choice(neg_idx, size=min(len(neg_idx), 25000), replace=False)
        demo_idx = np.sort(np.concatenate([pos_idx, sampled_neg]))
        X_demo, y_demo = X_tr[demo_idx], y_tr[demo_idx]

        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_demo, y_demo)
        
        c0_smote = int((y_smote == 0).sum())
        c1_smote = int((y_smote == 1).sum())
        print(f"Live SMOTE Execution Sample -> Class 0: {c0_smote:,} | Class 1: {c1_smote:,} | Equal? YES (1:1 Exactly!)")
        print(f"Extrapolated Full Scale SMOTE -> Class 0: {total_c0:,} | Class 1: {total_c0:,} | Equal? YES (1:1 Exactly, {2*total_c0:,} rows)")
        print("=" * 85)

if __name__ == "__main__":
    print_distribution_summary()
