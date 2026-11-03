"""Benchmark and Comparative Evaluation of All 10 Deep Learning & Machine Learning Algorithms.

Student: Moorthy Chetan (Reg No: 24BIT0246)
Project: Predictive Maintenance in Metro Trains Using Machine Learning and Deep Learning
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, accuracy_score
import joblib

from src.models import MODEL_REGISTRY, ALGORITHM_DISPLAY_NAMES
from src.windowing import SequenceWindowDataset
from src.preprocessing import FEATURE_COLS


def train_single_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    t_start = time.perf_counter()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    total_eval_time = time.perf_counter() - t_start
    latency_per_sample_ms = (total_eval_time / len(all_targets)) * 1000

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred_50 = (y_prob >= 0.5).astype(int)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_curve, precision_curve))
    rec = float(recall_score(y_true, y_pred_50, zero_division=0))
    prec = float(precision_score(y_true, y_pred_50, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_50, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred_50))

    return {
        "loss": total_loss / len(loader.dataset),
        "pr_auc": pr_auc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "accuracy": acc,
        "latency_ms": latency_per_sample_ms,
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist()
    }


def run_10_algorithm_benchmark(dataset_name: str = "metropt2", epochs: int = 3, batch_size: int = 128) -> Dict[str, Any]:
    """Execute complete end-to-end training and evaluation across all 10 algorithms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================================")
    print(f"  EXECUTING 10-ALGORITHM BENCHMARK ON DATASET: {dataset_name.upper()}")
    print(f"  Hardware Device: {device} | Epochs per model: {epochs}")
    print(f"=======================================================================\n")

    # Step 1: Explicitly log and verify dataset paths
    train_path = REPO_ROOT / f"data/processed/{dataset_name}/train.parquet"
    val_path = REPO_ROOT / f"data/processed/{dataset_name}/val.parquet"
    test_path = REPO_ROOT / f"data/processed/{dataset_name}/test.parquet"
    scaler_path = REPO_ROOT / f"models/{dataset_name}_scaler.joblib"

    print("--- [Step 1: Dataset Partition Paths] ---")
    print(f"  • Train Split:      {train_path.resolve()}")
    print(f"  • Validation Split: {val_path.resolve()}")
    print(f"  • Test Split:       {test_path.resolve()}")
    print(f"  • Scaler Checkpoint:{scaler_path.resolve()}")

    # Step 2: Load Parquet splits
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    scaler = joblib.load(scaler_path)

    # Step 3: Transform features out-of-sample using fitted StandardScaler
    print("\n--- [Step 2: Function Called After Splitting: fit_and_scale & SequenceWindowDataset] ---")
    print(f"  Calling StandardScaler.transform() across {len(FEATURE_COLS)} APU telemetry channels...")
    X_train = scaler.transform(train_df[FEATURE_COLS])
    X_val = scaler.transform(val_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    # Step 4: Build sliding window datasets (W=60, S=6)
    print("  Instantiating SequenceWindowDataset(window_size=60, step_size=6)...")
    train_dataset = SequenceWindowDataset(X_train, y_train, window_size=60, step_size=6, balance_ratio=1.0)
    val_dataset = SequenceWindowDataset(X_val, y_val, window_size=60, step_size=6, balance_ratio=None)
    test_dataset = SequenceWindowDataset(X_test, y_test, window_size=60, step_size=6, balance_ratio=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"  • Train Sequences (1:1 Balanced): {len(train_dataset):,} windows")
    print(f"  • Validation Sequences (Clean):   {len(val_dataset):,} windows")
    print(f"  • Test Sequences (Imbalanced):    {len(test_dataset):,} windows")
    print(f"-----------------------------------------------------------------------\n")

    criterion = nn.BCEWithLogitsLoss()
    benchmark_results = {}

    for idx, (model_key, model_cls) in enumerate(MODEL_REGISTRY.items(), 1):
        display_name = ALGORITHM_DISPLAY_NAMES[model_key]
        print(f"[{idx}/10] Training & Evaluating {display_name}...")
        
        # Set seeds for deterministic comparison
        torch.manual_seed(42)
        np.random.seed(42)

        model = model_cls().to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        t_train_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            loss = train_single_epoch(model, train_loader, criterion, optimizer, device)
        total_train_time = time.perf_counter() - t_train_start

        # Evaluate on test set
        eval_metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Save model checkpoint
        ckpt_path = REPO_ROOT / f"models/{dataset_name}_{model_key}.pt"
        torch.save(model.state_dict(), ckpt_path)

        benchmark_results[model_key] = {
            "rank": 0,
            "name": model.name,
            "display_name": display_name,
            "parameters": num_params,
            "train_time_sec": round(total_train_time, 2),
            "latency_ms": round(eval_metrics["latency_ms"], 3),
            "pr_auc": round(eval_metrics["pr_auc"], 4),
            "recall": round(eval_metrics["recall"], 4),
            "precision": round(eval_metrics["precision"], 4),
            "f1": round(eval_metrics["f1"], 4),
            "accuracy": round(eval_metrics["accuracy"], 4),
            "test_loss": round(eval_metrics["loss"], 4),
            "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
            "y_prob": eval_metrics["y_prob"],
            "y_true": eval_metrics["y_true"]
        }

        print(f"       -> PR-AUC: {eval_metrics['pr_auc']:.4f} | Recall: {eval_metrics['recall']:.4f} | "
              f"Precision: {eval_metrics['precision']:.4f} | F1: {eval_metrics['f1']:.4f} | Time: {total_train_time:.1f}s")

    # Step 5: Rank all 10 algorithms by PR-AUC and F1-Score
    sorted_models = sorted(
        benchmark_results.keys(),
        key=lambda k: (benchmark_results[k]["pr_auc"], benchmark_results[k]["f1"]),
        reverse=True
    )
    for rank, k in enumerate(sorted_models, 1):
        benchmark_results[k]["rank"] = rank

    top3_keys = sorted_models[:3]

    print("\n=======================================================================")
    print("                        🏆 TOP 3 BEST ALGORITHMS                        ")
    print("=======================================================================")
    for r, k in enumerate(top3_keys, 1):
        m = benchmark_results[k]
        print(f"  #{r}: {m['display_name']} -> PR-AUC: {m['pr_auc']:.4f} | F1: {m['f1']:.4f} | Recall: {m['recall']:.4f}")
    print("=======================================================================\n")

    # Export benchmark tables & JSON
    export_benchmark_artifacts(benchmark_results, sorted_models, dataset_name)
    return benchmark_results


def export_benchmark_artifacts(results: Dict[str, Any], sorted_keys: List[str], dataset_name: str):
    """Save markdown table and JSON artifact."""
    tables_dir = REPO_ROOT / "reports/tables"
    results_dir = REPO_ROOT / "results"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON (without bulky y_prob for lightweight summary)
    summary_dict = {}
    for k in sorted_keys:
        item = results[k].copy()
        item.pop("y_prob", None)
        item.pop("y_true", None)
        summary_dict[k] = item

    with open(results_dir / f"all_10_algorithms_{dataset_name}.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    # 2. Markdown Table
    md_lines = [
        f"# Complete 10-Algorithm Benchmark Comparison Matrix ({dataset_name.upper()})\n",
        f"**Student**: Moorthy Chetan (`24BIT0246`)  \n",
        f"**Evaluation Partition**: Held-Out Chronological Test Partition (23,668 sequences)  \n",
        f"**Sorting Criteria**: Ranked by PR-AUC (Primary) and F1-Score (Secondary)\n\n",
        "| Rank | Algorithm | Parameters | Train Time | Latency (ms) | Recall | Precision | F1-Score | PR-AUC | Status |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for k in sorted_keys:
        m = results[k]
        badge = "🥇 **TOP 1**" if m["rank"] == 1 else ("🥈 **TOP 2**" if m["rank"] == 2 else ("🥉 **TOP 3**" if m["rank"] == 3 else "Evaluated"))
        md_lines.append(
            f"| {m['rank']} | **{m['display_name']}** | {m['parameters']:,} | {m['train_time_sec']}s | {m['latency_ms']:.2f} | "
            f"{m['recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | **{m['pr_auc']:.4f}** | {badge} |"
        )

    md_lines.append("\n\n### 🏆 Top 3 Algorithms Selected:")
    for r in range(3):
        k = sorted_keys[r]
        m = results[k]
        md_lines.append(f"{r+1}. **{m['display_name']}** — PR-AUC: `{m['pr_auc']:.4f}`, Recall: `{m['recall']:.4f}`, F1: `{m['f1']:.4f}`")

    table_path = tables_dir / f"all_10_algorithms_{dataset_name}_benchmark.md"
    with open(table_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Saved benchmark summary to: {table_path}")


if __name__ == "__main__":
    run_10_algorithm_benchmark(dataset_name="metropt2", epochs=3)
