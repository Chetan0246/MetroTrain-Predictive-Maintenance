"""Model architectures (LSTM, GRU, TCN) and training execution pipeline."""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score

from src.windowing import SequenceWindowDataset
from src.preprocessing import FEATURE_COLS

# ---------------------------------------------------------------------------
# 1. Architecture Definitions
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """2-layer Long Short-Term Memory network for sequence classification."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        out, (hn, cn) = self.lstm(x)
        # Use representation at last timestep
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


class GRUClassifier(nn.Module):
    """2-layer Gated Recurrent Unit network for sequence classification."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hn = self.gru(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


class ChausalConv1d(nn.Module):
    """Dilated Causal 1D Convolution: pads only in the past to prevent lookahead."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, seq_len]
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual block for Temporal Convolutional Network."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        return self.final_relu(out + res)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network with dilated causal residual blocks."""

    def __init__(self, input_dim: int = 16, num_channels: list = [32, 64, 128], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=dilation_size, dropout=dropout))
            
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [batch, seq_len, features]; TCN expects [batch, features, seq_len]
        x_transposed = x.transpose(1, 2)
        feat = self.network(x_transposed)
        # Global pooling over sequence length
        pooled = feat.mean(dim=-1)
        logits = self.fc(pooled).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# 2. Training and Evaluation Functions
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
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


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_batch.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = float(auc(recall, precision))
    
    # Binary predictions at threshold 0.5
    binary_preds = (all_preds >= 0.5).astype(int)
    f1 = float(f1_score(all_targets, binary_preds, zero_division=0))
    rec = float(recall_score(all_targets, binary_preds, zero_division=0))
    prec = float(precision_score(all_targets, binary_preds, zero_division=0))
    
    return {
        "pr_auc": pr_auc,
        "f1": f1,
        "recall": rec,
        "precision": prec,
        "y_true": all_targets.tolist(),
        "y_prob": all_preds.tolist()
    }


def run_experiment(
    dataset_name: str,
    model_type: str,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    balance_ratio: float = 1.0,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Train and evaluate a specific (dataset, model) combination."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cpu")
    print(f"\n{'='*60}\nStarting Experiment: Dataset={dataset_name.upper()} | Model={model_type.upper()}\n{'='*60}")
    
    # Load processed splits
    data_dir = Path(f"data/processed/{dataset_name}")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    train_feat, train_lbl = train_df[FEATURE_COLS].values, train_df["target"].values
    val_feat, val_lbl = val_df[FEATURE_COLS].values, val_df["target"].values
    test_feat, test_lbl = test_df[FEATURE_COLS].values, test_df["target"].values
    
    # Create datasets (Training uses balanced sampling; Val and Test use natural distribution)
    train_dataset = SequenceWindowDataset(train_feat, train_lbl, window_size=60, step_size=6, balance_ratio=balance_ratio, random_seed=random_seed)
    val_dataset = SequenceWindowDataset(val_feat, val_lbl, window_size=60, step_size=6, balance_ratio=None)
    test_dataset = SequenceWindowDataset(test_feat, test_lbl, window_size=60, step_size=6, balance_ratio=None)
    
    print(f"Windows -> Train (Balanced {balance_ratio}:1): {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate Model
    input_dim = len(FEATURE_COLS)
    if model_type == "lstm":
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    elif model_type == "gru":
        model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    elif model_type == "tcn":
        model = TCNClassifier(input_dim=input_dim, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Training Loop
    best_val_loss = float("inf")
    best_weights = None
    start_time = time.time()
    epoch_history = []
    
    for ep in range(1, epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(yb)
        val_loss /= len(val_loader.dataset)
        
        ep_duration = time.time() - ep_start
        epoch_history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "duration": ep_duration})
        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {ep_duration:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            
    total_train_time = time.time() - start_time
    print(f"Training completed in {total_train_time:.1f}s. Evaluating on unseen test set...")
    
    # Load best weights and evaluate on test set
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    # Save checkpoint
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / f"{dataset_name}_{model_type}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved best model weights to {model_save_path}")
    
    test_results = evaluate_model(model, test_loader, device)
    print(f"Test Set Performance -> PR-AUC: {test_results['pr_auc']:.4f} | Recall: {test_results['recall']:.4f} | F1: {test_results['f1']:.4f} | Precision: {test_results['precision']:.4f}")
    
    results = {
        "dataset": dataset_name,
        "model": model_type,
        "epochs": epochs,
        "total_train_time": total_train_time,
        "epoch_history": epoch_history,
        "test_metrics": {
            "pr_auc": test_results["pr_auc"],
            "recall": test_results["recall"],
            "f1": test_results["f1"],
            "precision": test_results["precision"]
        },
        "y_true": test_results["y_true"],
        "y_prob": test_results["y_prob"]
    }
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{dataset_name}_{model_type}_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved experiment results to {result_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train LSTM, GRU, or TCN on MetroPT datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=["metropt3", "metropt2"])
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "tcn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--balance_ratio", type=float, default=1.0)
    args = parser.parse_args()
    
    run_experiment(
        dataset_name=args.dataset,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        balance_ratio=args.balance_ratio
    )


if __name__ == "__main__":
    main()
