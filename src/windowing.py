"""PyTorch sequence dataset and windowing utilities for time-series modeling."""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceWindowDataset(Dataset):
    """Memory-efficient PyTorch Dataset for sliding time-series windows without array duplication."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int = 60,
        step_size: int = 6,
        balance_ratio: Optional[float] = None,
        random_seed: int = 42
    ):
        """Args:
            features: 2D numpy array [N, C] of scaled sensor measurements.
            labels: 1D numpy array [N] of binary target labels.
            window_size: Number of timesteps per window (e.g. 60 = 10 min at 10s steps).
            step_size: Stride between consecutive windows (e.g. 6 = 1 min).
            balance_ratio: If set (e.g. 1.0 for 1:1, 5.0 for 5:1), under-samples healthy windows
                           to achieve balanced class distribution for training.
            random_seed: Fixed seed for reproducible window sampling.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.window_size = window_size
        self.step_size = step_size
        
        # Calculate valid window starting indices
        num_records = len(features)
        raw_starts = list(range(0, num_records - window_size + 1, step_size))
        
        # Determine window-level labels: label of the last step or max within window
        # We assign positive if impending warning/failure occurs at the window end
        window_labels = []
        for s in raw_starts:
            e = s + window_size
            # Window is positive if the final target step is positive
            window_labels.append(int(labels[e - 1]))
            
        window_labels = np.array(window_labels)
        
        if balance_ratio is not None:
            np.random.seed(random_seed)
            pos_indices = np.where(window_labels == 1)[0]
            neg_indices = np.where(window_labels == 0)[0]
            
            n_pos = len(pos_indices)
            n_neg_sampled = min(len(neg_indices), int(n_pos * balance_ratio))
            
            sampled_neg_indices = np.random.choice(neg_indices, size=n_neg_sampled, replace=False)
            selected_indices = np.sort(np.concatenate([pos_indices, sampled_neg_indices]))
            
            self.window_starts = [raw_starts[i] for i in selected_indices]
            self.window_labels = window_labels[selected_indices]
        else:
            self.window_starts = raw_starts
            self.window_labels = window_labels
            
    def __len__(self) -> int:
        return len(self.window_starts)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.window_starts[idx]
        end = start + self.window_size
        x = self.features[start:end]
        y = torch.tensor(self.window_labels[idx], dtype=torch.float32)
        return x, y

    def get_class_counts(self) -> Dict[str, int]:
        pos = int((self.window_labels == 1).sum())
        neg = int((self.window_labels == 0).sum())
        return {"normal": neg, "positive": pos, "total": len(self.window_labels)}


def compute_window_distribution_table(
    train_feat: np.ndarray, train_lbl: np.ndarray,
    val_feat: np.ndarray, val_lbl: np.ndarray,
    test_feat: np.ndarray, test_lbl: np.ndarray,
    window_size: int = 60,
    step_size: int = 6
) -> Dict[str, Any]:
    """Compute original vs balanced window counts across splits."""
    ds_train_orig = SequenceWindowDataset(train_feat, train_lbl, window_size, step_size, balance_ratio=None)
    ds_train_bal = SequenceWindowDataset(train_feat, train_lbl, window_size, step_size, balance_ratio=1.0)
    ds_val = SequenceWindowDataset(val_feat, val_lbl, window_size, step_size, balance_ratio=None)
    ds_test = SequenceWindowDataset(test_feat, test_lbl, window_size, step_size, balance_ratio=None)
    
    return {
        "train_original": ds_train_orig.get_class_counts(),
        "train_balanced": ds_train_bal.get_class_counts(),
        "val": ds_val.get_class_counts(),
        "test": ds_test.get_class_counts()
    }


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 60,
    step_size: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """Create 3D sequence array [samples, window_size, features] and corresponding labels."""
    num_samples = data.shape[0]
    X_list = []
    y_list = []
    
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        X_list.append(data[start:end])
        y_list.append(labels[end - 1])
        
    return np.array(X_list), np.array(y_list)

