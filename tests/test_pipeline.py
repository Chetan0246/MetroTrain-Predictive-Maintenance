"""Unit tests for pipeline validation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_inspection import inspect_dataframe
from src.labeling import label_failure_intervals, add_early_warning_labels
from src.windowing import create_sliding_windows


def test_label_failure_intervals():
    """Verify ground truth failure interval labeling."""
    timestamps = pd.date_range("2022-01-01 00:00:00", periods=10, freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "val": range(10)})
    
    # Define failure between minute 3 and 5
    failure_intervals = [("2022-01-01 00:03:00", "2022-01-01 00:05:00")]
    labeled_df = label_failure_intervals(df, failure_intervals, timestamp_col="timestamp", label_col="failure")
    
    assert "failure" in labeled_df.columns
    assert labeled_df["failure"].sum() == 3
    assert list(labeled_df.loc[3:5, "failure"]) == [1, 1, 1]
    assert labeled_df.loc[0, "failure"] == 0


def test_early_warning_labeling():
    """Verify early warning lead-time window logic."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01 00:00:00", periods=10, freq="1min"),
        "failure": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    })
    # 2 minutes lead time
    df = add_early_warning_labels(df, lead_time_steps=2, label_col="failure", early_warning_col="warning")
    assert "warning" in df.columns
    # Index 4 and 5 should be flagged as early warning before failure at index 6
    assert df.loc[4, "warning"] == 1
    assert df.loc[5, "warning"] == 1


def test_create_sliding_windows():
    """Verify sliding sequence generator shapes."""
    data = np.random.randn(100, 5)
    labels = np.random.randint(0, 2, size=(100,))
    window_size = 10
    step_size = 2
    
    X, y = create_sliding_windows(data, labels, window_size=window_size, step_size=step_size)
    expected_windows = (100 - window_size) // step_size + 1
    
    assert X.shape == (expected_windows, window_size, 5)
    assert y.shape == (expected_windows,)
