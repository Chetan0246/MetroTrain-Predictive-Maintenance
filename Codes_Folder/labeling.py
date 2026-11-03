"""Labeling utilities for failure events, anomalies, and lead-time windows."""

from typing import List, Tuple
import pandas as pd


def label_failure_intervals(
    df: pd.DataFrame,
    failure_intervals: List[Tuple[str, str]],
    timestamp_col: str = "timestamp",
    label_col: str = "failure"
) -> pd.DataFrame:
    """Label time ranges corresponding to component failures.

    Args:
        df: Input DataFrame with timestamp column.
        failure_intervals: List of (start_timestamp, end_timestamp) strings or datetimes.
        timestamp_col: Name of the timestamp column.
        label_col: Name of the binary target label column.
    """
    df = df.copy()
    df[label_col] = 0
    ts = pd.to_datetime(df[timestamp_col])
    
    for start, end in failure_intervals:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        mask = (ts >= start_ts) & (ts <= end_ts)
        df.loc[mask, label_col] = 1
        
    return df


def add_early_warning_labels(
    df: pd.DataFrame,
    lead_time_steps: int,
    label_col: str = "failure",
    early_warning_col: str = "early_warning",
    include_failure: bool = True
) -> pd.DataFrame:
    """Create lead-time / early warning labels preceding failure periods.

    Args:
        df: Input DataFrame with failure label.
        lead_time_steps: Number of time steps before failure onset to flag.
        label_col: Existing binary failure column.
        early_warning_col: Target column name for early warning.
        include_failure: Whether failure period itself is included in positive label.
    """
    df = df.copy()
    rev = df[label_col].iloc[::-1]
    warning = rev.shift(1).rolling(window=lead_time_steps, min_periods=1).max().fillna(0).iloc[::-1].astype(int)
    
    if include_failure:
        df[early_warning_col] = ((warning == 1) | (df[label_col] == 1)).astype(int)
    else:
        df[early_warning_col] = ((warning == 1) & (df[label_col] == 0)).astype(int)
        
    return df


if __name__ == "__main__":
    print("Labeling module initialized.")
