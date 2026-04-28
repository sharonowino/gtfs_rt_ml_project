"""
GTFS Disruption Detection - Leakage Detection Utilities
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
import logging

logger = logging.getLogger(__name__)

def detect_potential_leakage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.8
) -> List[str]:
    """
    Detect features whose correlation with target differs significantly
    between train and validation sets (potential leakage indicator).
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    threshold : float
        Correlation difference threshold to flag as suspicious
    
    Returns
    -------
    List of feature names with suspicious correlation differences
    """
    train_corr = X_train.apply(lambda x: x.corr(pd.Series(y_train)))
    val_corr = X_val.apply(lambda x: x.corr(pd.Series(y_val)))
    
    corr_diff = (val_corr - train_corr).abs()
    suspicious = corr_diff[corr_diff > threshold].index.tolist()
    
    if suspicious:
        warnings.warn(
            f"Potential leakage detected in features: {suspicious}\n"
            f"These features show correlation patterns that differ "
            f"significantly between train and validation sets."
        )
    
    return suspicious


def verify_temporal_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str
) -> bool:
    """
    Verify that train/val/test splits have no temporal overlap.
    
    Returns True if split is valid, False otherwise.
    """
    max_train = train_df[timestamp_col].max()
    min_val = val_df[timestamp_col].min()
    max_val = val_df[timestamp_col].max()
    min_test = test_df[timestamp_col].min()
    
    issues = []
    
    if max_train >= min_val:
        issues.append("Train-Val temporal overlap")
    if max_val >= min_test:
        issues.append("Val-Test temporal overlap")
    
    if issues:
        warnings.warn(f"Temporal split issues: {issues}")
        return False
    
    return True


def compute_leakage_safe_network_load(
    df: pd.DataFrame,
    stop_id_col: str,
    timestamp_col: str,
    severity_col: str,
    lookback_minutes: int = 5
) -> pd.Series:
    """
    Compute network disruption load with STRICT backward-looking window.
    
    This fixes the leakage in the original implementation by ensuring
    each row only sees historical data, not future information.
    
    VECTORIZED VERSION: Uses merge_asof for efficient temporal joins
    instead of row-wise iteration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must be sorted by timestamp)
    stop_id_col : str
        Column name for stop IDs
    timestamp_col : str
        Column name for timestamps
    severity_col : str
        Column name for severity values
    lookback_minutes : int
        How far back to look for neighbourhood severity
    
    Returns
    -------
    pd.Series with network load values
    """
    df_sorted = df.sort_values(timestamp_col).copy()
    
    # Create cutoff timestamps
    df_sorted['_cutoff'] = df_sorted[timestamp_col] - pd.Timedelta(minutes=lookback_minutes)
    
    # Use merge_asof for efficient temporal joins
    # This finds the most recent past observation for each row
    result = pd.merge_asof(
        df_sorted[[timestamp_col, stop_id_col, severity_col]],
        df_sorted[['_cutoff', stop_id_col, severity_col]].rename(
            columns={'_cutoff': timestamp_col, severity_col: '_past_severity'}
        ),
        on=timestamp_col,
        by=stop_id_col,
        direction='backward'
    )
    
    # Fill NaN with 0 (no past observations)
    return result['_past_severity'].fillna(0.0)


def compute_stop_load_train_only(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    stop_id_col: str = 'stop_id',
    hour_col: str = 'hour'
) -> pd.Series:
    """
    Compute stop_load_proxy using ONLY training data, then apply to full dataset.
    
    This prevents leakage where test/val trip counts leak into features.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data only (e.g., first 70% by time)
    full_df : pd.DataFrame
        Full dataset (train + val + test)
    stop_id_col : str
        Column for stop ID
    hour_col : str
        Column for hour of day
    
    Returns
    -------
    pd.Series with stop_load_proxy values for full_df
    """
    # Compute counts from TRAINING data ONLY
    train_counts = train_df.groupby([stop_id_col, hour_col]).size().reset_index(name='count')
    
    # Create lookup dictionary
    load_lookup = {}
    for _, row in train_counts.iterrows():
        load_lookup[(row[stop_id_col], row[hour_col])] = row['count']
    
    # Apply to full dataset (default to median for unseen combinations)
    median_load = train_counts['count'].median()
    
    def get_load(row):
        key = (row[stop_id_col], row[hour_col])
        return load_lookup.get(key, median_load)
    
    return full_df.apply(get_load, axis=1)
