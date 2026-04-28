"""
================================================================================
EARLY WARNING METRICS MODULE
================================================================================

This module provides comprehensive metrics for evaluating early warning systems:
- PR-AUC: Precision-Recall Area Under Curve (rare event evaluation)
- F1: Balanced detection (harmonic mean of precision/recall)
- ROC-AUC: General discrimination ability
- Lead Time Gain: Early warning ability (how far ahead predictions are made)
- Detection Delay: How early disruptions are caught

Add this to early_warning_app.py for comprehensive model evaluation.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def calculate_early_warning_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    lead_times: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive early warning metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 = no disruption, 1 = disruption)
    y_pred : array-like
        Predicted binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    lead_times : array-like, optional
        Lead time in minutes for each prediction
        (how early the prediction was made)

    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        average_precision_score, roc_auc_score,
        confusion_matrix, precision_recall_curve, roc_curve
    )

    metrics = {}

    # Basic metrics
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)

    # PR-AUC (best for rare events)
    metrics['pr_auc'] = average_precision_score(y_true, y_proba)

    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    except:
        metrics['roc_auc'] = np.nan

    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    # Specificity (true negative rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Lead time metrics (if available)
    if lead_times is not None:
        metrics.update(_calculate_lead_time_metrics(y_true, y_pred, lead_times))

    return metrics


def _calculate_lead_time_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lead_times: np.ndarray
) -> Dict[str, float]:
    """Calculate lead time specific metrics."""

    metrics = {}

    # Only consider true positives for lead time analysis
    tp_mask = (y_true == 1) & (y_pred == 1)

    if tp_mask.sum() > 0:
        tp_lead_times = lead_times[tp_mask]

        metrics['mean_lead_time'] = float(np.mean(tp_lead_times))
        metrics['median_lead_time'] = float(np.median(tp_lead_times))
        metrics['min_lead_time'] = float(np.min(tp_lead_times))
        metrics['max_lead_time'] = float(np.max(tp_lead_times))

        # Lead time at different thresholds
        for threshold in [5, 10, 15, 30, 60]:
            pct = (tp_lead_times >= threshold).mean() * 100
            metrics[f'lead_time_pct_{threshold}min'] = pct

        # Lead time gain (average early warning time)
        metrics['lead_time_gain'] = metrics['mean_lead_time']

        # Detection delay (inverse of lead time - lower is better)
        # If we predict 30 min early, delay is -30 min
        metrics['detection_delay'] = -metrics['mean_lead_time']
    else:
        metrics['mean_lead_time'] = 0
        metrics['median_lead_time'] = 0
        metrics['lead_time_gain'] = 0
        metrics['detection_delay'] = 0

    return metrics


def calculate_rolling_metrics(
    df: pd.DataFrame,
    timestamp_col: str,
    prediction_col: str,
    true_label_col: str,
    probability_col: str,
    window_days: int = 7,
) -> pd.DataFrame:
    """
    Calculate metrics over rolling time windows.

    Parameters
    ----------
    df : DataFrame
        Data with predictions and true labels
    timestamp_col : str
        Column name for timestamps
    prediction_col : str
        Column name for predictions
    true_label_col : str
        Column name for true labels
    probability_col : str
        Column name for probabilities
    window_days : int
        Rolling window size in days

    Returns
    -------
    DataFrame
        Metrics per time window
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)

    results = []

    min_time = df[timestamp_col].min()
    max_time = df[timestamp_col].max()

    current_start = min_time
    while current_start < max_time:
        current_end = current_start + timedelta(days=window_days)

        mask = (df[timestamp_col] >= current_start) & (df[timestamp_col] < current_end)
        window_df = df[mask]

        if len(window_df) > 0:
            y_true = window_df[true_label_col].values
            y_pred = window_df[prediction_col].values
            y_proba = window_df[probability_col].values

            metrics = calculate_early_warning_metrics(y_true, y_pred, y_proba)
            metrics['window_start'] = current_start
            metrics['window_end'] = current_end
            metrics['n_samples'] = len(window_df)

            results.append(metrics)

        current_start = current_end

    return pd.DataFrame(results)


def get_metrics_summary(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable summary string."""

    summary = f"""
     EARLY WARNING METRICS SUMMARY
    ================================

    Detection Performance:
    • F1 Score:        {metrics.get('f1_score', 0):.4f}
    • Precision:       {metrics.get('precision', 0):.4f}
    • Recall:          {metrics.get('recall', 0):.4f}
    • Specificity:     {metrics.get('specificity', 0):.4f}

    Rare Event Metrics:
    • PR-AUC:          {metrics.get('pr_auc', 0):.4f}
    • ROC-AUC:         {metrics.get('roc_auc', 0):.4f}

    Lead Time Analysis:
    • Mean Lead Time:  {metrics.get('mean_lead_time', 0):.1f} minutes
    • Lead Time Gain:  {metrics.get('lead_time_gain', 0):.1f} minutes
    • Detection Delay: {metrics.get('detection_delay', 0):.1f} minutes

    Lead Time Coverage:
    • ≥5 min:   {metrics.get('lead_time_pct_5min', 0):.1f}%
    • ≥10 min:  {metrics.get('lead_time_pct_10min', 0):.1f}%
    • ≥15 min:  {metrics.get('lead_time_pct_15min', 0):.1f}%
    • ≥30 min:  {metrics.get('lead_time_pct_30min', 0):.1f}%
    • ≥60 min:  {metrics.get('lead_time_pct_60min', 0):.1f}%

    Confusion Matrix:
    • TP: {metrics.get('true_positives', 0):,}
    • TN: {metrics.get('true_negatives', 0):,}
    • FP: {metrics.get('false_positives', 0):,}
    • FN: {metrics.get('false_negatives', 0):,}
    """

    return summary



# STREAMLIT INTEGRATION


def display_early_warning_metrics_streamlit(metrics: Dict[str, float]):
    """Display metrics in Streamlit format."""
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator

    # Detection metrics
    st.markdown("###  Detection Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    col2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    col4.metric("Specificity", f"{metrics.get('specificity', 0):.4f}")

    # Rare event metrics
    st.markdown("###  Rare Event Metrics")
    col1, col2 = st.columns(2)
    col1.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}",
                help="Precision-Recall Area Under Curve - best for imbalanced data")
    col2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}",
                help="Receiver Operating Characteristic - general discrimination")

    # Lead time metrics
    st.markdown("###  Early Warning Lead Time")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Lead Time", f"{metrics.get('mean_lead_time', 0):.1f} min",
                help="Average time before disruption that prediction is made")
    col2.metric("Lead Time Gain", f"{metrics.get('lead_time_gain', 0):.1f} min",
                help="How much earlier than baseline (5min) predictions are made")
    col3.metric("Detection Delay", f"{metrics.get('detection_delay', 0):.1f} min",
                help="Negative value means early detection (good!)")

    # Lead time coverage
    st.markdown("#### Lead Time Coverage")
    lt_cols = st.columns(5)
    lt_cols[0].metric("≥5 min", f"{metrics.get('lead_time_pct_5min', 0):.1f}%")
    lt_cols[1].metric("≥10 min", f"{metrics.get('lead_time_pct_10min', 0):.1f}%")
    lt_cols[2].metric("≥15 min", f"{metrics.get('lead_time_pct_15min', 0):.1f}%")
    lt_cols[3].metric("≥30 min", f"{metrics.get('lead_time_pct_30min', 0):.1f}%")
    lt_cols[4].metric("≥60 min", f"{metrics.get('lead_time_pct_60min', 0):.1f}%")

    # Confusion matrix
    st.markdown("###  Confusion Matrix")
    cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
    cm_col1.metric("True Positives", f"{metrics.get('true_positives', 0):,}")
    cm_col2.metric("True Negatives", f"{metrics.get('true_negatives', 0):,}")
    cm_col3.metric("False Positives", f"{metrics.get('false_positives', 0):,}")
    cm_col4.metric("False Negatives", f"{metrics.get('false_negatives', 0):,}")



# USAGE EXAMPLE


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 1000

    # Simulated data
    y_true = np.random.choice([0, 1], n, p=[0.9, 0.1])
    y_proba = np.random.beta(2, 5, n)
    y_pred = (y_proba > 0.5).astype(int)

    # Simulated lead times (in minutes)
    lead_times = np.random.exponential(15, n)

    # Calculate metrics
    metrics = calculate_early_warning_metrics(y_true, y_pred, y_proba, lead_times)

    # Print summary
    print(get_metrics_summary(metrics))