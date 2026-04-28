"""
Debugging Guide for GTFS Disruption Pipeline
==================================

Common issues and how to debug them.

Issue Categories:
1. Data Issues
2. Model Training Issues
3. Leakage Issues
4. Performance Issues
5. Deployment Issues
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================================
# 1. DATA ISSUES
# =========================================================================

def debug_data_quality(df: pd.DataFrame) -> Dict:
    """Check data quality issues."""
    
    issues = {}
    
    # Missing values
    missing = df.isnull().sum()
    issues['columns_with_missing'] = missing[missing > 0].to_dict()
    
    # Duplicate timestamps
    if 'feed_timestamp' in df.columns:
        dup_ts = df['feed_timestamp'].duplicated().sum()
        issues['duplicate_timestamps'] = dup_ts
    
    # Column type mismatches
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            issues[f'{col}_has_nan_in_numeric'] = True
    
    # Timestamp issues
    ts_cols = ['feed_timestamp', 'timestamp', 'event_time']
    for col in ts_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                issues[f'{col}_range'] = (df[col].min(), df[col].max())
            except:
                issues[f'{col}_parse_error'] = True
    
    if not issues:
        logger.info("No data quality issues detected")
    else:
        logger.warning(f"Data quality issues: {issues}")
    
    return issues


def debug_feature_distribution(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Check feature distributions."""
    
    distributions = {}
    
    for col in feature_cols:
        if col in df.columns:
            distributions[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'zeros': (df[col] == 0).sum(),
                'nans': df[col].isna().sum(),
            }
    
    return distributions


# =========================================================================
# 2. MODEL TRAINING ISSUES
# =========================================================================

def debug_model_training(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict:
    """Debug model training issues."""
    
    issues = {}
    
    # Check for NaN/Inf in features
    if np.isnan(X_train).any():
        issues['nan_in_X_train'] = np.isnan(X_train).sum()
    if np.isinf(X_train).any():
        issues['inf_in_X_train'] = np.isinf(X_train).sum()
    
    # Check for class imbalance
    pos_rate = y_train.mean()
    issues['train_positive_rate'] = pos_rate
    if pos_rate < 0.01:
        issues['extreme_class_imbalance'] = True
    
    # Check train/val distribution mismatch
    train_mean = X_train.mean(axis=0)
    val_mean = X_val.mean(axis=0)
    
    if len(train_mean) == len(val_mean):
        mean_diff = np.abs(train_mean - val_mean).mean()
        issues['train_val_mean_diff'] = mean_diff
        
        if mean_diff > 0.5:
            logger.warning("Significant train/val feature distribution mismatch!")
    
    # Check model convergence
    if hasattr(model, 'n_estimators'):
        issues['n_estimators'] = model.n_estimators
    
    return issues


def debug_leakage(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict:
    """Check for potential leakage."""
    
    leakage = {}
    
    # Correlation difference between train and val
    train_corr = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])])
    val_corr = np.array([np.corrcoef(X_val[:, i], y_val)[0, 1] for i in range(X_val.shape[1])])
    
    # High correlation difference indicates potential leakage
    corr_diff = np.abs(train_corr - val_corr)
    leakage['features_high_corr_diff'] = corr_diff.tolist()
    
    # Check if any difference is suspiciously high
    if (corr_diff > 0.3).any():
        leakage['potential_leakage_features'] = np.where(corr_diff > 0.3)[0].tolist()
        logger.warning(f"Potential leakage in features: {leakage['potential_leakage_features']}")
    
    return leakage


# =========================================================================
# 3. PERFORMANCE ISSUES
# =========================================================================

def debug_prediction_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """Debug prediction performance issues."""
    
    metrics = {}
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['roc_auc'] = np.nan
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Check for specific issues
    if metrics['recall'] < 0.5:
        logger.warning("Low recall - model missing positive cases!")
    
    if metrics['precision'] < 0.3:
        logger.warning("Low precision - too many false positives!")
    
    if np.isnan(metrics['roc_auc']):
        logger.warning("ROC AUC is NaN - check probability predictions")
    
    return metrics


# =========================================================================
# 4. GNN SPECIFIC ISSUES
# =========================================================================

def debug_gnn_graph(
    edge_index: np.ndarray,
    num_nodes: int,
) -> Dict:
    """Debug GNN graph construction."""
    
    issues = {}
    
    # Check for isolated nodes
    all_nodes = set(range(num_nodes))
    connected = set(edge_index.flatten())
    isolated = all_nodes - connected
    
    issues['num_isolated_nodes'] = len(isolated)
    issues['isolated_node_samples'] = list(isolated)[:10]
    
    # Check for disconnected components
    from collections import defaultdict
    
    adj = defaultdict(set)
    for src, dst in edge_index.T:
        adj[src].add(dst)
        adj[dst].add(src)
    
    components = []
    visited = set()
    
    for start in adj:
        if start not in visited:
            component = {start}
            queue = [start]
            
            while queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if neighbor not in component:
                        component.add(neighbor)
                        queue.append(neighbor)
            
            visited.update(component)
            components.append(component)
    
    issues['num_components'] = len(components)
    issues['component_sizes'] = sorted(len(c) for c in components)[-5:]
    
    # Check for proper graph structure
    if len(components) > 1:
        logger.warning(f"Graph has {len(components)} disconnected components!")
    
    # Check edge density
    num_edges = edge_index.shape[1]
    max_edges = num_nodes * (num_nodes - 1)
    density = num_edges / max_edges if max_edges > 0 else 0
    
    issues['edge_density'] = density
    
    if density < 0.01:
        logger.warning("Very sparse graph - may need better edges")
    
    return issues


def debug_sequence_model(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> Dict:
    """Debug sequence model data."""
    
    issues = {}
    
    # Check sequence lengths
    issues['sequence_length'] = sequence_length
    issues['num_sequences'] = len(X) // sequence_length
    
    # Check for padding issues
    if (X == 0).all(axis=0).any():
        issues['padding_in_data'] = True
    
    # Check temporal patterns
    if len(y) > sequence_length:
        corr_lag = []
        for lag in range(1, min(20, sequence_length)):
            corr = np.corrcoef(y[:-lag], y[lag:])[0, 1]
            corr_lag.append(corr)
        
        issues['temporal_correlation_lags'] = corr_lag
        
        # Find best lag
        best_lag = np.argmax(corr_lag) + 1
        issues['best_temporal_lag'] = best_lag
    
    return issues


# =========================================================================
# 5. DEPLOYMENT ISSUES
# =========================================================================

def debug_latency(
    predictions: List[float],
    timestamps: List[pd.Timestamp],
) -> Dict:
    """Debug prediction latency."""
    
    if len(predictions) < 2:
        return {}
    
    latencies = []
    for i in range(1, len(timestamps)):
        diff = (timestamps[i] - timestamps[i-1]).total_seconds() * 1000
        latencies.append(diff)
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'max_latency_ms': np.max(latencies),
    }


def debug_model_drift(
    historical_metrics: List[Dict],
    recent_metrics: List[Dict],
    threshold: float = 0.1,
) -> Dict:
    """Debug model drift over time."""
    
    drift = {}
    
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        if metric in historical_metrics[0] and metric in recent_metrics[-1]:
            hist = np.mean([m[metric] for m in historical_metrics])
            recent = recent_metrics[-1][metric]
            
            drift[metric] = {
                'historical': hist,
                'recent': recent,
                'change': recent - hist,
                'percent_change': (recent - hist) / hist if hist > 0 else 0,
            }
            
            if abs(drift[metric]['percent_change']) > threshold:
                logger.warning(f"{metric} drift detected: {drift[metric]}")
    
    return drift


# =========================================================================
# HELPER FUNCTION
# =========================================================================

def run_full_diagnostics(
    df: pd.DataFrame,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_val_prob: np.ndarray,
) -> Dict:
    """Run all diagnostic checks."""
    
    logger.info("Running full diagnostics...")
    
    diagnostics = {}
    
    # Data quality
    diagnostics['data_quality'] = debug_data_quality(df)
    
    # Model training
    diagnostics['training'] = debug_model_training(
        model, X_train, y_train, X_val, y_val
    )
    
    # Leakage
    diagnostics['leakage'] = debug_leakage(X_train, y_train, X_val, y_val)
    
    # Performance
    diagnostics['performance'] = debug_prediction_performance(
        y_val, y_val_pred, y_val_prob
    )
    
    # Summary
    logger.info("=== Diagnostic Summary ===")
    for category, issues in diagnostics.items():
        if issues:
            logger.info(f"{category}: {issues}")
    
    return diagnostics


__all__ = [
    'debug_data_quality',
    'debug_feature_distribution',
    'debug_model_training',
    'debug_leakage',
    'debug_prediction_performance',
    'debug_gnn_graph',
    'debug_sequence_model',
    'debug_latency',
    'debug_model_drift',
    'run_full_diagnostics',
]