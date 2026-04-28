"""
GTFS Disruption Detection - Evaluation Module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

def safe_roc_auc(y_true, y_proba, min_positives: int = 15):
    """
    Compute ROC-AUC with safeguard for small positive counts.
    
    When fewer than ~15 positives exist, ROC-AUC becomes unreliable
    and has extremely high variance.
    """
    from sklearn.metrics import roc_auc_score
    
    n_pos = int(y_true.sum())
    if n_pos < min_positives:
        warnings.warn(
            f"roc_auc_score unreliable: only {n_pos} positives in split "
            f"(minimum recommended: {min_positives}). Using PR-AUC instead."
        )
        return np.nan
    return roc_auc_score(y_true, y_proba)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities for positive class
    
    Returns
    -------
    Dict with metric names and values
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = safe_roc_auc(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    
    return metrics


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute multi-label classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels (binary matrix)
    y_pred : array-like
        Predicted labels (binary matrix)
    
    Returns
    -------
    Dict with metric names and values
    """
    from sklearn.metrics import f1_score, hamming_loss
    
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred),
    }
    
    return metrics


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 multi_label: bool = False) -> Dict[str, float]:
    """
    Compute comprehensive metrics based on latest literature (2024-2025).
    
    Based on papers:
    - Multi-Head + LSTM (Scientific Reports 2025): F1, AUC-ROC, MCC, Precision, Recall
    - GTFS Delay Prediction (Passarella 2025): F1, AUC, MAE
    - Traffic Anomaly Detection (Jadhav 2025): MCC, G-Mean, Balanced Accuracy
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities
    multi_label : bool
        If True, compute multi-label metrics (Jaccard, Subset Accuracy)
    
    Returns
    -------
    Dict with all metric names and values
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        hamming_loss, jaccard_score, balanced_accuracy_score,
        matthews_corrcoef, confusion_matrix
    )
    import numpy as np
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle multi-dim arrays
    if y_true.ndim > 1:
        y_true_1d = y_true.ravel()
        y_pred_1d = y_pred.ravel()
    else:
        y_true_1d = y_true
        y_pred_1d = y_pred
    
    metrics = {}
    
    # === Core Classification Metrics ===
    try:
        metrics['accuracy'] = accuracy_score(y_true_1d, y_pred_1d)
    except:
        metrics['accuracy'] = np.nan
    
    try:
        metrics['precision'] = precision_score(y_true_1d, y_pred_1d, zero_division=0, average='binary')
        metrics['precision_macro'] = precision_score(y_true_1d, y_pred_1d, zero_division=0, average='macro')
    except:
        metrics['precision'] = np.nan
        metrics['precision_macro'] = np.nan
    
    try:
        metrics['recall'] = recall_score(y_true_1d, y_pred_1d, zero_division=0, average='binary')
        metrics['recall_macro'] = recall_score(y_true_1d, y_pred_1d, zero_division=0, average='macro')
    except:
        metrics['recall'] = np.nan
        metrics['recall_macro'] = np.nan
    
    try:
        metrics['f1'] = f1_score(y_true_1d, y_pred_1d, zero_division=0, average='binary')
        metrics['f1_macro'] = f1_score(y_true_1d, y_pred_1d, zero_division=0, average='macro')
        metrics['f1_weighted'] = f1_score(y_true_1d, y_pred_1d, zero_division=0, average='weighted')
        metrics['f1_samples'] = f1_score(y_true_1d, y_pred_1d, zero_division=0, average='samples')
    except:
        metrics['f1'] = np.nan
        metrics['f1_macro'] = np.nan
        metrics['f1_weighted'] = np.nan
        metrics['f1_samples'] = np.nan
    
    # === Probability-based Metrics ===
    if y_proba is not None:
        y_proba = np.array(y_proba).ravel()
        # Clip probabilities to valid range
        y_proba = np.clip(y_proba, 0, 1)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_1d, y_proba)
        except:
            metrics['roc_auc'] = np.nan
        try:
            metrics['pr_auc'] = average_precision_score(y_true_1d, y_proba)
        except:
            metrics['pr_auc'] = np.nan
    
    # === Advanced Metrics (Literature 2024-2025) ===
    try:
        metrics['mcc'] = matthews_corrcoef(y_true_1d, y_pred_1d)  # Jadhav 2025
    except:
        metrics['mcc'] = np.nan
    
    try:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_1d, y_pred_1d)
    except:
        metrics['balanced_accuracy'] = np.nan
    
    # === G-Mean (Geometric Mean) ===
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_1d, y_pred_1d).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['g_mean'] = np.sqrt(sensitivity * specificity)
    except:
        metrics['g_mean'] = np.nan
    
    # === Multi-Label Metrics ===
    if multi_label and y_true.ndim > 1:
        try:
            metrics['jaccard_index'] = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
        except:
            metrics['jaccard_index'] = np.nan
        try:
            # Subset accuracy (exact match) - key for multi-label
            metrics['subset_accuracy'] = np.mean(np.all(y_true == y_pred, axis=1))
        except:
            metrics['subset_accuracy'] = np.nan
        try:
            metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        except:
            metrics['hamming_loss'] = np.nan
        
        # === Additional Multi-Label (Nature 2025) ===
        try:
            metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        except:
            metrics['f1_micro'] = np.nan
        try:
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        except:
            metrics['f1_macro'] = np.nan
        try:
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            metrics['f1_weighted'] = np.nan
        try:
            metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        except:
            metrics['precision_micro'] = np.nan
        try:
            metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        except:
            metrics['recall_micro'] = np.nan
    
    # === True Negative Rate (Specificity) ===
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_1d, y_pred_1d).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = specificity
        metrics['true_negative_rate'] = specificity
    except:
        metrics['specificity'] = np.nan
        metrics['true_negative_rate'] = np.nan
    
    # === Cohen's Kappa (multi-class agreement) ===
    try:
        from sklearn.metrics import cohen_kappa_score
        metrics['kappa'] = cohen_kappa_score(y_true_1d, y_pred_1d)
    except:
        metrics['kappa'] = np.nan
    
    # === Error Rate ===
    try:
        metrics['error_rate'] = 1 - metrics.get('accuracy', np.nan)
    except:
        metrics['error_rate'] = np.nan
    
    return metrics


def compute_delay_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics for delay prediction (in seconds).
    
    Based on Teo 2022 (MAE, RMSE) for GTFS delay prediction.
    
    Parameters
    ----------
    y_true : array-like
        True delays in seconds
    y_pred : array-like
        Predicted delays in seconds
    
    Returns
    -------
    Dict with metric names and values
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    
    return metrics


def compute_ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for ordinal classification (e.g., severity levels).
    
    Based on Shalit 2022 for ordinal GTFS classification.
    Pareto Accuracy measures how close ordinal predictions are.
    
    Parameters
    ----------
    y_true : array-like
        True ordinal labels (e.g., 0,1,2,3 for severity)
    y_pred : array-like
        Predicted ordinal labels
    
    Returns
    -------
    Dict with ordinal metrics
    """
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np
    
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    metrics = {}
    
    # Standard accuracy
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except:
        metrics['accuracy'] = np.nan
    
    # F1 for ordinal (weighted by class distance)
    try:
        metrics['f1_ordinal'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    except:
        metrics['f1_ordinal'] = np.nan
    
    # Pareto Accuracy (Shalit 2022)
    # Measures how close predictions are to true labels
    try:
        abs_diff = np.abs(y_true - y_pred)
        # Within 1 class tolerance
        metrics['pareto_accuracy_1'] = np.mean(abs_diff <= 1)
        # Within 2 class tolerance
        metrics['pareto_accuracy_2'] = np.mean(abs_diff <= 2)
    except:
        metrics['pareto_accuracy_1'] = np.nan
        metrics['pareto_accuracy_2'] = np.nan
    
    # Mean Absolute Distance (MAD)
    try:
        metrics['mean_abs_distance'] = np.mean(abs_diff)
    except:
        metrics['mean_abs_distance'] = np.nan
    
    # Class distribution for analysis
    try:
        metrics['true_class_mean'] = np.mean(y_true)
        metrics['pred_class_mean'] = np.mean(y_pred)
    except:
        metrics['true_class_mean'] = np.nan
        metrics['pred_class_mean'] = np.nan
    
    return metrics


def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute detailed sensitivity (recall/TPR) and specificity (TNR) per class.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    
    Returns
    -------
    Dict with per-class sensitivity and specificity
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    metrics = {}
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        classes = np.unique(y_true)
        
        for i, cls in enumerate(classes):
            # For binary, cm is 2x2
            if len(classes) == 2:
                tn, fp, fn, tp = cm.ravel()
            else:
                # Multi-class: extract binary stats for this class vs rest
                tp = cm[i, i] if i < cm.shape[0] else 0
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics[f'sensitivity_class_{cls}'] = sensitivity
            metrics[f'specificity_class_{cls}'] = specificity
            
            # Also add under original names
            metrics[f'recall_class_{cls}'] = sensitivity
            metrics[f'tnr_class_{cls}'] = specificity
        
        # Overall averages
        sens_values = [v for k, v in metrics.items() if 'sensitivity_class' in k]
        spec_values = [v for k, v in metrics.items() if 'specificity_class' in k]
        
        metrics['sensitivity_mean'] = np.mean(sens_values) if sens_values else np.nan
        metrics['specificity_mean'] = np.mean(spec_values) if spec_values else np.nan
        
    except Exception as e:
        logger.warning(f"  Sensitivity/specificity failed: {e}")
    
    return metrics


def compute_operational_metrics(inference_times: List[float],
                                 sample_count: int) -> Dict[str, float]:
    """
    Compute operational metrics for deployment.
    
    Based on Digiitus 2024 for real-time systems.
    
    Parameters
    ----------
    inference_times : list
        List of inference times in milliseconds
    sample_count : int
        Number of samples processed
    
    Returns
    -------
    Dict with operational metrics
    """
    import numpy as np
    
    inference_times = np.array(inference_times)
    
    metrics = {
        'latency_mean_ms': np.mean(inference_times),
        'latency_median_ms': np.median(inference_times),
        'latency_p95_ms': np.percentile(inference_times, 95),
        'latency_p99_ms': np.percentile(inference_times, 99),
        'latency_min_ms': np.min(inference_times),
        'latency_max_ms': np.max(inference_times),
        'throughput_samples_per_sec': sample_count / (np.sum(inference_times) / 1000),
    }
    
    return metrics


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                 target_names: Optional[List[str]] = None) -> str:
    """
    Generate sklearn classification report.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list, optional
        Names for each class
    
    Returns
    -------
    str : Formatted classification report
    """
    from sklearn.metrics import classification_report
    
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         normalize: bool = True) -> None:
    """
    Plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Class labels
    normalize : bool
        Whether to normalize by row (true labels)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                  label: str = 'Model') -> float:
    """
    Plot ROC curve and return AUC score.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities for positive class
    label : str
        Label for the curve
    
    Returns
    -------
    float : AUC score
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return roc_auc


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                               label: str = 'Model') -> float:
    """
    Plot Precision-Recall curve and return average precision.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities for positive class
    label : str
        Label for the curve
    
    Returns
    -------
    float : Average precision score
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{label} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return avg_precision


# Import interpretability module
from .interpretability import (
    SHAPExplainer,
    FeatureImportanceAnalyzer,
    AttentionVisualizer,
    explain_model
)

# Import significance testing module
from .significance import (
    SignificanceTester,
    compare_models
)

# Import fairness analysis module
from .fairness import (
    FairnessAnalyzer,
    analyze_fairness
)

# Import spatial maps module
from .spatial_maps import (
    generate_all_spatial_maps,
    plot_disruption_density_map,
    plot_severity_map,
    plot_spatial_lag_map,
    plot_interactive_map,
    plot_temporal_evolution_map,
    plot_hotspots_map,
)

__all__ = [
    'safe_roc_auc',
    'compute_metrics',
    'compute_multilabel_metrics',
    'generate_classification_report',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'SHAPExplainer',
    'FeatureImportanceAnalyzer',
    'AttentionVisualizer',
    'explain_model',
    'SignificanceTester',
    'compare_models',
    'FairnessAnalyzer',
    'analyze_fairness',
    'generate_all_spatial_maps',
    'plot_disruption_density_map',
    'plot_severity_map',
    'plot_spatial_lag_map',
    'plot_interactive_map',
    'plot_temporal_evolution_map',
    'plot_hotspots_map',
]
