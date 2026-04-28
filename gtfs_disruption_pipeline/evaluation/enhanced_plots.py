"""
================================================================================
ENHANCED EVALUATION PLOTS FOR ALL MODELS
================================================================================

This module provides visualization functions for extended enhanced evaluation metrics:
- PR-AUC curves
- ROC curves
- Lead Time Analysis
- Detection Delay
- Model Comparison Charts

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, f1_score, average_precision_score, roc_auc_score
)


def plot_pr_auc_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Precision-Recall Curves - All Models",
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot PR-AUC curves for all models.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to (y_true, y_proba) tuples
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (model_name, (y_true, y_proba)), color in zip(results_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

        ax.plot(recall, precision, color=color, lw=2,
                label=f'{model_name} (PR-AUC = {pr_auc:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves - All Models",
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot ROC curves for all models.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to (y_true, y_proba) tuples
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (model_name, (y_true, y_proba)), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (ROC-AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_lead_time_analysis(
    results_df: pd.DataFrame,
    metric_col: str = 'mean_lead_time',
    title: str = "Lead Time Analysis - All Models",
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot lead time analysis for all models.

    Parameters
    ----------
    results_df : DataFrame
        DataFrame with model results including lead time metrics
    metric_col : str
        Column to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_df)))
    bars = ax1.bar(results_df['model'], results_df[metric_col], color=colors)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Lead Time (minutes)', fontsize=12)
    ax1.set_title('Mean Lead Time by Model', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, results_df[metric_col]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    ax2 = axes[1]
    sorted_df = results_df.sort_values(metric_col, ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_df)))
    ax2.barh(sorted_df['model'], sorted_df[metric_col], color=colors)
    ax2.set_xlabel('Lead Time (minutes)', fontsize=12)
    ax2.set_title('Model Ranking by Lead Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_detection_delay(
    results_df: pd.DataFrame,
    title: str = "Detection Delay Analysis - All Models",
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot detection delay for all models (negative = early detection).

    Parameters
    ----------
    results_df : DataFrame
        DataFrame with model results including detection_delay
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sorted_df = results_df.sort_values('detection_delay', ascending=True)

    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in sorted_df['detection_delay']]

    bars = ax.barh(sorted_df['model'], sorted_df['detection_delay'], color=colors)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel('Detection Delay (minutes)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Early Detection (< 0)'),
        Patch(facecolor='#e74c3c', label='Late Detection (> 0)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Model Comparison - All Metrics",
    figsize: Tuple[int, int] = (14, 8),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot comprehensive model comparison across all metrics.

    Parameters
    ----------
    results_df : DataFrame
        DataFrame with model results
    metrics : list
        List of metric columns to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib Figure
    """
    if metrics is None:
        metrics = ['f1_score', 'pr_auc', 'roc_auc', 'precision', 'recall']

    available_metrics = [m for m in metrics if m in results_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]

        sorted_df = results_df.sort_values(metric, ascending=False)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_df)))

        bars = ax.bar(sorted_df['model'], sorted_df[metric], color=colors)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, sorted_df[metric]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrices(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Confusion Matrices - All Models",
    figsize: Tuple[int, int] = (15, 5),
    save_path: str = None,
) -> plt.Figure:
    """
    Plot confusion matrices for all models.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to (y_true, y_pred) tuples
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib Figure
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, (y_true, y_pred)) in zip(axes, results_dict.items()):
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Disruption', 'Disruption'],
                   yticklabels=['No Disruption', 'Disruption'])
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_evaluation_dashboard(
    results_df: pd.DataFrame,
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_dir: str = None,
) -> Dict[str, plt.Figure]:
    """
    Create a complete evaluation dashboard with all plots.

    Parameters
    ----------
    results_df : DataFrame
        DataFrame with model results
    results_dict : dict
        Dictionary mapping model names to (y_true, y_proba) tuples
    save_dir : str, optional
        Directory to save plots

    Returns
    -------
    dict
        Dictionary of matplotlib Figures
    """
    figures = {}

    figures['model_comparison'] = plot_model_comparison(
        results_df,
        title="Enhanced Evaluation - All Models",
        save_path=f"{save_dir}/model_comparison.png" if save_dir else None
    )

    figures['pr_auc'] = plot_pr_auc_curves(
        results_dict,
        title="Precision-Recall Curves - All Models",
        save_path=f"{save_dir}/pr_auc_curves.png" if save_dir else None
    )

    figures['roc'] = plot_roc_curves(
        results_dict,
        title="ROC Curves - All Models",
        save_path=f"{save_dir}/roc_curves.png" if save_dir else None
    )

    if 'mean_lead_time' in results_df.columns:
        figures['lead_time'] = plot_lead_time_analysis(
            results_df,
            title="Lead Time Analysis - All Models",
            save_path=f"{save_dir}/lead_time.png" if save_dir else None
        )

    if 'detection_delay' in results_df.columns:
        figures['detection_delay'] = plot_detection_delay(
            results_df,
            title="Detection Delay - All Models",
            save_path=f"{save_dir}/detection_delay.png" if save_dir else None
        )

    return figures


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000

    models = ['RandomForest', 'XGBoost', 'LightGBM', 'STGCN', 'DCRNN']

    results_data = {
        'model': models,
        'f1_score': np.random.uniform(0.3, 0.7, len(models)),
        'precision': np.random.uniform(0.3, 0.7, len(models)),
        'recall': np.random.uniform(0.3, 0.7, len(models)),
        'specificity': np.random.uniform(0.5, 0.9, len(models)),
        'pr_auc': np.random.uniform(0.3, 0.7, len(models)),
        'roc_auc': np.random.uniform(0.5, 0.9, len(models)),
        'mean_lead_time': np.random.uniform(10, 30, len(models)),
        'lead_time_gain': np.random.uniform(10, 30, len(models)),
        'detection_delay': np.random.uniform(-20, 0, len(models)),
    }

    results_df = pd.DataFrame(results_data)

    print("Creating evaluation dashboard...")

    fig1 = plot_model_comparison(results_df)
    plt.show()

    fig2 = plot_lead_time_analysis(results_df)
    plt.show()

    fig3 = plot_detection_delay(results_df)
    plt.show()

    print("\nDashboard complete!")