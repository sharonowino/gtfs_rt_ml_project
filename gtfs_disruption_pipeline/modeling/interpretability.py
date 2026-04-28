"""
GTFS Disruption Interpretability Module
==============================
SHAP and LIME integration for model interpretability.

Usage:
------
from gtfs_disruption.modeling.interpretability import DisruptionExplainer, explain_model

# Create explainer
explainer = DisruptionExplainer(model, X_train, model_type='lightgbm')
shap_values = explainer.explain(X_test)

# Get feature importance
explainer.plot_importance(shap_values, feature_names)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("shap not installed - interpretability disabled")
    SHAP_AVAILABLE = False


class DisruptionExplainer:
    """
    SHAP explainer for disruption models.
    
    Supports tree-based and neural network models.
    """
    
    def __init__(self, model, X_train: pd.DataFrame, model_type: str = 'auto'):
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        
        if model_type == 'auto':
            self.model_type = self._detect_model_type()
    
    def _detect_model_type(self) -> str:
        """Detect model type from model name."""
        model_name = type(self.model).__name__.lower()
        
        if 'lgb' in model_name:
            return 'lightgbm'
        elif 'xgb' in model_name:
            return 'xgboost'
        elif 'forest' in model_name or 'tree' in model_name:
            return 'randomforest'
        elif 'mlp' in model_name or 'neural' in model_name:
            return 'deep'
        return 'kernel'
    
    def create_explainer(self) -> 'shap.Explainer':
        """Create SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            raise ImportError("shap required: pip install shap")
        
        if self.model_type == 'lightgbm':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'xgboost':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'randomforest':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, self.X_train.values)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self.X_train.values
            )
        
        return self.explainer
    
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """Get SHAP values for test data."""
        if self.explainer is None:
            self.create_explainer()
        
        X_values = X.values if hasattr(X, 'values') else X
        
        if self.model_type in ['lightgbm', 'xgboost', 'randomforest']:
            self.shap_values = self.explainer.shap_values(X_values)
        elif self.model_type == 'deep':
            self.shap_values = self.explainer.shap_values(X_values)
        else:
            self.shap_values = self.explainer.shap_values(X_values)
        
        return self.shap_values
    
    def get_feature_importance(self, feature_names: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """Get top features by mean absolute SHAP value."""
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_abs)[-top_k:][::-1]
        
        return [(feature_names[i], mean_abs[i]) for i in indices]
    
    def plot_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 20,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot feature importance bar chart."""
        if not SHAP_AVAILABLE:
            return
        
        import matplotlib.pyplot as plt
        
        mean_abs = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(mean_abs)[-top_k:][::-1]
        
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(top_k),
            [mean_abs[i] for i in indices],
            [feature_names[i] for i in indices]
        )
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Feature')
        plt.title('Top Feature Importances (SHAP)')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        if show:
            plt.show()
    
    def plot_dependence(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        feature_idx: int,
        feature_name: str,
        color_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """Plot SHAP dependence for a feature."""
        if not SHAP_AVAILABLE:
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        X_values = X.values if hasattr(X, 'values') else X
        
        plt.scatter(X_values[:, feature_idx], shap_values[:, feature_idx], alpha=0.3)
        plt.xlabel(feature_name)
        plt.ylabel('SHAP Value')
        plt.title(f'SHAP Dependence: {feature_name}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
    
    def get_local_explanation(
        self,
        instance_idx: int,
        feature_names: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get local explanation for a single prediction."""
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        instance_shap = self.shap_values[instance_idx]
        indices = np.argsort(np.abs(instance_shap))[-top_k:][::-1]
        
        return [(feature_names[i], instance_shap[i]) for i in indices]


def explain_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_type: str = 'auto',
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, 'shap.Explainer']:
    """
    Convenience function to get SHAP explanations.
    
    Returns
    -------
    tuple (shap_values, explainer)
    """
    if feature_names is None:
        feature_names = X_train.columns.tolist()
    
    explainer = DisruptionExplainer(model, X_train, model_type)
    shap_values = explainer.explain(X_test)
    
    return shap_values, explainer


def get_top_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 20
) -> List[Tuple[str, float]]:
    """Get top features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs)[-top_k:][::-1]
    
    return [(feature_names[i], mean_abs[i]) for i in indices]


def print_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 20
):
    """Print feature importance table."""
    top_features = get_top_features(shap_values, feature_names, top_k)
    
    print(f"\n{'='*60}")
    print(f"Top {top_k} Features by SHAP Importance")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Feature':<30} {'SHAP':<10}")
    print(f"{'-'*60}")
    
    for i, (feat, score) in enumerate(top_features, 1):
        print(f"{i:<6} {feat:<30} {score:+.4f}")