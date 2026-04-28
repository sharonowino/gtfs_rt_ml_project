"""
GTFS Disruption Detection - Interpretability Module
==================================================
SHAP values, feature importance, and attention visualization.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model interpretability.
    
    Provides global and local feature importance explanations
    for tree-based and linear models.
    """
    
    def __init__(self, model, X_train: np.ndarray, feature_names: List[str]):
        """
        Initialize SHAP explainer.
        
        Parameters
        ----------
        model : trained model
            Model to explain
        X_train : np.ndarray
            Training data for background
        feature_names : List[str]
            Feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        # Try to import SHAP
        try:
            import shap
            self.shap_available = True
        except ImportError:
            self.shap_available = False
            warnings.warn("SHAP not installed. Install with: pip install shap")
    
    def compute_shap_values(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Parameters
        ----------
        X : np.ndarray
            Data to explain
        max_samples : int
            Maximum number of samples to use
        
        Returns
        -------
        np.ndarray of SHAP values
        """
        if not self.shap_available:
            logger.warning("SHAP not available. Returning zeros.")
            return np.zeros_like(X)
        
        import shap
        
        # Subsample if needed
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create explainer based on model type
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based model
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Linear model or other
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    self.X_train[:100]  # Background sample
                )
            
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-output
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Use positive class
            
            return self.shap_values
            
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return np.zeros_like(X_sample)
    
    def get_global_importance(self, X: np.ndarray, top_n: int = 20) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Parameters
        ----------
        X : np.ndarray
            Data to explain
        top_n : int
            Number of top features to return
        
        Returns
        -------
        pd.DataFrame with feature importance
        """
        shap_values = self.compute_shap_values(X)
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def get_local_explanation(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Get local explanation for a single sample.
        
        Parameters
        ----------
        X : np.ndarray
            Data containing the sample
        sample_idx : int
            Index of sample to explain
        
        Returns
        -------
        Dict with local explanation
        """
        shap_values = self.compute_shap_values(X[sample_idx:sample_idx+1])
        
        # Get feature contributions
        contributions = []
        for i, (feat_name, shap_val) in enumerate(zip(self.feature_names, shap_values[0])):
            contributions.append({
                'feature': feat_name,
                'shap_value': shap_val,
                'feature_value': X[sample_idx, i] if i < X.shape[1] else np.nan
            })
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'sample_idx': sample_idx,
            'base_value': self.explainer.expected_value if self.explainer else 0,
            'prediction': self.model.predict_proba(X[sample_idx:sample_idx+1])[0, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X[sample_idx:sample_idx+1])[0],
            'top_contributions': contributions[:10]
        }
    
    def plot_summary(self, X: np.ndarray, plot_type: str = 'beeswarm', max_display: int = 20):
        """
        Plot SHAP summary.
        
        Parameters
        ----------
        X : np.ndarray
            Data to explain
        plot_type : str
            'beeswarm', 'bar', or 'violin'
        max_display : int
            Maximum features to display
        """
        if not self.shap_available:
            logger.warning("SHAP not available. Cannot plot.")
            return
        
        import shap
        import matplotlib.pyplot as plt
        
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'beeswarm':
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, max_display=max_display, show=False)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, plot_type='bar', max_display=max_display, show=False)
        elif plot_type == 'violin':
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, plot_type='violin', max_display=max_display, show=False)
        
        plt.tight_layout()
        plt.show()


class FeatureImportanceAnalyzer:
    """
    Feature importance analysis using multiple methods.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize feature importance analyzer.
        
        Parameters
        ----------
        model : trained model
            Model to analyze
        feature_names : List[str]
            Feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    def get_tree_importance(self) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Returns
        -------
        pd.DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                   n_repeats: int = 10) -> pd.DataFrame:
        """
        Get permutation importance.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        n_repeats : int
            Number of repeats for permutation
        
        Returns
        -------
        pd.DataFrame with permutation importance
        """
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(result.importances_mean)],
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def plot_importance(self, importance_df: pd.DataFrame, top_n: int = 20, 
                       title: str = 'Feature Importance'):
        """
        Plot feature importance.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            Feature importance DataFrame
        top_n : int
            Number of top features to plot
        title : str
            Plot title
        """
        import matplotlib.pyplot as plt
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class AttentionVisualizer:
    """
    Visualize attention weights from neural network models.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize attention visualizer.
        
        Parameters
        ----------
        model : trained model
            Neural network model with attention
        feature_names : List[str]
            Feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Extract attention weights from model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        
        Returns
        -------
        np.ndarray of attention weights
        """
        # This is model-specific - implement based on your architecture
        if hasattr(self.model, 'attention_weights'):
            return self.model.attention_weights
        elif hasattr(self.model, 'get_attention'):
            return self.model.get_attention(X)
        else:
            logger.warning("Model does not have attention weights")
            return np.zeros((len(X), len(self.feature_names)))
    
    def plot_attention_heatmap(self, X: np.ndarray, sample_idx: int = 0):
        """
        Plot attention heatmap for a sample.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        sample_idx : int
            Index of sample to visualize
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        attention_weights = self.get_attention_weights(X)
        
        if len(attention_weights) == 0:
            logger.warning("No attention weights available")
            return
        
        # Get attention for sample
        if len(attention_weights.shape) == 3:
            # Multi-head attention
            attn = attention_weights[sample_idx].mean(axis=0)  # Average across heads
        else:
            attn = attention_weights[sample_idx]
        
        # Reshape if needed
        if len(attn) != len(self.feature_names):
            logger.warning(f"Attention shape {attn.shape} doesn't match features {len(self.feature_names)}")
            return
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            attn.reshape(1, -1),
            xticklabels=self.feature_names,
            yticklabels=['Attention'],
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title(f'Attention Weights for Sample {sample_idx}')
        plt.tight_layout()
        plt.show()
    
    def plot_attention_distribution(self, X: np.ndarray, top_n: int = 10):
        """
        Plot distribution of attention weights across features.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        top_n : int
            Number of top features to show
        """
        import matplotlib.pyplot as plt
        
        attention_weights = self.get_attention_weights(X)
        
        if len(attention_weights) == 0:
            logger.warning("No attention weights available")
            return
        
        # Average attention across samples
        if len(attention_weights.shape) == 3:
            avg_attention = attention_weights.mean(axis=(0, 1))  # Average across samples and heads
        else:
            avg_attention = attention_weights.mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(avg_attention)[-top_n:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]
        top_attention = avg_attention[top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_attention)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Average Attention Weight')
        plt.title(f'Top {top_n} Features by Attention')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


def explain_model(model, X_train: np.ndarray, X_test: np.ndarray, 
                 feature_names: List[str], method: str = 'shap') -> Dict[str, Any]:
    """
    Convenience function to explain model predictions.
    
    Parameters
    ----------
    model : trained model
        Model to explain
    X_train : np.ndarray
        Training data
    X_test : np.ndarray
        Test data to explain
    feature_names : List[str]
        Feature names
    method : str
        'shap', 'permutation', or 'attention'
    
    Returns
    -------
    Dict with explanation results
    """
    results = {}
    
    if method == 'shap':
        explainer = SHAPExplainer(model, X_train, feature_names)
        results['global_importance'] = explainer.get_global_importance(X_test)
        results['local_explanation'] = explainer.get_local_explanation(X_test, sample_idx=0)
        
    elif method == 'permutation':
        analyzer = FeatureImportanceAnalyzer(model, feature_names)
        results['permutation_importance'] = analyzer.get_permutation_importance(
            X_test, np.zeros(len(X_test))  # Dummy y for unsupervised
        )
        
    elif method == 'attention':
        visualizer = AttentionVisualizer(model, feature_names)
        results['attention_weights'] = visualizer.get_attention_weights(X_test)
    
    return results
