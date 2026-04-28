"""
GTFS Disruption Detection - Modeling Module
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
import logging

logger = logging.getLogger(__name__)

def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    sort_first: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform temporal (chronological) split of DataFrame.
    
    CRITICAL: This prevents data leakage that occurs with random splits.
    With random splits, rolling window features can "see" future data during
    training, leading to overly optimistic evaluation metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    timestamp_col : str
        Column name containing timestamps (will be sorted by this)
    train_ratio : float
        Proportion for training set (default: 0.70)
    val_ratio : float
        Proportion for validation set (default: 0.15)
    test_ratio : float
        Proportion for test set (default: 0.15)
    sort_first : bool
        Whether to sort by timestamp before splitting
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) in chronological order
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    df_split = df.copy()
    
    if sort_first:
        if timestamp_col not in df_split.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")
        df_split = df_split.sort_values(timestamp_col).reset_index(drop=True)
    
    n = len(df_split)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_split.iloc[:train_end]
    val_df = df_split.iloc[train_end:val_end]
    test_df = df_split.iloc[val_end:]
    
    logger.info(f"Chronological split:")
    logger.info(f"  Train: {len(train_df):,} rows ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df):,} rows ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df):,} rows ({len(test_df)/n*100:.1f}%)")
    
    # Verify no temporal overlap
    if sort_first:
        max_train_time = train_df[timestamp_col].max()
        min_val_time = val_df[timestamp_col].min()
        max_val_time = val_df[timestamp_col].max()
        min_test_time = test_df[timestamp_col].min()
        
        if max_train_time >= min_val_time:
            warnings.warn("Temporal overlap detected between train and validation!")
        if max_val_time >= min_test_time:
            warnings.warn("Temporal overlap detected between validation and test!")
    
    return train_df, val_df, test_df


def temporal_train_val_test_split(
    df: pd.DataFrame,
    timestamp_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Alternative API: temporal split with test/val size specification.
    
    Train = 1 - test_size - val_size
    Val = val_size
    Test = test_size
    """
    return chronological_split(
        df, timestamp_col,
        train_ratio=1.0 - test_size - val_size,
        val_ratio=val_size,
        test_ratio=test_size
    )


class TemporalAwareBalancer:
    """
    Handles class imbalance WITHOUT breaking temporal ordering.
    
    Unlike SMOTE which generates synthetic samples at random positions,
    this class uses class weights which preserve the original temporal
    structure of the data.
    
    For tree-based models (RF, XGBoost, LightGBM), use class_weight parameter.
    For neural networks, use sample_weight during fit.
    """
    
    def __init__(self, strategy: str = "class_weight"):
        """
        Parameters
        ----------
        strategy : str
            "class_weight" - use balanced class weights (recommended)
            "oversample" - temporal-preserving oversampling
            "none" - no balancing
        """
        self.strategy = strategy
        self.class_weights_ = None
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights."""
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        
        weights = {}
        for cls, count in zip(classes, counts):
            weights[cls] = n_samples / (n_classes * count)
        
        self.class_weights_ = weights
        return weights
    
    def get_estimator_params(self, model_type: str) -> Dict:
        """Get appropriate parameters for different model types."""
        if self.strategy == "none" or self.class_weights_ is None:
            return {}
        
        if model_type in ["random_forest", "rf"]:
            return {"class_weight": "balanced"}
        
        elif model_type in ["xgboost", "xgb"]:
            if len(self.class_weights_) == 2:
                neg_weight = self.class_weights_.get(0, 1.0)
                pos_weight = self.class_weights_.get(1, 1.0)
                return {"scale_pos_weight": neg_weight / pos_weight}
            return {"scale_pos_weight": "balanced"}
        
        elif model_type in ["lightgbm", "lgb"]:
            return {"class_weight": "balanced"}
        
        elif model_type in ["sklearn", "any"]:
            return {"class_weight": self.class_weights_}
        
        return {}
    
    def fit(self, y: np.ndarray) -> 'TemporalAwareBalancer':
        """Fit the balancer on training labels."""
        if self.strategy == "class_weight":
            self.compute_class_weights(y)
        return self


class WalkForwardCV:
    """
    Walk-forward cross-validation for temporal data.
    
    Unlike standard k-fold CV, this preserves temporal ordering:
    - Each fold uses past data to predict future
    - Mimics real production deployment
    - Appropriate for time-series with limited data
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = None, gap: int = 0):
        """
        Parameters
        ----------
        n_splits : int
            Number of walk-forward folds
        test_size : int
            Size of test set per fold (in rows). If None, uses 1/n_splits
        gap : int
            Gap between train and test (in rows) to prevent leakage
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y=None, timestamps=None):
        """
        Generate train/test indices for walk-forward CV.
        
        Yields
        ------
        tuple
            (train_indices, test_indices)
        """
        n = len(X)
        
        if self.test_size is None:
            self.test_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = (i + 1) * self.test_size
            test_end = min(test_start + self.test_size, n)
            train_end = test_start - self.gap
            
            if train_end <= 0 or test_end > n:
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
    
    def get_n_splits(self):
        return self.n_splits


# Import adaptive_split module
from .adaptive_split import AdaptiveSplitter, SplitConfig, SplitResult, adaptive_split

# Import feature selection module
from .feature_selection import FeatureSelector, select_features

# Import hyperparameter optimization module
from .hyperparameter_optimization import (
    HyperparameterOptimizer,
    get_default_param_space,
    optimize_model
)

from .multi_output import (
    MultiOutputChainTrainer,
    MultiLabelWrapper,
    train_multilabel_model,
    evaluate_multilabel,
    compare_chain_vs_multioutput
)

try:
    from .gnn_models import (
        STARNGAT,
        STGAT,
        TransitGNNDataLoader,
        make_gnn_model,
        train_gnn_model,
        GraphAttentionLayer,
        ResidualGATBlock,
        SpatiotemporalAttention,
        TORCH_AVAILABLE
    )
    GNN_AVAILABLE = True
except (ImportError, OSError, AttributeError):
    GNN_AVAILABLE = False

try:
    from .interpretability import (
        DisruptionExplainer,
        explain_model,
        get_top_features,
        print_feature_importance,
        SHAP_AVAILABLE
    )
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False

__all__ = [
    'chronological_split',
    'temporal_train_val_test_split',
    'TemporalAwareBalancer',
    'WalkForwardCV',
    'AdaptiveSplitter',
    'SplitConfig',
    'SplitResult',
    'adaptive_split',
    'FeatureSelector',
    'select_features',
    'HyperparameterOptimizer',
    'get_default_param_space',
    'optimize_model',
    'MultiOutputChainTrainer',
    'MultiLabelWrapper',
    'train_multilabel_model',
    'evaluate_multilabel',
    'compare_chain_vs_multioutput',
    'STARNGAT',
    'STGAT',
    'make_gnn_model',
    'train_gnn_model',
    'GNN_AVAILABLE',
    'DisruptionExplainer',
    'explain_model',
    'get_top_features',
    'print_feature_importance',
    'INTERPRETABILITY_AVAILABLE',
]
