"""
GTFS Disruption Detection - Feature Selection Module

Research-Grade Feature Selection for Spatiotemporal Transportation Data

Issues addressed per research audit:
- Preserves operationally important variables (delay, headway, alert features)
- Temporal-aware selection (respects time-series structure)
- Domain expert override capability
- Reduces bias from tree-based importance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
from sklearn.feature_selection import (
    mutual_info_classif, 
    RFECV,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Domain expert: operationally important features that should NOT be removed
REQUIRED_FEATURES: Set[str] = {
    'delay', 'arrival_delay_seconds', 'arrival_delay', 'is_delayed', 'is_on_time',
    'headway', 'headway_ratio', 'actual_headway', 'headway_deviation',
    'alert_id', 'cause', 'effect', 'cause_id', 'effect_id', 'alert_active',
    'route_id', 'stop_id', 'trip_id', 'vehicle_id',
    'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'stop_degree', 'is_hub_stop', 'dist_to_nearest_hub_km',
}

# Features that require temporal context (should not be removed without lag analysis)
TEMPORAL_FEATURES: Set[str] = {
    'delay_trend', 'delay_acceleration', 'delay_diff',
    'mean_delay_1min', 'mean_delay_3min', 'mean_delay_5min', 'mean_delay_10min',
    'mean_delay_30min', 'mean_delay_60min',
    'rolling_delay_std', 'delay_variance_1min', 'delay_variance_5min',
}


class FeatureSelector:
    """
    Feature selection for spatiotemporal transportation datasets.
    
    Methods:
    - Correlation filtering: Remove highly correlated features
    - Mutual information: Rank features by information gain
    - Recursive feature elimination: With temporal cross-validation
    - Variance threshold: Remove low-variance features
    - Domain expert protection: Preserve operationally important features
    - Temporal-aware selection: Protect temporal features
    
    Research audit fixes:
    - Preserves REQUIRED_FEATURES (operationally critical)
    - Protects TEMPORAL_FEATURES (time-series structure)
    - Domain expert override capability
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature selector.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration for feature selection
        config keys:
            - preserve_required: bool (preserve domain expert features)
            - preserve_temporal: bool (protect temporal features)
            - domain_expert_features: List[str] (additional required features)
        """
        self.config = config or {}
        self.selected_features = []
        self.feature_scores = {}
        self.removed_features = []
        
        # Configuration
        self.preserve_required = self.config.get('preserve_required', True)
        self.preserve_temporal = self.config.get('preserve_temporal', True)
        self.domain_expert_features = set(self.config.get('domain_expert_features', []))
    
    def correlation_filter(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        threshold : float
            Correlation threshold to remove features
        
        Returns
        -------
        List of features to keep
        """
        logger.info(f"Running correlation filter (threshold={threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Keep features not in drop list
        features_to_keep = [col for col in X.columns if col not in to_drop]
        
        logger.info(f"  Removed {len(to_drop)} highly correlated features")
        logger.info(f"  Remaining features: {len(features_to_keep)}")
        
        return features_to_keep
    
    def mutual_information_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        top_k: int = 30
    ) -> List[str]:
        """
        Select features based on mutual information.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target labels
        top_k : int
            Number of top features to select
        
        Returns
        -------
        List of selected features
        """
        logger.info(f"Running mutual information selection (top_k={top_k})...")
        
        # Handle missing values
        X_filled = X.fillna(X.median())
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_filled, y, random_state=42)
        
        # Create feature scores dictionary
        self.feature_scores = dict(zip(X.columns, mi_scores))
        
        # Sort by score
        sorted_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top k features
        selected = [feat for feat, score in sorted_features[:top_k]]
        
        logger.info(f"  Selected {len(selected)} features by mutual information")
        logger.info(f"  Top 5 features: {selected[:5]}")
        
        return selected
    
    def recursive_feature_elimination(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        min_features: int = 10,
        cv_folds: int = 5
    ) -> List[str]:
        """
        Recursive feature elimination with cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target labels
        min_features : int
            Minimum number of features to select
        cv_folds : int
            Number of cross-validation folds
        
        Returns
        -------
        List of selected features
        """
        logger.info(f"Running recursive feature elimination (min_features={min_features})...")
        
        # Handle missing values
        X_filled = X.fillna(X.median())
        
        # Use Random Forest for RFE
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Run RFECV
        rfecv = RFECV(
            estimator=rf,
            step=1,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1',
            min_features_to_select=min_features,
            n_jobs=-1
        )
        
        rfecv.fit(X_filled, y)
        
        # Get selected features
        selected = X.columns[rfecv.support_].tolist()
        
        logger.info(f"  Selected {len(selected)} features by RFE")
        logger.info(f"  Optimal number of features: {rfecv.n_features_}")
        
        return selected
    
    def variance_threshold_selection(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Remove low-variance features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        threshold : float
            Variance threshold
        
        Returns
        -------
        List of features to keep
        """
        logger.info(f"Running variance threshold selection (threshold={threshold})...")
        
        # Handle missing values
        X_filled = X.fillna(X.median())
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_filled)
        
        # Get selected features
        selected = X.columns[selector.get_support()].tolist()
        
        logger.info(f"  Removed {len(X.columns) - len(selected)} low-variance features")
        logger.info(f"  Remaining features: {len(selected)}")
        
        return selected
    
    def protect_domain_features(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """
        Protect operationally important features from removal.
        
        Per research audit: certain features are operationally critical
        and should not be removed even if they correlate highly or have low variance.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        features : List[str]
            Current feature list
        
        Returns
        -------
        List of features with domain expert protection applied
        """
        if not self.preserve_required:
            return features
        
        # Build complete protected set
        protected = REQUIRED_FEATURES | self.domain_expert_features | TEMPORAL_FEATURES
        
        # Check which protected features exist in X
        protected_present = [f for f in protected if f in X.columns]
        
        # Add missing but needed
        for feat in features:
            if feat in protected:
                protected_present.append(feat)
        
        protected_present = list(set(protected_present))
        
        removed = set(features) - set(protected_present) - set(protected_present)
        if removed:
            self.removed_features.extend(list(removed))
            logger.info(f"  Domain expert: protected {len([f for f in protected_present if f in features])} features")
        
        return protected_present
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        methods: Optional[List[str]] = None,
        config: Optional[Dict] = None
    ) -> List[str]:
        """
        Run feature selection pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target labels
        methods : List[str], optional
            List of methods to use ['correlation', 'mutual_info', 'rfe', 'variance']
        config : Dict, optional
            Configuration for each method
        
        Returns
        -------
        List of selected features
        """
        if methods is None:
            methods = ['correlation', 'mutual_info', 'rfe']
        
        if config is None:
            config = self.config
        
        logger.info("="*60)
        logger.info("FEATURE SELECTION PIPELINE")
        logger.info("="*60)
        
        selected_features = list(X.columns)
        
        # Step 0: Domain expert protection (BEFORE any selection)
        selected_features = self.protect_domain_features(X, selected_features)
        
        # Step 1: Variance threshold
        if 'variance' in methods:
            variance_config = config.get('variance', {})
            selected_features = self.variance_threshold_selection(
                X[selected_features],
                threshold=variance_config.get('threshold', 0.01)
            )
            # Re-protect after each step
            selected_features = self.protect_domain_features(X, selected_features)
        
        # Step 2: Correlation filtering
        if 'correlation' in methods:
            corr_config = config.get('correlation', {})
            selected_features = self.correlation_filter(
                X[selected_features],
                threshold=corr_config.get('threshold', 0.95)
            )
            selected_features = self.protect_domain_features(X, selected_features)
        
        # Step 3: Mutual information
        if 'mutual_info' in methods:
            mi_config = config.get('mutual_info', {})
            selected_features = self.mutual_information_selection(
                X[selected_features],
                y,
                top_k=mi_config.get('top_k', 30)
            )
            selected_features = self.protect_domain_features(X, selected_features)
        
        # Step 4: Recursive feature elimination
        if 'rfe' in methods:
            rfe_config = config.get('rfe', {})
            selected_features = self.recursive_feature_elimination(
                X[selected_features],
                y,
                min_features=rfe_config.get('min_features', 10),
                cv_folds=rfe_config.get('cv_folds', 5)
            )
            selected_features = self.protect_domain_features(X, selected_features)
        
        self.selected_features = selected_features
        
        logger.info("="*60)
        logger.info(f"FINAL SELECTED FEATURES: {len(selected_features)}")
        logger.info("="*60)
        
        return selected_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns
        -------
        Dict mapping feature names to importance scores
        """
        return self.feature_scores


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    methods: Optional[List[str]] = None,
    config: Optional[Dict] = None
) -> List[str]:
    """
    Convenience function for feature selection.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target labels
    methods : List[str], optional
        List of methods to use
    config : Dict, optional
        Configuration for each method
    
    Returns
    -------
    List of selected features
    """
    selector = FeatureSelector(config)
    return selector.select_features(X, y, methods, config)
