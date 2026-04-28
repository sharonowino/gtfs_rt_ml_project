"""
GTFS Multi-Output Classification Module
======================================
Train and evaluate multi-output classifiers for disruption prediction.

This module supports:
1. MultiOutputClassifier - parallel independent classifiers
2. ClassifierChain - sequential classifiers with dependency modeling

Key difference in predict_proba:
- MultiOutputClassifier: returns list of L arrays, each shape (n, 2)
- ClassifierChain: returns single 2D array, shape (n, L)

Usage:
------
from gtfs_disruption.modeling.multi_output import (
    MultiOutputChainTrainer,
    train_multilabel_model,
    evaluate_multilabel
)
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    hamming_loss, accuracy_score
)

logger = logging.getLogger(__name__)


class MultiOutputChainTrainer:
    """
    Trainer for multi-output classification models.
    
    Supports both MultiOutputClassifier and ClassifierChain with
    explicit handling of their different predict_proba outputs.
    
    Parameters
    ----------
    model_type : str
        "multi_output" or "chain"
    base_estimator : BaseEstimator
        Base classifier (e.g., RandomForest, XGBoost, LightGBM)
    order : list, optional
        Specific label order for ClassifierChain
    """
    
    def __init__(
        self,
        model_type: str = "chain",
        base_estimator: Optional[BaseEstimator] = None,
        order: Optional[List[str]] = None
    ):
        self.model_type = model_type
        self.base_estimator = base_estimator
        self.order = order
        self.model_ = None
        self.label_names_ = []
        self._is_fitted = False
    
    def _create_model(self, estimator) -> Union[MultiOutputClassifier, ClassifierChain]:
        """Create the multi-output model."""
        if self.model_type == "multi_output":
            model = MultiOutputClassifier(estimator)
        elif self.model_type == "chain":
            model = ClassifierChain(estimator, order=self.order)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return model
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'MultiOutputChainTrainer':
        """
        Fit the multi-output model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Label matrix (n_samples, n_labels)
        sample_weight : np.ndarray, optional
            Sample weights for training
            
        Returns
        -------
        self
        """
        if self.base_estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            self.base_estimator = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        self.model_ = self._create_model(self.base_estimator)
        
        logger.info(f"Fitting {self.model_type} model...")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        
        if sample_weight is not None:
            self.model_.fit(X, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X, y)
        
        self._is_fitted = True
        
        if self.model_type == "chain" and self.order is None:
            if hasattr(self.model_, 'order_'):
                self.order = self.model_.order_
                logger.info(f"  Chain order: {self.order}")
        
        logger.info(f"  Model fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model_.predict(X)
    
    def predict_proba(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Predict probabilities with explicit handling.
        
        Returns
        -------
        tuple
            (proba, is_chain_format)
            
            - MultiOutputClassifier: returns list of L arrays, each (n, 2)
            - ClassifierChain: returns single array (n, L)
            
        Examples
        --------
        >>> proba, is_chain = model.predict_proba(X_test)
        >>> if is_chain:
        ...     # Shape: (n_samples, n_labels)
        ...     probs = proba[:, 1]  # Positive class probability
        ... else:
        ...     # Shape: list of n_labels arrays, each (n_samples, 2)
        ...     probs = np.array([p[:, 1] for p in proba]).T
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        raw_proba = self.model_.predict_proba(X)
        
        if self.model_type == "chain":
            if isinstance(raw_proba, list):
                proba = np.array(raw_proba)
            else:
                proba = raw_proba
            return proba, True
        else:
            if isinstance(raw_proba, list):
                return raw_proba, False
            else:
                return [raw_proba], False
    
    def predict_proba_binary(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get positive class probabilities in standard (n_samples, n_labels) format.
        
        Handle both MultiOutputClassifier and ClassifierChain outputs.
        """
        proba, is_chain = self.predict_proba(X)
        
        if is_chain:
            if proba.ndim == 2:
                return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
            else:
                return proba
        else:
            if isinstance(proba, list):
                return np.array([p[:, 1] for p in proba]).T
            else:
                return proba[:, 1] if proba.ndim > 1 else proba
    
    def score(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y_true : np.ndarray
            True labels (n_samples, n_labels)
        metrics : list
            Metrics to compute: 'f1', 'precision', 'recall', 'hamming'
            
        Returns
        -------
        dict
            Metric name -> score
        """
        if metrics is None:
            metrics = ['f1', 'precision', 'recall', 'hamming']
        
        y_pred = self.predict(X)
        
        results = {}
        
        for metric in metrics:
            if metric == 'f1':
                results['f1_macro'] = f1_score(
                    y_true, y_pred, average='macro', zero_division=0
                )
                results['f1_micro'] = f1_score(
                    y_true, y_pred, average='micro', zero_division=0
                )
                results['f1_samples'] = f1_score(
                    y_true, y_pred, average='samples', zero_division=0
                )
            elif metric == 'precision':
                results['precision_macro'] = precision_score(
                    y_true, y_pred, average='macro', zero_division=0
                )
                results['precision_micro'] = precision_score(
                    y_true, y_pred, average='micro', zero_division=0
                )
            elif metric == 'recall':
                results['recall_macro'] = recall_score(
                    y_true, y_pred, average='macro', zero_division=0
                )
                results['recall_micro'] = recall_score(
                    y_true, y_pred, average='micro', zero_division=0
                )
            elif metric == 'hamming':
                results['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        results['exact_match'] = accuracy_score(y_true, y_pred)
        
        return results


def train_multilabel_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "chain",
    base_estimator: Optional[BaseEstimator] = None,
    label_order: Optional[List[str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    **estimator_kwargs
) -> MultiOutputChainTrainer:
    """
    Convenience function to train a multi-output model.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels (n_samples, n_labels)
    model_type : str
        "multi_output" or "chain"
    base_estimator : BaseEstimator
        Base classifier instance
    label_order : list
        Order for ClassifierChain
    sample_weight : np.ndarray
        Sample weights
    **estimator_kwargs
        Passed to base_estimator if not provided
        
    Returns
    -------
    MultiOutputChainTrainer
        Fitted model
    """
    if base_estimator is None:
        from sklearn.ensemble import RandomForestClassifier
        base_estimator = RandomForestClassifier(
            n_estimators=estimator_kwargs.get('n_estimators', 100),
            class_weight=estimator_kwargs.get('class_weight', 'balanced'),
            random_state=estimator_kwargs.get('random_state', 42),
            n_jobs=estimator_kwargs.get('n_jobs', -1)
        )
    
    trainer = MultiOutputChainTrainer(
        model_type=model_type,
        base_estimator=base_estimator,
        order=label_order
    )
    
    trainer.fit(X_train, y_train, sample_weight)
    
    return trainer


def evaluate_multilabel(
    model: MultiOutputChainTrainer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate multi-label model with comprehensive metrics.
    
    Parameters
    ----------
    model : MultiOutputChainTrainer
        Fitted model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        True test labels
    label_names : list
        Names of labels for reporting
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    results = model.score(X_test, y_test)
    
    if label_names is not None:
        logger.info("\nPer-label metrics:")
        
        for i, name in enumerate(label_names):
            if i < y_test.shape[1]:
                f1 = f1_score(
                    y_test[:, i], y_pred[:, i],
                    average='binary', zero_division=0
                )
                prec = precision_score(
                    y_test[:, i], y_pred[:, i],
                    average='binary', zero_division=0
                )
                rec = recall_score(
                    y_test[:, i], y_pred[:, i],
                    average='binary', zero_division=0
                )
                logger.info(f"  {name}: F1={f1:.3f}, P={prec:.3f}, R={rec:.3f}")
    
    return results


def compare_chain_vs_multioutput(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_estimator,
    label_order: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare MultiOutputClassifier vs ClassifierChain performance.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    base_estimator : BaseEstimator
        Base classifier
    label_order : list
        Label order for chain
        
    Returns
    -------
    dict
        {'multi_output': {...}, 'chain': {...}}
    """
    results = {}
    
    for model_type in ['multi_output', 'chain']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type}...")
        logger.info(f"{'='*50}")
        
        model = train_multilabel_model(
            X_train, y_train,
            model_type=model_type,
            base_estimator=base_estimator,
            label_order=label_order
        )
        
        eval_results = evaluate_multilabel(model, X_test, y_test)
        results[model_type] = eval_results
        
        logger.info(f"\nResults for {model_type}:")
        for k, v in eval_results.items():
            logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"\n{'='*50}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*50}")
    
    for metric in ['f1_macro', 'hamming_loss', 'exact_match']:
        if metric in results.get('multi_output', {}):
            mo = results['multi_output'].get(metric, 0)
            ch = results['chain'].get(metric, 0)
            winner = "chain" if ch > mo else "multi_output" if mo > ch else "tie"
            logger.info(f"  {metric}: multi_output={mo:.4f}, chain={ch:.4f} -> {winner}")
    
    return results


class MultiLabelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to convert single-output model to multi-label.
    
    Useful for models that don't natively support multi-label.
    """
    
    def __init__(
        self,
        estimator: BaseEstimator,
        strategy: str = "copy"
    ):
        self.estimator = estimator
        self.strategy = strategy
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiLabelWrapper':
        """Fit separate models for each label."""
        n_labels = y.shape[1] if y.ndim > 1 else 1
        
        self.models_ = []
        
        for i in range(n_labels):
            if y.ndim > 1:
                y_i = y[:, i]
            else:
                y_i = y
            
            model_i = self._clone_estimator(self.estimator)
            model_i.fit(X, y_i)
            self.models_.append(model_i)
        
        return self
    
    def _clone_estimator(self, est):
        """Clone estimator for each label."""
        from sklearn.base import clone
        return clone(est)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        predictions = []
        
        for model in self.models_:
            predictions.append(model.predict(X))
        
        if len(predictions) == 1:
            return predictions[0].reshape(-1, 1)
        
        return np.column_stack(predictions)
    
    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """Predict probabilities for each label."""
        probas = []
        
        for model in self.models_:
            if hasattr(model, 'predict_proba'):
                probas.append(model.predict_proba(X))
            else:
                pred = model.predict(X)
                proba = np.zeros((len(X), 2))
                proba[:, 1] = pred
                proba[:, 0] = 1 - pred
                probas.append(proba)
        
        return probas