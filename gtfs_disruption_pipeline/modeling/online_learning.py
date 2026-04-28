"""
Online Learning & Concept Drift Detection
====================================

For production transit disruption systems that must adapt to:
1. Seasonal changes (timetable changes, holidays)
2. Infrastructure changes (new routes, closures)
3. External events (weather, emergencies)
4. Gradual degradation (aging systems)

This module provides:
1. Incremental model training
2. Concept drift detection (ADWIN, DDM)
3. Model update triggers
4. Performance monitoring

Usage:
------
from gtfs_disruption.modeling.online_learning import (
    OnlineDriftDetector,
    IncrementalModel,
    monitor_performance,
)

# Initialize drift detector
detector = OnlineDriftDetector(
    drift_detector='adwin',
    warning_threshold=0.05,
    drift_threshold=0.1,
)

# In production loop
for batch in streaming_data:
    detector.update(batch)
    if detector.drift_detected:
        retrain_model()
"""
import logging
from typing import Dict, List, Optional, Tuple, Literal, Callable, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for river library (online learning)
RIVER_AVAILABLE = False

try:
    from river import compose, drift, metrics, tree, forest
    from river.datasets import Stream
    RIVER_AVAILABLE = True
except ImportError:
    river = None
    drift = None
    metrics = None
    tree = None
    forest = None


# =========================================================================
# DRIFT DETECTION ALGORITHMS
# =========================================================================

class DriftDetector:
    """Base class for drift detection."""
    
    def __init__(self, warning_threshold: float = 0.05, drift_threshold: float = 0.1):
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.warning_detected = False
        self.drift_detected = False
        self.warning_history = []
        self.drift_history = []
    
    def update(self, prediction: float, actual: float) -> Dict[str, bool]:
        """Update with new prediction and actual value."""
        raise NotImplementedError
    
    def reset(self):
        """Reset detector state."""
        self.warning_detected = False
        self.drift_detected = False
        self.warning_history = []
        self.drift_history = []


class ADWINDriftDetector(DriftDetector):
    """
    ADaptive WINdows (ADWIN) drift detector.
    
    Maintains variable-length window of recent items.
    Detects change in data distribution.
    
    Based on: Bifet and Gavalas (2008)
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.05,
        drift_threshold: float = 0.1,
        delta_confidence: float = 0.002,
    ):
        super().__init__(warning_threshold, drift_threshold)
        
        self.delta = delta_confidence
        
        # ADWIN parameters
        self.window = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        
        # Detection variables
        self._epsilon_cut = 0.0
    
    def update(self, prediction: float, actual: float) -> Dict[str, bool]:
        """Update detector with new observation."""
        error = float(prediction != actual)
        
        # Add to window
        self.window.append(error)
        self.width += 1
        
        # Update statistics
        mean = np.mean(self.window)
        var = np.var(self.window) if len(self.window) > 1 else 0
        
        # Compute cut threshold
        self._epsilon_cut = self._compute_cut(mean, var)
        
        # Check for drift
        self._check_drift()
        
        return {
            'warning': self.warning_detected,
            'drift': self.drift_detected,
            'error_rate': mean,
        }
    
    def _compute_cut(self, mean: float, variance: float) -> float:
        """Compute epsilon cut value."""
        if self.width < 10:
            return 0.0
        
        delta_prime = self.delta / np.log(self.width)
        
        # ADWIN formula
        epsilon = np.sqrt(
            (1.0 / (2.0 * self.width)) * 
            np.log(4.0 / delta_prime)
        )
        
        return epsilon
    
    def _check_drift(self):
        """Check for drift in window."""
        
        if self.width < 20:
            self.warning_detected = False
            self.drift_detected = False
            return
        
        # Compare recent window to older window
        half = self.width // 2
        recent = self.window[-half:]
        older = self.window[:half]
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        drift = abs(recent_mean - older_mean)
        
        self.warning_history.append(drift)
        self.drift_history.append(drift)
        
        # Thresholds
        self.warning_detected = drift > self.warning_threshold
        self.drift_detected = drift > self.drift_threshold
    
    def reset(self):
        """Reset detector."""
        super().reset()
        self.window = []
        self.width = 0


class DDMDriftDetector(DriftDetector):
    """
    Drift Detection Method (DDM).
    
    Based on: Gama et al. (2004)
    
    Monitors error rate with statistical significance.
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.05,
        drift_threshold: float = 0.1,
        min_instances: int = 30,
    ):
        super().__init__(warning_threshold, drift_threshold)
        
        self.min_instances = min_instances
        
        # Statistics
        self.errors = 0
        self.total = 0
        self.error_rate = 0.0
        
        # Warning tracking
        self._in_warning = False
        self._warning_start = 0
    
    def update(self, prediction: float, actual: float) -> Dict[str, bool]:
        """Update detector with new observation."""
        error = int(prediction != actual)
        
        self.errors += error
        self.total += 1
        
        if self.total < self.min_instances:
            return {'warning': False, 'drift': False}
        
        # Update error rate
        self.error_rate = self.errors / self.total
        
        # Standard deviation
        std = np.sqrt(self.error_rate * (1 - self.error_rate) / self.total)
        
        # Check thresholds
        warning_level = self.error_rate + std
        drift_level = self.error_rate + 2 * std
        
        self.warning_detected = warning_level > self.warning_threshold
        self.drift_detected = drift_level > self.drift_threshold
        
        self.warning_history.append(float(self.warning_detected))
        self.drift_history.append(float(self.drift_detected))
        
        return {
            'warning': self.warning_detected,
            'drift': self.drift_detected,
            'error_rate': self.error_rate,
        }
    
    def reset(self):
        """Reset detector."""
        super().reset()
        self.errors = 0
        self.total = 0
        self.error_rate = 0.0


class HDDMDriftDetector(DriftDetector):
    """
    Hierarchical DDM for detecting drift in high-dimensional data.
    
    Based on: Dulasu et al. (2019)
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.05,
        drift_threshold: float = 0.1,
    ):
        super().__init__(warning_threshold, drift_threshold)
        
        # Page-Hinkley test
        self.phi = 0.0
        self.alpha = 0.005  # Threshold for change
        self.delta = 0.01  # False positive rate
        
        # Feature-wise detectors
        self.feature_detectors = {}
    
    def update(self, prediction: float, actual: float, features: Optional[np.ndarray] = None) -> Dict[str, bool]:
        """Update detector with optional features."""
        
        error = float(prediction != actual)
        
        # Update Page-Hinkley test
        self.phi = self.phi + error - self.alpha
        
        # Drift detection
        self.drift_detected = self.phi > self.delta
        
        self.drift_history.append(float(self.drift_detected))
        
        # Feature-wise drift
        if features is not None:
            self._update_feature_drift(features)
        
        return {
            'warning': self.warning_detected,
            'drift': self.drift_detected,
        }
    
    def _update_feature_drift(self, features: np.ndarray):
        """Update drift detection for each feature."""
        
        for i, feat in enumerate(features):
            if i not in self.feature_detectors:
                self.feature_detectors[i] = {
                    'correct': 0,
                    'total': 0,
                    'phi': 0.0,
                }
            
            d = self.feature_detectors[i]
            is_correct = int(feat > 0.5)
            d['correct'] += is_correct
            d['total'] += 1
            
            if d['total'] > 10:
                d['phi'] += (1 - is_correct) - self.alpha
    
    def get_feature_importance_drift(self) -> List[Tuple[int, float]]:
        """Get features with drift detected."""
        
        drifts = []
        for i, d in self.feature_detectors.items():
            if d['phi'] > self.delta:
                drifts.append((i, d['phi']))
        
        return sorted(drifts, key=lambda x: x[1], reverse=True)


# =========================================================================
# ONLINE MODEL WRAPPER
# =========================================================================

class IncrementalModel:
    """
    Incremental learning model wrapper.
    
    Provides consistent API for online learning with:
    1. Partial fit (update with new data)
    2. Predict (streaming prediction)
    3. Drift handling
    """
    
    def __init__(
        self,
        model_type: Literal['river', 'sklearn', 'lightgbm'] = 'river',
        model_config: Optional[Dict] = None,
    ):
        self.model_type = model_type
        self.model_config = model_config or {}
        self.model = None
        self.is_fitted = False
        
        self._build_model()
    
    def _build_model(self):
        """Build the model."""
        
        if self.model_type == 'river':
            try:
                # Use Hoeffding Tree for online learning
                self.model = tree.HoeffdingTreeClassifier(
                    **self.model_config
                )
                logger.info("  Using River HoeffdingTreeClassifier")
            except Exception as e:
                logger.warning(f"  River not available: {e}")
                self.model_type = 'sklearn'
        
        if self.model_type == 'sklearn':
            from sklearn.linear_model import SGDClassifier
            
            self.model = SGDClassifier(
                loss='log_loss',
                random_state=42,
                warm_start=True,
                max_iter=1,
                **self.model_config
            )
            logger.info("  Using SGDClassifier")
        
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            
            self.model = lgb.LGBMClassifier(
                n_estimators=1,
                learning_rate=0.1,
                **self.model_config
            )
            logger.info("  Using LightGBM (incremental)")
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Incrementally update model."""
        
        if not self.is_fitted:
            # Initial fit
            self.model.fit(X, y)
            self.is_fitted = True
        else:
            # Incremental update
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(X, y)
            elif hasattr(self.model, 'fit_partial'):
                self.model.fit_partial(X, y)
            elif hasattr(self.model, 'increment_learn_state'):
                self.model = self.model.increment_learn_state(X, y)
            else:
                # For sklearn models, refit
                self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        
        if not self.is_fitted:
            return np.zeros(len(X))
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        
        if not self.is_fitted:
            return np.ones(len(X)) * 0.5
        
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(X)
            return prob[:, 1] if prob.shape[1] > 1 else prob.flatten()
        
        # Fallback
        preds = self.predict(X)
        return preds.astype(float)


class ModelUpdateManager:
    """
    Manages model updates based on drift detection.
    
    Strategies:
    - immediate: Retrain as soon as drift detected
    - batch: Accumulate until minimum samples
    - window: Use sliding window of recent data
    """
    
    def __init__(
        self,
        model: IncrementalModel,
        drift_detector: DriftDetector,
        update_strategy: Literal['immediate', 'batch', 'window'] = 'batch',
        min_samples: int = 100,
        window_size: int = 1000,
    ):
        self.model = model
        self.drift_detector = drift_detector
        self.update_strategy = update_strategy
        self.min_samples = min_samples
        self.window_size = window_size
        
        # Buffer for batch updates
        self.buffer_X = []
        self.buffer_y = []
        
        # Sliding window
        self.window_X = []
        self.window_y = []
        
        self.total_updates = 0
    
    def update(self, X: np.ndarray, y: np.ndarray, 
              pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Update with new batch of data.
        
        Returns dict with:
        - should_retrain: bool
        - drift_detected: bool
        - warning_detected: bool
        """
        
        # Update drift detector
        if pred is not None:
            drift_status = self.drift_detector.update(pred, y)
        else:
            pred = self.model.predict(X)
            drift_status = self.drift_detector.update(pred, y)
        
        should_retrain = False
        
        # Buffer update (for batch strategy)
        self.buffer_X.extend(X)
        self.buffer_y.extend(y)
        
        # Sliding window update
        self.window_X.extend(X)
        self.window_y.extend(y)
        
        # Keep window size
        if len(self.window_X) > self.window_size:
            n_remove = len(self.window_X) - self.window_size
            self.window_X = self.window_X[n_remove:]
            self.window_y = self.window_y[n_remove:]
        
        # Check retrain condition
        if self.update_strategy == 'immediate':
            should_retrain = drift_status['drift']
        
        elif self.update_strategy == 'batch':
            should_retrain = (
                len(self.buffer_X) >= self.min_samples and
                (drift_status['drift'] or drift_status['warning'])
            )
        
        elif self.update_strategy == 'window':
            should_retrain = drift_status['drift']
        
        # Perform retrain if needed
        if should_retrain:
            self._retrain()
        
        return {
            'should_retrain': should_retrain,
            'drift_detected': drift_status['drift'],
            'warning_detected': drift_status['warning'],
            'buffer_size': len(self.buffer_X),
            'window_size': len(self.window_X),
        }
    
    def _retrain(self):
        """Retrain model."""
        
        logger.info("Retraining model due to drift...")
        
        # Prepare data
        if self.update_strategy == 'window':
            X = np.array(self.window_X[-self.window_size:])
            y = np.array(self.window_y[-self.window_size:])
        else:
            X = np.array(self.buffer_X)
            y = np.array(self.buffer_y)
        
        # Retrain
        self.model.partial_fit(X, y)
        
        # Update statistics
        self.total_updates += 1
        
        # Clear buffer
        self.buffer_X = []
        self.buffer_y = []
        
        # Reset drift detector
        self.drift_detector.reset()
        
        logger.info(f"  Model retrained (total updates: {self.total_updates})")


# =========================================================================
# PERFORMANCE MONITORING
# =========================================================================

class PerformanceMonitor:
    """
    Monitor model performance over time.
    
    Tracks:
    1. Accuracy, precision, recall, F1
    2. Prediction latency
    3. Feature drift
    """
    
    def __init__(
        self,
        window_size: int = 1000,
    ):
        self.window_size = window_size
        
        # Prediction history
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
        # Metrics history
        self.metrics_history = []
    
    def record(
        self,
        prediction: Any,
        actual: Any,
        timestamp: Optional[pd.Timestamp] = None,
    ):
        """Record prediction and actual."""
        
        self.predictions.append(prediction)
        self.actuals.append(actual)
        
        if timestamp is not None:
            self.timestamps.append(timestamp)
        
        # Maintain window
        if len(self.predictions) > self.window_size:
            self.predictions = self.predictions[-self.window_size:]
            self.actuals = self.actuals[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        
        if len(self.predictions) < 10:
            return {}
        
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, 
            recall_score, f1_score, 
            confusion_matrix
        )
        
        # Handle different prediction types
        if preds.dtype in [np.float32, np.float64]:
            # Regression
            return {
                'mae': np.mean(np.abs(preds - acts)),
                'rmse': np.sqrt(np.mean((preds - acts) ** 2)),
            }
        else:
            # Classification
            return {
                'accuracy': accuracy_score(acts, preds),
                'precision': precision_score(acts, preds, zero_division=0),
                'recall': recall_score(acts, preds, zero_division=0),
                'f1': f1_score(acts,_preds, zero_division=0),
            }
    
    def check_performance_degradation(
        self,
        baseline_metrics: Dict[str, float],
        threshold: float = 0.1,
    ) -> Tuple[bool, List[str]]:
        """Check if performance has degraded."""
        
        current = self.get_metrics()
        
        degraded = []
        
        for metric, baseline in baseline_metrics.items():
            if metric in current:
                current_val = current[metric]
                
                if metric in ['mae', 'rmse']:
                    # Higher is worse
                    if current_val > baseline * (1 + threshold):
                        degraded.append(metric)
                else:
                    # Lower is worse
                    if current_val < baseline * (1 - threshold):
                        degraded.append(metric)
        
        return len(degraded) > 0, degraded


# =========================================================================
# FACTORY FUNCTIONS
# =========================================================================

def create_drift_detector(
    detector_type: Literal['adwin', 'ddm', 'hddm'] = 'adwin',
    warning_threshold: float = 0.05,
    drift_threshold: float = 0.1,
    **kwargs
) -> DriftDetector:
    """Factory function to create drift detector."""
    
    if detector_type == 'adwin':
        return ADWINDriftDetector(
            warning_threshold=warning_threshold,
            drift_threshold=drift_threshold,
            **kwargs
        )
    elif detector_type == 'ddm':
        return DDMDriftDetector(
            warning_threshold=warning_threshold,
            drift_threshold=drift_threshold,
            **kwargs
        )
    elif detector_type == 'hddm':
        return HDDMDriftDetector(
            warning_threshold=warning_threshold,
            drift_threshold=drift_threshold,
        )
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


__all__ = [
    'DriftDetector',
    'ADWINDriftDetector',
    'DDMDriftDetector',
    'HDDMDriftDetector',
    'IncrementalModel',
    'ModelUpdateManager',
    'PerformanceMonitor',
    'create_drift_detector',
    'RIVER_AVAILABLE',
]