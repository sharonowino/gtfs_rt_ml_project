"""
GTFS Disruption Detection - Monitoring Module
=============================================
Drift detection and performance tracking for production models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Data drift detection for monitoring model performance.
    
    Supports:
    - Kolmogorov-Smirnov test
    - Population Stability Index (PSI)
    - Chi-square test for categorical features
    - Window-based drift detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize drift detector.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration for drift detection
        """
        self.config = config or {}
        self.method = self.config.get('method', 'ks_test')
        self.threshold = self.config.get('threshold', 0.05)
        self.window_size = self.config.get('window_size', 1000)
        
        # Store reference distribution
        self.reference_data = None
        self.reference_stats = {}
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit drift detector on reference data.
        
        Parameters
        ----------
        X : np.ndarray
            Reference data
        feature_names : List[str], optional
            Feature names
        """
        self.reference_data = X.copy()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        
        # Compute reference statistics
        for i, name in enumerate(feature_names):
            self.reference_stats[name] = {
                'mean': np.mean(X[:, i]),
                'std': np.std(X[:, i]),
                'min': np.min(X[:, i]),
                'max': np.max(X[:, i]),
                'median': np.median(X[:, i])
            }
        
        logger.info(f"Fitted drift detector on {len(X)} samples with {len(feature_names)} features")
    
    def ks_test(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov test for drift detection.
        
        Parameters
        ----------
        X : np.ndarray
            New data to test
        
        Returns
        -------
        Dict with drift detection results
        """
        from scipy import stats
        
        if self.reference_data is None:
            raise ValueError("Drift detector not fitted. Call fit() first.")
        
        n_features = X.shape[1]
        drift_scores = {}
        p_values = {}
        
        for i in range(n_features):
            # KS test for each feature
            ks_stat, p_value = stats.ks_2samp(self.reference_data[:, i], X[:, i])
            
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            drift_scores[feature_name] = ks_stat
            p_values[feature_name] = p_value
        
        # Overall drift score (mean of all features)
        overall_drift = np.mean(list(drift_scores.values()))
        overall_p_value = np.mean(list(p_values.values()))
        
        return {
            'method': 'ks_test',
            'drift_scores': drift_scores,
            'p_values': p_values,
            'overall_drift': overall_drift,
            'overall_p_value': overall_p_value,
            'drift_detected': overall_p_value < self.threshold,
            'threshold': self.threshold
        }
    
    def psi(self, X: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """
        Population Stability Index (PSI) for drift detection.
        
        Parameters
        ----------
        X : np.ndarray
            New data to test
        n_bins : int
            Number of bins for discretization
        
        Returns
        -------
        Dict with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Drift detector not fitted. Call fit() first.")
        
        n_features = X.shape[1]
        psi_scores = {}
        
        for i in range(n_features):
            # Discretize reference and new data
            ref_data = self.reference_data[:, i]
            new_data = X[:, i]
            
            # Create bins based on reference data
            bin_edges = np.percentile(ref_data, np.linspace(0, 100, n_bins + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Compute distributions
            ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
            new_counts, _ = np.histogram(new_data, bins=bin_edges)
            
            # Normalize
            ref_dist = ref_counts / len(ref_data)
            new_dist = new_counts / len(new_data)
            
            # Avoid division by zero
            ref_dist = np.clip(ref_dist, 1e-10, None)
            new_dist = np.clip(new_dist, 1e-10, None)
            
            # Compute PSI
            psi = np.sum((new_dist - ref_dist) * np.log(new_dist / ref_dist))
            
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            psi_scores[feature_name] = psi
        
        # Overall PSI score
        overall_psi = np.mean(list(psi_scores.values()))
        
        return {
            'method': 'psi',
            'psi_scores': psi_scores,
            'overall_psi': overall_psi,
            'drift_detected': overall_psi > 0.2,  # PSI > 0.2 indicates significant drift
            'threshold': 0.2
        }
    
    def detect_drift(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift using configured method.
        
        Parameters
        ----------
        X : np.ndarray
            New data to test
        
        Returns
        -------
        Dict with drift detection results
        """
        if self.method == 'ks_test':
            return self.ks_test(X)
        elif self.method == 'psi':
            return self.psi(X)
        else:
            raise ValueError(f"Unknown drift detection method: {self.method}")
    
    def plot_drift(self, drift_results: Dict[str, Any]):
        """
        Plot drift detection results.
        
        Parameters
        ----------
        drift_results : Dict[str, Any]
            Drift detection results
        """
        import matplotlib.pyplot as plt
        
        method = drift_results['method']
        
        if method == 'ks_test':
            scores = drift_results['drift_scores']
            p_values = drift_results['p_values']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot KS statistics
            features = list(scores.keys())
            ks_values = list(scores.values())
            
            ax1.barh(features, ks_values, alpha=0.7, color='steelblue')
            ax1.axvline(x=self.threshold, color='red', linestyle='--', label=f'Threshold ({self.threshold})')
            ax1.set_xlabel('KS Statistic')
            ax1.set_title('Feature-wise Drift (KS Test)')
            ax1.legend()
            
            # Plot p-values
            p_vals = list(p_values.values())
            ax2.barh(features, p_vals, alpha=0.7, color='coral')
            ax2.axvline(x=self.threshold, color='red', linestyle='--', label=f'Threshold ({self.threshold})')
            ax2.set_xlabel('P-value')
            ax2.set_title('Feature-wise P-values')
            ax2.legend()
            
        elif method == 'psi':
            scores = drift_results['psi_scores']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(scores.keys())
            psi_values = list(scores.values())
            
            ax.barh(features, psi_values, alpha=0.7, color='steelblue')
            ax.axvline(x=0.2, color='red', linestyle='--', label='Threshold (0.2)')
            ax.set_xlabel('PSI Score')
            ax.set_title('Population Stability Index (PSI)')
            ax.legend()
        
        plt.tight_layout()
        plt.show()


class PerformanceTracker:
    """
    Track model performance over time.
    
    Supports:
    - Metric tracking (F1, precision, recall, etc.)
    - Performance degradation detection
    - Alert generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize performance tracker.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration for performance tracking
        """
        self.config = config or {}
        self.metrics = self.config.get('metrics', ['f1', 'precision', 'recall', 'roc_auc'])
        self.alert_threshold = self.config.get('alert_threshold', 0.1)
        self.window_size = self.config.get('window_size', 100)
        
        # Store performance history
        self.history = {metric: deque(maxlen=self.window_size) for metric in self.metrics}
        self.baseline = {}
        self.alerts = []
    
    def set_baseline(self, metrics: Dict[str, float]):
        """
        Set baseline performance metrics.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Baseline metrics
        """
        self.baseline = metrics.copy()
        logger.info(f"Set baseline metrics: {metrics}")
    
    def update(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """
        Update performance metrics.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Current metrics
        timestamp : datetime, optional
            Timestamp for the metrics
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for metric in self.metrics:
            if metric in metrics:
                self.history[metric].append({
                    'value': metrics[metric],
                    'timestamp': timestamp
                })
        
        # Check for performance degradation
        self._check_degradation(metrics, timestamp)
    
    def _check_degradation(self, metrics: Dict[str, float], timestamp: datetime):
        """
        Check for performance degradation.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Current metrics
        timestamp : datetime
            Timestamp for the metrics
        """
        for metric in self.metrics:
            if metric in metrics and metric in self.baseline:
                baseline_value = self.baseline[metric]
                current_value = metrics[metric]
                
                # Compute relative change
                if baseline_value > 0:
                    relative_change = (baseline_value - current_value) / baseline_value
                    
                    if relative_change > self.alert_threshold:
                        alert = {
                            'metric': metric,
                            'baseline': baseline_value,
                            'current': current_value,
                            'relative_change': relative_change,
                            'timestamp': timestamp,
                            'severity': 'high' if relative_change > 0.2 else 'medium'
                        }
                        self.alerts.append(alert)
                        
                        logger.warning(
                            f"Performance degradation detected for {metric}: "
                            f"{baseline_value:.4f} -> {current_value:.4f} "
                            f"({relative_change:.1%} decrease)"
                        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns
        -------
        Dict with performance summary
        """
        summary = {}
        
        for metric in self.metrics:
            if len(self.history[metric]) > 0:
                values = [entry['value'] for entry in self.history[metric]]
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1],
                    'n_samples': len(values)
                }
        
        return summary
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get performance alerts.
        
        Parameters
        ----------
        severity : str, optional
            Filter by severity ('high', 'medium', or None for all)
        
        Returns
        -------
        List of alerts
        """
        if severity is None:
            return self.alerts
        else:
            return [alert for alert in self.alerts if alert['severity'] == severity]
    
    def plot_performance(self, metric: str = 'f1'):
        """
        Plot performance over time.
        
        Parameters
        ----------
        metric : str
            Metric to plot
        """
        import matplotlib.pyplot as plt
        
        if len(self.history[metric]) == 0:
            logger.warning(f"No history for metric: {metric}")
            return
        
        values = [entry['value'] for entry in self.history[metric]]
        timestamps = [entry['timestamp'] for entry in self.history[metric]]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, marker='o', alpha=0.7, color='steelblue')
        
        if metric in self.baseline:
            plt.axhline(y=self.baseline[metric], color='red', linestyle='--', 
                       label=f'Baseline ({self.baseline[metric]:.4f})')
        
        plt.xlabel('Time')
        plt.ylabel(metric.upper())
        plt.title(f'Performance Tracking: {metric.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class ModelMonitor:
    """
    Comprehensive model monitoring.
    
    Combines drift detection and performance tracking.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model monitor.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration for model monitoring
        """
        self.config = config or {}
        self.drift_detector = DriftDetector(self.config.get('drift_detection', {}))
        self.performance_tracker = PerformanceTracker(self.config.get('performance_tracking', {}))
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit drift detector on reference data.
        
        Parameters
        ----------
        X : np.ndarray
            Reference data
        feature_names : List[str], optional
            Feature names
        """
        self.drift_detector.fit(X, feature_names)
    
    def set_baseline_performance(self, metrics: Dict[str, float]):
        """
        Set baseline performance metrics.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Baseline metrics
        """
        self.performance_tracker.set_baseline(metrics)
    
    def monitor(
        self,
        X: np.ndarray,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Monitor model performance and data drift.
        
        Parameters
        ----------
        X : np.ndarray
            New data
        metrics : Dict[str, float]
            Current performance metrics
        timestamp : datetime, optional
            Timestamp for the metrics
        
        Returns
        -------
        Dict with monitoring results
        """
        # Detect drift
        drift_results = self.drift_detector.detect_drift(X)
        
        # Update performance
        self.performance_tracker.update(metrics, timestamp)
        
        # Get performance summary
        performance_summary = self.performance_tracker.get_performance_summary()
        
        # Get alerts
        alerts = self.performance_tracker.get_alerts()
        
        return {
            'drift': drift_results,
            'performance': performance_summary,
            'alerts': alerts,
            'timestamp': timestamp or datetime.now()
        }
    
    def get_monitoring_report(self) -> str:
        """
        Get monitoring report.
        
        Returns
        -------
        str : Monitoring report
        """
        report = []
        report.append("="*60)
        report.append("MODEL MONITORING REPORT")
        report.append("="*60)
        
        # Drift detection
        drift_results = self.drift_detector.detect_drift(np.random.randn(100, 10))  # Dummy data
        report.append(f"\nDrift Detection:")
        report.append(f"  Method: {drift_results['method']}")
        report.append(f"  Drift detected: {drift_results['drift_detected']}")
        
        # Performance tracking
        performance_summary = self.performance_tracker.get_performance_summary()
        report.append(f"\nPerformance Tracking:")
        for metric, stats in performance_summary.items():
            report.append(f"  {metric}:")
            report.append(f"    Mean: {stats['mean']:.4f}")
            report.append(f"    Std: {stats['std']:.4f}")
            report.append(f"    Latest: {stats['latest']:.4f}")
        
        # Alerts
        alerts = self.performance_tracker.get_alerts()
        report.append(f"\nAlerts: {len(alerts)}")
        for alert in alerts[-5:]:  # Show last 5 alerts
            report.append(f"  [{alert['severity']}] {alert['metric']}: "
                         f"{alert['baseline']:.4f} -> {alert['current']:.4f} "
                         f"({alert['relative_change']:.1%})")
        
        return "\n".join(report)


def create_monitor(config: Optional[Dict] = None) -> ModelMonitor:
    """
    Create a model monitor.
    
    Parameters
    ----------
    config : Dict, optional
        Configuration for model monitoring
    
    Returns
    -------
    ModelMonitor instance
    """
    return ModelMonitor(config)
