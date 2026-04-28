"""
GTFS Disruption Detection - Experiment Tracking Module
======================================================
MLflow integration for experiment tracking and model management.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Experiment tracking using MLflow.
    
    Supports:
    - Parameter logging
    - Metric logging
    - Artifact logging
    - Model logging
    - Experiment comparison
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize experiment tracker.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration for experiment tracking
        """
        self.config = config or {}
        self.experiment_name = self.config.get('experiment_name', 'gtfs_disruption_detection')
        self.tracking_uri = self.config.get('tracking_uri', 'http://localhost:5000')
        self.enabled = self.config.get('enabled', False)
        
        # Try to import MLflow
        try:
            import mlflow
            self.mlflow_available = True
            if self.enabled:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
        except ImportError:
            self.mlflow_available = False
            if self.enabled:
                warnings.warn("MLflow not installed. Install with: pip install mlflow")
    
    def start_run(self, run_name: Optional[str] = None) -> Any:
        """
        Start a new MLflow run.
        
        Parameters
        ----------
        run_name : str, optional
            Name of the run
        
        Returns
        -------
        MLflow run object or None
        """
        if not self.enabled or not self.mlflow_available:
            return None
        
        import mlflow
        
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def end_run(self):
        """End the current MLflow run."""
        if not self.enabled or not self.mlflow_available:
            return
        
        import mlflow
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to log
        """
        if not self.enabled or not self.mlflow_available:
            return
        
        import mlflow
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metrics to log
        step : int, optional
            Step number for the metrics
        """
        if not self.enabled or not self.mlflow_available:
            return
        
        import mlflow
        
        for key, value in metrics.items():
            if not np.isnan(value):
                mlflow.log_metric(key, value, step=step)
        
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to MLflow.
        
        Parameters
        ----------
        local_path : str
            Path to the artifact file
        artifact_path : str, optional
            Path within the artifact store
        """
        if not self.enabled or not self.mlflow_available:
            return
        
        import mlflow
        
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """
        Log a model to MLflow.
        
        Parameters
        ----------
        model : trained model
            Model to log
        artifact_path : str
            Path within the artifact store
        """
        if not self.enabled or not self.mlflow_available:
            return
        
        import mlflow
        
        # Log model based on type
        if hasattr(model, 'save_model'):
            # LightGBM
            mlflow.lightgbm.log_model(model, artifact_path, **kwargs)
        elif hasattr(model, 'save'):
            # XGBoost
            mlflow.xgboost.log_model(model, artifact_path, **kwargs)
        elif hasattr(model, 'predict'):
            # Scikit-learn
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        else:
            # Generic pickle
            mlflow.pyfunc.log_model(artifact_path, python_model=model, **kwargs)
        
        logger.info(f"Logged model to: {artifact_path}")
    
    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure to MLflow.
        
        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure to log
        artifact_file : str
            Filename for the artifact
        """
        if not self.enabled or not self.mlflow_available:
            return
        
        import mlflow
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            figure.savefig(f.name, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(f.name, artifact_file)
        
        logger.info(f"Logged figure: {artifact_file}")
    
    def log_dataframe(self, df: pd.DataFrame, artifact_file: str):
        """
        Log a DataFrame to MLflow.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to log
        artifact_file : str
            Filename for the artifact
        """
        if not self.enabled or not self.mlflow_available:
            return
        
        import mlflow
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, artifact_file)
        
        logger.info(f"Logged DataFrame: {artifact_file}")
    
    def get_experiment_runs(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get all runs from an experiment.
        
        Parameters
        ----------
        experiment_name : str, optional
            Name of the experiment
        
        Returns
        -------
        pd.DataFrame with run information
        """
        if not self.enabled or not self.mlflow_available:
            return pd.DataFrame()
        
        import mlflow
        
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return pd.DataFrame()
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return runs
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Parameters
        ----------
        run_ids : List[str]
            List of run IDs to compare
        metrics : List[str]
            List of metrics to compare
        
        Returns
        -------
        pd.DataFrame with comparison
        """
        if not self.enabled or not self.mlflow_available:
            return pd.DataFrame()
        
        import mlflow
        
        comparison = []
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            run_data = {
                'run_id': run_id,
                'run_name': run.info.run_name,
                'status': run.info.status
            }
            
            for metric in metrics:
                run_data[metric] = run.data.metrics.get(metric, np.nan)
            
            comparison.append(run_data)
        
        return pd.DataFrame(comparison)


class ExperimentManager:
    """
    High-level experiment management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize experiment manager.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration for experiment management
        """
        self.config = config or {}
        self.tracker = ExperimentTracker(config)
        self.current_run = None
    
    def run_experiment(
        self,
        experiment_name: str,
        model_name: str,
        params: Dict[str, Any],
        train_fn: callable,
        evaluate_fn: callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run a complete experiment with tracking.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment
        model_name : str
            Name of the model
        params : Dict[str, Any]
            Model parameters
        train_fn : callable
            Function to train the model
        evaluate_fn : callable
            Function to evaluate the model
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        
        Returns
        -------
        Dict with experiment results
        """
        logger.info(f"Running experiment: {experiment_name} with model: {model_name}")
        
        # Start run
        run_name = f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_run = self.tracker.start_run(run_name)
        
        try:
            # Log parameters
            self.tracker.log_params(params)
            self.tracker.log_params({
                'model_name': model_name,
                'experiment_name': experiment_name,
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test)
            })
            
            # Train model
            model = train_fn(X_train, y_train, params)
            
            # Evaluate on validation set
            val_metrics = evaluate_fn(model, X_val, y_val)
            self.tracker.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
            
            # Evaluate on test set
            test_metrics = evaluate_fn(model, X_test, y_test)
            self.tracker.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
            
            # Log model
            self.tracker.log_model(model, f"models/{model_name}")
            
            results = {
                'run_id': self.current_run.info.run_id if self.current_run else None,
                'run_name': run_name,
                'model_name': model_name,
                'params': params,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'model': model
            }
            
            return results
            
        finally:
            # End run
            self.tracker.end_run()
            self.current_run = None
    
    def get_best_run(
        self,
        experiment_name: str,
        metric: str = 'test_f1',
        ascending: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run from an experiment.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment
        metric : str
            Metric to optimize
        ascending : bool
            Whether to sort in ascending order
        
        Returns
        -------
        Dict with best run information
        """
        runs = self.tracker.get_experiment_runs(experiment_name)
        
        if runs.empty:
            return None
        
        # Sort by metric
        if metric in runs.columns:
            runs = runs.sort_values(f'metrics.{metric}', ascending=ascending)
            best_run = runs.iloc[0]
            
            return {
                'run_id': best_run['run_id'],
                'run_name': best_run['tags.mlflow.runName'],
                'metric_value': best_run[f'metrics.{metric}'],
                'params': {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.')}
            }
        
        return None


def create_experiment_tracker(config: Optional[Dict] = None) -> ExperimentTracker:
    """
    Create an experiment tracker.
    
    Parameters
    ----------
    config : Dict, optional
        Configuration for experiment tracking
    
    Returns
    -------
    ExperimentTracker instance
    """
    return ExperimentTracker(config)


def create_experiment_manager(config: Optional[Dict] = None) -> ExperimentManager:
    """
    Create an experiment manager.
    
    Parameters
    ----------
    config : Dict, optional
        Configuration for experiment management
    
    Returns
    -------
    ExperimentManager instance
    """
    return ExperimentManager(config)
