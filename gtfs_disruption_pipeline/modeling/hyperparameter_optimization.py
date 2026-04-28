"""
GTFS Disruption Detection - Hyperparameter Optimization Module
=============================================================
Optuna-based hyperparameter optimization for transit disruption models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import warnings

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Supports:
    - Bayesian optimization with TPE sampler
    - Multi-objective optimization
    - Pruning of unpromising trials
    - Parallel execution
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize hyperparameter optimizer.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration for optimization
        """
        self.config = config or {}
        self.study = None
        self.best_params = None
        self.best_value = None
        
        # Try to import Optuna
        try:
            import optuna
            self.optuna_available = True
        except ImportError:
            self.optuna_available = False
            warnings.warn("Optuna not installed. Install with: pip install optuna")
    
    def create_objective(
        self,
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_space: Dict[str, Any],
        metric_fn: Callable,
        metric_name: str = 'f1'
    ) -> Callable:
        """
        Create objective function for Optuna.
        
        Parameters
        ----------
        model_class : class
            Model class to optimize
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        param_space : Dict[str, Any]
            Parameter search space
        metric_fn : Callable
            Metric function to optimize
        metric_name : str
            Name of metric
        
        Returns
        -------
        Callable objective function
        """
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Create and train model
            try:
                model = model_class(**params, random_state=42)
                model.fit(X_train, y_train)
                
                # Predict
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_val)[:, 1]
                    y_pred = (y_proba >= 0.5).astype(int)
                else:
                    y_pred = model.predict(X_val)
                    y_proba = None
                
                # Compute metric
                if metric_name == 'f1':
                    from sklearn.metrics import f1_score
                    score = f1_score(y_val, y_pred, zero_division=0)
                elif metric_name == 'roc_auc':
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y_val, y_proba) if y_proba is not None else 0
                elif metric_name == 'pr_auc':
                    from sklearn.metrics import average_precision_score
                    score = average_precision_score(y_val, y_proba) if y_proba is not None else 0
                else:
                    score = metric_fn(y_val, y_pred, y_proba)
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        return objective
    
    def optimize(
        self,
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_space: Dict[str, Any],
        metric_fn: Optional[Callable] = None,
        metric_name: str = 'f1',
        n_trials: int = 100,
        timeout: Optional[int] = None,
        direction: str = 'maximize',
        study_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Parameters
        ----------
        model_class : class
            Model class to optimize
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        param_space : Dict[str, Any]
            Parameter search space
        metric_fn : Callable, optional
            Metric function to optimize
        metric_name : str
            Name of metric
        n_trials : int
            Number of trials
        timeout : int, optional
            Timeout in seconds
        direction : str
            'maximize' or 'minimize'
        study_name : str, optional
            Name of study
        
        Returns
        -------
        Dict with optimization results
        """
        if not self.optuna_available:
            logger.warning("Optuna not available. Returning default parameters.")
            return {
                'best_params': {},
                'best_value': 0.0,
                'n_trials': 0
            }
        
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        
        logger.info("="*60)
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info("="*60)
        
        # Create objective
        objective = self.create_objective(
            model_class, X_train, y_train, X_val, y_val,
            param_space, metric_fn, metric_name
        )
        
        # Create study
        if study_name is None:
            study_name = f"{model_class.__name__}_optimization"
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"\nOptimization complete:")
        logger.info(f"  Best {metric_name}: {self.best_value:.4f}")
        logger.info(f"  Best parameters: {self.best_params}")
        logger.info(f"  Number of trials: {len(self.study.trials)}")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def get_param_importance(self) -> pd.DataFrame:
        """
        Get parameter importance from optimization study.
        
        Returns
        -------
        pd.DataFrame with parameter importance
        """
        if self.study is None:
            logger.warning("No optimization study available")
            return pd.DataFrame()
        
        import optuna
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            
            importance_df = pd.DataFrame({
                'parameter': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
            return pd.DataFrame()
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            logger.warning("No optimization study available")
            return
        
        import optuna
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()
    
    def plot_param_importances(self):
        """Plot parameter importances."""
        if self.study is None:
            logger.warning("No optimization study available")
            return
        
        import optuna
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.show()
    
    def plot_parallel_coordinate(self):
        """Plot parallel coordinate plot."""
        if self.study is None:
            logger.warning("No optimization study available")
            return
        
        import optuna
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        fig.show()


def get_default_param_space(model_name: str) -> Dict[str, Any]:
    """
    Get default parameter space for common models.
    
    Parameters
    ----------
    model_name : str
        Name of model
    
    Returns
    -------
    Dict with parameter space
    """
    param_spaces = {
        'RandomForest': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 50},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
        },
        'XGBoost': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 50},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
        },
        'LightGBM': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 50},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'num_leaves': {'type': 'int', 'low': 10, 'high': 100},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 50},
        },
        'MLP': {
            'hidden_layer_sizes': {'type': 'categorical', 'choices': [
                (64, 32), (128, 64), (256, 128), (64, 32, 16), (128, 64, 32)
            ]},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh']},
            'alpha': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
            'learning_rate_init': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
        },
    }
    
    return param_spaces.get(model_name, {})


def optimize_model(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_space: Optional[Dict[str, Any]] = None,
    metric_name: str = 'f1',
    n_trials: int = 100,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter optimization.
    
    Parameters
    ----------
    model_class : class
        Model class to optimize
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    param_space : Dict[str, Any], optional
        Parameter search space
    metric_name : str
        Name of metric to optimize
    n_trials : int
        Number of trials
    timeout : int, optional
        Timeout in seconds
    
    Returns
    -------
    Dict with optimization results
    """
    optimizer = HyperparameterOptimizer()
    
    if param_space is None:
        param_space = get_default_param_space(model_class.__name__)
    
    return optimizer.optimize(
        model_class, X_train, y_train, X_val, y_val,
        param_space, metric_name=metric_name,
        n_trials=n_trials, timeout=timeout
    )
