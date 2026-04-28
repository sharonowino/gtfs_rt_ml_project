"""
Multi-Task Learning for Transit Disruption
==================================

Joint model for multiple disruption prediction tasks:
1. Binary detection (is disruption?)
2. Severity classification (minor/major/cancelled/...)
3. Duration prediction (minutes)
4. Recovery time estimation

Multi-task learning provides:
- Shared representations across tasks
- Better generalization
- Improved data efficiency

Usage:
------
from gtfs_disruption.modeling.multi_task import (
    DisruptionMultiTaskModel,
    MultiTaskLoss,
    task_weights,
)

model = DisruptionMultiTaskModel(
    input_dim=64,
    shared_dim=128,
    task_heads={
        'detection': ('binary', 1),
        'severity': ('classification', 6),
        'duration': ('regression', 1),
        'recovery': ('regression', 1),
    }
)
"""
import logging
from typing import Dict, List, Optional, Tuple, Literal, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available - multi-task models disabled")
    torch = None
    nn = None


if not TORCH_AVAILABLE:
    class DisruptionMultiTaskModel:
        pass
    class MultiTaskLoss:
        pass
    
    __all__ = ['DisruptionMultiTaskModel', 'MultiTaskLoss', 'task_weights', 'TORCH_AVAILABLE']
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    
    # =========================================================================
    # TASK DEFINITIONS
    # =========================================================================
    
    @dataclass
    class TaskConfig:
        """Configuration for a single task."""
        name: str
        task_type: Literal['binary', 'classification', 'regression']
        output_dim: int
        weight: float = 1.0
        loss_fn: Optional[str] = None
    
    
    # Default task configurations
    DEFAULT_TASKS = {
        'detection': TaskConfig(
            'detection', 'binary', output_dim=1, weight=1.0
        ),
        'severity': TaskConfig(
            'severity', 'classification', output_dim=6, weight=0.8,
            loss_fn='cross_entropy'
        ),
        'duration': TaskConfig(
            'duration', 'regression', output_dim=1, weight=0.5,
            loss_fn='mse'
        ),
        'recovery': TaskConfig(
            'recovery', 'regression', output_dim=1, weight=0.3,
            loss_fn='mse'
        ),
    }
    
    # Task weights (可以根据类别不平衡调整)
    task_weights: Dict[str, float] = {
        'detection': 1.0,
        'severity': 0.8,
        'duration': 0.5,
        'recovery': 0.3,
    }
    
    
    # =========================================================================
    # MULTI-TASK MODEL
    # =========================================================================
    
    class DisruptionMultiTaskModel(nn.Module):
        """
        Multi-task model for transit disruption prediction.
        
        Architecture:
        1. Shared encoder (MLP/GNN)
        2. Task-specific feature layers
        3. Separate prediction heads for each task
        
        Tasks:
        - detection: Binary disruption status
        - severity: Multi-class (6 disruption types)
        - duration: Regression (minutes)
        - recovery: Regression (minutes)
        """
        
        def __init__(
            self,
            input_dim: int,
            shared_dim: int = 128,
            hidden_dim: int = 64,
            num_layers: int = 3,
            dropout: float = 0.1,
            task_configs: Optional[Dict[str, TaskConfig]] = None,
        ):
            super().__init__()
            
            self.tasks = task_configs or DEFAULT_TASKS
            
            # Shared encoder
            encoder_layers = []
            in_ch = input_dim
            for i in range(num_layers):
                out_ch = shared_dim if i < num_layers - 1 else hidden_dim
                encoder_layers.extend([
                    nn.Linear(in_ch, out_ch),
                    nn.LayerNorm(out_ch),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_ch = out_ch
            
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Task-specific layers
            self.task_layers = nn.ModuleDict()
            self.task_heads = nn.ModuleDict()
            
            for task_name, config in self.tasks.items():
                # Task-specific feature layer
                self.task_layers[task_name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                
                # Task head
                if config.task_type == 'binary':
                    self.task_heads[task_name] = nn.Linear(hidden_dim, 1)
                elif config.task_type == 'classification':
                    self.task_heads[task_name] = nn.Linear(hidden_dim, config.output_dim)
                else:  # regression
                    self.task_heads[task_name] = nn.Linear(hidden_dim, config.output_dim)
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: [batch, input_dim]
            
            Returns:
                dict of task predictions
            """
            # Shared encoding
            shared = self.encoder(x)
            
            outputs = {}
            
            for task_name, config in self.tasks.items():
                # Task-specific features
                task_feat = self.task_layers[task_name](shared)
                
                # Task prediction
                out = self.task_heads[task_name](task_feat)
                
                if config.task_type == 'binary':
                    outputs[task_name] = out.squeeze(-1)
                elif config.task_type == 'classification':
                    outputs[task_name] = out
                else:
                    outputs[task_name] = out.squeeze(-1) if config.output_dim == 1 else out
            
            return outputs
        
        def predict(self, x: torch.Tensor) -> Dict[str, Any]:
            """Get predictions in usable format."""
            out = self.forward(x)
            
            predictions = {}
            
            for task_name, config in self.tasks.items():
                pred = out[task_name]
                
                if config.task_type == 'binary':
                    predictions[task_name] = {
                        'prob': torch.sigmoid(pred),
                        'class': (torch.sigmoid(pred) > 0.5).long(),
                    }
                elif config.task_type == 'classification':
                    predictions[task_name] = {
                        'logits': pred,
                        'class': pred.argmax(dim=-1),
                        'prob': F.softmax(pred, dim=-1),
                    }
                else:  # regression
                    predictions[task_name] = {
                        'value': pred,
                    }
            
            return predictions
    
    
    class MultiTaskLoss(nn.Module):
        """
        Combined loss for multi-task learning.
        
        Weighted sum of task-specific losses.
        """
        
        def __init__(
            self,
            task_configs: Optional[Dict[str, TaskConfig]] = None,
            weights: Optional[Dict[str, float]] = None,
            use_uncertainty_weighting: bool = False,
        ):
            super().__init__()
            
            self.tasks = task_configs or DEFAULT_TASKS
            self.weights = weights or task_weights
            self.use_uncertainty_weighting = use_uncertainty_weighting
            
            # Loss functions for each task type
            self.binary_loss_fn = nn.BCEWithLogitsLoss()
            self.class_loss_fn = nn.CrossEntropyLoss()
            self.mse_loss_fn = nn.MSELoss()
            self.l1_loss_fn = nn.L1Loss()
            
            # Learnable uncertainty weights (if enabled)
            if use_uncertainty_weighting:
                self.log_vars = nn.Parameter(
                    torch.zeros(len(self.tasks))
                )
        
        def forward(
            self, 
            predictions: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """Compute weighted multi-task loss."""
            
            total_loss = 0
            task_losses = {}
            
            for task_name, config in self.tasks.items():
                if task_name not in predictions or task_name not in targets:
                    continue
                
                pred = predictions[task_name]
                target = targets[task_name]
                
                # Task-specific loss
                if config.task_type == 'binary':
                    loss = self.binary_loss_fn(pred, target)
                elif config.task_type == 'classification':
                    loss = self.class_loss_fn(pred, target.long())
                else:  # regression
                    loss = self.mse_loss_fn(pred, target)
                
                # Uncertainty weighting (Kendall et al., 2018)
                if self.use_uncertainty_weighting:
                    task_idx = list(self.tasks.keys()).index(task_name)
                    precision = torch.exp(-self.log_vars[task_idx])
                    weighted_loss = precision * loss + self.log_vars[task_idx]
                    loss = weighted_loss
                else:
                    weighted_loss = loss * self.weights.get(task_name, 1.0)
                
                total_loss += weighted_loss
                task_losses[task_name] = loss.item()
            
            return total_loss, task_losses


    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    
    class MultiTaskDataset(Dataset):
        """
        Dataset with multiple target columns.
        """
        
        def __init__(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            task_targets: Dict[str, str],
        ):
            self.df = df.copy()
            self.feature_cols = feature_cols
            self.task_targets = task_targets
            
            # Prepare tensors
            self.X = torch.from_numpy(
                df[feature_cols].fillna(0).values.astype(np.float32)
            )
            
            self.targets = {}
            for task, col in task_targets.items():
                if col in df.columns:
                    vals = df[col].fillna(0).values
                    
                    if task in ['severity', 'disruption_class']:
                        # Classification: convert to integers
                        self.targets[task] = torch.from_numpy(vals.astype(np.int64))
                    else:
                        self.targets[task] = torch.from_numpy(vals.astype(np.float32))
                else:
                    self.targets[task] = torch.zeros(len(df))
        
        def __len__(self) -> int:
            return len(self.X)
        
        def __getitem__(self, idx: int):
            x = self.X[idx]
            targets = {k: v[idx] for k, v in self.targets.items()}
            return x, targets


    # =========================================================================
    # TRAINING UTILITIES
    # =========================================================================
    
    def train_multi_task_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: MultiTaskLoss,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, float]:
        """Train one epoch."""
        model.train()
        total_loss = 0
        task_losses = {k: 0 for k in model.tasks.keys()}
        n_batches = 0
        
        for batch in loader:
            X, targets = batch
            X = X.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            
            predictions = model(X)
            loss, losses = loss_fn(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            for k, v in losses.items():
                task_losses[k] += v
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            **{k: v / n_batches for k, v in task_losses.items()},
        }


    def evaluate_multi_task(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, Any]:
        """Evaluate multi-task model."""
        model.eval()
        
        results = {
            'detection': {'preds': [], 'targets': []},
            'severity': {'preds': [], 'targets': []},
            'duration': {'preds': [], 'targets': []},
            'recovery': {'preds': [], 'targets': []},
        }
        
        with torch.no_grad():
            for batch in loader:
                X, targets = batch
                X = X.to(device)
                
                predictions = model(X)
                
                for task_name in results.keys():
                    if task_name in predictions:
                        pred = predictions[task_name].cpu().numpy()
                        if task_name == 'detection':
                            pred = (torch.sigmoid(predictionstask_name) > 0.5).numpy()
                        elif task_name == 'severity':
                            pred = pred.argmax(axis=-1)
                        
                        results[task_name]['preds'].extend(pred.flatten())
                    
                    if task_name in targets:
                        tgt = targets[task_name].cpu().numpy()
                        results[task_name]['targets'].extend(tgt.flatten())
        
        # Compute metrics per task
        metrics = {}
        
        for task_name, res in results.items():
            preds = np.array(res['preds'])
            targets = np.array(res['targets'])
            
            if len(preds) == 0:
                continue
            
            if task_name == 'detection':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                metrics[f'{task_name}_accuracy'] = accuracy_score(targets, preds)
                metrics[f'{task_name}_precision'] = precision_score(targets, preds, zero_division=0)
                metrics[f'{task_name}_recall'] = recall_score(targets, preds, zero_division=0)
                metrics[f'{task_name}_f1'] = f1_score(targets, preds, zero_division=0)
            
            elif task_name == 'severity':
                from sklearn.metrics import accuracy_score, f1_score
                metrics[f'{task_name}_accuracy'] = accuracy_score(targets, preds)
                metrics[f'{task_name}_f1'] = f1_score(targets, preds, average='macro', zero_division=0)
            
            else:  # regression
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                metrics[f'{task_name}_mae'] = mean_absolute_error(targets, preds)
                metrics[f'{task_name}_rmse'] = np.sqrt(mean_squared_error(targets, preds))
                metrics[f'{task_name}_r2'] = r2_score(targets, preds)
        
        return metrics


    __all__ = [
        'DisruptionMultiTaskModel',
        'MultiTaskLoss',
        'TaskConfig',
        'DEFAULT_TASKS',
        'task_weights',
        'MultiTaskDataset',
        'train_multi_task_epoch',
        'evaluate_multi_task',
        'TORCH_AVAILABLE',
    ]