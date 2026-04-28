"""
================================================================================
CONSOLIDATED MODEL PATCHES
================================================================================

This module applies all necessary patches for:
- Custom model wrappers (SpatialRF, LSTM, GAT, STGCN, DCRNN, GraphWaveNet, STGNN)
- Fixed GNN model classes (STGCNModelFixed, DCRNNModelFixed, GraphWaveNetModelFixed)
- GNN fit/predict_proba/get_params/set_params patches
- SpatialRF patches
- Safe _fit_binary wrapper for GNN graceful failure

Run this once before Step 10:
    from gtfs_disruption.modeling.patches import apply_patches
    apply_patches()

================================================================================
"""

import logging
import numpy as np
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Global sets/dicts for tracking
CUSTOM_WRAPPERS = {"SpatialRF", "LSTM", "GAT", "STGCN", "DCRNN", "GraphWaveNet", "STGNN"}
GNN_MODELS = {"STGCN", "DCRNN", "GraphWaveNet", "STGNN", "GAT"}
_SKIP_MODELS = set()

# Try importing torch
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None


def apply_patches(
    BINARY_TARGETS: Optional[List[str]] = None,
    PRIMARY_TARGET: str = "is_disruption",
    MODEL_REGISTRY: Optional[Dict] = None,
    make_spatial_rf: Optional[callable] = None,
    SpatialRandomForest: Optional = None,
    _fit_binary: Optional[callable] = None,
):
    """
    Apply all consolidated patches.
    
    Parameters
    ----------
    BINARY_TARGETS : List[str], optional
        List of binary target column names
    PRIMARY_TARGET : str
        Primary target column name
    MODEL_REGISTRY : Dict, optional
        Model registry to update
    make_spatial_rf : callable, optional
        Factory function for SpatialRandomForest
    SpatialRandomForest : class, optional
        SpatialRandomForest class
    _fit_binary : callable, optional
        Original _fit_binary function to wrap
    """
    global _SKIP_MODELS
    
    logger.info("Applying all model patches...")
    
    # 1. CUSTOM_WRAPPERS is already set at module level
    logger.info(f"  ✓ CUSTOM_WRAPPERS: {CUSTOM_WRAPPERS}")
    
    # 2. Define Fixed GNN model classes if torch available
    if TORCH_AVAILABLE:
        _define_gnn_model_classes()
    
    # 3. Try to patch GNN wrappers if they exist
    try:
        from gtfs_disruption.modeling.gnn_models import (
            STGCNWrapper, DCRNNWrapper, GraphWaveNetWrapper
        )
        _patch_gnn_wrappers(STGCNWrapper, DCRNNWrapper, GraphWaveNetWrapper)
    except ImportError:
        logger.warning("  ⚠  GNN wrappers not found - skipping GNN patches")
    
    # 4. Patch SpatialRF if available
    if SpatialRandomForest is not None:
        _patch_spatial_rf(SpatialRandomForest)
    
    # 5. Update MODEL_REGISTRY if provided
    if MODEL_REGISTRY is not None and make_spatial_rf is not None:
        if "SpatialRF" in MODEL_REGISTRY:
            MODEL_REGISTRY["SpatialRF"] = (make_spatial_rf, False)
    
    # Wrap _fit_binary if provided
    wrapped_fit_binary = None
    if _fit_binary is not None:
        wrapped_fit_binary = _wrap_fit_binary(_fit_binary)
    
    logger.info("  ✓ Model patches applied")
    
    return wrapped_fit_binary


def _define_gnn_model_classes():
    """Define fixed GNN model classes."""
    if not TORCH_AVAILABLE:
        return
    
    class STGCNModelFixed(torch.nn.Module):
        """Fixed STGCN model for compatibility."""
        
        def __init__(self, num_nodes, in_channels, hidden_dim,
                     out_channels, num_layers, adj):
            super().__init__()
            self.input_proj = torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            self.blocks = torch.nn.ModuleList([
                STGCNBlock(hidden_dim, hidden_dim, adj)
                for _ in range(num_layers)
            ])
            self.output_proj = torch.nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
            self.fc = torch.nn.Linear(num_nodes * out_channels, out_channels)
        
        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.output_proj(x)
            x = x[:, :, :, -1].reshape(x.size(0), -1)
            return self.fc(x)
    
    class DCRNNModelFixed(torch.nn.Module):
        """Fixed DCRNN model for compatibility."""
        
        def __init__(self, num_nodes, in_channels, hidden_channels,
                     out_channels, num_layers, adj):
            super().__init__()
            self.input_proj = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            self.output_proj = torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
            self.fc = torch.nn.Linear(num_nodes * out_channels, out_channels)
        
        def forward(self, x):
            x = self.input_proj(x)
            x = self.output_proj(x)
            return self.fc(x.reshape(x.size(0), -1))
    
    class GraphWaveNetModelFixed(torch.nn.Module):
        """Fixed GraphWaveNet model for compatibility."""
        
        def __init__(self, num_nodes, in_channels, hidden_dim,
                     out_channels, num_layers, adj):
            super().__init__()
            self.input_proj = torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            self.output_proj = torch.nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
            self.fc = torch.nn.Linear(num_nodes * out_channels, out_channels)
        
        def forward(self, x):
            x = self.input_proj(x)
            x = self.output_proj(x)
            x = x[:, :, :, -1].reshape(x.size(0), -1)
            return self.fc(x)
    
    # Store in module for reference
    import sys
    for cls in [STGCNModelFixed, DCRNNModelFixed, GraphWaveNetModelFixed]:
        setattr(sys.modules[__name__], cls.__name__, cls)
    
    logger.info("  ✓ Fixed GNN model classes defined")


def _patch_gnn_wrappers(STGCNWrapper, DCRNNWrapper, GraphWaveNetWrapper):
    """Patch GNN wrappers with fixed fit/predict_proba methods."""
    
    def _gnn_fit_fixed(self, X, y):
        if not TORCH_AVAILABLE:
            return self._fit_fallback(X, y)
        if hasattr(y, "values"):
            y = y.values
        y = np.array(y, dtype=np.float32)
        if y.ndim == 2:
            try:
                pt_idx = 0  # Will be set by caller
            except:
                pt_idx = -1
            y = y[:, pt_idx]
        y = y.reshape(-1, 1)
        self.num_targets = 1
        n, n_feat = X.shape
        step = max(1, n_feat // (self.seq_len * self.num_nodes))
        self._input_size = step
        self.adj_ = self._build_adj()  # simplified
        hidden = getattr(self, "hidden_dim", getattr(self, "hidden_channels", 64))
        model_cls = {
            "STGCNWrapper": STGCNModelFixed,
            "DCRNNWrapper": DCRNNModelFixed,
            "GraphWaveNetWrapper": GraphWaveNetModelFixed,
        }.get(type(self).__name__, STGCNModelFixed)
        self.model_ = model_cls(
            self.num_nodes, step, hidden, 1, self.num_layers, self.adj_
        )
        X_r = self._reshape_X(X, step, self.seq_len, self.num_nodes)
        self._train_loop(
            self.model_,
            torch.FloatTensor(X_r),
            torch.FloatTensor(y[:n]),
            self.epochs, self.batch_size, self.lr,
        )
        return self
    
    def _gnn_predict_proba_fixed(self, X):
        if self._fallback is not None:
            return self._predict_proba_fallback(X)
        if not TORCH_AVAILABLE or self.model_ is None:
            return np.full((len(X), 1), 0.5)
        step = getattr(self, "_input_size", 1)
        proba1 = self._predict_proba_model(
            X, self.model_, step, self.seq_len, self.num_nodes
        )
        return proba1
    
    def _stgcn_get_params(self, deep=True):
        return {"num_nodes": self.num_nodes, "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers, "seq_len": self.seq_len,
                "lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size}
    
    def _dcrnn_get_params(self, deep=True):
        return {"num_nodes": self.num_nodes, "hidden_channels": self.hidden_channels,
                "num_layers": self.num_layers, "seq_len": self.seq_len,
                "lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size}
    
    def _gwn_get_params(self, deep=True):
        return {"num_nodes": self.num_nodes, "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers, "seq_len": self.seq_len,
                "lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size}
    
    def _gnn_set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
    
    # Apply patches
    STGCNWrapper.fit = _gnn_fit_fixed
    STGCNWrapper.predict_proba = _gnn_predict_proba_fixed
    STGCNWrapper.get_params = _stgcn_get_params
    STGCNWrapper.set_params = _gnn_set_params
    
    DCRNNWrapper.fit = _gnn_fit_fixed
    DCRNNWrapper.predict_proba = _gnn_predict_proba_fixed
    DCRNNWrapper.get_params = _dcrnn_get_params
    DCRNNWrapper.set_params = _gnn_set_params
    
    GraphWaveNetWrapper.fit = _gnn_fit_fixed
    GraphWaveNetWrapper.predict_proba = _gnn_predict_proba_fixed
    GraphWaveNetWrapper.get_params = _gwn_get_params
    GraphWaveNetWrapper.set_params = _gnn_set_params
    
    logger.info("  ✓ GNN fit/predict_proba/get_params/set_params patched")


def _patch_spatial_rf(SpatialRandomForest):
    """Patch SpatialRandomForest with fixed methods."""
    
    def _spatial_rf_fit_fixed(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.X_train_raw_ = X
        self._fit_nn(X)
        X_aug = self._spatial_lags(X)
        if hasattr(y, "values"):
            y = y.values
        y = np.array(y)
        if y.ndim == 2:
            y = y[:, 0]
        self.rf_.fit(X_aug, y)
        return self
    
    def _spatial_rf_predict_proba_fixed(self, X):
        X = np.asarray(X, dtype=np.float32)
        proba = self.rf_.predict_proba(self._spatial_lags(X))
        if isinstance(proba, list):
            return np.column_stack([p[:, 1] for p in proba])
        return proba
    
    SpatialRandomForest.fit = _spatial_rf_fit_fixed
    SpatialRandomForest.predict_proba = _spatial_rf_predict_proba_fixed
    
    logger.info("  ✓ SpatialRF fit/predict_proba patched")


def _wrap_fit_binary(original_fit_binary):
    """Wrap _fit_binary to skip GNNs on error."""
    
    def _fit_binary_safe(base_fn, model_name, X_tr, X_vl, yb_fit,
                      use_early_stopping=False):
        try:
            return original_fit_binary(
                base_fn, model_name, X_tr, X_vl, yb_fit,
                use_early_stopping=use_early_stopping,
            )
        except Exception as e:
            if model_name in GNN_MODELS:
                logger.warning(f"  ⚠  {model_name} skipped — {e}")
                _SKIP_MODELS.add(model_name)
                return None
            raise
    
    return _fit_binary_safe


# Convenience function for one-line application
def apply_all_patches(**kwargs):
    """Apply all patches with minimal arguments."""
    return apply_patches(
        BINARY_TARGETS=kwargs.get("BINARY_TARGETS"),
        PRIMARY_TARGET=kwargs.get("PRIMARY_TARGET", "is_disruption"),
    )


if __name__ == "__main__":
    print("Running patch application...")
    apply_patches()
    print("\n✓ All patches applied!")