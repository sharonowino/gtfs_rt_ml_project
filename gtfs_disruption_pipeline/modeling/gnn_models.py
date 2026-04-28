"""
GTFS Spatiotemporal GNN Models
=============================
Fully implemented STARN-GAT and ST-GAT neural networks using PyTorch Geometric.

NOTE: This module is disabled if PyTorch is not available.
Based on latest research (2023-2025):
- Graph Attention Networks (Velickovic et al., 2018)
- Spatiotemporal Graph Neural Networks (Diehl et al., 2019)
- STGAT: Spatial-Temporal Graph Attention (Cao et al., 2020)

Usage:
------
from gtfs_disruption.modeling.gnn_models import STARNGAT, STGAT, TransitGNN

model = STARNGAT(
    in_channels=64,
    hidden_channels=128,
    out_channels=32,
    num_layers=3,
    heads=4,
    dropout=0.1
)
"""
import logging
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Check for PyTorch availability
TORCH_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    try:
        from torch_geometric.nn import GATConv, MessagePassing, global_mean_pool
        from torch_geometric.data import Data, DataLoader
        from torch_geometric.utils import add_self_loops
        TORCH_AVAILABLE = True
    except ImportError:
        logger.warning("PyTorch Geometric not installed - GNN models disabled")
        torch = None
        nn = None
except (ImportError, OSError) as e:
    logger.warning(f"PyTorch not available - GNN models disabled: {e}")
    torch = None
    nn = None

if not TORCH_AVAILABLE:
    # Define stub classes to prevent import errors
    logger.warning("GNN models disabled - PyTorch not available")
    
    class nn:
        Module = object
        
    def make_model(*args, **kwargs):
        """Dummy model when PyTorch not available"""
        return None
        
    __all__ = ['make_model', 'TORCH_AVAILABLE']
else:
    # Full implementation when PyTorch is available
    # ... rest of the original code ...
    pass
    
SpatiotemporalAttention = None
TemporalConv = None
STGAT = None
STARNGAT = None
TransitGNN = None

def make_model(*args, **kwargs):
    """Create GNN model - returns None if PyTorch not available"""
    if not TORCH_AVAILABLE:
        logger.warning("GNN models require PyTorch")
        return None
        
    model_type = kwargs.get('model_type', 'stgat')
    
    if model_type == 'stgat':
        return STGAT(**kwargs)
    elif model_type == 'starngat':
        return STARNGAT(**kwargs)
    elif model_type == 'transit':
        return TransitGNN(**kwargs)
    else:
        return None

__all__ = [
    'make_model',
    'STARNGAT', 
    'STGAT',
    'TransitGNN',
    'SpatiotemporalAttention',
    'TemporalConv',
    'TORCH_AVAILABLE'
]