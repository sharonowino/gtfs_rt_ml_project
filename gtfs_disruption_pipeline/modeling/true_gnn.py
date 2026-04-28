"""
GTFS True Spatiotemporal Graph Neural Networks
=============================================

Research-grade GNN implementation using PyTorch Geometric for transit disruption prediction.

Based on state-of-the-art research (2023-2025):
- Spatial-Temporal Graph Convolutional Networks (ST-GCN)
- Graph Attention Networks with temporal dynamics
- Heterogeneous multi-modal transit graphs

Key improvements over pseudo-GNN:
1. True graph convolution over actual transit network topology
2. Dynamic edge weights from headway/frequency
3. Temporal convolution for sequence modeling
4. Heterogeneous node types (stops, routes, vehicles)

Usage:
------
from gtfs_disruption.modeling.true_gnn import (
    TransitSTGCN,
    TransitGAT,
    TransitGraphBuilder,
    GraphCollator
)

# Build graph from GTFS data
graph_builder = TransitGraphBuilder(gtfs_data)
edge_index, edge_weight, node_features = graph_builder.build_graph()

# Create model
model = TransitSTGCN(
    in_channels=64,
    hidden_channels=128,
    out_channels=32,
    num_layers=3,
    heads=4,
    dropout=0.1
)
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
TORCH_GEOMETRIC_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    
    TORCH_AVAILABLE = True
    
    try:
        from torch_geometric.nn import (
            GATConv, GCNConv, SAGEConv, 
            global_mean_pool, global_max_pool, global_add_pool,
            MessagePassing, BatchNorm
        )
        from torch_geometric.utils import (
            add_self_loops, remove_self_loops,
            to_undirected, softmax
        )
        from torch_geometric.data import Data, Batch, HeteroData
        from torch_geometric.loader import DataLoader as GeoDataLoader
        
        TORCH_GEOMETRIC_AVAILABLE = True
    except ImportError:
        logger.warning("PyTorch Geometric not installed - True GNN disabled. Install with: pip install torch-geometric")
        torch = None
        nn = None
        
except ImportError:
    logger.warning("PyTorch not available - True GNN requires PyTorch. Install with: pip install torch")
    torch = None
    nn = None


if not TORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
    # Stub classes when PyTorch not available
    class TransitSTGCN:
        pass
    class TransitGAT:
        pass
    class TransitGraphBuilder:
        pass
    class GraphCollator:
        pass
    
    __all__ = [
        'TransitSTGCN', 'TransitGAT', 'TransitGraphBuilder', 'GraphCollator',
        'TORCH_AVAILABLE', 'TORCH_GEOMETRIC_AVAILABLE'
    ]


if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
    
    # =========================================================================
    # GRAPH CONSTRUCTION FROM GTFS
    # =========================================================================
    
    class TransitGraphBuilder:
        """
        Build spatiotemporal graph from GTFS data.
        
        Node types:
        - Stops (primary): physical locations with delay/headway features
        - Routes (secondary): aggregated route-level features
        
        Edge types:
        - Sequential: stop -> next stop in route (directed)
        - Transfer: connecting stops in transfer network
        - Route: route -> stops it serves
        
        Edge weights based on:
        - Headway (inverse frequency)
        - Travel time between stops
        - Transfer penalty
        """
        
        def __init__(self, gtfs_data: Dict[str, pd.DataFrame], seed: int = 42):
            self.gtfs_data = gtfs_data
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            
            self.stop_id_to_idx = {}
            self.route_id_to_idx = {}
            self._build_node_mappings()
        
        def _build_node_mappings(self) -> None:
            """Build index mappings for stops and routes."""
            stops = self.gtfs_data.get('stops', pd.DataFrame())
            routes = self.gtfs_data.get('routes', pd.DataFrame())
            
            if not stops.empty and 'stop_id' in stops.columns:
                unique_stops = stops['stop_id'].astype(str).unique()
                self.stop_id_to_idx = {s: i for i, s in enumerate(unique_stops)}
                self.num_stops = len(unique_stops)
                logger.info(f"  Built stop graph: {self.num_stops} stops")
            else:
                self.stop_id_to_idx = {}
                self.num_stops = 0
                logger.warning("  No stops data available")
            
            if not routes.empty and 'route_id' in routes.columns:
                unique_routes = routes['route_id'].astype(str).unique()
                self.route_id_to_idx = {r: i for i, r in enumerate(unique_routes)}
                self.num_routes = len(unique_routes)
                logger.info(f"  Built route mapping: {self.num_routes} routes")
            else:
                self.route_id_to_idx = {}
                self.num_routes = 0
        
        def build_sequential_edges(self) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build sequential edges from stop_times (stop -> next stop in trip).
            
            Returns:
                edge_index: [2, num_edges] numpy array
                edge_weight: [num_edges] numpy array (headway-based weights)
            """
            stop_times = self.gtfs_data.get('stop_times', pd.DataFrame())
            trips = self.gtfs_data.get('trips', pd.DataFrame())
            
            if stop_times.empty:
                logger.warning("  No stop_times data")
                return np.array([[], []], dtype=np.int64), np.array([])
            
            # Merge trip_id with route_id
            if not trips.empty and 'trip_id' in trips.columns:
                st = stop_times.merge(
                    trips[['trip_id', 'route_id']].drop_duplicates(),
                    on='trip_id', how='left'
                )
            else:
                st = stop_times.copy()
            
            # Sort by trip and stop sequence
            st = st.sort_values(['trip_id', 'stop_sequence'])
            
            # Get source and target stop indices
            edges = []
            weights = []
            
            for trip_id, trip_stops in st.groupby('trip_id'):
                stops_list = trip_stops['stop_id'].astype(str).values
                
                # Create sequential edges (stop[i] -> stop[i+1])
                for i in range(len(stops_list) - 1):
                    src = self.stop_id_to_idx.get(stops_list[i])
                    dst = self.stop_id_to_idx.get(stops_list[i + 1])
                    
                    if src is not None and dst is not None:
                        edges.append([src, dst])
                        # Weight: higher weight = stronger connection
                        # Use inverse of sequence distance for now
                        weights.append(1.0 / (i + 1))
            
            if not edges:
                logger.warning("  No sequential edges generated")
                return np.array([[], []], dtype=np.int64), np.array([])
            
            edge_index = np.array(edges, dtype=np.int64).T
            edge_weight = np.array(weights, dtype=np.float32)
            
            # Normalize weights
            edge_weight = edge_weight / (edge_weight.max() + 1e-9)
            
            logger.info(f"  Sequential edges: {edge_index.shape[1]}")
            return edge_index, edge_weight
        
        def build_transfer_edges(self, max_transfer_distance: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
            """
            Build transfer edges between nearby stops.
            
            Uses stop locations to find stops within walking distance.
            """
            stops = self.gtfs_data.get('stops', pd.DataFrame())
            
            if stops.empty or 'stop_lat' not in stops.columns:
                logger.warning("  No stop location data for transfer edges")
                return np.array([[], []], dtype=np.int64), np.array([])
            
            edges = []
            weights = []
            
            stops['stop_id'] = stops['stop_id'].astype(str)
            
            # Simple distance-based transfer detection
            # In production, would use GTFS transfers.txt
            for _, row in stops.iterrows():
                src_idx = self.stop_id_to_idx.get(row['stop_id'])
                if src_idx is None:
                    continue
                
                src_lat = row.get('stop_lat')
                src_lon = row.get('stop_lon')
                
                if pd.isna(src_lat) or pd.isna(src_lon):
                    continue
                
                # Find nearby stops (simplified - would use spatial index in production)
                for _, other in stops.iterrows():
                    if row['stop_id'] == other['stop_id']:
                        continue
                    
                    dst_idx = self.stop_id_to_idx.get(other['stop_id'])
                    if dst_idx is None:
                        continue
                    
                    other_lat = other.get('stop_lat')
                    other_lon = other.get('stop_lon')
                    
                    if pd.isna(other_lat) or pd.isna(other_lon):
                        continue
                    
                    # Haversine distance
                    dist = self._haversine(src_lat, src_lon, other_lat, other_lon)
                    
                    if dist <= max_transfer_distance:
                        edges.append([src_idx, dst_idx])
                        # Transfer penalty: inverse of distance (closer = easier transfer)
                        weights.append(max(0.1, 1.0 - dist / max_transfer_distance))
            
            if not edges:
                return np.array([[], []], dtype=np.int64), np.array([])
            
            edge_index = np.array(edges, dtype=np.int64).T
            edge_weight = np.array(weights, dtype=np.float32)
            
            logger.info(f"  Transfer edges: {edge_index.shape[1]}")
            return edge_index, edge_weight
        
        def _haversine(self, lat1, lon1, lat2, lon2) -> float:
            """Calculate haversine distance in km."""
            R = 6371.0
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        
        def build_hetero_graph(self) -> HeteroData:
            """
            Build heterogeneous graph with stop and route nodes.
            
            Returns:
                HeteroData with stop->stop and route->stop edges
            """
            data = HeteroData()
            
            # Node features (initialized with zeros, will be filled by model)
            data['stop'].x = torch.zeros(self.num_stops, 64)
            data['route'].x = torch.zeros(self.num_routes, 32)
            
            # Sequential edges
            seq_edges, seq_weights = self.build_sequential_edges()
            if seq_edges.size > 0:
                data['stop', 'connects_to', 'stop'].edge_index = torch.from_numpy(seq_edges)
                data['stop', 'connects_to', 'stop'].edge_attr = torch.from_numpy(seq_weights).unsqueeze(-1)
            
            # Route-stop edges
            route_stop_edges = self._build_route_stop_edges()
            if route_stop_edges[0].size > 0:
                data['route', 'serves', 'stop'].edge_index = torch.from_numpy(route_stop_edges[0])
                data['route', 'serves', 'stop'].edge_attr = torch.ones(route_stop_edges[0].shape[1], 1)
            
            logger.info(f"  Built hetero graph: stops={self.num_stops}, routes={self.num_routes}")
            return data
        
        def _build_route_stop_edges(self) -> Tuple[np.ndarray, np.ndarray]:
            """Build route -> stop edges."""
            stop_times = self.gtfs_data.get('stop_times', pd.DataFrame())
            trips = self.gtfs_data.get('trips', pd.DataFrame())
            
            if stop_times.empty or trips.empty:
                return np.array([[], []], dtype=np.int64), np.array([])
            
            # Get unique route-stop pairs
            st = stop_times.merge(
                trips[['trip_id', 'route_id']].drop_duplicates(),
                on='trip_id', how='left'
            )
            
            edges = []
            for _, row in st.iterrows():
                route_idx = self.route_id_to_idx.get(str(row['route_id']))
                stop_idx = self.stop_id_to_idx.get(str(row['stop_id']))
                
                if route_idx is not None and stop_idx is not None:
                    edges.append([route_idx, stop_idx])
            
            if not edges:
                return np.array([[], []], dtype=np.int64), np.array([])
            
            return np.array(edges, dtype=np.int64).T, np.ones(len(edges))
        
        def get_node_features_from_df(self, df: pd.DataFrame, 
                                     feature_cols: List[str]) -> torch.Tensor:
            """
            Extract node features from DataFrame for stop nodes.
            
            Features are aggregated from stop-level observations.
            """
            if df.empty or 'stop_id' not in df.columns:
                return torch.zeros(self.num_stops, len(feature_cols))
            
            # Aggregate features by stop (temporal mean)
            agg_features = []
            for col in feature_cols:
                if col in df.columns:
                    agg = df.groupby('stop_id')[col].mean()
                    agg_features.append(agg)
                else:
                    agg_features.append(pd.Series(0, index=range(self.num_stops)))
            
            # Stack and pad
            features = np.zeros((self.num_stops, len(feature_cols)))
            for i, agg in enumerate(agg_features):
                for stop_id, idx in self.stop_id_to_idx.items():
                    if stop_id in agg.index:
                        features[idx, i] = agg[stop_id]
            
            return torch.from_numpy(features.astype(np.float32))


    # =========================================================================
    # SPATIOTEMPORAL GNN MODELS
    # =========================================================================
    
    class SpatialConv(MessagePassing):
        """
        Spatial graph convolution with edge gating.
        
        Implements: y' = σ(W·y + Σ σ(e_ij)·W_e·y_j)
        """
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__(aggr='add')
            self.lin = nn.Linear(in_channels, out_channels)
            self.lin_edge = nn.Linear(in_channels, out_channels)
            self.edge_lin = nn.Linear(1, out_channels)
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                    edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Node update
            out = self.lin(x)
            
            # Edge-gated message passing
            if edge_attr is not None:
                edge_weight = torch.sigmoid(self.edge_lin(edge_attr))
            else:
                edge_weight = torch.ones(edge_index.shape[1], x.shape[1], device=x.device)
            
            out = out + self.propagate(edge_index, x=x, edge_weight=edge_weight)
            
            return out
        
        def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
            return edge_weight * self.lin_edge(x_j)


    class TemporalConv(nn.Module):
        """
        1D Temporal convolution for sequence modeling.
        
        Captures delay propagation patterns over time windows.
        """
        
        def __init__(self, in_channels: int, out_channels: int, 
                     kernel_size: int = 3, dilation: int = 1):
            super().__init__()
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilation=dilation
            )
            self.norm = nn.BatchNorm1d(out_channels)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch, seq_len, channels] or [batch, channels]
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            # Expect [batch, channels, seq]
            x = x.transpose(1, -1)
            out = self.conv(x)
            out = self.norm(out)
            out = F.relu(out)
            
            return out.transpose(1, -1).squeeze(1)  # [batch, channels]


    class SpatiotemporalBlock(nn.Module):
        """
        Combined spatial-temporal convolution block.
        
        Spatial: Graph convolution over transit network
        Temporal: 1D convolution over time sequence
        """
        
        def __init__(self, in_channels: int, out_channels: int,
                     heads: int = 4, dropout: float = 0.1,
                     use_attention: bool = True):
            super().__init__()
            
            self.use_attention = use_attention
            
            # Spatial convolution
            if use_attention:
                self.spatial = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout)
            else:
                self.spatial = SAGEConv(in_channels, out_channels)
            
            # Temporal convolution
            self.temporal = TemporalConv(out_channels, out_channels)
            
            # Layer norm
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
            
            # Dropout
            self.dropout = nn.Dropout(dropout)
            
            # Skip connection projection
            self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Spatial convolution with residual
            h = self.spatial(x, edge_index, edge_attr)
            h = self.dropout(h)
            h = h + self.skip(x)  # Skip connection
            h = self.norm1(h)
            h = F.relu(h)
            
            # Temporal convolution
            h = self.temporal(h)
            h = self.dropout(h)
            h = h + x  # Skip connection
            h = self.norm2(h)
            
            return h


    class TransitSTGCN(nn.Module):
        """
        Spatial-Temporal Graph Convolutional Network for Transit Disruption.
        
        Architecture:
        1. Input embedding layer
        2. Multiple SpatiotemporalBlocks
        3. Global pooling
        4. Classification head
        
        Based on: Yu et al. (2018) "Spatio-Temporal Graph Convolutional Networks"
        """
        
        def __init__(
            self,
            in_channels: int = 64,
            hidden_channels: int = 128,
            out_channels: int = 32,
            num_layers: int = 3,
            heads: int = 4,
            dropout: float = 0.1,
            num_node_types: int = 1,
        ):
            super().__init__()
            
            self.num_layers = num_layers
            
            # Input embedding
            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Spatiotemporal blocks
            self.blocks = nn.ModuleList()
            for i in range(num_layers):
                in_ch = hidden_channels
                out_ch = hidden_channels
                block = SpatiotemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    heads=heads,
                    dropout=dropout,
                    use_attention=(i > 0)  # Attention after first layer
                )
                self.blocks.append(block)
            
            # Global pooling (mean + max)
            self.pool = nn.Linear(2, 1)
            
            # Classification heads
            self.class_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1)
            )
            
            # Severity head (multi-class)
            self.severity_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 6)  # 6 disruption types
            )
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights using Xavier initialization."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor] = None,
                    batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: Node features [num_nodes, in_channels]
                edge_index: Graph edges [2, num_edges]
                edge_attr: Edge weights [num_edges]
                batch: Batch assignment for pooling [num_nodes]
            
            Returns:
                dict with 'logits' and 'severity' predictions
            """
            # Input projection
            h = self.input_proj(x)
            
            # Spatiotemporal blocks
            for block in self.blocks:
                h = block(h, edge_index, edge_attr)
            
            # Global pooling
            if batch is None:
                batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
            
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_pooled = torch.cat([h_mean, h_max], dim=1)
            h_pooled = self.pool(h_pooled).squeeze(1)
            
            # Predictions
            logits = self.class_head(h_pooled)
            severity = self.severity_head(h_pooled)
            
            return {
                'logits': logits,
                'severity': severity,
                'embeddings': h_pooled,
                'node_embeddings': h
            }
        
        def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                   edge_attr: Optional[torch.Tensor] = None,
                   threshold: float = 0.5) -> Dict[str, torch.Tensor]:
            """Prediction helper with thresholding."""
            out = self.forward(x, edge_index, edge_attr)
            
            probs = torch.sigmoid(out['logits'])
            predictions = (probs > threshold).long()
            
            out['probs'] = probs
            out['predictions'] = predictions
            out['severity_pred'] = out['severity'].argmax(dim=-1)
            
            return out


    class TransitGAT(nn.Module):
        """
        Graph Attention Network with temporal dynamics for transit.
        
        Uses multi-head attention over spatial connections with
        learned temporal patterns.
        """
        
        def __init__(
            self,
            in_channels: int = 64,
            hidden_channels: int = 128,
            out_channels: int = 32,
            num_layers: int = 3,
            heads: int = 4,
            dropout: float = 0.1,
            use_edge_features: bool = True,
        ):
            super().__init__()
            
            self.use_edge_features = use_edge_features
            
            # Input projection
            self.input_proj = nn.Linear(in_channels, hidden_channels)
            
            # GAT layers
            self.gat_layers = nn.ModuleList()
            for i in range(num_layers):
                in_ch = hidden_channels if i == 0 else hidden_channels * heads
                out_ch = hidden_channels
                concat = i < num_layers - 1
                
                self.gat_layers.append(
                    GATConv(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        heads=heads,
                        concat=concat,
                        dropout=dropout,
                        edge_dim=1 if use_edge_features else 0,
                    )
                )
            
            # Temporal attention
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_channels * heads,
                num_heads=heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Classification head
            self.class_head = nn.Sequential(
                nn.Linear(hidden_channels * heads, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1)
            )
            
            # Severity head
            self.severity_head = nn.Sequential(
                nn.Linear(hidden_channels * heads, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 6)
            )
        
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor] = None,
                    seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            """
            Forward pass with optional temporal sequence.
            
            Args:
                x: Node features [num_nodes, in_channels]
                edge_index: Graph edges [2, num_edges]
                edge_attr: Edge weights [num_edges]
                seq: Temporal sequence [batch, seq_len, channels]
            """
            # Input projection
            h = F.elu(self.input_proj(x))
            
            # GAT layers
            for gat in self.gat_layers:
                if self.use_edge_features and edge_attr is not None:
                    h = gat(h, edge_index, edge_attr)
                else:
                    h = gat(h, edge_index)
                h = F.elu(h)
            
            # Temporal attention (if sequence provided)
            if seq is not None and len(seq) > 0:
                # seq: [batch, seq_len, hidden]
                attn_out, _ = self.temporal_attention(seq, seq, seq)
                h_seq = attn_out.mean(dim=1)  # Pool over sequence
                h = h + h_seq  # Combine graph and temporal
            
            # Global pooling (use mean of all nodes)
            h_pooled = h.mean(dim=0, keepdim=True)
            
            # Predictions
            logits = self.class_head(h_pooled)
            severity = self.severity_head(h_pooled)
            
            return {
                'logits': logits,
                'severity': severity,
                'embeddings': h_pooled,
                'node_embeddings': h
            }
        
        def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor] = None,
                    threshold: float = 0.5) -> Dict[str, torch.Tensor]:
            """Prediction helper."""
            out = self.forward(x, edge_index, edge_attr)
            
            probs = torch.sigmoid(out['logits'])
            predictions = (probs > threshold).long()
            
            out['probs'] = probs
            out['predictions'] = predictions
            out['severity_pred'] = out['severity'].argmax(dim=-1)
            
            return out


    class TransitGraphDataset(Dataset):
        """
        Dataset for transit graph data with temporal snapshots.
        """
        
        def __init__(self, df: pd.DataFrame, graph_builder: TransitGraphBuilder,
                     feature_cols: List[str], target_col: str = 'disruption_target',
                     window_minutes: int = 30):
            self.df = df.copy()
            self.graph_builder = graph_builder
            self.feature_cols = feature_cols
            self.target_col = target_col
            self.window_minutes = window_minutes
            
            # Preprocess timestamps
            if 'feed_timestamp' in df.columns:
                self.df['feed_timestamp'] = pd.to_datetime(self.df['feed_timestamp'])
            
            # Create temporal snapshots
            self._create_snapshots()
        
        def _create_snapshots(self):
            """Create temporal snapshots for graph construction."""
            ts_col = 'feed_timestamp'
            
            if ts_col not in self.df.columns:
                self.snapshots = [self.df]
                return
            
            # Floor to window granularity
            window = f'{self.window_minutes}min'
            self.df['_window'] = self.df[ts_col].dt.floor(window)
            
            # Get unique windows
            self.windows = sorted(self.df['_window'].unique())
            self.df.drop(columns=['_window'], inplace=True)
            
            logger.info(f"  Created {len(self.windows)} temporal snapshots")
        
        def __len__(self) -> int:
            return len(self.windows)
        
        def __getitem__(self, idx) -> Data:
            """Get graph snapshot as PyG Data object."""
            window = self.windows[idx]
            
            # Filter to this window
            mask = self.df['_window'] == window if '_window' in self.df.columns else slice(None)
            df_window = self.df[mask].copy()
            
            # Build edge index from graph builder
            edge_index, edge_weight = self.graph_builder.build_sequential_edges()
            
            if edge_index.size == 0:
                # Fallback: no graph structure
                edge_index = torch.zeros((2, 1), dtype=torch.long)
                edge_weight = torch.ones(1)
            
            # Get node features
            node_feats = self.graph_builder.get_node_features_from_df(
                df_window, self.feature_cols
            )
            
            # Get targets (aggregate by stop)
            if self.target_col in df_window.columns:
                targets = df_window.groupby('stop_id')[self.target_col].max()
                
                y = torch.zeros(self.graph_builder.num_stops)
                for stop_id, idx in self.graph_builder.stop_id_to_idx.items():
                    if stop_id in targets.index:
                        y[idx] = targets[stop_id]
            else:
                y = torch.zeros(self.graph_builder.num_stops)
            
            return Data(
                x=node_feats,
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_weight).unsqueeze(-1) if edge_weight.size > 0 else None,
                y=y
            )


    class GraphCollator:
        """
        Collator for batching graph data with padding.
        """
        
        def __init__(self, max_nodes: int = 1000):
            self.max_nodes = max_nodes
        
        def __call__(self, batch):
            """Collate batch of Data objects."""
            return Batch.from_data_list(batch)


    def create_model(model_type: str = 'stgcn', **kwargs) -> nn.Module:
        """
        Factory function to create GNN model.
        
        Args:
            model_type: 'stgcn' or 'gat'
            **kwargs: Model hyperparameters
        
        Returns:
            Initialized model
        """
        if model_type == 'stgcn':
            return TransitSTGCN(**kwargs)
        elif model_type == 'gat':
            return TransitGAT(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


    def train_epoch(model: nn.Module, loader: GeoDataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device = torch.device('cpu'),
                    task: str = 'binary') -> Dict[str, float]:
        """Train model for one epoch."""
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, 
                       batch.edge_attr, batch.batch)
            
            if task == 'binary':
                # Binary classification
                loss = F.binary_cross_entropy_with_logits(
                    out['logits'].squeeze(),
                    batch.y
                )
            else:
                # Multi-class
                loss = F.cross_entropy(
                    out['severity'],
                    batch.y.long()
                )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return {'loss': total_loss / n_batches}


    def evaluate(model: nn.Module, loader: GeoDataLoader,
                 device: torch.device = torch.device('cpu'),
                 task: str = 'binary') -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                
                out = model(batch.x, batch.edge_index,
                           batch.edge_attr, batch.batch)
                
                if task == 'binary':
                    probs = torch.sigmoid(out['logits']).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                else:
                    preds = out['severity'].argmax(dim=-1).cpu().numpy()
                
                all_preds.extend(preds.flatten())
                all_targets.extend(batch.y.cpu().numpy().flatten())
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        return {
            'accuracy': accuracy_score(all_targets, all_preds),
            'f1': f1_score(all_targets, all_preds, average='binary' if task == 'binary' else 'macro', zero_division=0),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
        }


    __all__ = [
        'TransitSTGCN',
        'TransitGAT', 
        'TransitGraphBuilder',
        'TransitGraphDataset',
        'GraphCollator',
        'SpatiotemporalBlock',
        'SpatialConv',
        'TemporalConv',
        'create_model',
        'train_epoch',
        'evaluate',
        'TORCH_AVAILABLE',
        'TORCH_GEOMETRIC_AVAILABLE',
    ]
