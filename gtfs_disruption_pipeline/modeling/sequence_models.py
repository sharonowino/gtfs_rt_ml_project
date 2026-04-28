"""
Sequence Models for Transit Delay Propagation
=====================================

LSTM/GRU models for capturing temporal delay patterns along routes.

Delay propagates spatially (along route direction) and temporally
(over time). These models capture:
1. Sequential delay patterns within trips
2. Delay cascade timing along route direction  
3. Temporal dependencies for early warning

Based on research:
- LSTM/GRU for time-series (Greff et al., 2017)
- Sequence-to-sequence for forecasting ( Sutskever et al., 2014)
- Attention for interpretability (Bahdanau et al., 2014)

Usage:
------
from gtfs_disruption.modeling.sequence_models import (
    DelayLSTM,
    DelayGRU,
    SequencePredictor,
    create_sequence_model
)

model = create_sequence_model('lstm', input_dim=64, hidden_dim=128)
"""
import logging
from typing import Dict, List, Optional, Tuple, Literal
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
    logger.warning("PyTorch not available - sequence models disabled")
    torch = None
    nn = None


if not TORCH_AVAILABLE:
    # Stub classes
    class DelayLSTM:
        pass
    class DelayGRU:
        pass
    class SequencePredictor:
        pass
    
    __all__ = ['DelayLSTM', 'DelayGRU', 'SequencePredictor', 'create_sequence_model', 'TORCH_AVAILABLE']
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    
    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    
    class DelaySequenceDataset(Dataset):
        """
        Dataset of delay sequences along routes.
        
        Creates sequences for predicting future delays
        from historical delay patterns.
        """
        
        def __init__(
            self,
            df: pd.DataFrame,
            trip_col: str = 'trip_id',
            stop_col: str = 'stop_id',
            time_col: str = 'feed_timestamp',
            delay_col: str = 'delay_sec',
            feature_cols: Optional[List[str]] = None,
            sequence_length: int = 10,
            prediction_horizon: int = 1,
            min_sequence_length: int = 3,
        ):
            self.df = df.copy()
            self.trip_col = trip_col
            self.stop_col = stop_col
            self.time_col = time_col
            self.delay_col = delay_col
            self.sequence_length = sequence_length
            self.prediction_horizon = prediction_horizon
            self.min_sequence_length = min_sequence_length
            
            # Parse timestamps
            if time_col in self.df.columns:
                self.df[time_col] = pd.to_datetime(self.df[time_col])
            
            # Get feature columns
            self.feature_cols = feature_cols or [
                'delay_sec', 'speed', 'headway_sec',
                'stop_sequence', 'hour', 'day_of_week'
            ]
            
            # Build sequences
            self.sequences = self._build_sequences()
            self.targets = self._build_targets()
        
        def _build_sequences(self) -> List[torch.Tensor]:
            """Build delay sequences grouped by trip."""
            
            # Sort by trip and time
            sort_cols = [c for c in [self.trip_col, self.time_col] 
                        if c in self.df.columns]
            
            if sort_cols:
                self.df = self.df.sort_values(sort_cols)
            
            sequences = []
            
            # Group by trip
            for trip_id, trip_df in self.df.groupby(self.trip_col):
                if len(trip_df) < self.min_sequence_length:
                    continue
                
                # Get feature matrix
                feat_vals = []
                for col in self.feature_cols:
                    if col in trip_df.columns:
                        vals = trip_df[col].fillna(0).values
                    else:
                        vals = np.zeros(len(trip_df))
                    feat_vals.append(vals)
                
                feat_matrix = np.stack(feat_vals, axis=1)
                
                # Create sequences with sliding window
                for i in range(len(trip_df) - self.sequence_length - self.prediction_horizon + 1):
                    seq = feat_matrix[i:i + self.sequence_length]
                    sequences.append(torch.from_numpy(seq.astype(np.float32)))
            
            logger.info(f"  Built {len(sequences)} sequences")
            return sequences
        
        def _build_targets(self) -> torch.Tensor:
            """Build prediction targets."""
            
            targets = []
            
            for seq in self.sequences:
                # Target is delay at prediction_horizon steps ahead
                target_idx = -self.prediction_horizon
                target = seq[target_idx, 0]  # delay column
                targets.append(target)
            
            if not targets:
                return torch.zeros(len(self.sequences))
            
            return torch.stack(targets)
        
        def __len__(self) -> int:
            return len(self.sequences)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.sequences[idx], self.targets[idx]
    
    
    class DelayPredictionDataset(Dataset):
        """
        Dataset for delay prediction (multi-step forecasting).
        
        Predicts delays at multiple future horizons.
        """
        
        def __init__(
            self,
            df: pd.DataFrame,
            horizons: List[int] = [1, 3, 5, 10],  # minutes ahead
            **kwargs
        ):
            self.horizons = horizons
            
            # Build base dataset
            self.base = DelaySequenceDataset(
                df, 
                prediction_horizon=max(horizons),
                **kwargs
            )
        
        def __len__(self) -> int:
            return len(self.base)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            seq, target = self.base[idx]
            
            # For multi-step, target is vector of delays at each horizon
            # (simplified - would compute properly in production)
            multi_target = target.repeat(len(self.horizons))
            
            return seq, multi_target


    # =========================================================================
    # MODELS
    # =========================================================================
    
    class DelayEncoder(nn.Module):
        """
        Encoder LSTM for delay sequences.
        
        Encodes historical delay pattern into hidden state.
        """
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 2,
            dropout: float = 0.1,
            bidirectional: bool = True,
        ):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Encode sequence.
            
            Args:
                x: [batch, seq_len, input_size]
            
            Returns:
                output: [batch, seq_len, hidden_size * directions]
                (h_n, c_n): hidden states
            """
            output, (h_n, c_n) = self.lstm(x)
            
            # Use last layer hidden state
            direction_mult = 2 if self.bidirectional else 1
            h_last = h_n[-direction_mult:]
            
            return output, (h_last, c_n)


    class AttentionAggregator(nn.Module):
        """
        Attention mechanism for sequence aggregation.
        
        Allows model to focus on relevant parts of delay history.
        """
        
        def __init__(self, hidden_size: int, num_heads: int = 4):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        def forward(self, x: torch.Tensor,
                   key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Apply attention pooling.
            
            Args:
                x: [batch, seq_len, hidden_size]
                key_padding_mask: [batch, seq_len] - True for padding
            
            Returns:
                pooled: [batch, hidden_size]
            """
            # Self-attention
            attn_out, _ = self.attention(
                x, x, x,
                key_padding_mask=key_padding_mask
            )
            
            # Residual connection + layer norm
            out = self.layer_norm(x + attn_out)
            
            # Mean pooling over sequence
            pooled = out.mean(dim=1)
            
            return pooled


    class DelayLSTM(nn.Module):
        """
        LSTM model for delay prediction.
        
        Architecture:
        1. Input embedding
        2. BiLSTM encoder
        3. Attention pooling
        4. Prediction head
        """
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_heads: int = 4,
            output_size: int = 1,
        ):
            super().__init__()
            
            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
            # Encoder
            self.encoder = DelayEncoder(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
            )
            
            # Attention pooling
            self.attention = AttentionAggregator(
                hidden_size * 2,  # bidirectional
                num_heads=num_heads,
            )
            
            # Prediction head
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )
        
        def forward(self, x: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: [batch, seq_len, input_size]
                mask: [batch, seq_len] - True for padding positions
            
            Returns:
                predictions dictionary
            """
            # Input projection
            h = self.input_proj(x)
            
            # Encode
            encoded, _ = self.encoder(h)
            
            # Attention pooling (use mask if provided)
            pooled = self.attention(encoded, key_padding_mask=mask)
            
            # Predict
            prediction = self.predictor(pooled).squeeze(-1)
            
            return {
                'prediction': prediction,
                'encoding': pooled,
                'sequence_encoding': encoded,
            }
        
        def predict(self, x: torch.Tensor,
                   threshold: float = 300.0) -> Dict[str, torch.Tensor]:
            """Prediction helper for classification."""
            out = self.forward(x)
            preds = (out['prediction'] > threshold).long()
            
            out['class_prediction'] = preds
            out['is_delayed'] = preds.bool()
            
            return out


    class DelayGRU(nn.Module):
        """
        GRU model for delay prediction.
        
        Generally faster to train than LSTM, may perform similarly.
        """
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_heads: int = 4,
            output_size: int = 1,
        ):
            super().__init__()
            
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
            # GRU encoder
            self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
                batch_first=True,
            )
            
            # Attention
            self.attention = AttentionAggregator(
                hidden_size * 2,
                num_heads=num_heads,
            )
            
            # Prediction head
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            h = self.input_proj(x)
            encoded, _ = self.gru(h)
            pooled = self.attention(encoded)
            prediction = self.predictor(pooled).squeeze(-1)
            
            return {
                'prediction': prediction,
                'encoding': pooled,
                'sequence_encoding': encoded,
            }


    class MultiStepPredictor(nn.Module):
        """
        Multi-step ahead delay prediction.
        
        Predicts delays at multiple future horizons using:
        1. Shared encoder
        2. Separate heads for each horizon
        """
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            horizons: List[int] = [1, 3, 5, 10],  # minutes
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.horizons = horizons
            self.num_horizons = len(horizons)
            
            # Shared encoder
            self.encoder = DelayEncoder(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
            )
            
            # Attention
            self.attention = AttentionAggregator(
                hidden_size * 2,
                num_heads=4,
            )
            
            # Separate heads per horizon
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
                for _ in horizons
            ])
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            encoded, _ = self.encoder(x)
            pooled = self.attention(encoded)
            
            # Predict each horizon
            predictions = []
            for head in self.heads:
                predictions.append(head(pooled).squeeze(-1))
            
            predictions = torch.stack(predictions, dim=1)
            
            return {
                'predictions': predictions,  # [batch, num_horizons]
                'encoding': pooled,
            }


    class TemporalFusionModel(nn.Module):
        """
        Temporal Fusion Transformer lite for transit.
        
        Combines:
        1. LSTM encoder for delay history
        2. Static features (route, stop)
        3. Temporal features (time of day, day of week)
        4. Attention for interpretability
        """
        
        def __init__(
            self,
            sequence_input_size: int,
            static_input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_horizons: int = 4,
        ):
            super().__init__()
            
            # Sequence LSTM
            self.sequence_encoder = DelayEncoder(
                sequence_input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
            )
            
            # Static feature projection
            self.static_proj = nn.Sequential(
                nn.Linear(static_input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
            # Temporal attention (for interpretability)
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            
            # Fusion gate
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_size * 2 + hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            
            # Prediction heads
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size * 2 + hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(num_horizons)
            ])
        
        def forward(self, 
                   sequence: torch.Tensor,
                   static: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                sequence: [batch, seq_len, seq_input_size]
                static: [batch, static_input_size]
            """
            # Encode sequence
            seq_enc, _ = self.sequence_encoder(sequence)
            
            # Temporal attention
            attn_out, attn_weights = self.temporal_attention(
                seq_enc, seq_enc, seq_enc
            )
            
            # Static features
            static_enc = self.static_proj(static)
            
            # Fuse using gate
            concat = torch.cat([attn_out.mean(dim=1), static_enc], dim=1)
            gate = self.fusion_gate(concat)
            fused = gate * static_enc
            
            # Predict each horizon
            predictions = []
            for head in self.heads:
                pred = head(torch.cat([attn_out.mean(dim=1), fused], dim=1))
                predictions.append(pred.squeeze(-1))
            
            predictions = torch.stack(predictions, dim=1)
            
            return {
                'predictions': predictions,
                'attention_weights': attn_weights,
                'fusion_gate': gate,
            }


    # =========================================================================
    # FACTORY AND TRAINING
    # =========================================================================
    
    def create_sequence_model(
        model_type: Literal['lstm', 'gru', 'multistep', 'fusion'],
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ) -> nn.Module:
        """Factory function to create sequence model."""
        
        if model_type == 'lstm':
            return DelayLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                output_size=kwargs.get('output_size', 1),
            )
        elif model_type == 'gru':
            return DelayGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                output_size=kwargs.get('output_size', 1),
            )
        elif model_type == 'multistep':
            return MultiStepPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                horizons=kwargs.get('horizons', [1, 3, 5, 10]),
            )
        elif model_type == 'fusion':
            return TemporalFusionModel(
                sequence_input_size=input_size,
                static_input_size=kwargs.get('static_input_size', 10),
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                num_horizons=kwargs.get('num_horizons', 4),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


    def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            sequences, targets = batch
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            out = model(sequences)
            loss = criterion(out.get('prediction', out.get('predictions', out['logits'])), targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return {'loss': total_loss / n_batches}


    def evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                sequences, targets = batch
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                out = model(sequences)
                loss = criterion(out.get('prediction', out.get('predictions', out['logits'])), targets)
                
                total_loss += loss.item()
                
                preds = out.get('prediction', out.get('predictions', out['logits']))
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                
                all_preds.extend(preds.flatten())
                all_targets.extend(targets.flatten())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        return {
            'loss': total_loss / len(loader),
            'mae': mean_absolute_error(all_targets, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
            'r2': r2_score(all_targets, all_preds),
        }


    __all__ = [
        'DelaySequenceDataset',
        'DelayPredictionDataset',
        'DelayLSTM',
        'DelayGRU',
        'MultiStepPredictor',
        'TemporalFusionModel',
        'create_sequence_model',
        'train_epoch',
        'evaluate',
        'TORCH_AVAILABLE',
    ]