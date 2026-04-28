"""
Adaptive Data Splitting Pipeline for GTFS-RT Traffic Disruption Data
=====================================================================
Automatically selects optimal chronological splitting strategy based on
incoming data volume to prevent temporal leakage while maximizing
training efficiency.

Supports:
- Fixed-ratio chronological split (70/15/15) for medium/large datasets
- Walk-forward validation for small datasets
- Gap buffer between splits to prevent feature window straddling
- Multi-stream alignment before splitting
- Disruption class balance checking
- MinIO partition-aware loading
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for adaptive splitting."""
    # Thresholds for dataset size classification
    small_dataset_max_points: int = 10000
    small_dataset_max_days: float = 7.0
    
    # Fixed-ratio split parameters
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Walk-forward parameters
    walk_forward_min_train_days: int = 4
    
    # Gap buffer (minutes) between splits to prevent feature window straddling
    gap_buffer_minutes: int = 30
    
    # Multi-stream alignment bin size (minutes)
    alignment_bin_minutes: int = 5
    
    # Strategy override: 'auto', 'fixed_ratio', 'walk_forward'
    strategy_override: str = 'auto'
    
    # MinIO configuration
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_secure: bool = True
    
    # Specific time boundaries for 8-day window (GMT+3)
    # Mon 23 Mar 2026, 21:34 → Tue 31 Mar 2026, 06:27
    use_specific_boundaries: bool = False
    train_end_time: Optional[str] = None  # e.g., "2026-03-30 12:00:00"
    val_end_time: Optional[str] = None    # e.g., "2026-03-31 00:00:00"
    test_end_time: Optional[str] = None   # e.g., "2026-03-31 06:27:00"


@dataclass
class SplitResult:
    """Result of adaptive splitting operation."""
    strategy_used: str
    split_type: str  # 'fixed_ratio' or 'walk_forward'
    
    # For fixed-ratio split
    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    
    # For walk-forward validation
    fold_indices: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    total_points: int = 0
    total_days: float = 0.0
    disruption_balance: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)
    
    # Time boundaries used
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    val_start: Optional[datetime] = None
    val_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None


class AdaptiveSplitter:
    """
    Adaptive data splitting pipeline for GTFS-RT traffic disruption data.
    
    Automatically selects between:
    - Fixed-ratio chronological split (70/15/15) for medium/large datasets
    - Walk-forward validation for small datasets
    
    Ensures temporal ordering is preserved and prevents data leakage.
    """
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize the adaptive splitter.
        
        Parameters
        ----------
        config : SplitConfig, optional
            Configuration for splitting. Uses defaults if not provided.
        """
        self.config = config or SplitConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not np.isclose(
            self.config.train_ratio + self.config.val_ratio + self.config.test_ratio,
            1.0
        ):
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0"
            )
        
        if self.config.strategy_override not in ['auto', 'fixed_ratio', 'walk_forward']:
            raise ValueError(
                "strategy_override must be 'auto', 'fixed_ratio', or 'walk_forward'"
            )
    
    def _assess_data_volume(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> Tuple[int, float, str]:
        """
        Assess data volume and determine dataset size category.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        timestamp_col : str
            Name of timestamp column
        
        Returns
        -------
        Tuple of (total_points, total_days, size_category)
        """
        total_points = len(df)
        
        # Calculate time span
        timestamps = pd.to_datetime(df[timestamp_col])
        time_span = timestamps.max() - timestamps.min()
        total_days = time_span.total_seconds() / (24 * 3600)
        
        # Classify dataset size
        if (total_points < self.config.small_dataset_max_points or 
            total_days < self.config.small_dataset_max_days):
            size_category = 'small'
        else:
            size_category = 'medium_large'
        
        logger.info(f"Data volume assessment:")
        logger.info(f"  Total points: {total_points:,}")
        logger.info(f"  Time span: {total_days:.2f} days")
        logger.info(f"  Size category: {size_category}")
        
        return total_points, total_days, size_category
    
    def _temporal_leakage_guard(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.DataFrame:
        """
        Sort data by timestamp to prevent temporal leakage.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        timestamp_col : str
            Name of timestamp column
        
        Returns
        -------
        pd.DataFrame sorted by timestamp
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in dataframe")
        
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Verify no duplicate timestamps at boundaries
        n_duplicates = df_sorted[timestamp_col].duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates} duplicate timestamps")
        
        return df_sorted
    
    def _align_multi_streams(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        stream_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Align multiple data streams onto common time bins.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with multiple streams
        timestamp_col : str
            Name of timestamp column
        stream_cols : List[str], optional
            Columns representing different data streams
        
        Returns
        -------
        pd.DataFrame with aligned time bins
        """
        df = df.copy()
        
        # Create time bins
        bin_seconds = self.config.alignment_bin_minutes * 60
        df['_time_bin'] = pd.to_datetime(df[timestamp_col]).dt.floor(
            f'{self.config.alignment_bin_minutes}min'
        )
        
        # If stream columns specified, aggregate within bins
        if stream_cols:
            # Group by time bin and aggregate streams
            agg_dict = {}
            for col in stream_cols:
                if col in df.columns:
                    if df[col].dtype in ['object', 'category']:
                        agg_dict[col] = 'first'
                    else:
                        agg_dict[col] = 'mean'
            
            if agg_dict:
                df = df.groupby('_time_bin').agg(agg_dict).reset_index()
                df = df.rename(columns={'_time_bin': timestamp_col})
        
        return df
    
    def _check_disruption_balance(
        self,
        splits: Dict[str, pd.DataFrame],
        disruption_col: str = 'disruption_type'
    ) -> Dict[str, float]:
        """
        Check disruption class balance across splits.
        
        Parameters
        ----------
        splits : Dict[str, pd.DataFrame]
            Dictionary of split dataframes
        disruption_col : str
            Name of disruption type column
        
        Returns
        -------
        Dict with disruption distribution per split
        """
        balance = {}
        
        for split_name, split_df in splits.items():
            if disruption_col in split_df.columns:
                # Calculate disruption rate (non-ON_TIME events)
                disruption_rate = (
                    split_df[disruption_col] != 'ON_TIME'
                ).mean()
                balance[split_name] = disruption_rate
            else:
                balance[split_name] = np.nan
        
        return balance
    
    def _apply_gap_buffer(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        split_point: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Apply gap buffer around split point to prevent feature window straddling.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        timestamp_col : str
            Name of timestamp column
        split_point : pd.Timestamp
            Point where split occurs
        
        Returns
        -------
        pd.DataFrame with gap buffer applied
        """
        gap_delta = timedelta(minutes=self.config.gap_buffer_minutes)
        
        # Remove records within gap buffer of split point
        mask = (
            (pd.to_datetime(df[timestamp_col]) < split_point - gap_delta) |
            (pd.to_datetime(df[timestamp_col]) > split_point + gap_delta)
        )
        
        return df[mask].reset_index(drop=True)
    
    def _fixed_ratio_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute 70/15/15 chronological split.
        
        Parameters
        ----------
        df : pd.DataFrame
            Sorted dataframe
        timestamp_col : str
            Name of timestamp column
        
        Returns
        -------
        Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Apply gap buffers if configured
        if self.config.gap_buffer_minutes > 0:
            # Gap between train and val
            if len(val_df) > 0:
                val_start = pd.to_datetime(val_df[timestamp_col].iloc[0])
                train_df = self._apply_gap_buffer(train_df, timestamp_col, val_start)
            
            # Gap between val and test
            if len(test_df) > 0:
                test_start = pd.to_datetime(test_df[timestamp_col].iloc[0])
                val_df = self._apply_gap_buffer(val_df, timestamp_col, test_start)
        
        logger.info(f"Fixed-ratio split:")
        logger.info(f"  Train: {len(train_df):,} rows ({len(train_df)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df):,} rows ({len(val_df)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} rows ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def _fixed_ratio_split_with_boundaries(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute fixed-ratio split using specific time boundaries.
        
        Parameters
        ----------
        df : pd.DataFrame
            Sorted dataframe
        timestamp_col : str
            Name of timestamp column
        
        Returns
        -------
        Tuple of (train_df, val_df, test_df)
        """
        # Parse time boundaries
        train_end = pd.to_datetime(self.config.train_end_time)
        val_end = pd.to_datetime(self.config.val_end_time)
        test_end = pd.to_datetime(self.config.test_end_time)
        
        # Apply gap buffers
        gap_delta = timedelta(minutes=self.config.gap_buffer_minutes)
        
        # Split by time boundaries
        train_mask = pd.to_datetime(df[timestamp_col]) < train_end - gap_delta
        val_mask = (
            (pd.to_datetime(df[timestamp_col]) > train_end + gap_delta) &
            (pd.to_datetime(df[timestamp_col]) < val_end - gap_delta)
        )
        test_mask = pd.to_datetime(df[timestamp_col]) > val_end + gap_delta
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        
        n = len(df)
        logger.info(f"Fixed-ratio split (specific boundaries):")
        logger.info(f"  Train: {len(train_df):,} rows ({len(train_df)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df):,} rows ({len(val_df)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} rows ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def _walk_forward_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> List[Dict[str, Any]]:
        """
        Execute walk-forward validation with expanding windows.
        
        Parameters
        ----------
        df : pd.DataFrame
            Sorted dataframe
        timestamp_col : str
            Name of timestamp column
        
        Returns
        -------
        List of fold dictionaries with train/test indices
        """
        # Get unique dates
        dates = pd.to_datetime(df[timestamp_col]).dt.date.unique()
        dates = sorted(dates)
        total_days = len(dates)
        
        min_train_days = self.config.walk_forward_min_train_days
        
        if total_days <= min_train_days:
            raise ValueError(
                f"Dataset has only {total_days} days, need at least "
                f"{min_train_days + 1} days for walk-forward validation"
            )
        
        # Number of folds: total_days - min_train_days
        n_folds = total_days - min_train_days
        
        folds = []
        for i in range(n_folds):
            # Train: Day 1 to Day (min_train_days + i)
            train_end_date = dates[min_train_days + i - 1]
            # Test: Day (min_train_days + i + 1)
            test_date = dates[min_train_days + i]
            
            train_mask = pd.to_datetime(df[timestamp_col]).dt.date <= train_end_date
            test_mask = pd.to_datetime(df[timestamp_col]).dt.date == test_date
            
            train_indices = df[train_mask].index.tolist()
            test_indices = df[test_mask].index.tolist()
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                folds.append({
                    'fold': i + 1,
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                    'train_end_date': train_end_date,
                    'test_date': test_date,
                    'train_size': len(train_indices),
                    'test_size': len(test_indices)
                })
        
        logger.info(f"Walk-forward validation:")
        logger.info(f"  Total days: {total_days}")
        logger.info(f"  Folds: {len(folds)}")
        logger.info(f"  Min train days: {min_train_days}")
        
        for fold in folds[:3]:  # Log first 3 folds
            logger.info(
                f"  Fold {fold['fold']}: "
                f"Train to {fold['train_end_date']} "
                f"({fold['train_size']:,} rows), "
                f"Test on {fold['test_date']} "
                f"({fold['test_size']:,} rows)"
            )
        
        if len(folds) > 3:
            logger.info(f"  ... and {len(folds) - 3} more folds")
        
        return folds
    
    def split(
        self,
        data: Union[pd.DataFrame, str],
        timestamp_col: str = 'feed_timestamp',
        disruption_col: str = 'disruption_type',
        stream_cols: Optional[List[str]] = None
    ) -> SplitResult:
        """
        Execute adaptive splitting pipeline.
        
        Parameters
        ----------
        data : pd.DataFrame or str
            Input dataframe or MinIO path
        timestamp_col : str
            Name of timestamp column
        disruption_col : str
            Name of disruption type column for balance checking
        stream_cols : List[str], optional
            Columns representing different data streams for alignment
        
        Returns
        -------
        SplitResult with split datasets or fold indices
        """
        logger.info("="*60)
        logger.info("ADAPTIVE SPLITTING PIPELINE")
        logger.info("="*60)
        
        # Load data if path provided
        if isinstance(data, str):
            logger.info(f"Loading data from: {data}")
            if data.startswith('s3://') or data.startswith('minio://'):
                # MinIO path handling
                df = self._load_from_minio(data)
            else:
                # Local file path
                df = pd.read_parquet(data)
        else:
            df = data.copy()
        
        # Step 1: Temporal leakage guard
        logger.info("Step 1: Applying temporal leakage guard...")
        df = self._temporal_leakage_guard(df, timestamp_col)
        
        # Step 2: Multi-stream alignment
        if stream_cols:
            logger.info("Step 2: Aligning multi-stream data...")
            df = self._align_multi_streams(df, timestamp_col, stream_cols)
        else:
            logger.info("Step 2: Skipping multi-stream alignment (no stream columns specified)")
        
        # Step 3: Assess data volume
        logger.info("Step 3: Assessing data volume...")
        total_points, total_days, size_category = self._assess_data_volume(
            df, timestamp_col
        )
        
        # Step 4: Select strategy
        if self.config.strategy_override != 'auto':
            strategy = self.config.strategy_override
            logger.info(f"Step 4: Using override strategy: {strategy}")
        elif size_category == 'small':
            strategy = 'walk_forward'
            logger.info(
                f"Step 4: Selected walk-forward validation "
                f"(dataset: {total_points:,} points, {total_days:.1f} days)"
            )
        else:
            strategy = 'fixed_ratio'
            logger.info(
                f"Step 4: Selected fixed-ratio split "
                f"(dataset: {total_points:,} points, {total_days:.1f} days)"
            )
        
        # Step 5: Execute split
        logger.info("Step 5: Executing split...")
        
        if strategy == 'fixed_ratio':
            # Use specific boundaries if configured
            if self.config.use_specific_boundaries and self.config.train_end_time:
                train_df, val_df, test_df = self._fixed_ratio_split_with_boundaries(
                    df, timestamp_col
                )
            else:
                train_df, val_df, test_df = self._fixed_ratio_split(df, timestamp_col)
            
            # Check disruption balance
            splits = {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }
            disruption_balance = self._check_disruption_balance(splits, disruption_col)
            
            # Check for warnings
            warnings = []
            if disruption_col in df.columns:
                # Check if major disruption is confined to training set
                train_disruptions = train_df[disruption_col].value_counts()
                val_disruptions = val_df[disruption_col].value_counts()
                test_disruptions = test_df[disruption_col].value_counts()
                
                for disruption_type in ['MAJOR_DELAY', 'CANCELLED']:
                    if disruption_type in train_disruptions:
                        train_count = train_disruptions[disruption_type]
                        val_count = val_disruptions.get(disruption_type, 0)
                        test_count = test_disruptions.get(disruption_type, 0)
                        
                        if train_count > 0 and val_count == 0 and test_count == 0:
                            warnings.append(
                                f"Major disruption '{disruption_type}' found only in training set"
                            )
            
            # Get time boundaries
            train_start = pd.to_datetime(train_df[timestamp_col]).min() if len(train_df) > 0 else None
            train_end = pd.to_datetime(train_df[timestamp_col]).max() if len(train_df) > 0 else None
            val_start = pd.to_datetime(val_df[timestamp_col]).min() if len(val_df) > 0 else None
            val_end = pd.to_datetime(val_df[timestamp_col]).max() if len(val_df) > 0 else None
            test_start = pd.to_datetime(test_df[timestamp_col]).min() if len(test_df) > 0 else None
            test_end = pd.to_datetime(test_df[timestamp_col]).max() if len(test_df) > 0 else None
            
            result = SplitResult(
                strategy_used=strategy,
                split_type='fixed_ratio',
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                total_points=total_points,
                total_days=total_days,
                disruption_balance=disruption_balance,
                warnings=warnings,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end
            )
            
        else:  # walk_forward
            fold_indices = self._walk_forward_split(df, timestamp_col)
            
            # Check disruption balance across folds
            disruption_balance = {}
            for fold in fold_indices:
                train_df_fold = df.iloc[fold['train_indices']]
                test_df_fold = df.iloc[fold['test_indices']]
                
                if disruption_col in df.columns:
                    train_rate = (train_df_fold[disruption_col] != 'ON_TIME').mean()
                    test_rate = (test_df_fold[disruption_col] != 'ON_TIME').mean()
                    disruption_balance[f"fold_{fold['fold']}_train"] = train_rate
                    disruption_balance[f"fold_{fold['fold']}_test"] = test_rate
            
            result = SplitResult(
                strategy_used=strategy,
                split_type='walk_forward',
                fold_indices=fold_indices,
                total_points=total_points,
                total_days=total_days,
                disruption_balance=disruption_balance,
                warnings=[]
            )
        
        # Log summary
        logger.info("="*60)
        logger.info("SPLIT SUMMARY")
        logger.info("="*60)
        logger.info(f"Strategy used: {result.strategy_used}")
        logger.info(f"Split type: {result.split_type}")
        logger.info(f"Total points: {result.total_points:,}")
        logger.info(f"Total days: {result.total_days:.2f}")
        
        if result.train_start and result.train_end:
            logger.info(f"Train period: {result.train_start} to {result.train_end}")
        if result.val_start and result.val_end:
            logger.info(f"Val period: {result.val_start} to {result.val_end}")
        if result.test_start and result.test_end:
            logger.info(f"Test period: {result.test_start} to {result.test_end}")
        
        if result.disruption_balance:
            logger.info("Disruption balance:")
            for split_name, rate in result.disruption_balance.items():
                logger.info(f"  {split_name}: {rate:.3f}")
        
        if result.warnings:
            logger.warning("Warnings:")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
        
        return result
    
    def _load_from_minio(self, path: str) -> pd.DataFrame:
        """
        Load data from MinIO/S3 path.
        
        Parameters
        ----------
        path : str
            MinIO/S3 path (e.g., s3://bucket/path/to/data/)
        
        Returns
        -------
        pd.DataFrame
        """
        try:
            import s3fs
            
            fs = s3fs.S3FileSystem(
                endpoint_url=self.config.minio_endpoint,
                key=self.config.minio_access_key,
                secret=self.config.minio_secret_key,
                secure=self.config.minio_secure
            )
            
            # List all parquet files in path
            files = fs.glob(f"{path}/*.parquet")
            
            if not files:
                raise FileNotFoundError(f"No parquet files found at {path}")
            
            # Load and concatenate
            dfs = []
            for file in files:
                with fs.open(file, 'rb') as f:
                    dfs.append(pd.read_parquet(f))
            
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(df):,} rows from {len(files)} files")
            
            return df
            
        except ImportError:
            raise ImportError(
                "s3fs package required for MinIO access. "
                "Install with: pip install s3fs"
            )


def adaptive_split(
    data: Union[pd.DataFrame, str],
    timestamp_col: str = 'feed_timestamp',
    disruption_col: str = 'disruption_type',
    stream_cols: Optional[List[str]] = None,
    config: Optional[SplitConfig] = None
) -> SplitResult:
    """
    Convenience function for adaptive splitting.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        Input dataframe or MinIO path
    timestamp_col : str
        Name of timestamp column
    disruption_col : str
        Name of disruption type column
    stream_cols : List[str], optional
        Columns representing different data streams
    config : SplitConfig, optional
        Configuration for splitting
    
    Returns
    -------
    SplitResult with split datasets or fold indices
    """
    splitter = AdaptiveSplitter(config)
    return splitter.split(data, timestamp_col, disruption_col, stream_cols)
