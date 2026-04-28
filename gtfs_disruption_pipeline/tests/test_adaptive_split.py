"""
Tests for adaptive data splitting pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from gtfs_disruption.modeling.adaptive_split import (
    AdaptiveSplitter,
    SplitConfig,
    SplitResult,
    adaptive_split
)


@pytest.fixture
def small_dataset():
    """Create a small dataset (< 10,000 points, < 7 days)."""
    np.random.seed(42)
    n_points = 5000
    base_time = datetime(2026, 3, 23, 21, 34)
    
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_points)]
    
    return pd.DataFrame({
        'feed_timestamp': timestamps,
        'speed': np.random.uniform(0, 60, n_points),
        'delay_sec': np.random.uniform(-60, 600, n_points),
        'disruption_type': np.random.choice(
            ['ON_TIME', 'MINOR_DELAY', 'MAJOR_DELAY'],
            n_points,
            p=[0.8, 0.15, 0.05]
        )
    })


@pytest.fixture
def large_dataset():
    """Create a large dataset (> 10,000 points, > 7 days)."""
    np.random.seed(42)
    n_points = 50000
    base_time = datetime(2026, 3, 1, 0, 0)
    
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_points)]
    
    return pd.DataFrame({
        'feed_timestamp': timestamps,
        'speed': np.random.uniform(0, 60, n_points),
        'delay_sec': np.random.uniform(-60, 600, n_points),
        'disruption_type': np.random.choice(
            ['ON_TIME', 'MINOR_DELAY', 'MAJOR_DELAY'],
            n_points,
            p=[0.8, 0.15, 0.05]
        )
    })


@pytest.fixture
def multi_stream_dataset():
    """Create dataset with multiple data streams."""
    np.random.seed(42)
    n_points = 10000
    base_time = datetime(2026, 3, 23, 21, 34)
    
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_points)]
    
    return pd.DataFrame({
        'feed_timestamp': timestamps,
        'vehicle_speed': np.random.uniform(0, 60, n_points),
        'trip_delay': np.random.uniform(-60, 600, n_points),
        'alert_severity': np.random.choice([0, 1, 2], n_points),
        'disruption_type': np.random.choice(
            ['ON_TIME', 'MINOR_DELAY', 'MAJOR_DELAY'],
            n_points,
            p=[0.8, 0.15, 0.05]
        )
    })


class TestSplitConfig:
    """Test SplitConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SplitConfig()
        assert config.small_dataset_max_points == 10000
        assert config.small_dataset_max_days == 7.0
        assert config.train_ratio == 0.70
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.walk_forward_min_train_days == 4
        assert config.gap_buffer_minutes == 30
        assert config.alignment_bin_minutes == 5
        assert config.strategy_override == 'auto'
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SplitConfig(
            small_dataset_max_points=5000,
            small_dataset_max_days=5.0,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            gap_buffer_minutes=60
        )
        assert config.small_dataset_max_points == 5000
        assert config.small_dataset_max_days == 5.0
        assert config.train_ratio == 0.80
        assert config.val_ratio == 0.10
        assert config.test_ratio == 0.10
        assert config.gap_buffer_minutes == 60


class TestAdaptiveSplitter:
    """Test AdaptiveSplitter class."""
    
    def test_init_default(self):
        """Test initialization with default config."""
        splitter = AdaptiveSplitter()
        assert splitter.config is not None
        assert splitter.config.small_dataset_max_points == 10000
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = SplitConfig(small_dataset_max_points=5000)
        splitter = AdaptiveSplitter(config)
        assert splitter.config.small_dataset_max_points == 5000
    
    def test_invalid_config_ratios(self):
        """Test that invalid ratio sum raises error."""
        config = SplitConfig(train_ratio=0.80, val_ratio=0.15, test_ratio=0.15)
        with pytest.raises(ValueError, match="must equal 1.0"):
            AdaptiveSplitter(config)
    
    def test_invalid_strategy_override(self):
        """Test that invalid strategy override raises error."""
        config = SplitConfig(strategy_override='invalid')
        with pytest.raises(ValueError, match="must be 'auto'"):
            AdaptiveSplitter(config)
    
    def test_assess_data_volume_small(self, small_dataset):
        """Test data volume assessment for small dataset."""
        splitter = AdaptiveSplitter()
        total_points, total_days, size_category = splitter._assess_data_volume(
            small_dataset, 'feed_timestamp'
        )
        
        assert total_points == 5000
        assert total_days < 7.0
        assert size_category == 'small'
    
    def test_assess_data_volume_large(self, large_dataset):
        """Test data volume assessment for large dataset."""
        splitter = AdaptiveSplitter()
        total_points, total_days, size_category = splitter._assess_data_volume(
            large_dataset, 'feed_timestamp'
        )
        
        assert total_points == 50000
        assert total_days > 7.0
        assert size_category == 'medium_large'
    
    def test_temporal_leakage_guard(self, small_dataset):
        """Test temporal leakage guard sorts data."""
        splitter = AdaptiveSplitter()
        
        # Shuffle data
        shuffled = small_dataset.sample(frac=1, random_state=42)
        
        # Apply guard
        sorted_df = splitter._temporal_leakage_guard(shuffled, 'feed_timestamp')
        
        # Verify sorted
        timestamps = pd.to_datetime(sorted_df['feed_timestamp'])
        assert (timestamps.diff().dropna() >= timedelta(0)).all()
    
    def test_temporal_leakage_guard_missing_column(self, small_dataset):
        """Test temporal leakage guard with missing column."""
        splitter = AdaptiveSplitter()
        
        with pytest.raises(ValueError, match="not found"):
            splitter._temporal_leakage_guard(small_dataset, 'missing_column')
    
    def test_align_multi_streams(self, multi_stream_dataset):
        """Test multi-stream alignment."""
        splitter = AdaptiveSplitter()
        
        aligned = splitter._align_multi_streams(
            multi_stream_dataset,
            'feed_timestamp',
            ['vehicle_speed', 'trip_delay', 'alert_severity']
        )
        
        # Verify time bins created
        assert 'feed_timestamp' in aligned.columns
        # Verify data was aggregated (fewer rows than original)
        assert len(aligned) <= len(multi_stream_dataset)
    
    def test_check_disruption_balance(self, small_dataset):
        """Test disruption balance checking."""
        splitter = AdaptiveSplitter()
        
        splits = {
            'train': small_dataset.iloc[:4000],
            'val': small_dataset.iloc[4000:4500],
            'test': small_dataset.iloc[4500:]
        }
        
        balance = splitter._check_disruption_balance(splits, 'disruption_type')
        
        assert 'train' in balance
        assert 'val' in balance
        assert 'test' in balance
        assert all(0 <= v <= 1 for v in balance.values())
    
    def test_fixed_ratio_split(self, large_dataset):
        """Test fixed-ratio chronological split."""
        config = SplitConfig(gap_buffer_minutes=0)  # Disable gap buffer for test
        splitter = AdaptiveSplitter(config)
        
        train_df, val_df, test_df = splitter._fixed_ratio_split(
            large_dataset, 'feed_timestamp'
        )
        
        # Verify sizes
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(large_dataset)
        
        # Verify chronological order
        train_max = pd.to_datetime(train_df['feed_timestamp']).max()
        val_min = pd.to_datetime(val_df['feed_timestamp']).min()
        val_max = pd.to_datetime(val_df['feed_timestamp']).max()
        test_min = pd.to_datetime(test_df['feed_timestamp']).min()
        
        assert train_max < val_min
        assert val_max < test_min
    
    def test_walk_forward_split(self, small_dataset):
        """Test walk-forward validation split."""
        config = SplitConfig(walk_forward_min_train_days=2)
        splitter = AdaptiveSplitter(config)
        
        folds = splitter._walk_forward_split(small_dataset, 'feed_timestamp')
        
        # Verify folds created
        assert len(folds) > 0
        
        # Verify fold structure
        for fold in folds:
            assert 'fold' in fold
            assert 'train_indices' in fold
            assert 'test_indices' in fold
            assert 'train_end_date' in fold
            assert 'test_date' in fold
            assert 'train_size' in fold
            assert 'test_size' in fold
            
            # Verify no overlap
            train_set = set(fold['train_indices'])
            test_set = set(fold['test_indices'])
            assert len(train_set & test_set) == 0
    
    def test_walk_forward_split_too_small(self):
        """Test walk-forward split with too few days."""
        # Create dataset with only 3 days
        np.random.seed(42)
        n_points = 1000
        base_time = datetime(2026, 3, 23, 0, 0)
        
        timestamps = [base_time + timedelta(hours=i) for i in range(n_points)]
        df = pd.DataFrame({
            'feed_timestamp': timestamps,
            'disruption_type': ['ON_TIME'] * n_points
        })
        
        config = SplitConfig(walk_forward_min_train_days=4)
        splitter = AdaptiveSplitter(config)
        
        with pytest.raises(ValueError, match="only 3 days"):
            splitter._walk_forward_split(df, 'feed_timestamp')
    
    def test_split_small_dataset_auto(self, small_dataset):
        """Test automatic strategy selection for small dataset."""
        splitter = AdaptiveSplitter()
        result = splitter.split(small_dataset, 'feed_timestamp', 'disruption_type')
        
        assert result.strategy_used == 'walk_forward'
        assert result.split_type == 'walk_forward'
        assert result.fold_indices is not None
        assert len(result.fold_indices) > 0
        assert result.total_points == 5000
        assert result.total_days < 7.0
    
    def test_split_large_dataset_auto(self, large_dataset):
        """Test automatic strategy selection for large dataset."""
        config = SplitConfig(gap_buffer_minutes=0)
        splitter = AdaptiveSplitter(config)
        result = splitter.split(large_dataset, 'feed_timestamp', 'disruption_type')
        
        assert result.strategy_used == 'fixed_ratio'
        assert result.split_type == 'fixed_ratio'
        assert result.train_df is not None
        assert result.val_df is not None
        assert result.test_df is not None
        assert result.total_points == 50000
        assert result.total_days > 7.0
    
    def test_split_override_fixed_ratio(self, small_dataset):
        """Test strategy override to fixed-ratio for small dataset."""
        config = SplitConfig(
            strategy_override='fixed_ratio',
            gap_buffer_minutes=0
        )
        splitter = AdaptiveSplitter(config)
        result = splitter.split(small_dataset, 'feed_timestamp', 'disruption_type')
        
        assert result.strategy_used == 'fixed_ratio'
        assert result.split_type == 'fixed_ratio'
        assert result.train_df is not None
    
    def test_split_override_walk_forward(self, large_dataset):
        """Test strategy override to walk-forward for large dataset."""
        config = SplitConfig(strategy_override='walk_forward')
        splitter = AdaptiveSplitter(config)
        result = splitter.split(large_dataset, 'feed_timestamp', 'disruption_type')
        
        assert result.strategy_used == 'walk_forward'
        assert result.split_type == 'walk_forward'
        assert result.fold_indices is not None
    
    def test_split_with_multi_stream_alignment(self, multi_stream_dataset):
        """Test split with multi-stream alignment."""
        config = SplitConfig(gap_buffer_minutes=0)
        splitter = AdaptiveSplitter(config)
        result = splitter.split(
            multi_stream_dataset,
            'feed_timestamp',
            'disruption_type',
            stream_cols=['vehicle_speed', 'trip_delay', 'alert_severity']
        )
        
        assert result.strategy_used == 'fixed_ratio'
        assert result.train_df is not None
    
    def test_split_disruption_balance_check(self, large_dataset):
        """Test disruption balance checking in split."""
        config = SplitConfig(gap_buffer_minutes=0)
        splitter = AdaptiveSplitter(config)
        result = splitter.split(large_dataset, 'feed_timestamp', 'disruption_type')
        
        assert result.disruption_balance is not None
        assert 'train' in result.disruption_balance
        assert 'val' in result.disruption_balance
        assert 'test' in result.disruption_balance


class TestAdaptiveSplitFunction:
    """Test convenience function."""
    
    def test_adaptive_split_small(self, small_dataset):
        """Test convenience function with small dataset."""
        result = adaptive_split(small_dataset, 'feed_timestamp', 'disruption_type')
        
        assert isinstance(result, SplitResult)
        assert result.strategy_used == 'walk_forward'
    
    def test_adaptive_split_large(self, large_dataset):
        """Test convenience function with large dataset."""
        config = SplitConfig(gap_buffer_minutes=0)
        result = adaptive_split(
            large_dataset,
            'feed_timestamp',
            'disruption_type',
            config=config
        )
        
        assert isinstance(result, SplitResult)
        assert result.strategy_used == 'fixed_ratio'
    
    def test_adaptive_split_with_config(self, small_dataset):
        """Test convenience function with custom config."""
        config = SplitConfig(
            strategy_override='fixed_ratio',
            gap_buffer_minutes=0
        )
        result = adaptive_split(
            small_dataset,
            'feed_timestamp',
            'disruption_type',
            config=config
        )
        
        assert result.strategy_used == 'fixed_ratio'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
