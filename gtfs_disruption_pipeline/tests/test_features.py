"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from gtfs_disruption.features import DisruptionFeatureBuilder
from gtfs_disruption.features.classifier import DisruptionClassifier
from gtfs_disruption.features.analyzer import DisruptionAnalyzer

class TestDisruptionFeatureBuilder:
    """Test DisruptionFeatureBuilder class."""
    
    def test_init(self):
        """Test initialization."""
        merged_df = pd.DataFrame({'trip_id': ['t1', 't2'], 'stop_id': ['s1', 's2']})
        gtfs_data = {'trips': pd.DataFrame(), 'routes': pd.DataFrame()}
        builder = DisruptionFeatureBuilder(merged_df, gtfs_data)
        assert builder.merged_df is not None
        assert builder.gtfs is not None
    
    def test_ensure_str(self):
        """Test string conversion."""
        merged_df = pd.DataFrame({'trip_id': [1, 2], 'stop_id': [3, 4]})
        gtfs_data = {}
        builder = DisruptionFeatureBuilder(merged_df, gtfs_data)
        result = builder._ensure_str(merged_df, ['trip_id', 'stop_id'])
        assert result['trip_id'].dtype == object
        assert result['stop_id'].dtype == object

class TestDisruptionClassifier:
    """Test DisruptionClassifier class."""
    
    def test_init(self):
        """Test initialization with default values."""
        clf = DisruptionClassifier()
        assert clf.delay_major_sec == 600
        assert clf.delay_minor_sec == 120
        assert clf.delay_early_sec == -60
        assert clf.speed_stopped_kmh == 2.0
        assert clf.speed_slow_kmh == 10.0
    
    def test_init_custom(self):
        """Test initialization with custom values."""
        clf = DisruptionClassifier(
            delay_major_sec=300,
            delay_minor_sec=60,
            delay_early_sec=-30,
            speed_stopped_kmh=1.0,
            speed_slow_kmh=5.0
        )
        assert clf.delay_major_sec == 300
        assert clf.delay_minor_sec == 60
        assert clf.delay_early_sec == -30
        assert clf.speed_stopped_kmh == 1.0
        assert clf.speed_slow_kmh == 5.0
    
    def test_is_cancelled(self):
        """Test cancellation detection."""
        clf = DisruptionClassifier()
        
        # Test CANCELED vehicle status
        row1 = pd.Series({'vehicle_status': 'CANCELED'})
        assert clf._is_cancelled(row1) == True
        
        # Test NO_SERVICE alert effect
        row2 = pd.Series({'alert_effect': 'NO_SERVICE'})
        assert clf._is_cancelled(row2) == True
        
        # Test normal status
        row3 = pd.Series({'vehicle_status': 'IN_TRANSIT_TO', 'alert_effect': 'DETOUR'})
        assert clf._is_cancelled(row3) == False
    
    def test_classify(self):
        """Test classification logic."""
        clf = DisruptionClassifier()
        
        # Create test data
        df = pd.DataFrame({
            'delay_sec': [0, 150, 700, -80, 0],
            'speed': [30, 5, 0, 25, 15],
            'speed_flag': ['normal', 'slow', 'stopped', 'normal', 'normal'],
            'has_alert': [False, False, False, False, True],
            'vehicle_status': ['IN_TRANSIT_TO'] * 5,
            'alert_effect': [''] * 5
        })
        
        result = clf.classify(df)
        
        # Check that columns were added
        assert 'disruption_type' in result.columns
        assert 'severity_score' in result.columns
        
        # Check specific classifications
        assert result.loc[0, 'disruption_type'] == 'ON_TIME'  # normal
        assert result.loc[1, 'disruption_type'] == 'MINOR_DELAY'  # delay > 120
        assert result.loc[2, 'disruption_type'] == 'MAJOR_DELAY'  # delay > 600
        assert result.loc[3, 'disruption_type'] == 'EARLY'  # delay < -60
        assert result.loc[4, 'disruption_type'] == 'SERVICE_ALERT'  # has alert

class TestDisruptionAnalyzer:
    """Test DisruptionAnalyzer class."""
    
    def test_init(self):
        """Test initialization."""
        df = pd.DataFrame({'disruption_type': ['ON_TIME', 'MINOR_DELAY']})
        analyzer = DisruptionAnalyzer(df)
        assert analyzer.df is not None
    
    def test_schema(self, caplog):
        """Test schema printing."""
        df = pd.DataFrame({
            'trip_id': ['t1', 't2'],
            'stop_id': ['s1', 's2'],
            'disruption_type': ['ON_TIME', 'MINOR_DELAY']
        })
        analyzer = DisruptionAnalyzer(df)
        analyzer.schema()
        assert 'FEATURE SCHEMA' in caplog.text
    
    def test_hot_spots(self):
        """Test hot spots analysis."""
        df = pd.DataFrame({
            'stop_id': ['s1', 's1', 's2', 's2'],
            'severity_score': [5, 7, 3, 2],
            'disruption_type': ['MINOR_DELAY', 'MAJOR_DELAY', 'SLOW_TRAFFIC', 'ON_TIME']
        })
        analyzer = DisruptionAnalyzer(df)
        hot_spots = analyzer.hot_spots(top_n=2)
        
        assert len(hot_spots) == 2
        assert hot_spots.loc[0, 'stop_id'] == 's1'  # highest severity
        assert hot_spots.loc[0, 'avg_severity'] == 6.0  # (5+7)/2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
