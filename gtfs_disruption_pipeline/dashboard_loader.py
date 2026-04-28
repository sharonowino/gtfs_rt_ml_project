"""
Dashboard Data Loader - Uses gtfs_disruption modules properly.
===============================================
Integrates with DisruptionPipeline, DisruptionFeatureBuilder,
DisruptionClassifier, and ingestion modules.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_pipeline_data(
    config_path: str = "config.yaml",
    realtime_url: str = None,
    static_url: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and process data using gtfs_disruption modules.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (feature_df, pipeline_results) - Enriched DataFrame and results dict
    """
    try:
        from gtfs_disruption import DisruptionPipeline
        from gtfs_disruption.features import DisruptionFeatureBuilder
        from gtfs_disruption.ingestion import (
            ingest_combined, ingest_local, ingest_live,
            fetch_static_gtfs, load_static_gtfs_from_zip,
            DEFAULT_FEED_URLS, DEFAULT_STATIC_GTFS_URL, DEFAULT_LOCAL_DIR
        )

        # Step 1: Load real-time data
        merged_df = pd.DataFrame()
        gtfs_data = {}

        with st.spinner("Loading real-time feed data..."):
            if realtime_url:
                merged_df = ingest_live(realtime_url)
            elif DEFAULT_LOCAL_DIR:
                try:
                    merged_df = ingest_local(DEFAULT_LOCAL_DIR)
                except FileNotFoundError:
                    merged_df = ingest_live(list(DEFAULT_FEED_URLS.values()))
            else:
                merged_df = ingest_live(list(DEFAULT_FEED_URLS.values()))

        if merged_df.empty:
            st.warning("No real-time data loaded")
            return pd.DataFrame(), {}

        st.success(f"Loaded {len(merged_df)} real-time records")

        # Step 2: Load static GTFS data
        with st.spinner("Loading static GTFS data..."):
            try:
                gtfs_data = load_static_gtfs_from_zip(static_url or DEFAULT_STATIC_GTFS_URL)
            except Exception as e:
                logger.warning(f"Could not load static GTFS: {e}")
                gtfs_data = fetch_static_gtfs()

        if gtfs_data.get('stops') is not None and not gtfs_data['stops'].empty:
            st.success(f"Loaded static GTFS with {len(gtfs_data['stops'])} stops")
        else:
            st.info("No static GTFS data - using real-time only")

        # Step 3: Build features using DisruptionFeatureBuilder
        with st.spinner("Building features..."):
            builder = DisruptionFeatureBuilder(merged_df, gtfs_data)
            feature_df = builder.build()

        if feature_df.empty:
            st.warning("Feature building produced empty DataFrame")
            return pd.DataFrame(), {}

        st.success(f"Built {len(feature_df)} feature records with {len(feature_df.columns)} columns")

        # Step 4: Run pipeline for classification and analysis
        with st.spinner("Running disruption classification..."):
            try:
                pipeline = DisruptionPipeline(config_path)
                results = pipeline.run_full_pipeline(
                    merged_df=merged_df,
                    gtfs_data=gtfs_data,
                    model_type="best",
                    use_adaptive_split=False  # Faster for dashboard
                )

                # Extract classified_df if available
                classified_df = results.get('classified_df', feature_df)

                # Merge predictions back to feature_df if available
                if 'disruption_type' in classified_df.columns and 'disruption_type' not in feature_df.columns:
                    feature_df['disruption_type'] = classified_df['disruption_type']
                if 'disruption_class' in classified_df.columns and 'disruption_class' not in feature_df.columns:
                    feature_df['disruption_class'] = classified_df['disruption_class']

                results['feature_df'] = feature_df
                return feature_df, results

            except Exception as e:
                logger.warning(f"Pipeline classification failed: {e}")
                # Return feature_df with basic disruption_type from delay
                feature_df = _infer_disruption_type(feature_df)
                return feature_df, {'feature_df': feature_df}

    except ImportError as e:
        st.error(f"Missing module: {e}")
        return pd.DataFrame(), {}
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return pd.DataFrame(), {}


def _infer_disruption_type(df: pd.DataFrame) -> pd.DataFrame:
    """Infer disruption_type from delay if not present."""
    df = df.copy()

    if 'disruption_type' in df.columns:
        return df

    if 'delay_min' in df.columns:
        df['disruption_type'] = pd.cut(
            df['delay_min'],
            bins=[-np.inf, 2, 5, np.inf],
            labels=['ON_TIME', 'MINOR_DELAY', 'MAJOR_DELAY']
        )
    elif 'delay_sec' in df.columns:
        df['delay_min'] = df['delay_sec'] / 60
        df['disruption_type'] = pd.cut(
            df['delay_min'],
            bins=[-np.inf, 2, 5, np.inf],
            labels=['ON_TIME', 'MINOR_DELAY', 'MAJOR_DELAY']
        )
    else:
        df['disruption_type'] = 'ON_TIME'
        df['delay_min'] = 0.0

    return df


def load_live_data_only() -> pd.DataFrame:
    """Load just the live feed without full pipeline."""
    try:
        from gtfs_disruption.ingestion import ingest_live, DEFAULT_FEED_URLS

        merged_df = ingest_live(list(DEFAULT_FEED_URLS.values()))
        if merged_df.empty:
            return pd.DataFrame()

        return _infer_disruption_type(merged_df)
    except Exception as e:
        st.warning(f"Could not load live feed: {e}")
        return pd.DataFrame()


def load_parquet_with_pipeline(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a parquet file and process through the pipeline.

    Parameters
    ----------
    file_path : str
        Path to parquet file or file-like object from Streamlit uploader

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (feature_df, results)
    """
    try:
        from gtfs_disruption.features import DisruptionFeatureBuilder
        from gtfs_disruption.ingestion import fetch_static_gtfs

        # Load parquet file
        if hasattr(file_path, 'read'):
            import io
            df = pd.read_parquet(io.BytesIO(file_path.read()))
        else:
            df = pd.read_parquet(file_path)

        st.success(f"Loaded {len(df)} records from parquet")

        # Load minimal static GTFS
        with st.spinner("Loading static GTFS..."):
            gtfs_data = fetch_static_gtfs()

        # Build features
        with st.spinner("Building features..."):
            builder = DisruptionFeatureBuilder(df, gtfs_data)
            feature_df = builder.build()

        # Infer disruption type from delay if needed
        feature_df = _infer_disruption_type(feature_df)

        return feature_df, {'feature_df': feature_df}

    except Exception as e:
        st.error(f"Error loading parquet: {e}")
        return pd.DataFrame(), {}


def prepare_dashboard_data(df: pd.DataFrame) -> Dict:
    """
    Prepare data for dashboard using gtfs_disruption modules.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from pipeline

    Returns
    -------
    Dict
        Dashboard-ready data dictionary
    """
    if df is None or df.empty:
        return _generate_sample_data()

    data = {}

    # Required columns with fallbacks
    delay_col = 'delay_min' if 'delay_min' in df.columns else 'delay_sec'
    timestamp_col = None
    for col in ['feed_timestamp', 'timestamp', 'timestamp_unix', 'event_time', 'arrival_time']:
        if col in df.columns:
            timestamp_col = col
            break

    disruption_col = 'disruption_type' if 'disruption_type' in df.columns else None

    # Active disruptions count
    if disruption_col and disruption_col in df.columns:
        active = df[df[disruption_col] != 'ON_TIME']
        data['active_disruptions'] = len(active)
        data['alert_distribution'] = df[disruption_col].value_counts().to_dict()
    else:
        # Infer from delay
        if delay_col and delay_col in df.columns:
            threshold = 2  # 2 minutes
            delayed = df[df[delay_col] > threshold] if df[delay_col].dtype != 'object' else df
            data['active_disruptions'] = len(delayed)
            data['alert_distribution'] = {
                'Delayed': len(delayed),
                'On-Time': len(df) - len(delayed)
            }
        else:
            data['active_disruptions'] = 0
            data['alert_distribution'] = {'Total Records': len(df)}

    # Average delay
    if delay_col and delay_col in df.columns:
        data['avg_delay'] = df[delay_col].mean()
    else:
        data['avg_delay'] = 0.0

    # Timeline
    if timestamp_col and timestamp_col in df.columns:
        try:
            timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
            df_ts = df.copy()
            df_ts['hour'] = timestamps.dt.floor('h')
            timeline = df_ts.groupby('hour').size().reset_index(name='disruptions')
            data['timeline'] = timeline
        except Exception as e:
            logger.warning(f"Timeline error: {e}")
            data['timeline'] = pd.DataFrame({'hour': [], 'disruptions': []})
    else:
        # Generate synthetic timeline
        from datetime import datetime
        n = min(len(df), 24)
        hours = pd.date_range(end=datetime.now(), periods=n, freq='h')
        data['timeline'] = pd.DataFrame({
            'hour': hours,
            'disruptions': np.random.poisson(15, n)
        })

    # Route performance
    if 'route_id' in df.columns:
        if disruption_col and disruption_col in df.columns:
            route_perf = df.groupby('route_id').apply(
                lambda x: (x[disruption_col] != 'ON_TIME').mean()
            ).reset_index(name='Disruption Rate')
        else:
            route_perf = df.groupby('route_id').size().reset_index(name='Trip Count')
            route_perf.columns = ['Route', 'Trip Count']
        data['route_performance'] = route_perf
    else:
        data['route_performance'] = pd.DataFrame({'Route': [], 'On-Time %': []})

    # Active alerts table
    alert_cols = []
    available = []
    for col in ['feed_timestamp', 'timestamp', 'route_id', 'delay_min', 'disruption_type', 'alert_cause', 'alert_effect']:
        if col in df.columns:
            alert_cols.append(col)
            available.append(col)
    data['active_alerts'] = df[available].head(50) if available else pd.DataFrame()

    # Stop hotspots
    if 'stop_id' in df.columns and delay_col in df.columns:
        hotspots = df.groupby('stop_id')[delay_col].mean().sort_values(ascending=False).head(10)
        data['hotspots'] = hotspots
    elif 'latitude' in df.columns and 'longitude' in df.columns and delay_col in df.columns:
        df['grid_lat'] = (df['latitude'] * 100).round() / 100
        df['grid_lon'] = (df['longitude'] * 100).round() / 100
        hotspots = df.groupby(['grid_lat', 'grid_lon'])[delay_col].mean()
        data['hotspots'] = hotspots
    else:
        data['hotspots'] = pd.Series()

    # Model performance (if predictions available)
    if 'predicted_disruption' in df.columns and disruption_col and disruption_col in df.columns:
        actual = (df[disruption_col] != 'ON_TIME').astype(int)
        predicted = df['predicted_disruption']
        correct = (actual == predicted).sum()
        data['prediction_f1'] = correct / len(actual) if len(actual) > 0 else 0
    else:
        data['prediction_f1'] = 0.0

    # Attach DataFrame for panels
    data['df'] = df

    return data


def _generate_sample_data() -> Dict:
    """Generate sample data for demonstration."""
    from datetime import datetime

    hours = pd.date_range(end=datetime.now(), periods=24, freq='h')

    return {
        'active_disruptions': 12,
        'avg_delay': 8.2,
        'prediction_f1': 0.87,
        'timeline': pd.DataFrame({
            'hour': hours,
            'disruptions': np.random.poisson(15, 24)
        }),
        'alert_distribution': {
            'Technical': 35,
            'Weather': 25,
            'Construction': 20,
            'Strike': 10,
            'Other': 10
        },
        'route_performance': pd.DataFrame({
            'Route': [f'Route {i}' for i in 'ABCDE'],
            'On-Time %': [0.85, 0.72, 0.91, 0.68, 0.79]
        }),
        'active_alerts': pd.DataFrame({
            'Time': ['10:23', '09:45', '08:12', '07:30'],
            'Route': ['A12', 'B5', 'C3', 'R17'],
            'Type': ['Delay', 'Cancellation', 'Delay', 'Alert'],
            'Severity': ['Medium', 'High', 'Low', 'Medium'],
            'Expected Clear': ['11:30', 'TBD', '09:00', '10:15']
        }),
        'hotspots': pd.Series({
            f'Stop {i}': np.random.randint(5, 20) for i in range(1, 11)
        }),
        'df': pd.DataFrame()
    }