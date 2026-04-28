"""
Comprehensive Transit Feature Engineering Module
============================================
Covers 5 feature families:
1. Temporal features
2. Geospatial and operational features  
3. Headway and propagation features (NEW)
4. Network features
5. NLP features from service alerts

This module provides features optimized for disruption detection
and 30-minute early warning prediction.

Author: Kilo Transit ML Team
"""

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    
    # Feature families to compute
    enable_temporal: bool = True
    enable_geospatial: bool = True
    enable_headway: bool = True
    enable_network: bool = True
    enable_nlp: bool = True
    
    # Headway parameters
    headway_window_minutes: int = 60
    propagation_window_minutes: int = 30
    min_stops_for_headway: int = 3
    
    # Network parameters
    compute_centrality: bool = True
    compute_pagerank: bool = False
    enable_graph_features: bool = True
    
    # NLP parameters
    enable_sentiment: bool = True
    enable_topic: bool = False
    enable_ner: bool = False
    
    # Temporal window sizes
    lag_windows: List[int] = field(default_factory=lambda: [5, 15, 30, 60])
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Geo parameters  
    geo_bins_km: float = 10.0
    buffer_radius_km: float = 5.0
    
    # Output
    output_prefix: str = ""


# =============================================================================
# 1. TEMPORAL FEATURES
# =============================================================================

class TemporalFeatureEngineer:
    """
    Creates temporal features from GTFS-RT timestamps.
    
    Features created:
    - Time-of-day cyclical encoding (sin/cos hour)
    - Day-of-week cyclical encoding
    - Time bucket (peak/off-peak/night)
    - Service period (weekday/weekend/holiday)
    - Trend features (delay acceleration)
    - Rolling statistics
    - Lag features
    
    Usage:
    ------
    engineer = TemporalFeatureEngineer(config)
    df = engineer.add_features(df)
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.cfg = config or FeatureConfig()
        
    def add_features(self, df: pd.DataFrame, 
                   timestamp_col: str = "feed_timestamp",
                   delay_col: str = "delay_sec") -> pd.DataFrame:
        """Add all temporal features to the DataFrame."""
        logger.info("Adding temporal features...")
        df = df.copy()
        
        # Ensure timestamp is datetime
        ts_col = timestamp_col if timestamp_col in df.columns else "timestamp"
        if ts_col not in df.columns:
            logger.warning(f"Timestamp column '{ts_col}' not found")
            return df
            
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        
        # Extract time components
        df["hour"] = df[ts_col].dt.hour
        df["day_of_week"] = df[ts_col].dt.dayofweek
        df["day_name"] = df[ts_col].dt.day_name()
        df["date"] = df[ts_col].dt.date
        df["month"] = df[ts_col].dt.month
        df["week_of_year"] = df[ts_col].dt.isocalendar().week
        
        # Cyclical encoding
        df = self._add_cyclical_features(df)
        
        # Time buckets
        df = self._add_time_buckets(df)
        
        # Lag features (if delay column exists)
        if delay_col in df.columns:
            df = self._add_lag_features(df, delay_col)
            df = self._add_rolling_features(df, delay_col)
            df = self._add_trend_features(df, delay_col)
        
        return df
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical sin/cos encoding for time features."""
        # Hour of day cyclical (24-hour cycle)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # Day of week cyclical (7-day cycle)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Month cyclical (12-month cycle)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
        
        # Week of year cyclical (~52 weeks)
        df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        
        return df
    
    def _add_time_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time period buckets."""
        hour = df["hour"]
        
        # Peak hours (morning 7-9, evening 16-19)
        df["is_peak"] = (((hour >= 7) & (hour <= 9)) | 
                        ((hour >= 16) & (hour <= 19))).astype(int)
        
        # Morning peak
        df["is_morning_peak"] = ((hour >= 7) & (hour <= 9)).astype(int)
        
        # Evening peak
        df["is_evening_peak"] = ((hour >= 16) & (hour <= 19)).astype(int)
        
        # Off-peak
        df["is_offpeak"] = ((hour >= 10) & (hour <= 15)).astype(int)
        
        # Night
        df["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)
        
        # Early morning
        df["is_early_morning"] = ((hour >= 5) & (hour < 7)).astype(int)
        
        # Late evening
        df["is_late_evening"] = ((hour >= 20) & (hour < 22)).astype(int)
        
        # Weekend
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Weekday
        df["is_weekday"] = (~df["day_of_week"].isin([5, 6])).astype(int)
        
        # Rush hour indicator (peak on weekday)
        df["is_rush_hour"] = (df["is_weekday"] * df["is_peak"]).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, 
                        delay_col: str = "delay_sec") -> pd.DataFrame:
        """Add lag features for delay."""
        if "trip_id" not in df.columns or "stop_id" not in df.columns:
            return df
            
        group = df.groupby(["trip_id", "stop_id"], sort=False)
        
        for window in self.cfg.lag_windows:
            col_name = f"delay_lag_{window}"
            df[col_name] = group[delay_col].shift(window)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame,
                          delay_col: str = "delay_sec") -> pd.DataFrame:
        """Add rolling window statistics."""
        if "trip_id" not in df.columns:
            return df
            
        group = df.groupby("trip_id", sort=False)
        
        for window in self.cfg.rolling_windows:
            # Mean
            df[f"delay_rolling_mean_{window}"] = group[delay_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            # Std
            df[f"delay_rolling_std_{window}"] = group[delay_col].transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
            # Max
            df[f"delay_rolling_max_{window}"] = group[delay_col].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
            # Min
            df[f"delay_rolling_min_{window}"] = group[delay_col].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame,
                            delay_col: str = "delay_sec") -> pd.DataFrame:
        """Add trend/acceleration features."""
        if "trip_id" not in df.columns:
            return df
            
        group = df.groupby("trip_id", sort=False)
        
        # First difference
        df["delay_diff"] = group[delay_col].diff()
        
        # Second difference (acceleration)
        df["delay_diff2"] = group["delay_diff"].diff()
        
        # Proportional change
        df["delay_pct_change"] = group[delay_col].pct_change().clip(-10, 10)
        
        return df


# =============================================================================
# 2. GEOSPATIAL AND OPERATIONAL FEATURES  
# =============================================================================

class GeospatialFeatureEngineer:
    """
    Creates geospatial and operational features.
    
    Features created:
    - Stop location features (lat/lon derived)
    - Distance to major hubs
    - Geographic clustering
    - Operational complexity
    - Stop sequence position
    
    Usage:
    ------
    geo_engineer = GeospatialFeatureEngineer(config)
    df = geo_engineer.add_features(df)
    """
    
    MAJOR_HUBS = {
        "AMS": {"lat": 52.3791, "lon": 4.9003, "name": "Amsterdam Centraal"},
        "UT": {"lat": 52.0890, "lon": 5.1093, "name": "Utrecht Centraal"},
        "RT": {"lat": 51.9249, "lon": 4.4689, "name": "Rotterdam Centraal"},
        "DH": {"lat": 52.0807, "lon": 4.3247, "name": "Den Haag Centraal"},
    }
    
    def __init__(self, config: FeatureConfig = None):
        self.cfg = config or FeatureConfig()
    
    def add_features(self, df: pd.DataFrame,
                    lat_col: str = "stop_lat",
                    lon_col: str = "stop_lon") -> pd.DataFrame:
        """Add all geospatial features."""
        logger.info("Adding geospatial features...")
        df = df.copy()
        
        # Location features
        df = self._add_location_features(df, lat_col, lon_col)
        
        # Distance to hubs
        df = self._add_hub_distance(df, lat_col, lon_col)
        
        # Geographic bins
        df = self._add_geo_bins(df, lat_col, lon_col)
        
        # Operational complexity
        df = self._add_operational_features(df)
        
        return df
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance in km."""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def _add_location_features(self, df: pd.DataFrame,
                            lat_col: str, lon_col: str) -> pd.DataFrame:
        """Add basic location features."""
        if lat_col not in df.columns or lon_col not in df.columns:
            return df
            
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        
        # Latitude bin (roughly 10km bins at NL latitudes)
        df["lat_bin"] = (lat * 10).round().astype("Int64")
        
        # Longitude bin
        df["lon_bin"] = (lon * 10).round().astype("Int64")
        
        # Combined geo grid cell
        df["geo_cell"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
        
        # North-South indicator
        df["is_north"] = (lat > 52.3).astype(int)
        
        # West-East indicator
        df["is_west"] = (lon < 5.0).astype(int)
        
        # Randstad indicator (Amsterdam-Den Haag-Rotterdam triangle)
        randstad_lat = (lat > 51.85) & (lat < 52.55)
        randstad_lon = (lon > 4.3) & (lon < 5.0)
        df["is_randstad"] = (randstad_lat & randstad_lon).astype(int)
        
        # Near coast indicator
        df["near_coast"] = ((lon < 4.6) | (lon > 6.8)).astype(int)
        
        return df
    
    def _add_hub_distance(self, df: pd.DataFrame,
                        lat_col: str, lon_col: str) -> pd.DataFrame:
        """Calculate distance to major hubs."""
        if lat_col not in df.columns:
            return df
            
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        
        # Distance to each major hub
        for hub_name, hub_info in self.MAJOR_HUBS.items():
            df[f"dist_to_{hub_name}_km"] = self._haversine(
                lat.values, lon.values,
                hub_info["lat"], hub_info["lon"]
            ).round(1)
        
        # Minimum distance to any hub
        hub_cols = [c for c in df.columns if c.startswith("dist_to_") and c.endswith("_km")]
        if hub_cols:
            df["min_dist_to_hub_km"] = df[hub_cols].min(axis=1)
            df["nearest_hub"] = df[hub_cols].idxmin(axis=1).str.replace("dist_to_", "").str.replace("_km", "")
        
        return df
    
    def _add_geo_bins(self, df: pd.DataFrame,
                     lat_col: str, lon_col: str) -> pd.DataFrame:
        """Add spatial aggregation bins."""
        if lat_col not in df.columns:
            return df
            
        # Multi-scale bins
        for scale in [5, 10, 20]:  # km
            df[f"geo_bin_{scale}km"] = (
                (pd.to_numeric(df[lat_col], errors="coerce") * (100/scale)).round() * (100/scale)
            ).astype("Int64").astype(str) + "_" + (
                (pd.to_numeric(df[lon_col], errors="coerce") * (100/scale)).round() * (100/scale)
            ).astype("Int64").astype(str)
        
        return df
    
    def _add_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add operational complexity features."""
        # Sequence position features
        if "stop_sequence" in df.columns:
            seq = pd.to_numeric(df["stop_sequence"], errors="coerce")
            
            # Position ratio (0=first stop, 1=last stop)
            if "trip_id" in df.columns:
                max_seq = df.groupby("trip_id")["stop_sequence"].transform("max")
                df["stop_sequence_ratio"] = (seq / max_seq.replace(0, np.nan)).fillna(0)
            
            # Is first stop
            df["is_first_stop"] = (seq == 1).astype(int)
            
            # Is last stop
            df["is_last_stop"] = df["stop_sequence"] == df.groupby("trip_id")["stop_sequence"].transform("max")
            df["is_last_stop"] = df["is_last_stop"].astype(int)
            
            # Middle stop
            df["is_middle_stop"] = ((seq > 1) & (df["is_last_stop"] == 0)).astype(int)
        
        return df


# =============================================================================
# 3. HEADWAY AND PROPAGATION FEATURES (NEW)
# =============================================================================

class HeadwayFeatureEngineer:
    """
    Creates headway and disruption propagation features.
    
    This is a critical addition for disruption detection as it captures:
    - Headway variability (unplanned gaps between vehicles)
    - Cascading delay propagation along routes
    - Knock-on effects to connecting services
    
    Features created:
    - Scheduled vs actual headway
    - Headway deviation / gap
    - Downstream delay propagation
    - Knock-on disruption indicator
    - Recovery time estimation
    
    Usage:
    ------
    hw_engineer = HeadwayFeatureEngineer(config)
    df = hw_engineer.add_features(df)
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.cfg = config or FeatureConfig()
    
    def add_features(self, df: pd.DataFrame,
                    timestamp_col: str = "feed_timestamp",
                    trip_col: str = "trip_id",
                    stop_col: str = "stop_id",
                    delay_col: str = "delay_sec") -> pd.DataFrame:
        """Add all headway features."""
        logger.info("Adding headway features...")
        df = df.copy()
        
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return df
            
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        
        # Sort by trip and time
        if trip_col in df.columns and stop_col in df.columns:
            sort_cols = [timestamp_col, trip_col]
            if stop_col in df.columns:
                sort_cols.append(stop_col)
            df = df.sort_values(sort_cols)
        
        # Headway features
        df = self._compute_headway(df, timestamp_col, trip_col)
        
        # Gap features
        df = self._compute_gaps(df, timestamp_col, trip_col)
        
        # Propagation features (requires delay_col)
        if delay_col in df.columns:
            df = self._compute_propagation(df, delay_col, trip_col, stop_col)
            df = self._compute_knockon(df, delay_col, trip_col, stop_col)
        
        return df
    
    def _compute_headway(self, df: pd.DataFrame,
                         ts_col: str, trip_col: str) -> pd.DataFrame:
        """Compute headway between consecutive trips at same stop."""
        if trip_col not in df.columns or "stop_id" not in df.columns:
            return df
            
        # For each stop, compute time gap between consecutive trips
        df = df.sort_values(["stop_id", ts_col])
        
        # Time since last trip (seconds)
        df["time_since_last_trip"] = df.groupby("stop_id")[ts_col].diff().dt.total_seconds()
        
        # Trips per hour at this stop (in rolling window)
        df["trips_per_hour"] = (
            3600 / df["time_since_last_trip"].replace(0, np.nan)
        ).fillna(0).clip(0, 30)
        
        # Is large gap (>15 minutes)
        df["has_large_gap"] = (
            df["time_since_last_trip"] > 15 * 60
        ).astype(int)
        
        # Gap severity (minutes)
        df["gap_severity_min"] = (
            df["time_since_last_trip"] / 60
        ).clip(0, 120).round(1)
        
        return df
    
    def _compute_gaps(self, df: pd.DataFrame,
                     ts_col: str, trip_col: str) -> pd.DataFrame:
        """Compute irregular gaps as early disruption indicator."""
        if trip_col not in df.columns:
            return df
            
        # Group by trip to compute within-trip statistics
        group = df.groupby(trip_col, sort=False)
        
        # Inter-stop travel time
        if "stop_sequence" in df.columns:
            # Sort by trip and sequence
            df = df.sort_values([trip_col, "stop_sequence"])
            
            # Travel time between stops
            df["inter_stop_time"] = df.groupby(trip_col)[ts_col].diff().dt.total_seconds()
            
            # Expected travel time based on schedule
            df["expected_inter_stop"] = df.groupby(trip_col)["inter_stop_time"].shift(-1)
            
            # Actual vs expected
            df["travel_time_diff"] = (
                df["inter_stop_time"] - df["expected_inter_stop"]
            ).fillna(0)
        
        return df
    
    def _compute_propagation(self, df: pd.DataFrame,
                            delay_col: str, trip_col: str, 
                            stop_col: str) -> pd.DataFrame:
        """Compute delay propagation along route direction."""
        if stop_col not in df.columns:
            return df
            
        # Sort by trip and stop sequence
        df = df.sort_values([trip_col, "stop_sequence"])
        
        # Downstream delay (delay at downstream stops)
        df["delay_downstream_mean"] = df.groupby(trip_col)[delay_col].transform(
            lambda x: x.shift(-1).rolling(3, min_periods=1).mean()
        )
        
        # Delay trend along route
        df["delay_propagation_trend"] = (
            df[delay_col] - df["delay_downstream_mean"]
        )
        
        # Is delay increasing downstream
        df["is_propagating"] = (
            df["delay_propagation_trend"] > 60  # 1+ minute increase downstream
        ).astype(int)
        
        # Delay at next stop
        df["delay_next_stop"] = df.groupby(trip_col)[delay_col].shift(-1)
        
        # Delay at previous stop  
        df["delay_prev_stop"] = df.groupby(trip_col)[delay_col].shift(1)
        
        # Delay change between stops
        df["delay_stop_to_stop"] = df[delay_col] - df["delay_prev_stop"]
        
        return df
    
    def _compute_knockon(self, df: pd.DataFrame,
                      delay_col: str, trip_col: str,
                      stop_col: str) -> pd.DataFrame:
        """Compute knock-on effects to connecting services.
        
        Uses STRICT backward-looking temporal windows to prevent leakage.
        Each stop's statistics are computed from PAST observations only.
        """
        if delay_col not in df.columns or stop_col not in df.columns:
            return df
        
        # Ensure data is sorted by time
        ts_col = "feed_timestamp" if "feed_timestamp" in df.columns else "timestamp"
        if ts_col in df.columns:
            df = df.sort_values([stop_col, ts_col])
        
        # Group by stop to find connecting services
        group = df.groupby(stop_col, sort=False)
        
        # CRITICAL FIX: Use shift(1) to exclude current observation
        # Mean delay at this stop from PAST observations only
        df["_delay_shifted"] = group[delay_col].shift(1)
        df["stop_mean_delay"] = group["_delay_shifted"].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        ).fillna(0)
        
        # Delay variance at this stop from PAST observations only
        df["stop_delay_variance"] = group["_delay_shifted"].transform(
            lambda x: x.rolling(10, min_periods=1).var()
        ).fillna(0)
        
        # Number of delayed trips at this stop (from PAST only)
        df["_delay_shifted_binary"] = (group[delay_col].shift(1) > 60).astype(int)
        df["stop_delayed_trips"] = group["_delay_shifted_binary"].transform(
            lambda x: x.rolling(10, min_periods=1).sum()
        )
        
        # Delay threshold indicator (current observation only - no leakage)
        df["stop_critical_delay"] = (
            df[delay_col] > 300  # >5 minutes
        ).astype(int)
        
        # Recovery time estimate (based on historical backlog, not current)
        df["estimated_recovery_min"] = (
            df["stop_mean_delay"] / 60 * (1 + df["stop_delayed_trips"] / 10)
        ).clip(0, 240).round(1)
        
        # Clean up temporary columns
        temp_cols = ["_delay_shifted", "_delay_shifted_binary"]
        df.drop(columns=[c for c in temp_cols if c in df.columns], inplace=True, errors="ignore")
        
        return df


# =============================================================================
# 4. NETWORK FEATURES  
# =============================================================================

class NetworkFeatureEngineer:
    """
    Creates network topology features from GTFS graph structure.
    
    Features created:
    - Stop connectivity degree
    - Route coverage
    - Transfer complexity
    - PageRank (optional)
    - Betweenness proxies
    - Critical stop indicators
    
    Usage:
    ------
    net_engineer = NetworkFeatureEngineer(config)
    df = net_engineer.add_features(df, gtfs_data=gtfs_dict)
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.cfg = config or FeatureConfig()
        self._stop_route_map = None
        self._stop_degree = None
        self._transfer_stops = None
    
    def add_features(self, df: pd.DataFrame,
                   gtfs_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Add all network features."""
        logger.info("Adding network features...")
        df = df.copy()
        
        if gtfs_data is not None:
            df = self._add_connectivity_features(df, gtfs_data)
            df = self._add_transfer_features(df, gtfs_data)
            df = self._add_route_diversity(df, gtfs_data)
        
        # Fallback features when no GTFS data available
        df = self._add_fallback_features(df)
        
        return df
    
    def _compute_stop_route_map(self, gtfs_data: Dict[str, pd.DataFrame]):
        """Build stop-route mapping from GTFS."""
        if self._stop_route_map is not None:
            return
            
        trips = gtfs_data.get("trips", pd.DataFrame())
        stop_times = gtfs_data.get("stop_times", pd.DataFrame())
        
        if trips.empty or stop_times.empty:
            self._stop_route_map = pd.DataFrame()
            return
            
        # Map trip_id to route_id
        trip_route = trips[["trip_id", "route_id"]].drop_duplicates()
        trip_route["trip_id"] = trip_route["trip_id"].astype(str)
        trip_route["route_id"] = trip_route["route_id"].astype(str)
        
        # Stop-route mapping
        sr = stop_times[["trip_id", "stop_id"]].drop_duplicates()
        sr["stop_id"] = sr["stop_id"].astype(str)
        sr = sr.merge(trip_route, on="trip_id", how="left")
        
        self._stop_route_map = sr[["stop_id", "route_id"]].drop_duplicates()
        
        # Stop degree
        self._stop_degree = self._stop_route_map.groupby("stop_id")["route_id"].nunique()
        self._stop_degree.name = "stop_route_degree"
        
        # Route degree
        self._route_degree = self._stop_route_map.groupby("route_id")["stop_id"].nunique()
        self._route_degree.name = "route_stop_count"
        
    def _add_connectivity_features(self, df: pd.DataFrame,
                              gtfs_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add connectivity-based features."""
        self._compute_stop_route_map(gtfs_data)
        
        if self._stop_degree is not None and "stop_id" in df.columns:
            df["stop_id"] = df["stop_id"].astype(str)
            df["stop_route_degree"] = df["stop_id"].map(self._stop_degree).fillna(1)
            
            # Hub indicator (high connectivity)
            df["is_hub_stop"] = (df["stop_route_degree"] >= 5).astype(int)
            
            # Major hub
            df["is_major_hub"] = (df["stop_route_degree"] >= 10).astype(int)
        
        if self._route_degree is not None and "route_id" in df.columns:
            df["route_id"] = df["route_id"].astype(str)
            df["route_stop_count"] = df["route_id"].map(self._route_degree).fillna(1)
        
        return df
    
    def _add_transfer_features(self, df: pd.DataFrame,
                         gtfs_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add transfer-related features."""
        transfers = gtfs_data.get("transfers", pd.DataFrame())
        
        if transfers.empty:
            return df
            
        # Get transfer stops
        from_stops = set(transfers["from_stop_id"].dropna().astype(str))
        to_stops = set(transfers["to_stop_id"].dropna().astype(str))
        transfer_stops = from_stops | to_stops
        
        if "stop_id" in df.columns:
            df["stop_id"] = df["stop_id"].astype(str)
            df["is_transfer_stop"] = df["stop_id"].isin(transfer_stops).astype(int)
            
            # Number of transfer connections
            transfer_count = (
                transfers["from_stop_id"].astype(str).value_counts() +
                transfers["to_stop_id"].astype(str).value_counts()
            )
            df["transfer_connection_count"] = df["stop_id"].map(
                transfer_count.to_dict()
            ).fillna(0).astype(int)
        
        return df
    
    def _add_route_diversity(self, df: pd.DataFrame,
                        gtfs_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add route diversity features."""
        if "route_id" not in df.columns:
            return df
            
        routes = gtfs_data.get("routes", pd.DataFrame())
        
        if routes.empty or "route_type" not in routes.columns:
            return df
            
        route_types = routes[["route_id", "route_type"]].drop_duplicates()
        route_types["route_id"] = route_types["route_id"].astype(str)
        route_types["route_type"] = pd.to_numeric(
            route_types["route_type"], errors="coerce"
        ).fillna(-1).astype(int)
        
        df = df.merge(
            route_types[["route_id", "route_type"]],
            on="route_id", how="left"
        )
        
        # Transport mode indicators
        df["is_bus"] = (df["route_type"] == 3).astype(int)
        df["is_train"] = (df["route_type"] == 2).astype(int)
        df["is_tram"] = (df["route_type"] == 0).astype(int)
        df["is_metro"] = (df["route_type"] == 1).astype(int)
        df["is_ferry"] = (df["route_type"] == 4).astype(int)
        
        return df
    
    def _add_fallback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fallback features when no GTFS data available."""
        if "stop_id" in df.columns:
            if "stop_route_degree" not in df.columns:
                df["stop_route_degree"] = 1
            if "is_hub_stop" not in df.columns:
                df["is_hub_stop"] = 0
            if "is_transfer_stop" not in df.columns:
                df["is_transfer_stop"] = 0
        
        if "route_id" in df.columns:
            if "route_stop_count" not in df.columns:
                df["route_stop_count"] = 1
        
        # Transport mode defaults
        if "route_type" not in df.columns:
            df["route_type"] = 3  # Default to bus
            df["is_bus"] = 1
            df["is_train"] = 0
            df["is_tram"] = 0
            df["is_metro"] = 0
            df["is_ferry"] = 0
        
        return df


# =============================================================================
# 5. NLP FEATURES FROM SERVICE ALERTS
# =============================================================================

class NLPAlertFeatureEngineer:
    """
    Extracts features from GTFS service alert text.
    
    Uses alert_nlp.py pipeline for core NLP processing, then adds
    derived features optimized for disruption detection.
    
    Features created:
    - Language detection (language_code, language_confidence)
    - Sentiment urgency (sentiment, sentiment_score)
    - Named entities (all_entities, loc_entities, entity_counts)
    - Text complexity (length, word count, sentences)
    - Alert severity proxies (keyword-based urgency)
    - Duration features (if alert timing available)
    - Topic labels (if topic modeling enabled)
    
    Usage:
    ------
    nlp_engineer = NLPAlertFeatureEngineer(config)
    df = nlp_engineer.add_features(df)
    """
    
    # Keywords indicating severity - from GTFS-RT domain
    SEVERE_KEYWORDS = [
        "emergency", "critical", "serious", "fatal", "accident",
        "blocked", "closed", "canceled", "cancelled", "not running", 
        "out of service", "no service", "severe", "danger"
    ]
    
    MODERATE_KEYWORDS = [
        "delayed", "longer", "reduced", "fewer", "slower",
        "disruption", "problem", "issue", "warning", "alter"
    ]
    
    URGENCY_KEYWORDS = [
        "immediately", "urgent", "asap", "now", "instant"
    ]
    
    # Dutch transit-specific keywords
    TRANSIT_KEYWORDS = {
        "tram": ["tram", "tramlijn", "sneltram"],
        "bus": ["bus", "buslijn", "lijnbus"],
        "train": ["trein", "treinlijn", "ns", "rail"],
        "metro": ["metro", "metrolijn"],
        "ferry": ["veer", "veerboot", " ferry"]
    }
    
    def __init__(self, config: FeatureConfig = None):
        self.cfg = config or FeatureConfig()
        self._nlp_enricher = None
    
    def _get_nlp_enricher(self):
        """Lazy load the NLP enricher from alert_nlp.py"""
        if self._nlp_enricher is None:
            try:
                from gtfs_disruption.features.alert_nlp import AlertNLPEnricher
                self._nlp_enricher = AlertNLPEnricher(
                    enable_language_detection=True,
                    enable_ner=self.cfg.enable_ner,
                    enable_sentiment=self.cfg.enable_sentiment,
                    enable_topic=self.cfg.enable_topic,
                )
            except Exception as e:
                logger.warning(f"Failed to load AlertNLPEnricher: {e}")
        return self._nlp_enricher
    
    def add_features(self, df: pd.DataFrame,
                   text_col: str = "combined_text") -> pd.DataFrame:
        """Add all NLP alert features."""
        logger.info("Adding NLP alert features...")
        df = df.copy()
        
        # Check for text column
        text_columns = [text_col, 'description_text', 'header_text', 'alert_text', 'text']
        available_text_col = next((c for c in text_columns if c in df.columns), None)
        
        if available_text_col is None:
            logger.warning("No text column found for NLP features")
            return df
        
        # Run core NLP pipeline from alert_nlp.py
        enricher = self._get_nlp_enricher()
        if enricher is not None:
            try:
                df = enricher.enrich(df)
                logger.info("  Core NLP pipeline applied successfully")
            except Exception as e:
                logger.warning(f"  NLP pipeline failed: {e}")
        
        # Sentiment-based features (from pipeline output or derive)
        df = self._add_sentiment_derived_features(df)
        
        # Entity-based features
        df = self._add_entity_features(df, available_text_col)
        
        # Text complexity features
        df = self._add_text_complexity(df, available_text_col)
        
        # Keyword-based severity
        df = self._add_keyword_severity(df, available_text_col)
        
        # Transit-specific features
        df = self._add_transit_features(df, available_text_col)
        
        # Alert cause/effect encoding
        df = self._add_cause_effect_features(df)
        
        # Duration features (if available)
        df = self._add_duration_features(df)
        
        return df
    
    def _add_sentiment_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived sentiment features for urgency detection."""
        # Sentiment indicator
        if "sentiment" in df.columns:
            # Binary urgency: negative = potentially disruptive
            df["alert_negative"] = (df["sentiment"] == "Negative").astype(int)
            df["alert_positive"] = (df["sentiment"] == "Positive").astype(int)
            df["alert_neutral"] = (df["sentiment"] == "Neutral").astype(int)
            
            # Confidence-based urgency
            if "score" in df.columns:
                # High confidence negative = high urgency
                df["sentiment_urgent"] = (
                    (df["sentiment"] == "Negative") & (df["score"] > 0.7)
                ).astype(int)
                
                # Confidence level categories
                df["confidence_level"] = pd.cut(
                    df["score"].fillna(0),
                    bins=[0, 0.5, 0.7, 0.85, 1.0],
                    labels=["low", "medium", "high", "very_high"]
                )
                
                # Weighted urgency score
                df["urgency_score"] = df["score"].fillna(0) * df["alert_negative"]
        
        return df
    
    def _add_entity_features(self, df: pd.DataFrame,
                           text_col: str) -> pd.DataFrame:
        """Add entity count/density features."""
        # Entity count from NER
        if "all_entities" in df.columns:
            df["entity_count"] = df["all_entities"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            # Location entity count (most relevant for transit)
            df["location_entity_count"] = df["all_entities"].apply(
                lambda x: sum(1 for e in x if e.get("entity_type") == "LOC")
                if isinstance(x, list) else 0
            )
            
            # Organization entity count
            df["org_entity_count"] = df["all_entities"].apply(
                lambda x: sum(1 for e in x if e.get("entity_type") == "ORG")
                if isinstance(x, list) else 0
            )
            
            # Person entity count
            df["person_entity_count"] = df["all_entities"].apply(
                lambda x: sum(1 for e in x if e.get("entity_type") == "PER")
                if isinstance(x, list) else 0
            )
            
            # Has location entity (potentially actionable)
            df["has_location_entity"] = (df["location_entity_count"] > 0).astype(int)
            
            # Entity density (per 100 chars)
            text_lens = df[text_col].fillna("").str.len()
            df["entity_density"] = (
                df["entity_count"] / (text_lens / 100 + 1)
            ).round(3)
        
        # Geocoded location features
        if "first_lat" in df.columns and "first_lon" in df.columns:
            df["has_geocoded_location"] = (
                df["first_lat"].notna() & df["first_lon"].notna()
            ).astype(int)
            
            # Is in Netherlands bounds
            df["in_netherlands"] = (
                (df["first_lat"].between(50.75, 53.55)) &
                (df["first_lon"].between(3.30, 7.20))
            ).astype(int)
        
        return df
    
    def _add_text_complexity(self, df: pd.DataFrame,
                         text_col: str) -> pd.DataFrame:
        """Add text complexity features."""
        if text_col not in df.columns:
            return df
            
        text = df[text_col].fillna("")
        
        # Length features
        df["text_length"] = text.str.len()
        df["word_count"] = text.str.split().str.len().fillna(0)
        
        # Average word length
        df["avg_word_length"] = (
            text.str.len() / (df["word_count"] + 1)
        ).round(2)
        
        # Sentence count (rough)
        df["sentence_count"] = text.str.count(r"[.!?]").fillna(0) + 1
        
        # Average sentence length
        df["avg_sentence_length"] = (
            df["word_count"] / (df["sentence_count"] + 1)
        ).round(2)
        
        # Uppercase ratio (shouting/emphasis indicator)
        df["upper_ratio"] = (
            text.str.count(r"[A-Z]") / (df["text_length"] + 1)
        ).round(3)
        
        # Has all-caps words (emphasis)
        df["has_emphasis"] = (
            text.str.count(r"\b[A-Z]{2,}\b") > 0
        ).astype(int)
        
        # Number count (times, dates, amounts)
        df["number_count"] = text.str.count(r"\d")
        
        # Has time reference
        df["has_time_ref"] = (
            text.str.contains(r"\d{1,2}:\d{2}|am|pm|uur", case=False, na=False)
        ).astype(int)
        
        # Has location reference
        df["has_location_ref"] = (
            text.str.contains(r"station|stop|halte|plein|straat", case=False, na=False)
        ).astype(int)
        
        return df
    
    def _add_keyword_severity(self, df: pd.DataFrame,
                          text_col: str) -> pd.DataFrame:
        """Add keyword-based severity indicators."""
        if text_col not in df.columns:
            return df
            
        text_lower = df[text_col].fillna("").str.lower()
        
        # Severe keyword count (critical disruptions)
        df["severe_keyword_count"] = sum(
            text_lower.str.count(kw) for kw in self.SEVERE_KEYWORDS
        )
        
        # Moderate keyword count (moderate disruptions)
        df["moderate_keyword_count"] = sum(
            text_lower.str.count(kw) for kw in self.MODERATE_KEYWORDS
        )
        
        # Urgency keyword count
        df["urgency_keyword_count"] = sum(
            text_lower.str.count(kw) for kw in self.URGENCY_KEYWORDS
        )
        
        # Has severe keyword
        df["has_severe_keyword"] = (df["severe_keyword_count"] > 0).astype(int)
        
        # Has urgency keyword
        df["has_urgency_keyword"] = (df["urgency_keyword_count"] > 0).astype(int)
        
        # Composite keyword severity score
        df["keyword_severity"] = (
            df["severe_keyword_count"] * 3 + 
            df["moderate_keyword_count"] * 2 +
            df["urgency_keyword_count"] * 2
        ).clip(0, 30)
        
        # Severity level categories
        df["severity_level"] = pd.cut(
            df["keyword_severity"],
            bins=[-1, 0, 2, 5, 30],
            labels=["none", "low", "medium", "high"]
        )
        
        return df
    
    def _add_transit_features(self, df: pd.DataFrame,
                           text_col: str) -> pd.DataFrame:
        """Add transit-specific features from text."""
        if text_col not in df.columns:
            return df
            
        text_lower = df[text_col].fillna("").str.lower()
        
        # Transport mode detection
        for mode, keywords in self.TRANSIT_KEYWORDS.items():
            df[f"mentions_{mode}"] = (
                text_lower.str.contains("|".join(keywords), na=False)
            ).astype(int)
        
        # Combined transport mode count
        mode_cols = [c for c in df.columns if c.startswith("mentions_")]
        if mode_cols:
            df["transport_mode_count"] = df[mode_cols].sum(axis=1)
        
        return df
    
    def _add_cause_effect_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add encoded cause/effect features from GTFS-RT alerts."""
        # Cause encoding
        cause_map = {
            'UNKNOWN_CAUSE': 0, 'OTHER_CAUSE': 1, 'TECHNICAL_PROBLEM': 2,
            'STRIKE': 3, 'DEMONSTRATION': 4, 'ACCIDENT': 5,
            'HOLIDAY': 6, 'WEATHER': 7, 'MAINTENANCE': 8,
            'CONSTRUCTION': 9, 'POLICE_ACTIVITY': 10, 'MEDICAL_EMERGENCY': 11
        }
        
        if "cause" in df.columns:
            df["cause_encoded"] = df["cause"].map(cause_map).fillna(-1).astype(int)
        
        # Effect encoding
        effect_map = {
            'NO_SERVICE': 0, 'REDUCED_SERVICE': 1, 'SIGNIFICANT_DELAYS': 2,
            'DETOUR': 3, 'ADDITIONAL_SERVICE': 4, 'MODIFIED_SERVICE': 5,
            'OTHER_EFFECT': 6, 'UNKNOWN_EFFECT': 7, 'STOP_MOVED': 8
        }
        
        if "effect" in df.columns:
            df["effect_encoded"] = df["effect"].map(effect_map).fillna(-1).astype(int)
        
        # Composite severity from cause + effect
        if "cause_encoded" in df.columns and "effect_encoded" in df.columns:
            df["alert_severity_composite"] = (
                df["cause_encoded"].abs() + df["effect_encoded"].abs()
            )
        
        return df
    
    def _add_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add duration-aware features from alert timing."""
        # Find timestamp column
        ts_col = next((c for c in ['feed_timestamp', 'timestamp', 'id_time'] 
                      if c in df.columns), None)
        
        # Find start/end columns
        start_col = next((c for c in ['active_period_start', 'alert_start'] 
                         if c in df.columns), None)
        end_col = next((c for c in ['active_period_end', 'alert_end'] 
                       if c in df.columns), None)
        
        if ts_col and start_col:
            start = pd.to_datetime(df[start_col], errors='coerce')
            current = pd.to_datetime(df[ts_col], errors='coerce')
            
            # Alert age
            df["alert_age_seconds"] = (current - start).dt.total_seconds()
            df["alert_age_minutes"] = df["alert_age_seconds"] / 60
            df["alert_age_hours"] = df["alert_age_seconds"] / 3600
            
            # Is new alert (< 15 min)
            df["alert_is_new"] = (df["alert_age_minutes"] <= 15).astype(int)
            
            # Is aging (> 1 hour)
            df["alert_is_aging"] = (df["alert_age_hours"] > 1).astype(int)
        
        if ts_col and end_col:
            end = pd.to_datetime(df[end_col], errors='coerce')
            current = pd.to_datetime(df[ts_col], errors='coerce')
            
            # Remaining time
            df["alert_remaining_seconds"] = (end - current).dt.total_seconds()
            df["alert_remaining_minutes"] = df["alert_remaining_seconds"] / 60
            df["alert_remaining_hours"] = df["alert_remaining_seconds"] / 3600
            
            # Is expired
            df["alert_is_expired"] = (df["alert_remaining_seconds"] <= 0).astype(int)
            
            # Is expiring soon (< 30 min)
            df["alert_expiring_soon"] = (
                (df["alert_remaining_seconds"] > 0) & 
                (df["alert_remaining_seconds"] <= 1800)
            ).astype(int)
        
        if start_col and end_col:
            start = pd.to_datetime(df[start_col], errors='coerce')
            end = pd.to_datetime(df[end_col], errors='coerce')
            
            # Total duration
            df["alert_duration_seconds"] = (end - start).dt.total_seconds()
            df["alert_duration_minutes"] = df["alert_duration_seconds"] / 60
            df["alert_duration_hours"] = df["alert_duration_seconds"] / 3600
            df["alert_duration_days"] = df["alert_duration_seconds"] / 86400
            
            # Duration category
            df["duration_category"] = pd.cut(
                df["alert_duration_minutes"].fillna(0),
                bins=[-1, 30, 120, 480, float('inf')],
                labels=["short", "medium", "long", "extended"]
            )
        
        return df


# =============================================================================
# UNIFIED PIPELINE
# =============================================================================

class ComprehensiveFeatureEngineer:
    """
    Unified feature engineering pipeline combining all 5 feature families.
    
    Usage:
    ------
    engineer = ComprehensiveFeatureEngineer(config)
    df = engineer.fit_transform(df, gtfs_data=gtfs_dict)
    
    Or with default config:
    ------
    engineer = ComprehensiveFeatureEngineer()
    df = engineer.fit_transform(df)
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.cfg = config or FeatureConfig()
        
        # Initialize all feature engineers
        self.temporal = TemporalFeatureEngineer(self.cfg)
        self.geospatial = GeospatialFeatureEngineer(self.cfg)
        self.headway = HeadwayFeatureEngineer(self.cfg)
        self.network = NetworkFeatureEngineer(self.cfg)
        self.nlp = NLPAlertFeatureEngineer(self.cfg)
    
    def fit_transform(self, df: pd.DataFrame,
                    gtfs_data: Dict[str, pd.DataFrame] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with GTFS-RT merged data
        gtfs_data : dict, optional
            Static GTFS data dictionary
        **kwargs : keyword arguments
            Additional parameters passed to individual engineers
            (e.g., timestamp_col, lat_col, lon_col)
        
        Returns:
        --------
        pd.DataFrame : Enriched DataFrame
        """
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        original_cols = set(df.columns)
        
        # 1. Temporal features
        if self.cfg.enable_temporal:
            df = self.temporal.add_features(df, **kwargs)
            logger.info(f"  Temporal features added: {len(set(df.columns) - len(original_cols))} new columns")
        
        # 2. Geospatial features  
        if self.cfg.enable_geospatial:
            df = self.geospatial.add_features(df, **kwargs)
            logger.info(f"  Geospatial features: {len(set(df.columns) - len(original_cols))} new columns")
        
        # 3. Headway features
        if self.cfg.enable_headway:
            df = self.headway.add_features(df, **kwargs)
            logger.info(f"  Headway features: {len(set(df.columns) - len(original_cols))} new columns")
        
        # 4. Network features
        if self.cfg.enable_network:
            df = self.network.add_features(df, gtfs_data=gtfs_data)
            logger.info(f"  Network features: {len(set(df.columns) - len(original_cols))} new columns")
        
        # 5. NLP features
        if self.cfg.enable_nlp:
            df = self.nlp.add_features(df, **kwargs)
            logger.info(f"  NLP features: {len(set(df.columns) - len(original_cols))} new columns")
        
        new_cols = set(df.columns) - original_cols
        logger.info(f"\n  Total new features: {len(new_cols)}")
        logger.info(f"  Final columns: {len(df.columns)}")
        
        return df


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def add_comprehensive_features(
    df: pd.DataFrame,
    gtfs_data: Dict[str, pd.DataFrame] = None,
    config: FeatureConfig = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to add all comprehensive features.
    
    Usage:
    ------
    df = add_comprehensive_features(
        merged_df,
        gtfs_data=gtfs_dict,
        timestamp_col="feed_timestamp",
        delay_col="delay_sec"
    )
    """
    engineer = ComprehensiveFeatureEngineer(config)
    return engineer.fit_transform(df, gtfs_data=gtfs_data, **kwargs)