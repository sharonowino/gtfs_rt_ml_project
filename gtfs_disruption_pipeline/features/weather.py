"""
Weather Integration Module
======================

Adds weather-based features for disruption prediction.

Weather is a major disruption cause, especially in Netherlands:
- Heavy rain/snow: reduced visibility, slower speeds
- High winds: bridge closures, tram/metro issues
- Extreme temperatures: track expansion issues, door problems
- Ice/snow: service cancellations

This module provides:
1. Weather feature extraction from various sources
2. Impact categories
3. Historical weather patterns

Usage:
------
from gtfs_disruption.features.weather import (
    WeatherFeatures,
    WeatherImpactScorer,
    load_weather_data
)

# If you have weather data
df = weather_fetcher.get_weather_for_transitArea(df, lat_col='stop_lat', lon_col='stop_lon')

# Or use static impact features
df = add_weather_impact_features(df)
"""
import logging
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Weather condition thresholds for Netherlands transit
WEATHER_THRESHOLDS = {
    # Temperature (Celsius)
    'temp_freezing': 0.0,
    'temp_hot': 30.0,
    'temp_cold': -5.0,
    
    # Wind (m/s)
    'wind_gale': 20.0,
    'wind_strong': 14.0,
    'wind_moderate': 8.0,
    
    # Precipitation (mm/h)
    'rain_heavy': 10.0,
    'rain_moderate': 4.0,
    'snow': 2.0,
    
    # Visibility (m)
    'visibility_low': 200,
    'visibility_poor': 1000,
    
    # humidity (%)
    'humidity_high': 90,
}


@dataclass
class WeatherCondition:
    """Weather condition at a point in time."""
    timestamp: pd.Timestamp
    temperature: float  # Celsius
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    precipitation: float  # mm/h
    humidity: float  # %
    visibility: float  # m
    cloud_cover: int  # 0-8
    pressure: float  # hPa
    
    # Dutch-specific
    weather_code: int  # KNMI code
    
    def is_freezing(self) -> bool:
        return self.temperature <= WEATHER_THRESHOLDS['temp_freezing']
    
    def is_hot(self) -> bool:
        return self.temperature >= WEATHER_THRESHOLDS['temp_hot']
    
    def is_windy(self) -> bool:
        return self.wind_speed >= WEATHER_THRESHOLDS['wind_strong']
    
    def is_rainy(self) -> bool:
        return self.precipitation >= WEATHER_THRESHOLDS['rain_moderate']
    
    def is_snowy(self) -> bool:
        return self.is_freezing() and self.precipitation > 0


@dataclass  
class WeatherImpact:
    """Weather impact on transit operations."""
    visibility_impact: float  # 0-1
    speed_impact: float  # 0-1
    reliability_impact: float  # 0-1
    capacity_impact: float  # 0-1
    
    total_impact: float  # 0-1
    
    @classmethod
    def from_condition(cls, cond: WeatherCondition) -> 'WeatherImpact':
        """Calculate impact from weather condition."""
        # Visibility impact
        vis = cond.visibility
        if vis < WEATHER_THRESHOLDS['visibility_low']:
            vis_impact = 1.0
        elif vis < WEATHER_THRESHOLDS['visibility_poor']:
            vis_impact = 0.5
        else:
            vis_impact = 0.0
        
        # Speed impact (wind)
        spd = cond.wind_speed
        if spd >= WEATHER_THRESHOLDS['wind_gale']:
            spd_impact = 1.0
        elif spd >= WEATHER_THRESHOLDS['wind_strong']:
            spd_impact = 0.7
        elif spd >= WEATHER_THRESHOLDS['wind_moderate']:
            spd_impact = 0.3
        else:
            spd_impact = 0.0
        
        # Reliability impact (precipitation + temperature)
        rel_impact = 0.0
        if cond.is_snowy():
            rel_impact = 1.0
        elif cond.is_freezing() and cond.is_rainy():
            rel_impact = 0.8
        elif cond.is_rainy():
            rel_impact = 0.4
        elif cond.is_freezing():
            rel_impact = 0.3
        
        # Capacity impact (crowding due to weather)
        cap_impact = 0.0
        if cond.is_rainy() or cond.is_snowy():
            cap_impact = 0.3  # More people in vehicles
        
        total = (vis_impact + spd_impact + rel_impact + cap_impact) / 4
        
        return cls(
            visibility_impact=vis_impact,
            speed_impact=spd_impact,
            reliability_impact=rel_impact,
            capacity_impact=cap_impact,
            total_impact=total
        )


class WeatherFeatures:
    """
    Weather feature extractor for transit disruption.
    
    Provides features based on weather conditions that affect
    transit operations in the Netherlands.
    """
    
    def __init__(self, timestamp_col: str = 'feed_timestamp'):
        self.timestamp_col = timestamp_col
        
        # Default weather thresholds
        self.thresholds = WEATHER_THRESHOLDS
    
    def compute_weather_features(self, df: pd.DataFrame,
                                weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute weather features.
        
        If weather_df provided, merge weather data.
        Otherwise, add placeholder features.
        """
        logger.info("Computing weather features...")
        
        out = df.copy()
        
        if weather_df is not None and not weather_df.empty:
            # Merge weather data
            out = self._merge_weather_data(out, weather_df)
        else:
            # Add placeholder features (indicate missing weather data)
            out = self._add_placeholder_features(out)
        
        # Compute impact features
        out = self._compute_impact_features(out)
        
        new_cols = [c for c in out.columns if c.startswith('weather_')]
        logger.info(f"  Added {len(new_cols)} weather features")
        
        return out
    
    def _merge_weather_data(self, df: pd.DataFrame,
                           weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data based on timestamp and location."""
        
        ts_col = self.timestamp_col
        
        # Ensure timestamps are datetime
        if ts_col not in df.columns:
            logger.warning(f"  Timestamp column {ts_col} not found")
            return df
        
        df[ts_col] = pd.to_datetime(df[ts_col])
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        
        # Simple merge on nearest timestamp (would use pd.merge_asof in production)
        df = df.merge(weather_df, on='timestamp', how='left')
        
        return df
    
    def _add_placeholder_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add placeholder features when weather data unavailable."""
        
        ts_col = self.timestamp_col
        
        if ts_col in df.columns:
            # Use time-based approximation
            # (e.g., assume worse weather in winter/rainy hours)
            dt = df[ts_col]
            
            if pd.api.types.is_datetime64_any_dtype(dt):
                # Month (winter = worse)
                month = dt.dt.month
                df['weather_is_winter'] = month.isin([12, 1, 2, 3]).astype(int)
                df['weather_is_summer'] = month.isin([6, 7, 8]).astype(int)
                
                # Hour of day
                hour = dt.dt.hour
                # Rush hours more susceptible
                df['weather_rush_hour'] = (
                    ((hour >= 7) & (hour <= 9)) |
                    ((hour >= 16) & (hour <= 19))
                ).astype(int)
                
                # Day of week
                dow = dt.dt.dayofweek
                df['weather_weekend'] = (dow >= 5).astype(int)
            else:
                df['weather_is_winter'] = 0
                df['weather_is_summer'] = 0
                df['weather_rush_hour'] = 0
                df['weather_weekend'] = 0
        else:
            df['weather_is_winter'] = 0
            df['weather_is_summer'] = 0
            df['weather_rush_hour'] = 0
            df['weather_weekend'] = 0
        
        # Placeholder for actual weather
        df['weather_temp'] = 10.0  # Default moderate
        df['weather_wind'] = 5.0
        df['weather_precip'] = 0.0
        df['weather_humidity'] = 75
        df['weather_visibility'] = 10000
        
        # Placeholder impacts
        df['weather_visibility_impact'] = 0.0
        df['weather_speed_impact'] = 0.0
        df['weather_reliability_impact'] = 0.0
        df['weather_capacity_impact'] = 0.0
        df['weather_total_impact'] = 0.0
        df['weather_has_data'] = 0
        
        return df
    
    def _compute_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute weather impact features."""
        
        # Temperature features
        if 'temperature' in df.columns:
            temp = df['temperature'].fillna(10.0)
            
            df['weather_temp_cold'] = (temp < self.thresholds['temp_freezing']).astype(int)
            df['weather_temp_hot'] = (temp > self.thresholds['temp_hot']).astype(int)
            df['weather_temp_freezing'] = (temp <= 0).astype(int)
        
        # Wind features
        if 'wind_speed' in df.columns:
            wind = df['wind_speed'].fillna(0)
            
            df['weather_wind_gale'] = (
                wind >= self.thresholds['wind_gale']
            ).astype(int)
            df['weather_wind_strong'] = (
                wind >= self.thresholds['wind_strong']
            ).astype(int)
            df['weather_wind_moderate'] = (
                wind >= self.thresholds['wind_moderate']
            ).astype(int)
        
        # Precipitation feature
        if 'precipitation' in df.columns:
            precip = df['precipitation'].fillna(0)
            
            df['weather_rain_heavy'] = (
                precip >= self.thresholds['rain_heavy']
            ).astype(int)
            df['weather_rain_moderate'] = (
                precip >= self.thresholds['rain_moderate']
            ).astype(int)
        
        # Visibility feature
        if 'visibility' in df.columns:
            vis = df['visibility'].fillna(10000)
            
            df['weather_visibility_low'] = (
                vis < self.thresholds['visibility_low']
            ).astype(int)
            df['weather_visibility_poor'] = (
                vis < self.thresholds['visibility_poor']
            ).astype(int)
        
        # Weather impact score (composite)
        impact_cols = [
            c for c in df.columns 
            if c.startswith('weather_') and c.endswith('_impact')
        ]
        
        if impact_cols:
            df['weather_total_impact'] = df[impact_cols].mean(axis=1)
            df['weather_is_disrupting'] = (
                df['weather_total_impact'] > 0.5
            ).astype(int)
        
        return df


class WeatherImpactScorer:
    """
    Calculate weather impact scores for transit operations.
    
    Uses rule-based scoring based on transit operational
    impact research.
    """
    
    # Impact weights by mode
    MODE_WEIGHTS = {
        'bus': {
            'visibility': 0.3,
            'speed': 0.4,
            'reliability': 0.2,
            'capacity': 0.1,
        },
        'tram': {
            'visibility': 0.2,
            'speed': 0.3,
            'reliability': 0.3,
            'capacity': 0.2,
        },
        'metro': {
            'visibility': 0.3,
            'speed': 0.2,
            'reliability': 0.3,
            'capacity': 0.2,
        },
        'train': {
            'visibility': 0.2,
            'speed': 0.2,
            'reliability': 0.4,
            'capacity': 0.2,
        },
    }
    
    def __init__(self, mode: str = 'bus'):
        self.mode = mode
        self.weights = self.MODE_WEIGHTS.get(mode, self.MODE_WEIGHTS['bus'])
    
    def score_weather(self, weather_row: pd.Series) -> float:
        """Score weather impact for a single observation."""
        
        impact = 0.0
        
        # Visibility impact
        vis = weather_row.get('visibility', 10000)
        if vis < WEATHER_THRESHOLDS['visibility_low']:
            impact += self.weights['visibility'] * 1.0
        elif vis < WEATHER_THRESHOLDS['visibility_poor']:
            impact += self.weights['visibility'] * 0.5
        
        # Speed impact (wind)
        wind = weather_row.get('wind_speed', 0)
        if wind >= WEATHER_THRESHOLDS['wind_gale']:
            impact += self.weights['speed'] * 1.0
        elif wind >= WEATHER_THRESHOLDS['wind_strong']:
            impact += self.weights['speed'] * 0.7
        elif wind >= WEATHER_THRESHOLDS['wind_moderate']:
            impact += self.weights['speed'] * 0.3
        
        # Reliability impact
        temp = weather_row.get('temperature', 10)
        precip = weather_row.get('precipitation', 0)
        
        if temp <= 0 and precip > 0:  # Freezing rain
            impact += self.weights['reliability'] * 1.0
        elif temp <= 0:  # Freezing
            impact += self.weights['reliability'] * 0.3
        elif precip > WEATHER_THRESHOLDS['rain_heavy']:
            impact += self.weights['reliability'] * 0.5
        elif precip > 0:
            impact += self.weights['reliability'] * 0.2
        
        return min(impact, 1.0)
    
    def score_column(self, df: pd.DataFrame) -> pd.Series:
        """Score impact for entire DataFrame."""
        
        return df.apply(self.score_weather, axis=1)


class DutchWeatherPatterns:
    """
    Dutch-specific weather pattern features.
    
    Netherlands weather characteristics:
    - Always windy
    - Frequent rain (especially west)
    - Sudden weather changes
    - Fog in winter mornings
    """
    
    # KNMI weather codes interpretation
    KNMI_CODES = {
        0: 'clear',
        1: 'partly_cloudy',
        2: 'cloudy',
        3: 'overcast',
        45: 'fog',
        48: 'rime_fog',
        51: 'drizzle',
        53: 'drizzle',
        55: 'drizzle',
        56: 'freezing_drizzle',
        61: 'rain',
        63: 'rain',
        65: 'rain',
        66: 'freezing_rain',
        71: 'snow',
        73: 'snow',
        75: 'snow',
        77: 'snow_grains',
        80: 'rain_showers',
        81: 'rain_showers',
        82: 'rain_showers',
        85: 'snow_showers',
        86: 'snow_showers',
        95: 'thunderstorm',
        96: 'thunderstorm_hail',
        99: 'thunderstorm_hail',
    }
    
    # Disruption risk by weather code
    DISRUPTION_RISK = {
        0: 0.0, 1: 0.0, 2: 0.0, 3: 0.1,
        45: 0.3, 48: 0.4,
        51: 0.2, 53: 0.2, 55: 0.3, 56: 0.7,
        61: 0.3, 63: 0.4, 65: 0.6, 66: 0.8,
        71: 0.6, 73: 0.7, 75: 0.8, 77: 0.5,
        80: 0.3, 81: 0.4, 82: 0.5,
        85: 0.7, 86: 0.8,
        95: 0.6, 96: 0.8, 99: 0.9,
    }
    
    def add_knmi_features(self, df: pd.DataFrame,
                        weather_code_col: str = 'weather_code') -> pd.DataFrame:
        """Add features based on KNMI weather codes."""
        
        if weather_code_col not in df.columns:
            return df
        
        code = df[weather_code_col].fillna(0).astype(int)
        
        # Risk level
        df['weather_disruption_risk'] = code.map(self.DISRUPTION_RISK).fillna(0)
        
        # Is any disrupting condition
        df['weather_is_disrupting_knmi'] = (
            df['weather_disruption_risk'] > 0.4
        ).astype(int)
        
        # Code category
        df['weather_code_category'] = code.apply(self._code_category)
        
        return df
    
    def _code_category(self, code: int) -> str:
        """Categorize KNMI code."""
        if code == 0:
            return 'clear'
        elif code <= 3:
            return 'cloudy'
        elif code <= 48:
            return 'fog'
        elif code <= 55:
            return 'drizzle'
        elif code <= 66:
            return 'rain'
        elif code <= 77:
            return 'snow'
        elif code <= 82:
            return 'showers'
        else:
            return 'storm'


def add_weather_features(
    df: pd.DataFrame,
    timestamp_col: str = 'feed_timestamp',
    weather_df: Optional[pd.DataFrame] = None,
    mode: str = 'bus'
) -> pd.DataFrame:
    """
    Convenience function to add weather features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input transit data
    timestamp_col : str
        Timestamp column
    weather_df : pd.DataFrame, optional
        Weather data with columns: timestamp, temperature, wind_speed, etc.
    mode : str
        Transit mode for impact weights ('bus', 'tram', 'metro', 'train')
    
    Returns
    -------
    pd.DataFrame with weather features
    """
    logger.info("Adding weather features...")
    
    extractor = WeatherFeatures(timestamp_col=timestamp_col)
    out = extractor.compute_weather_features(df, weather_df)
    
    # Add mode-specific scoring
    scorer = WeatherImpactScorer(mode=mode)
    out['weather_impact_score'] = scorer.score_column(out)
    
    # Add KNMI features if available
    if 'weather_code' in out.columns:
        knmi = DutchWeatherPatterns()
        out = knmi.add_knmi_features(out)
    
    return out


def create_mock_weather_data(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    freq: str = '1H'
) -> pd.DataFrame:
    """
    Create mock weather data for testing.
    
    Generates realistic Dutch weather patterns.
    """
    logger.info("Creating mock weather data...")
    
    # Generate timestamps
    timestamps = pd.date_range(start_time, end_time, freq=freq)
    n = len(timestamps)
    
    rng = np.random.default_rng(42)
    
    # Temperature: Dutch average ~10C, range -10 to 35
    temp = rng.normal(10, 8, n).clip(-10, 35)
    
    # Wind: Dutch average ~5 m/s
    wind = rng.exponential(5, n)
    
    # Precipitation: mostly 0, occasional rain
    precip = rng.exponential(0.5, n)
    precip = (precip > 2).astype(float) * rng.choice([0, 2, 5, 10], n, p=[0.7, 0.15, 0.1, 0.05])
    
    # Humidity: Dutch is humid, average ~80%
    humidity = rng.normal(80, 15, n).clip(30, 100)
    
    # Visibility: typically good unless fog
    visibility = rng.normal(15000, 5000, n).clip(50, 30000)
    
    # Cloud cover (0-8)
    cloud = rng.choice(range(9), n, p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02])
    
    # Pressure (hPa)
    pressure = rng.normal(1013, 15, n)
    
    # Weather codes (simplified)
    weather_code = np.zeros(n, dtype=int)
    weather_code[precip > 0] = rng.choice([51, 61, 63], n)
    weather_code[wind > 15] = 80
    weather_code[cloud < 2] = rng.choice([0, 1], n, p=[0.7, 0.3])
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temp,
        'wind_speed': wind,
        'wind_direction': rng.uniform(0, 360, n),
        'precipitation': precip,
        'humidity': humidity,
        'visibility': visibility,
        'cloud_cover': cloud,
        'pressure': pressure,
        'weather_code': weather_code,
    })


__all__ = [
    'WeatherFeatures',
    'WeatherImpactScorer',
    'DutchWeatherPatterns',
    'WeatherCondition',
    'WeatherImpact',
    'WEATHER_THRESHOLDS',
    'add_weather_features',
    'create_mock_weather_data',
]