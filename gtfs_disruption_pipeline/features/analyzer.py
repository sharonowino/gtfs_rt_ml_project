"""
GTFS Disruption Detection - Disruption Analyzer
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DisruptionAnalyzer:
    """
    Higher-level analytics on top of the classified DataFrame.
    
    Methods
    -------
    schema()          Print column definitions for the feature DataFrame.
    hot_spots(n)      Top-N stops by avg disruption severity.
    time_profile()    Disruptions by hour of day.
    alert_breakdown() Frequency of each alert cause/effect pair.
    most_delayed(n)   Top-N most delayed trips.
    """
    
    FEATURE_SCHEMA = {
        # Identifiers
        'trip_id': 'Unique trip identifier (GTFS static)',
        'stop_id': 'Stop identifier',
        'stop_sequence': 'Order of this stop within the trip',
        'vehicle_id': 'Vehicle identifier from GTFS-RT',
        # Schedule vs Reality
        'scheduled_time_sec': 'Scheduled arrival (seconds since midnight)',
        'actual_time_sec': 'Actual arrival reported by GTFS-RT',
        'delay_sec': 'actual - scheduled (negative = early)',
        'delay_min': 'delay_sec / 60  (human-readable)',
        'delay_flag': '"late" | "early" | "on_time"  (raw signal)',
        'missing_schedule': 'True if no scheduled time was matched',
        # Vehicle kinematics
        'speed': 'Vehicle speed in km/h (GTFS-RT VehiclePosition)',
        'speed_flag': '"stopped" | "slow" | "normal"',
        'vehicle_status': 'IN_TRANSIT_TO | STOPPED_AT | …',
        'lat': 'Vehicle latitude',
        'lon': 'Vehicle longitude',
        'timestamp': 'Unix timestamp of the vehicle report',
        # Service alerts
        'has_alert': 'True if any alert active for this route',
        'alert_cause': 'GTFS-RT AlertCause  (ACCIDENT, STRIKE, …)',
        'alert_effect': 'GTFS-RT AlertEffect (DETOUR, NO_SERVICE, …)',
        'alert_text': 'Human-readable alert description',
        # Static GTFS enrichment
        'route_id': 'Route identifier',
        'route_short_name': 'Short public route name ("42", "M7", …)',
        'route_long_name': 'Full route name',
        'route_type': '0=tram 1=subway 2=rail 3=bus …',
        'direction_id': '0=outbound  1=inbound',
        'trip_headsign': 'Destination sign text',
        'stop_name': 'Human-readable stop name',
        'stop_lat': 'Stop latitude',
        'stop_lon': 'Stop longitude',
        'agency_name': 'Transit agency name',
        # Disruption labels
        'disruption_type': 'ON_TIME | MINOR_DELAY | MAJOR_DELAY | EARLY | '
                          'CANCELLED | STOPPED_ON_ROUTE | SLOW_TRAFFIC | SERVICE_ALERT',
        'severity_score': '0–10  (0=OK, 10=Cancelled/Critical)',
    }
    
    def __init__(self, classified_df: pd.DataFrame):
        self.df = classified_df
    
    def schema(self):
        """Print a readable description of every feature column."""
        logger.info("\n" + "="*70)
        logger.info("FEATURE SCHEMA — disruption detection DataFrame")
        logger.info("="*70)
        for col, desc in self.FEATURE_SCHEMA.items():
            flag = "✓" if col in self.df.columns else "✗"
            logger.info(f"  {flag}  {col:<26} {desc}")
    
    def hot_spots(self, top_n: int = 10) -> pd.DataFrame:
        """Top-N stops by average disruption severity."""
        grp = self.df.groupby('stop_id')
        agg = grp.agg(events=('stop_id', 'count'), avg_severity=('severity_score', 'mean'))
        if 'delay_min' in self.df.columns:
            agg['avg_delay_min'] = grp['delay_min'].mean()
        agg['top_disruption'] = grp['disruption_type'].agg(lambda x: x.value_counts().index[0])
        if 'stop_name' in self.df.columns:
            nm = self.df[['stop_id', 'stop_name']].drop_duplicates('stop_id').set_index('stop_id')['stop_name']
            agg['stop_name'] = agg.index.map(nm)
        return agg.sort_values('avg_severity', ascending=False).head(top_n).reset_index()
    
    def time_profile(self, timestamp_col: str = 'timestamp') -> Optional[pd.DataFrame]:
        """Disruption count by hour-of-day."""
        if timestamp_col not in self.df.columns:
            logger.warning(f"Column '{timestamp_col}' not found.")
            return None
        df = self.df.copy()
        df['hour'] = pd.to_datetime(df[timestamp_col], unit='s', utc=True).dt.hour
        df['is_disrupted'] = df['disruption_type'] != 'ON_TIME'
        profile = df.groupby('hour').agg(
            total_events=('hour', 'count'),
            disrupted_events=('is_disrupted', 'sum'),
        ).reset_index()
        profile['pct_disrupted'] = (profile['disrupted_events'] / profile['total_events'] * 100).round(1)
        return profile
    
    def alert_breakdown(self) -> pd.DataFrame:
        """Frequency table of alert_cause x alert_effect combinations."""
        cols = [c for c in ['alert_cause', 'alert_effect'] if c in self.df.columns]
        if not cols:
            return pd.DataFrame({'note': ['No alert columns found']})
        return (
            self.df[self.df['has_alert'] == True]
            .groupby(cols).size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )
    
    def most_delayed(self, top_n: int = 20) -> pd.DataFrame:
        """Top-N most delayed trips by mean delay."""
        if 'delay_min' not in self.df.columns:
            return pd.DataFrame()
        grp = self.df.groupby('trip_id')
        agg = grp.agg(avg_delay_min=('delay_min', 'mean'),
                      max_delay_min=('delay_min', 'max'),
                      stop_events=('trip_id', 'count'))
        agg['disruption_type_mode'] = grp['disruption_type'].agg(lambda x: x.value_counts().index[0])
        if 'route_short_name' in self.df.columns:
            nm = self.df[['trip_id', 'route_short_name']].drop_duplicates('trip_id').set_index('trip_id')
            agg['route_short_name'] = agg.index.map(nm['route_short_name'])
        return agg.sort_values('avg_delay_min', ascending=False).head(top_n).reset_index()
