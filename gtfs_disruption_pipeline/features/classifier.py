"""
GTFS Disruption Detection - Disruption Classifier
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DisruptionClassifier:
    """
    Assigns each stop event a disruption label and severity score.
    
    Classification logic (priority order):
    ┌─────────────────────────┬─────────────────────────────────────────────┐
    │ disruption_type         │ Condition                                   │
    ├─────────────────────────┼─────────────────────────────────────────────┤
    │ CANCELLED               │ vehicle_status='CANCELED' OR alert NO_SVC   │
    │ MAJOR_DELAY             │ delay_sec > 600 (10 min)                    │
    │ STOPPED_ON_ROUTE        │ speed<=2 km/h AND delay_sec > 120           │
    │ MINOR_DELAY             │ delay_sec > 120 (2 min)                     │
    │ SLOW_TRAFFIC            │ speed_flag == 'slow'                        │
    │ EARLY                   │ delay_sec < -60 (1 min early)               │
    │ SERVICE_ALERT           │ has_alert=True (no other signal)            │
    │ ON_TIME                 │ everything else                             │
    └─────────────────────────┴─────────────────────────────────────────────┘
    
    Severity score (0-10):
        10 → CANCELLED
         7 → MAJOR_DELAY
         5 → MINOR_DELAY / STOPPED_ON_ROUTE
         3 → SLOW_TRAFFIC / EARLY / SERVICE_ALERT
         0 → ON_TIME
    """
    
    DISRUPTION_SEVERITY = {
        'CANCELLED': 10, 'MAJOR_DELAY': 7, 'STOPPED_ON_ROUTE': 5,
        'MINOR_DELAY': 5, 'SLOW_TRAFFIC': 3, 'EARLY': 3,
        'SERVICE_ALERT': 3, 'ON_TIME': 0,
    }
    
    def __init__(self, delay_major_sec: int = 600, delay_minor_sec: int = 120,
                 delay_early_sec: int = -60, speed_stopped_kmh: float = 2.0,
                 speed_slow_kmh: float = 10.0):
        self.delay_major_sec = delay_major_sec
        self.delay_minor_sec = delay_minor_sec
        self.delay_early_sec = delay_early_sec
        self.speed_stopped_kmh = speed_stopped_kmh
        self.speed_slow_kmh = speed_slow_kmh
    
    def _is_cancelled(self, row: pd.Series) -> bool:
        """Check if trip is cancelled."""
        if str(row.get('vehicle_status', '')).upper() == 'CANCELED':
            return True
        if 'NO_SERVICE' in str(row.get('alert_effect', '')).upper():
            return True
        return False
    
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add disruption_type and severity_score columns.
        
        Parameters
        ----------
        df : output of DisruptionFeatureBuilder.build()
        
        Returns
        -------
        pd.DataFrame with new columns: disruption_type, severity_score
        """
        logger.info("DisruptionClassifier.classify() — assigning labels...")
        out = df.copy()
        
        for col in ['delay_sec', 'speed']:
            if col not in out.columns:
                out[col] = np.nan
        if 'has_alert' not in out.columns:
            out['has_alert'] = False
        if 'speed_flag' not in out.columns:
            out['speed_flag'] = 'unknown'
        
        conditions = [
            out.apply(self._is_cancelled, axis=1),
            out['delay_sec'] > self.delay_major_sec,
            (out['speed_flag'] == 'stopped') & (out['delay_sec'] > self.delay_minor_sec),
            out['delay_sec'] > self.delay_minor_sec,
            out['speed_flag'] == 'slow',
            out['delay_sec'] < self.delay_early_sec,
            out['has_alert'] == True,
        ]
        choices = ['CANCELLED', 'MAJOR_DELAY', 'STOPPED_ON_ROUTE', 'MINOR_DELAY',
                   'SLOW_TRAFFIC', 'EARLY', 'SERVICE_ALERT']
        
        out['disruption_type'] = np.select(conditions, choices, default='ON_TIME')
        out['severity_score'] = out['disruption_type'].map(self.DISRUPTION_SEVERITY)
        
        counts = out['disruption_type'].value_counts()
        logger.info("\nDisruption counts:")
        for k, v in counts.items():
            logger.info(f"  {k:<22} {v:>6,}")
        return out
    
    def summary(self, classified_df: pd.DataFrame) -> pd.DataFrame:
        """
        Route-level disruption summary.
        
        Returns
        -------
        pd.DataFrame with columns:
            route_id, total_stop_events, avg_delay_min, max_delay_min,
            avg_severity, max_severity, alert_count,
            pct_on_time, pct_minor_delay, pct_major_delay,
            pct_cancelled, pct_stopped_on_route, pct_slow_traffic,
            pct_early, pct_service_alert, route_short_name
        """
        if 'route_id' not in classified_df.columns:
            return pd.DataFrame()
        
        agg = classified_df.groupby('route_id').agg(
            total_stop_events=('route_id', 'count'),
            avg_severity=('severity_score', 'mean'),
            max_severity=('severity_score', 'max'),
            alert_count=('has_alert', 'sum'),
        ).reset_index()
        
        if 'delay_min' in classified_df.columns:
            delay_agg = classified_df.groupby('route_id').agg(
                avg_delay_min=('delay_min', 'mean'),
                max_delay_min=('delay_min', 'max'),
            ).reset_index()
            agg = agg.merge(delay_agg, on='route_id', how='left')
        
        for dtype in ['ON_TIME', 'MINOR_DELAY', 'MAJOR_DELAY', 'EARLY',
                      'CANCELLED', 'STOPPED_ON_ROUTE', 'SLOW_TRAFFIC', 'SERVICE_ALERT']:
            col = f'pct_{dtype.lower()}'
            pcts = classified_df.groupby('route_id').apply(
                lambda x: (x['disruption_type'] == dtype).mean() * 100
            ).reset_index(name=col)
            agg = agg.merge(pcts, on='route_id', how='left')
        
        if 'route_short_name' in classified_df.columns:
            names = classified_df[['route_id', 'route_short_name']].drop_duplicates('route_id').set_index('route_id')
            agg = agg.merge(names, on='route_id', how='left')
        
        agg = agg.sort_values('avg_severity', ascending=False)
        logger.info(f"\nRoute summary: {len(agg)} routes")
        return agg
