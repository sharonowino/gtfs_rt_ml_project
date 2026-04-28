"""
GTFS Time-based Features for Transit Disruption Detection
=========================================================
Time features including rush hour peaks, holidays, and temporal patterns.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Optional, Tuple
import os

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Peak hour windows (Netherlands transit)
MORNING_PEAK_START = dt_time(6, 0)
MORNING_PEAK_END = dt_time(9, 0)
EVENING_PEAK_START = dt_time(16, 0)
EVENING_PEAK_END = dt_time(19, 0)

# Peak bucket bins
PEAK_BUCKET_BINS = [-1, 0, 30, 60, 180, np.inf]
PEAK_BUCKET_LABELS = ["in_peak", "under_30min", "30_to_60min", "1_to_3hrs", "off_peak"]

DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
}


# =============================================================================
# HELPERS
# =============================================================================

def _strip_timezone(series: pd.Series) -> pd.Series:
    """Convert to datetime, strip timezone if present."""
    series = pd.to_datetime(series, errors="coerce")
    if series.dt.tz is not None:
        series = series.dt.tz_convert(None)
    return series


def _minutes_to_next_peak(ts: pd.Timestamp, country: str = "NL") -> float:
    """
    Returns minutes until the next rush hour peak window.
    Returns 0.0 if currently inside a peak window.
    Returns NaN for invalid timestamps.
    """
    if pd.isna(ts):
        return np.nan

    current_time = ts.time()
    today = ts.date()

    morning_start_dt = datetime.combine(today, MORNING_PEAK_START)
    evening_start_dt = datetime.combine(today, EVENING_PEAK_START)

    # Check if in peak window
    if MORNING_PEAK_START <= current_time <= MORNING_PEAK_END:
        return 0.0
    elif EVENING_PEAK_START <= current_time <= EVENING_PEAK_END:
        return 0.0
    elif current_time < MORNING_PEAK_START:
        # Before morning peak
        return (morning_start_dt - ts).total_seconds() / 60
    elif MORNING_PEAK_END < current_time < EVENING_PEAK_START:
        # Between peaks
        return (evening_start_dt - ts).total_seconds() / 60
    else:
        # After evening peak - next day's morning
        next_morning = datetime.combine(today + timedelta(days=1), MORNING_PEAK_START)
        return (next_morning - ts).total_seconds() / 60


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def build_time_features(
    df: pd.DataFrame,
    datetime_col: str = "timestamp",
    country_code: str = "NL",
    include_holidays: bool = True,
) -> pd.DataFrame:
    """
    Adds time-based features to a copy of df.
    
    New columns:
        - hour_of_day: 0-23
        - day_of_week: Monday-Sunday
        - day_of_week_numeric: 0-6
        - is_weekend: 1 if Sat/Sun
        - is_holiday: 1 if public holiday
        - date: date only
        - month: 1-12
        - days_since_start: days from earliest date
        - time_to_rush_hour_peak: minutes to next peak
        - time_to_peak_bucket: binned peak time
        - is_peak_hour: 1 if in peak window
    """
    df = df.copy()
    
    # 1. Parse datetime
    df[datetime_col] = _strip_timezone(df[datetime_col])
    
    # 2. Base time columns
    df["hour_of_day"] = df[datetime_col].dt.hour
    df["day_of_week"] = df[datetime_col].dt.day_name()
    df["date"] = df[datetime_col].dt.date
    df["month"] = df[datetime_col].dt.month
    
    # 3. Days since start (stable ordinal)
    min_date = df[datetime_col].min()
    if pd.notna(min_date):
        df["days_since_start"] = (df[datetime_col] - min_date).dt.days
    else:
        df["days_since_start"] = 0
    
    # 4. Day type flags
    df["day_of_week_numeric"] = df["day_of_week"].map(DAY_MAP)
    df["is_weekend"] = (df["day_of_week_numeric"] >= 5).astype(int)
    
    # 5. Holiday flag
    if include_holidays and HOLIDAYS_AVAILABLE:
        try:
            years = df[datetime_col].dt.year.dropna().unique().astype(int).tolist()
            if not years:
                years = [datetime.now().year]
            cal = holidays.country_holidays(country_code, years=years)
            df["is_holiday"] = df["date"].apply(lambda d: int(d in cal) if pd.notna(d) else 0)
        except Exception as e:
            df["is_holiday"] = 0
    else:
        df["is_holiday"] = 0
    
    # 6. Rush hour features
    is_off_day = (df["is_weekend"] == 1) | (df["is_holiday"] == 1)
    
    df["time_to_rush_hour_peak"] = df[datetime_col].apply(
        lambda ts: _minutes_to_next_peak(ts, country_code)
    )
    df.loc[is_off_day, "time_to_rush_hour_peak"] = np.nan
    
    # Binned version
    df["time_to_peak_bucket"] = pd.cut(
        df["time_to_rush_hour_peak"],
        bins=PEAK_BUCKET_BINS,
        labels=PEAK_BUCKET_LABELS,
    )
    df["time_to_peak_bucket"] = df["time_to_peak_bucket"].cat.add_categories(["off_day"])
    df.loc[is_off_day, "time_to_peak_bucket"] = "off_day"
    
    # Binary peak flag
    df["is_peak_hour"] = (df["time_to_rush_hour_peak"] == 0).astype("Int64")
    
    return df


# =============================================================================
# TIME DISTRIBUTION PLOTS
# =============================================================================

def plot_time_distributions(
    df: pd.DataFrame,
    save_dir: str = "visualizations",
    datetime_col: str = "timestamp",
) -> dict:
    """
    Plot alert counts by hour of day and day of week with peak windows.
    
    Returns:
        dict: Figure objects for display
    """
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure time features exist
    if datetime_col in df.columns and "hour_of_day" not in df.columns:
        df = build_time_features(df, datetime_col)
    
    results = {}
    
    # 1. Hourly distribution (Plotly)
    hourly = df.groupby("hour_of_day").size().reindex(range(24), fill_value=0)
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly.index,
        y=hourly.values,
        marker_color="steelblue",
        name="Alerts"
    ))
    
    # Add peak windows
    fig_hourly.add_vrect(
        x0=6, x1=9,
        fillcolor="orange", opacity=0.15,
        layer="below", line_width=0,
        annotation_text="AM Peak"
    )
    fig_hourly.add_vrect(
        x0=16, x1=19,
        fillcolor="red", opacity=0.15,
        layer="below", line_width=0,
        annotation_text="PM Peak"
    )
    
    fig_hourly.update_layout(
        title="Alerts by Hour of Day",
        xaxis_title="Hour (0-23)",
        yaxis_title="Alert Count",
        height=300,
        template="plotly_white"
    )
    results["hourly"] = fig_hourly
    
    # 2. Day of week distribution (Plotly)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby("day_of_week").size().reindex(day_order, fill_value=0)
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=daily.index,
        y=daily.values,
        marker_color="forestgreen",
        name="Alerts"
    ))
    
    fig_daily.update_layout(
        title="Alerts by Day of Week",
        xaxis_title="Day",
        yaxis_title="Alert Count",
        height=300,
        template="plotly_white"
    )
    results["daily"] = fig_daily
    
    # 3. Peak bucket distribution
    if "time_to_peak_bucket" in df.columns:
        bucket_counts = df["time_to_peak_bucket"].value_counts()
        
        fig_bucket = go.Figure()
        fig_bucket.add_trace(go.Pie(
            labels=bucket_counts.index,
            values=bucket_counts.values,
            hole=0.3,
            textinfo="percent"
        ))
        
        fig_bucket.update_layout(
            title="Disruption Timing Distribution",
            height=300
        )
        results["bucket"] = fig_bucket
    
    return results


# =============================================================================
# ADD TO DASHBOARD
# =============================================================================

def get_time_features_for_dashboard(
    df: pd.DataFrame,
    datetime_col: str = "timestamp"
) -> dict:
    """
    Get time-based metrics for dashboard display.
    """
    if df is None or df.empty:
        return {
            "hourly_counts": {},
            "daily_counts": {},
            "peak_percentage": 0,
            "weekend_percentage": 0,
            "holiday_percentage": 0
        }
    
    # Build features if needed
    if "hour_of_day" not in df.columns:
        df = build_time_features(df, datetime_col)
    
    # Aggregate
    hourly = df.groupby("hour_of_day").size().to_dict()
    daily = df.groupby("day_of_week").size().to_dict()
    
    peak_pct = (df["is_peak_hour"] == 1).mean() * 100 if "is_peak_hour" in df.columns else 0
    weekend_pct = df["is_weekend"].mean() * 100 if "is_weekend" in df.columns else 0
    holiday_pct = df["is_holiday"].mean() * 100 if "is_holiday" in df.columns else 0
    
    return {
        "hourly_counts": hourly,
        "daily_counts": daily,
        "peak_percentage": round(peak_pct, 1),
        "weekend_percentage": round(weekend_pct, 1),
        "holiday_percentage": round(holiday_pct, 1)
    }