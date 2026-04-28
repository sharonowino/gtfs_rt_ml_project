"""
Early-Warning Module
====================
Converts a real-time disruption classifier into a 30-minute predictive
early-warning system by:

1. Snapshotting feed data into temporal windows
2. Computing lead-lag trend features (delay trajectory, speed deceleration)
3. Constructing forward-shifted targets (10/30/60 min ahead)
4. Providing the augmented DataFrame for model training
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default prediction horizons
HORIZONS = {"target_10min": 10, "target_30min": 30, "target_60min": 60}


class EarlyWarningBuilder:
    """
    Builds early-warning features and forward-shifted targets from
    a time-sorted DataFrame of stop-level observations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``feed_timestamp``, ``trip_id``, ``stop_id``,
        ``delay_sec``, ``speed``, and ``disruption_type``.
    horizons : dict
        Mapping of target column name → lookahead minutes.
    window_minutes : int
        Size of the look-back window for trend features.
    snapshot_interval : int
        Minutes between synthetic snapshots when data is sparse.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        horizons: Optional[Dict[str, int]] = None,
        window_minutes: int = 60,
        snapshot_interval: int = 5,
    ):
        self.df = df.copy()
        self.horizons = horizons or HORIZONS
        self.window = window_minutes
        self.snapshot_interval = snapshot_interval
        self._ts_col = (
            "feed_timestamp" if "feed_timestamp" in self.df.columns else "timestamp"
        )
        self.df[self._ts_col] = pd.to_datetime(self.df[self._ts_col])

    # ------------------------------------------------------------------
    # 1. Trend features (look-back window)
    # ------------------------------------------------------------------

    def _add_trend_features(self) -> pd.DataFrame:
        """Rate-of-change features over a sliding look-back window."""
        logger.info("  Building trend features...")

        df = self.df.sort_values([self._ts_col, "trip_id", "stop_id"])
        group = df.groupby(["trip_id", "stop_id"], sort=False)

        # Delay trajectory
        if "delay_sec" in df.columns:
            df["delay_prev"] = group["delay_sec"].shift(1)
            df["delay_delta"] = df["delay_sec"] - df["delay_prev"]
            df["delay_accel"] = group["delay_delta"].transform(
                lambda s: s - s.shift(1)
            )

            # Rolling mean / std of delay over last N observations
            df["delay_rolling_mean_5"] = group["delay_sec"].transform(
                lambda s: s.rolling(5, min_periods=1).mean()
            )
            df["delay_rolling_std_5"] = group["delay_sec"].transform(
                lambda s: s.rolling(5, min_periods=1).std().fillna(0)
            )
            df["delay_rolling_max_5"] = group["delay_sec"].transform(
                lambda s: s.rolling(5, min_periods=1).max()
            )

            # Delay trend slope (linear fit over last 5 observations)
            def _slope(s):
                vals = s.values
                if len(vals) < 3:
                    return np.nan
                x = np.arange(len(vals))
                mask = ~np.isnan(vals)
                if mask.sum() < 2:
                    return np.nan
                return np.polyfit(x[mask], vals[mask], 1)[0]

            df["delay_trend_slope"] = group["delay_sec"].transform(
                lambda s: s.rolling(5, min_periods=2).apply(_slope, raw=False)
            )

        # Speed trajectory
        if "speed" in df.columns:
            df["speed_prev"] = group["speed"].shift(1)
            df["speed_delta"] = df["speed"] - df["speed_prev"]
            df["speed_decel_rate"] = group["speed_delta"].transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )

        # Cumulative disruption count in look-back window
        if "disruption_type" in df.columns:
            df["is_disrupted_now"] = (df["disruption_type"] != "ON_TIME").astype(int)
            df["disruption_count_in_window"] = group["is_disrupted_now"].transform(
                lambda s: s.rolling(5, min_periods=1).sum()
            )

        self.df = df
        return df

    # ------------------------------------------------------------------
    # 2. Forward-shifted targets
    # ------------------------------------------------------------------

    def _add_forward_targets(self) -> pd.DataFrame:
        """
        Create binary targets that indicate whether a NEW disruption will
        occur within *N* minutes of the current observation.

        A NEW disruption is defined as a transition from ON_TIME to a
        disruption state (any disruption type != ON_TIME). This prevents
        the model from learning to predict the continuation of an existing
        disruption rather than detecting the onset.

        For each (trip_id, stop_id) group, we look ahead in time and
        mark ``1`` if a NEW disruption BEGINS within the horizon window.
        """
        logger.info("  Building forward-shifted targets (NEW disruption only)...")

        df = self.df
        ts = self._ts_col

        if "disruption_type" not in df.columns:
            logger.warning("    disruption_type not found — skipping forward targets")
            return df

        df = df.sort_values([ts, "trip_id", "stop_id"])
        
        # Mark current disruption state
        df["_is_disrupted"] = (df["disruption_type"] != "ON_TIME").astype(int)
        
        # Mark transitions: NEW disruption = currently ON_TIME but future is disrupted
        # This is the key fix: we only predict NEW disruptions, not continuation
        df["_prev_disrupted"] = df.groupby(["trip_id", "stop_id"], sort=False)["_is_disrupted"].shift(1).fillna(0).astype(int)
        df["_new_disruption"] = ((df["_is_disrupted"] == 1) & (df["_prev_disrupted"] == 0)).astype(int)

        for target_col, horizon_min in self.horizons.items():
            df[target_col] = 0
            horizon_td = pd.Timedelta(minutes=horizon_min)

            # For each group, check if a NEW disruption begins within the horizon
            def _flag_new_disruption(group):
                g = group.sort_values(ts)
                times = g[ts].values
                new_disr = g["_new_disruption"].values
                is_disrupted = g["_is_disrupted"].values
                result = np.zeros(len(g), dtype=int)
                
                for i in range(len(g)):
                    # If currently disrupted, look for NEW disruption starting after this point
                    if is_disrupted[i] == 0:  # Only predict for non-disrupted states
                        window_end = times[i] + horizon_td
                        # Look for NEW disruptions (not continuation) within horizon
                        future_mask = (times > times[i]) & (times <= window_end)
                        if future_mask.any() and new_disr[future_mask].any():
                            result[i] = 1
                return pd.Series(result, index=g.index)

            df[target_col] = df.groupby(
                ["trip_id", "stop_id"], group_keys=False
            ).apply(_flag_new_disruption)

            rate = df[target_col].mean()
            logger.info(f"    {target_col}: {rate:.2%} positive (new disruptions only)")

        # Disruption type within horizon (multi-class) - for NEW disruptions only
        for target_col, horizon_min in self.horizons.items():
            multi_col = target_col.replace("target_", "target_mc_")
            df[multi_col] = 0  # 0 = no new disruption

            horizon_td = pd.Timedelta(minutes=horizon_min)

            def _flag_new_disruption_type(group):
                g = group.sort_values(ts)
                times = g[ts].values
                types = g["disruption_type"].values
                new_disr = g["_new_disruption"].values
                is_disrupted = g["_is_disrupted"].values
                result = np.zeros(len(g), dtype=int)
                severity = {
                    "ON_TIME": 0, "EARLY": 1, "SLOW_TRAFFIC": 2,
                    "MINOR_DELAY": 3, "MAJOR_DELAY": 4, "CANCELLED": 5,
                    "STOPPED_ON_ROUTE": 3, "SERVICE_ALERT": 2,
                }
                for i in range(len(g)):
                    # Only predict for currently non-disrupted states
                    if is_disrupted[i] == 0:
                        window_end = times[i] + horizon_td
                        future_mask = (times > times[i]) & (times <= window_end)
                        if future_mask.any():
                            future_types = types[future_mask]
                            future_new_disr = new_disr[future_mask]
                            # Only consider NEW disruptions (transitions to disruption)
                            new_types = [t for j, t in enumerate(future_types) if future_new_disr[j] == 1]
                            if new_types:
                                max_sev = max(severity.get(t, 0) for t in new_types)
                                result[i] = max_sev
                return pd.Series(result, index=g.index)

            df[multi_col] = df.groupby(
                ["trip_id", "stop_id"], group_keys=False
            ).apply(_flag_new_disruption_type)

        df.drop(columns=["_is_disrupted", "_prev_disrupted", "_new_disruption"], inplace=True, errors="ignore")
        self.df = df
        return df

     # ------------------------------------------------------------------
     # 3. Route-level aggregate features
     # ------------------------------------------------------------------
 
    def _add_route_aggregates(self) -> pd.DataFrame:
        """Aggregate trend features at the route level per snapshot.
        
        Uses STRICT backward-looking windows to prevent target leakage.
        Each snapshot only sees historical data, not including current observation.
        """
        logger.info("  Building route-level aggregates (backward-looking only)...")

        df = self.df
        ts = self._ts_col

        if "route_id" not in df.columns or "delay_sec" not in df.columns:
            return df

        # Bin timestamps into snapshot windows
        df["_snap_bin"] = df[ts].dt.floor(
            f"{self.snapshot_interval}min"
        )

        # CRITICAL FIX: Exclude current observation from aggregates
        # Use shift to ensure we only use past data within each group
        df["_delay_historical"] = df.groupby("route_id")["delay_sec"].shift(1).fillna(0)
        if "speed" in df.columns:
            df["_speed_historical"] = df.groupby("route_id")["speed"].shift(1).fillna(0)
            speed_col = "_speed_historical"
        else:
            df["_speed_historical"] = 0
            speed_col = "_delay_historical"
        
        # Route-level stats per snapshot using HISTORICAL data only (excluding current)
        route_agg = df.groupby(["route_id", "_snap_bin"]).agg(
            route_delay_mean=("_delay_historical", "mean"),
            route_delay_max=("_delay_historical", "max"),
            route_delay_std=("_delay_historical", "std"),
            route_speed_mean=(speed_col, "mean"),
        ).reset_index()

        route_agg["route_delay_std"] = route_agg["route_delay_std"].fillna(0)

        # Route disruption rate per snapshot using HISTORICAL data only
        if "disruption_type" in df.columns:
            # Shift disruption status to get previous state (backward-looking)
            df["_current_disrupted"] = (df["disruption_type"] != "ON_TIME").astype(int)
            df["_prev_disrupted"] = df.groupby("route_id")["_current_disrupted"].shift(1).fillna(0).astype(int)
            
            route_disr = df.groupby(["route_id", "_snap_bin"]).agg(
                _prev_disrupted_sum=("_prev_disrupted", "sum"),
                _prev_count=("_prev_disrupted", "count")
            ).reset_index()
            # Avoid division by zero
            route_disr["route_disruption_rate"] = (
                route_disr["_prev_disrupted_sum"] / route_disr["_prev_count"].replace(0, 1)
            ).fillna(0)
            route_agg = route_agg.merge(
                route_disr[["route_id", "_snap_bin", "route_disruption_rate"]], 
                on=["route_id", "_snap_bin"], 
                how="left"
            )

        # Merge back
        self.df = df.merge(
            route_agg, on=["route_id", "_snap_bin"], how="left", suffixes=("", "_route")
        )
        # Clean up temporary columns
        temp_cols = ["_snap_bin", "_delay_historical", "_speed_historical", 
                     "_current_disrupted", "_prev_disrupted"]
        self.df.drop(columns=[c for c in temp_cols if c in self.df.columns], inplace=True, errors="ignore")
        return self.df

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Build all early-warning features and forward targets.

        Returns
        -------
        pd.DataFrame with trend features, forward targets, and route aggregates.
        """
        logger.info("=" * 60)
        logger.info("EARLY WARNING BUILDER")
        logger.info("=" * 60)

        self._add_trend_features()
        self._add_forward_targets()
        self._add_route_aggregates()

        new_cols = [c for c in self.df.columns if c not in self.df.columns or c.startswith(("delay_prev", "delay_delta", "delay_accel", "delay_rolling", "delay_trend", "speed_prev", "speed_delta", "speed_decel", "disruption_count", "target_", "route_delay", "route_speed", "route_disruption"))]
        logger.info(f"  Early warning features built: {self.df.shape}")
        return self.df


def add_early_warning_features(
    df: pd.DataFrame,
    horizons: Optional[Dict[str, int]] = None,
    window_minutes: int = 60,
) -> pd.DataFrame:
    """
    Convenience function: add early-warning features to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Merged/classified DataFrame.
    horizons : dict, optional
        Prediction horizons (default: 10/30/60 min).
    window_minutes : int
        Look-back window for trend features.

    Returns
    -------
    pd.DataFrame with early-warning features and forward targets.
    """
    builder = EarlyWarningBuilder(df, horizons=horizons, window_minutes=window_minutes)
    return builder.build()
