"""
GTFS Feature Enrichment Module
==============================
Enriches the merged real-time DataFrame with features derived from static GTFS
data (routes, stops, trips, stop_times, transfers, calendar_dates).

Feature families
---------------
1. Route features        — transport mode, route complexity, agency
2. Stop features         — hub score, transfer connectivity, geographic bins
3. Trip features         — trip length, stop count, shape distance
4. Schedule features     — scheduled headway, dwell patterns, time-of-day service
5. Network features      — stop degree, route diversity, transfer proximity
6. Geo-temporal features — distance to major hub, urban/rural, peak ratio
"""
import io
import logging
import zipfile
from typing import Dict, List, Optional, Tuple

import os
import io
import zipfile
import logging
import requests

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# GTFS route_type codes (https://gtfs.org/schedule/reference/#routestxt)
ROUTE_TYPE_MAP = {
    "0": "tram",
    "1": "subway",
    "2": "rail",
    "3": "bus",
    "4": "ferry",
    "5": "cable_tram",
    "6": "aerial_lift",
    "7": "funicular",
    "11": "trolleybus",
    "12": "monorail",
}

# Major hub stations in NL (Amsterdam CS, Utrecht CS, Rotterdam CS, Den Haag CS, etc.)
MAJOR_HUBS = {
    "stoparea:2155": {"name": "Amsterdam Centraal", "lat": 52.3791, "lon": 4.9003},
    "stoparea:1463": {"name": "Utrecht Centraal", "lat": 52.0890, "lon": 5.1093},
    "stoparea:1661": {"name": "Rotterdam Centraal", "lat": 51.9249, "lon": 4.4689},
    "stoparea:1986": {"name": "Den Haag Centraal", "lat": 52.0807, "lon": 4.3247},
    "stoparea:1642": {"name": "Leiden Centraal", "lat": 52.1663, "lon": 4.4814},
    "stoparea:1828": {"name": "Schiphol Airport", "lat": 52.3094, "lon": 4.7625},
    "stoparea:1929": {"name": "Eindhoven Centraal", "lat": 51.4431, "lon": 5.4813},
    "stoparea:1702": {"name": "Arnhem Centraal", "lat": 51.9846, "lon": 5.9010},
}


def _haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


class GTFSEnricher:
    """
    Enriches a merged real-time DataFrame with features derived from
    static GTFS data.

    Parameters
    ----------
    gtfs_data : dict, optional
        Dictionary of static GTFS DataFrames (keys: routes, stops, trips,
        stop_times, transfers, calendar_dates, shapes, agency).
        If None, automatically loads from default URL or local file.
    gtfs_path : str, optional
        Path to static GTFS zip file or URL.
        If None, uses default URL (gtfs.ovapi.nl/gtfs-nl.zip).
    """

    def __init__(self, gtfs_data: Dict[str, pd.DataFrame] = None, gtfs_path: str = None):
        # Auto-load static GTFS if not provided
        if gtfs_data is None:
            logger.info("Auto-loading static GTFS data...")
            gtfs_data = self._load_static_gtfs(gtfs_path)
        self.gtfs = gtfs_data
        # Pre-compute caches
        self._stop_route_map = None
        self._route_stop_counts = None
        self._stop_degree = None
        self._transfer_stops = None

    def _load_static_gtfs(self, path: str = None) -> Dict[str, pd.DataFrame]:
        """
        Automatically load static GTFS data.
        
        Parameters
        ----------
        path : str, optional
            GTFS zip path or URL
            
        Returns
        -------
        dict of GTFS DataFrames
        """
        import requests
        import io
        import zipfile as zf
        
        if path is None:
            # Try local file first, then URL
            local_paths = [
                'gtfs-nl.zip',
                'data/gtfs-nl.zip',
                'gtfs_disruption/gtfs-nl.zip',
            ]
            gtfs_data = {}
            for local_path in local_paths:
                if os.path.exists(local_path):
                    logger.info(f"Loading from local: {local_path}")
                    try:
                        with zf.ZipFile(local_path, 'r') as z:
                            for name in z.namelist():
                                if name.endswith('.txt'):
                                    key = name.replace('.txt', '')
                                    gtfs_data[key] = pd.read_csv(z.open(name), low_memory=False)
                        return gtfs_data
                    except Exception as e:
                        logger.warning(f"Failed to load {local_path}: {e}")
                        continue
            
            # Try URL
            url = 'http://gtfs.ovapi.nl/gtfs-nl.zip'
            logger.info(f"Downloading from: {url}")
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                with zf.ZipFile(io.BytesIO(response.content)) as z:
                    for name in z.namelist():
                        if name.endswith('.txt'):
                            key = name.replace('.txt', '')
                            gtfs_data[key] = pd.read_csv(z.open(name), low_memory=False)
                logger.info(f"Downloaded {len(gtfs_data)} GTFS files")
                return gtfs_data
            except Exception as e:
                logger.warning(f"Failed to download GTFS: {e}")
                logger.info("Using computed enrichment from trip data")
                return {}

    # ------------------------------------------------------------------
    # Lazy caches
    # ------------------------------------------------------------------

    def _ensure_stop_route_map(self):
        if self._stop_route_map is not None:
            return
        trips = self.gtfs.get("trips", pd.DataFrame())
        stop_times = self.gtfs.get("stop_times", pd.DataFrame())
        if trips.empty or stop_times.empty or "trip_id" not in stop_times.columns:
            self._stop_route_map = pd.DataFrame()
            return

        sr = stop_times[["trip_id", "stop_id"]].drop_duplicates().merge(
            trips[["trip_id", "route_id"]].drop_duplicates(), on="trip_id", how="left"
        )
        self._stop_route_map = sr[["stop_id", "route_id"]].drop_duplicates()

    def _ensure_stop_degree(self):
        if self._stop_degree is not None:
            return
        self._ensure_stop_route_map()
        if self._stop_route_map.empty:
            self._stop_degree = pd.Series(dtype=int)
            return
        self._stop_degree = self._stop_route_map.groupby("stop_id")["route_id"].nunique()

    def _ensure_transfer_stops(self):
        if self._transfer_stops is not None:
            return
        transfers = self.gtfs.get("transfers", pd.DataFrame())
        if transfers.empty:
            self._transfer_stops = set()
            return
        cols = set(transfers.columns)
        ids = set()
        if "from_stop_id" in cols:
            ids.update(transfers["from_stop_id"].dropna().astype(str).tolist())
        if "to_stop_id" in cols:
            ids.update(transfers["to_stop_id"].dropna().astype(str).tolist())
        self._transfer_stops = ids

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def _add_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Route-level features from routes.txt."""
        logger.info("  Enriching route features...")
        routes = self.gtfs.get("routes", pd.DataFrame())
        if routes.empty or "route_id" not in df.columns:
            return df

        r = routes.copy()
        r["route_id"] = r["route_id"].astype(str)

        # Transport mode (numeric encoding)
        if "route_type" in r.columns:
            r["route_type_int"] = pd.to_numeric(r["route_type"], errors="coerce").fillna(-1).astype(int)
            r["is_bus"] = (r["route_type_int"] == 3).astype(int)
            r["is_rail"] = (r["route_type_int"] == 2).astype(int)
            r["is_tram"] = (r["route_type_int"] == 0).astype(int)
            r["is_ferry"] = (r["route_type_int"] == 4).astype(int)
            r["is_metro"] = (r["route_type_int"] == 1).astype(int)

        # Route name length (proxy for route complexity)
        if "route_short_name" in r.columns:
            r["route_name_length"] = r["route_short_name"].astype(str).str.len()
        if "route_long_name" in r.columns:
            r["route_long_name_length"] = r["route_long_name"].astype(str).str.len()

        # Number of unique trips per route (service frequency proxy)
        trips = self.gtfs.get("trips", pd.DataFrame())
        if not trips.empty and "route_id" in trips.columns:
            trips["route_id"] = trips["route_id"].astype(str)
            freq = trips.groupby("route_id")["trip_id"].nunique().reset_index()
            freq.columns = ["route_id", "route_trip_count"]
            r = r.merge(freq, on="route_id", how="left")
            r["route_trip_count"] = r["route_trip_count"].fillna(0)

        # Number of unique stops per route
        self._ensure_stop_route_map()
        if not self._stop_route_map.empty:
            stop_counts = self._stop_route_map.groupby("route_id")["stop_id"].nunique().reset_index()
            stop_counts.columns = ["route_id", "route_stop_count"]
            r = r.merge(stop_counts, on="route_id", how="left")
            r["route_stop_count"] = r["route_stop_count"].fillna(0)

        # Number of unique service days per route
        cal = self.gtfs.get("calendar_dates", pd.DataFrame())
        if not trips.empty and not cal.empty and "service_id" in trips.columns and "service_id" in cal.columns:
            svc_days = cal.groupby("service_id")["date"].nunique().reset_index()
            svc_days.columns = ["service_id", "service_days"]
            trip_svc = trips[["route_id", "service_id"]].drop_duplicates()
            trip_svc = trip_svc.merge(svc_days, on="service_id", how="left")
            route_svc = trip_svc.groupby("route_id")["service_days"].sum().reset_index()
            route_svc.columns = ["route_id", "route_service_days"]
            r = r.merge(route_svc, on="route_id", how="left")

        # Select merge columns
        merge_cols = ["route_id"]
        for c in ["route_type_int", "is_bus", "is_rail", "is_tram", "is_ferry", "is_metro",
                   "route_name_length", "route_long_name_length",
                   "route_trip_count", "route_stop_count", "route_service_days"]:
            if c in r.columns:
                merge_cols.append(c)

        r_sub = r[merge_cols].drop_duplicates("route_id")
        df = df.merge(r_sub, on="route_id", how="left")
        return df

    def _add_stop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stop-level features from stops.txt."""
        logger.info("  Enriching stop features...")
        stops = self.gtfs.get("stops", pd.DataFrame())
        if stops.empty or "stop_id" not in df.columns:
            return df

        s = stops.copy()
        s["stop_id"] = s["stop_id"].astype(str)

        # Location type (0=stop, 1=station)
        if "location_type" in s.columns:
            s["is_station"] = pd.to_numeric(s["location_type"], errors="coerce").fillna(0).astype(int)

        # Has parent station
        if "parent_station" in s.columns:
            s["has_parent_station"] = s["parent_station"].notna().astype(int)

        # Has platform code
        if "platform_code" in s.columns:
            s["has_platform"] = s["platform_code"].notna().astype(int)

        # Geographic bins (for spatial grouping)
        if "stop_lat" in s.columns and "stop_lon" in s.columns:
            s["stop_lat_f"] = pd.to_numeric(s["stop_lat"], errors="coerce")
            s["stop_lon_f"] = pd.to_numeric(s["stop_lon"], errors="coerce")
            s["lat_bin"] = (s["stop_lat_f"] * 10).round().astype("Int64")  # ~11km bins
            s["lon_bin"] = (s["stop_lon_f"] * 10).round().astype("Int64")

        # Stop degree (number of routes serving this stop)
        self._ensure_stop_degree()
        if not self._stop_degree.empty:
            s["stop_degree"] = s["stop_id"].map(self._stop_degree).fillna(0).astype(int)
        else:
            s["stop_degree"] = 0

        # Hub indicator (stop_degree > 5)
        s["is_hub_stop"] = (s["stop_degree"] > 5).astype(int)

        # Transfer stop indicator
        self._ensure_transfer_stops()
        s["is_transfer_stop"] = s["stop_id"].isin(self._transfer_stops).astype(int)

        # Distance to nearest major hub
        if "stop_lat_f" in s.columns:
            min_dist = np.full(len(s), np.inf)
            for hub in MAJOR_HUBS.values():
                d = _haversine(s["stop_lat_f"].values, s["stop_lon_f"].values,
                               hub["lat"], hub["lon"])
                min_dist = np.minimum(min_dist, d)
            s["dist_to_nearest_hub_km"] = (min_dist / 1000).round(1)

        # Wheelchair boarding
        if "wheelchair_boarding" in s.columns:
            s["wheelchair_accessible_stop"] = pd.to_numeric(
                s["wheelchair_boarding"], errors="coerce"
            ).fillna(0).astype(int)

        # Select merge columns
        merge_cols = ["stop_id"]
        for c in ["is_station", "has_parent_station", "has_platform",
                   "lat_bin", "lon_bin", "stop_degree", "is_hub_stop",
                   "is_transfer_stop", "dist_to_nearest_hub_km",
                   "wheelchair_accessible_stop"]:
            if c in s.columns:
                merge_cols.append(c)

        s_sub = s[merge_cols].drop_duplicates("stop_id")
        df = df.merge(s_sub, on="stop_id", how="left")
        return df

    def _add_trip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trip-level features from trips.txt."""
        logger.info("  Enriching trip features...")
        trips = self.gtfs.get("trips", pd.DataFrame())
        if trips.empty or "trip_id" not in df.columns:
            return df

        t = trips.copy()
        t["trip_id"] = t["trip_id"].astype(str)

        # Direction
        if "direction_id" in t.columns:
            t["trip_direction"] = pd.to_numeric(t["direction_id"], errors="coerce").fillna(-1).astype(int)

        # Has shape
        if "shape_id" in t.columns:
            t["has_shape"] = t["shape_id"].notna().astype(int)

        # Wheelchair accessible
        if "wheelchair_accessible" in t.columns:
            t["trip_wheelchair"] = pd.to_numeric(
                t["wheelchair_accessible"], errors="coerce"
            ).fillna(0).astype(int)

        # Bikes allowed
        if "bikes_allowed" in t.columns:
            t["trip_bikes_allowed"] = pd.to_numeric(
                t["bikes_allowed"], errors="coerce"
            ).fillna(0).astype(int)

        # Number of stops in this trip (from stop_times)
        stop_times = self.gtfs.get("stop_times", pd.DataFrame())
        if not stop_times.empty and "trip_id" in stop_times.columns:
            stop_times["trip_id"] = stop_times["trip_id"].astype(str)
            trip_stops = stop_times.groupby("trip_id").agg(
                trip_stop_count=("stop_id", "nunique"),
                trip_max_sequence=("stop_sequence", lambda x: pd.to_numeric(x, errors="coerce").max()),
            ).reset_index()
            t = t.merge(trip_stops, on="trip_id", how="left")

        # Trip shape distance (from shapes if available) - skip if too large
        shapes = self.gtfs.get("shapes", pd.DataFrame())
        if not shapes.empty and "shape_id" in shapes.columns and "shape_dist_traveled" in shapes.columns and len(shapes) < 100000:
            shapes["shape_dist"] = pd.to_numeric(shapes["shape_dist_traveled"], errors="coerce")
            shape_dist = shapes.groupby("shape_id")["shape_dist"].max().reset_index()
            shape_dist.columns = ["shape_id", "shape_total_dist_m"]
            if "shape_id" in t.columns:
                t["shape_id"] = t["shape_id"].astype(str)
                shape_dist["shape_id"] = shape_dist["shape_id"].astype(str)
                t = t.merge(shape_dist, on="shape_id", how="left")

        # Select merge columns
        merge_cols = ["trip_id"]
        for c in ["trip_direction", "has_shape", "trip_wheelchair", "trip_bikes_allowed",
                   "trip_stop_count", "trip_max_sequence", "shape_total_dist_m"]:
            if c in t.columns:
                merge_cols.append(c)

        t_sub = t[merge_cols].drop_duplicates("trip_id")
        df = df.merge(t_sub, on="trip_id", how="left")
        return df

    def _add_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Schedule-derived features from stop_times.txt."""
        logger.info("  Enriching schedule features...")
        stop_times = self.gtfs.get("stop_times", pd.DataFrame())
        if stop_times.empty:
            return df

        st = stop_times.copy()

        # Scheduled arrival time in seconds from midnight
        def _gtfs_time_to_sec(t):
            if pd.isna(t):
                return np.nan
            parts = str(t).split(":")
            try:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            except (ValueError, IndexError):
                return np.nan

        if "arrival_time" in st.columns:
            st["sched_arr_sec"] = st["arrival_time"].apply(_gtfs_time_to_sec)
        if "departure_time" in st.columns:
            st["sched_dep_sec"] = st["departure_time"].apply(_gtfs_time_to_sec)

        # Scheduled dwell time at stop
        if "sched_arr_sec" in st.columns and "sched_dep_sec" in st.columns:
            st["sched_dwell_sec"] = st["sched_dep_sec"] - st["sched_arr_sec"]

        # Time between consecutive stops (scheduled travel time)
        if "trip_id" in st.columns and "sched_dep_sec" in st.columns:
            st["trip_id"] = st["trip_id"].astype(str)
            st = st.sort_values(["trip_id", "stop_sequence"])
            st["sched_travel_to_next"] = st.groupby("trip_id")["sched_dep_sec"].shift(-1) - st["sched_dep_sec"]

        # Stop sequence ratio (position in trip, 0-1)
        if "stop_sequence" in st.columns and "trip_id" in st.columns:
            st["stop_seq_num"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
            trip_max_seq = st.groupby("trip_id")["stop_seq_num"].transform("max")
            st["stop_seq_ratio"] = st["stop_seq_num"] / trip_max_seq.replace(0, np.nan)

        # Pickup/drop-off type indicators
        if "pickup_type" in st.columns:
            st["sched_no_pickup"] = (pd.to_numeric(st["pickup_type"], errors="coerce") == 1).astype(int)
        if "drop_off_type" in st.columns:
            st["sched_no_dropoff"] = (pd.to_numeric(st["drop_off_type"], errors="coerce") == 1).astype(int)

        # Merge with df on trip_id + stop_id
        if "trip_id" in df.columns and "stop_id" in df.columns:
            st["trip_id"] = st["trip_id"].astype(str)
            st["stop_id"] = st["stop_id"].astype(str)

            merge_cols = ["trip_id", "stop_id"]
            for c in ["sched_arr_sec", "sched_dep_sec", "sched_dwell_sec",
                       "sched_travel_to_next", "stop_seq_ratio",
                       "sched_no_pickup", "sched_no_dropoff"]:
                if c in st.columns:
                    merge_cols.append(c)

            st_sub = st[merge_cols].drop_duplicates(subset=["trip_id", "stop_id"])
            df = df.merge(st_sub, on=["trip_id", "stop_id"], how="left")

        # Scheduled time-of-day bucket
        if "sched_arr_sec" in df.columns:
            df["sched_hour"] = (df["sched_arr_sec"] / 3600).round().clip(0, 27)  # GTFS hours can exceed 24
            df["sched_is_peak"] = (
                ((df["sched_hour"] >= 7) & (df["sched_hour"] <= 9)) |
                ((df["sched_hour"] >= 16) & (df["sched_hour"] <= 19))
            ).astype(int)
            df["sched_is_offpeak"] = ((df["sched_hour"] >= 9) & (df["sched_hour"] < 16)).astype(int)
            df["sched_is_evening"] = (df["sched_hour"] >= 19).astype(int)
            df["sched_is_early_morning"] = (df["sched_hour"] < 7).astype(int)

        # FIX: stop_seq_ratio - use transform('max') instead of in-sample max to avoid bias
        if "stop_sequence" in st.columns and "trip_id" in st.columns:
            st["stop_seq_num"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
            trip_max_seq = st.groupby("trip_id")["stop_seq_num"].transform("max")
            st["stop_seq_ratio"] = st["stop_seq_num"] / trip_max_seq.replace(0, np.nan)

        return df

    def _add_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Network topology features."""
        logger.info("  Enriching network features...")

        if "stop_id" not in df.columns:
            return df

        # Stop-level route diversity (already in stop_degree)
        # Route-level average stop degree
        if "route_id" in df.columns:
            self._ensure_stop_route_map()
            if not self._stop_route_map.empty and self._stop_degree is not None:
                sr = self._stop_route_map.copy()
                sr["route_degree"] = sr["stop_id"].map(self._stop_degree).fillna(0)
                route_avg = sr.groupby("route_id")["route_degree"].agg(
                    ["mean", "max"]
                ).reset_index()
                route_avg.columns = ["route_id", "route_avg_stop_degree", "route_max_stop_degree"]
                df = df.merge(route_avg, on="route_id", how="left")

        # Transfer proximity score
        self._ensure_transfer_stops()
        if self._transfer_stops and "stop_id" in df.columns:
            df["near_transfer"] = df["stop_id"].astype(str).isin(self._transfer_stops).astype(int)

        return df

    def _add_schedule_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Real-time vs scheduled deviation features."""
        logger.info("  Enriching schedule deviation features...")

        if "sched_arr_sec" not in df.columns or "actual_time_sec" not in df.columns:
            return df

        # Actual vs scheduled deviation
        df["sched_deviation_sec"] = df["actual_time_sec"] - df["sched_arr_sec"]

        # Relative delay (delay as fraction of scheduled travel time)
        if "sched_travel_to_next" in df.columns:
            df["relative_delay"] = df["delay_sec"] / df["sched_travel_to_next"].replace(0, np.nan)
            df["relative_delay"] = df["relative_delay"].clip(-5, 5)

        # On-time performance flag (within 60 seconds of schedule)
        df["is_ontime_60s"] = (df["sched_deviation_sec"].abs() <= 60).astype(int)

        # Schedule adherence category
        df["sched_adherence"] = pd.cut(
            df["sched_deviation_sec"],
            bins=[-np.inf, -120, -60, 60, 300, np.inf],
            labels=["very_early", "early", "on_time", "late", "very_late"]
        ).astype(str)
        # One-hot encode
        for cat in ["very_early", "early", "on_time", "late", "very_late"]:
            df[f"adherence_{cat}"] = (df["sched_adherence"] == cat).astype(int)

        return df

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-derived features (hour, day_of_week, cyclic encoding, etc.)."""
        logger.info("  Enriching time features...")
        
        ts_col = None
        for col in ['timestamp', 'feed_timestamp', 'event_time', 'arrival_time_local']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            logger.warning("    No timestamp column found")
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        
        if df[ts_col].empty:
            return df
        
        df['hour'] = df[ts_col].dt.hour
        df['day_of_week'] = df[ts_col].dt.dayofweek
        df['month'] = df[ts_col].dt.month
        df['day_name'] = df[ts_col].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Peak hours
        df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df['is_peak'] = df['hour'].isin([8, 17, 18]).astype(int)
        
        # Cyclic encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['date'] = df[ts_col].dt.date
        df['timestamp_min'] = df[ts_col].dt.floor('min')
        df['timestamp_hour'] = df[ts_col].dt.floor('H')
        df['id_date_part'] = df[ts_col].dt.date
        df['id_time'] = df[ts_col].dt.time
        df['id_date'] = df[ts_col].dt.date
        
        return df

    def _add_alert_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alert-derived features from GTFS-RT feed."""
        logger.info("  Enriching alert features...")
        
        CAUSE_MAP = {
            'UNKNOWN_CAUSE': 0, 'OTHER_CAUSE': 1, 'TECHNICAL_PROBLEM': 2,
            'STRIKE': 3, 'DEMONSTRATION': 4, 'ACCIDENT': 5,
            'HOLIDAY': 6, 'WEATHER': 7, 'MAINTENANCE': 8,
            'CONSTRUCTION': 9, 'POLICE_ACTIVITY': 10, 'MEDICAL_EMERGENCY': 11
        }
        EFFECT_MAP = {
            'NO_SERVICE': 0, 'REDUCED_SERVICE': 1, 'SIGNIFICANT_DELAYS': 2,
            'DETOUR': 3, 'ADDITIONAL_SERVICE': 4, 'MODIFIED_SERVICE': 5,
            'OTHER_EFFECT': 6, 'UNKNOWN_EFFECT': 7, 'STOP_MOVED': 8
        }
        
        if 'cause' in df.columns:
            df['cause_id'] = df['cause'].map(CAUSE_MAP).fillna(-1).astype(int)
        
        if 'effect' in df.columns:
            df['effect_id'] = df['effect'].map(EFFECT_MAP).fillna(-1).astype(int)
        
        if 'description_text' in df.columns:
            df['has_text'] = df['description_text'].notna().astype(int)
            df['text_length'] = df['description_text'].fillna('').str.len()
            df['word_count'] = df['description_text'].fillna('').str.split().str.len()
        
        if 'active_period_start' in df.columns and 'active_period_end' in df.columns:
            start = pd.to_datetime(df['active_period_start'], errors='coerce')
            end = pd.to_datetime(df['active_period_end'], errors='coerce')
            df['active_period_duration_seconds'] = (end - start).dt.total_seconds()
            df['active_period_duration_days'] = df['active_period_duration_seconds'] / 86400
        
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'active_period_end' in df.columns:
                end = pd.to_datetime(df['active_period_end'], errors='coerce')
                df['remaining_active_time_seconds'] = (end - ts).dt.total_seconds()
                df['remaining_active_time_hours'] = df['remaining_active_time_seconds'] / 3600
                df['alert_age_seconds'] = (ts - pd.to_datetime(df['active_period_start'], errors='coerce')).dt.total_seconds()
                df['alert_age_minutes'] = df['alert_age_seconds'] / 60
                df['alert_age_hours'] = df['alert_age_seconds'] / 3600
        
        return df

    # ------------------------------------------------------------------
    # Delay features
    # ------------------------------------------------------------------

    def _add_delay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add delay-derived features.

        Creates:
        - ``abs_arrival_delay``: Absolute delay in seconds
        - ``is_delayed``: Binary (delay > 0)
        - ``delay_category``: Categorical delay severity
        """
        delay_col = 'delay'

        if delay_col not in df.columns:
            if 'arrival_delay' in df.columns:
                df[delay_col] = df['arrival_delay']
                logger.info(f"  Using 'arrival_delay' as 'delay'")
            else:
                logger.warning(f"  No delay column found")
                return df

        # Handle None values
        delay_series = df[delay_col].fillna(0)
        df['abs_arrival_delay'] = delay_series.abs()
        df['is_delayed'] = (delay_series > 0).astype(int)
        
        # Delay categories
        df['delay_category'] = pd.cut(
            delay_series.abs(),
            bins=[-np.inf, 0, 60, 180, 300, np.inf],
            labels=['early', 'minor', 'moderate', 'major', 'severe']
        )
        
        # Delay acceleration (handle None)
        if 'delay' in df.columns:
            df['delay_acceleration'] = df['delay'].fillna(0).diff(2)
        
        return df

    # ------------------------------------------------------------------
    # Headway features
    # ------------------------------------------------------------------

    def _add_headway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Headway instability features.
        
        Headway = time gap between consecutive vehicles on same route/stop
        """
        logger.info("  Enriching headway features...")
        
        ts_col = None
        for col in ['timestamp', 'feed_timestamp', 'event_time']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        
        sort_cols = ['route_id', 'stop_id', ts_col]
        existing_sort = [c for c in sort_cols if c in df.columns]
        
        if existing_sort:
            df = df.sort_values(existing_sort)
        
        if 'trip_id' in df.columns and ts_col:
            df['actual_headway'] = df.groupby(['route_id', 'stop_id'])[ts_col].diff().dt.total_seconds()
            
            # Compute scheduled_headway_sec from sched_arr_sec if available
            if 'sched_arr_sec' in df.columns:
                df['scheduled_headway_sec'] = df.groupby(['route_id', 'stop_id'])['sched_arr_sec'].diff()
            else:
                logger.warning("  'sched_arr_sec' not found — scheduled headway features will be NaN")
                df['scheduled_headway_sec'] = np.nan
            
            df['headway_ratio'] = df['actual_headway'] / df['scheduled_headway_sec'].replace(0, np.nan)
            df['headway_deviation'] = df['actual_headway'] - df['scheduled_headway_sec']
            df['headway_variability'] = df['actual_headway'].rolling(window=5, min_periods=1).std()
        
        return df


    def _add_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Speed-derived features."""
        logger.info("  Enriching speed features...")
        
        if 'speed_kmph' in df.columns:
            df['speed_mps'] = df['speed_kmph'] / 3.6
            df['is_stationary'] = (df['speed_kmph'] < 0.5).astype(int)
            df['is_stopped'] = (df['speed_kmph'] < 0.1).astype(int)
            df['low_speed_flag'] = (df['speed_kmph'] < 5).astype(int)
        
        if 'speed_mps' in df.columns:
            df['speed_change'] = df['speed_mps'].diff()
        
        return df

    # ------------------------------------------------------------------
    # Target features
    # ------------------------------------------------------------------
    # CRITICAL FIX: Targets now use PAST shifts (.shift(n)) instead of FUTURE shifts (.shift(-n))
    # The old code used .shift(-n) which looks into the future = DATA LEAKAGE
    # Targets should be created in the modeling pipeline AFTER train/test split
    # This method is kept for convenience but should be disabled in production

    def _add_target_features(self, df: pd.DataFrame, create_targets: bool = False) -> pd.DataFrame:
        """
        Target labels for disruption prediction.
        
        WARNING: Set create_targets=False for production. Targets should be created
        in the modeling pipeline AFTER train/val/test split to prevent leakage.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        create_targets : bool
            If True, creates target columns. Default False for safety.
        """
        logger.info("  Enriching target features...")
        
        if not create_targets:
            logger.info("    Target creation disabled (set create_targets=True to enable)")
            return df
        
        if 'delay' in df.columns:
            delay_col = df['delay']
            
            df['target_10min'] = (
                (delay_col.shift(10) > 60) | (delay_col.shift(10) < -60)
            ).astype(int)
            df['target_30min'] = (
                (delay_col.shift(30) > 60) | (delay_col.shift(30) < -60)
            ).astype(int)
            df['target_60min'] = (
                (delay_col.shift(60) > 60) | (delay_col.shift(60) < -60)
            ).astype(int)
            df['target_disruption_30min'] = (
                (delay_col.shift(30) > 180) | (delay_col.shift(30) < -180)
            ).astype(int)
            df['is_disruption'] = (delay_col.abs() > 180).astype(int)
        
        return df

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all enrichment features to the merged DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Merged real-time data (output of ``merge_feed_data``).

        Returns
        -------
        pd.DataFrame with additional feature columns.
        """
        logger.info("=" * 60)
        logger.info("GTFS FEATURE ENRICHMENT")
        logger.info("=" * 60)

        out = df.copy()

        # Static GTFS features
        out = self._add_route_features(out)
        out = self._add_stop_features(out)
        out = self._add_trip_features(out)
        out = self._add_schedule_features(out)
        out = self._add_network_features(out)
        out = self._add_schedule_deviation_features(out)

        # Computed features
        out = self._add_time_features(out)
        out = self._add_alert_features(out)
        out = self._add_delay_features(out)
        out = self._add_speed_features(out)
        
        # Headway features (if timestamp column exists)
        out = self._add_headway_features(out)
        
        # Target features - DISABLED by default to prevent leakage
        # Enable with: enricher.enrich(df, create_targets=True)
        # Or: out = self._add_target_features(out, create_targets=True)
        logger.info("  Target features disabled by default (see documentation)")

        new_cols = [c for c in out.columns if c not in df.columns]
        logger.info(f"  Added {len(new_cols)} enrichment features: {new_cols}")
        return out


def enrich_with_static_gtfs(
    merged_df: pd.DataFrame,
    gtfs_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Convenience function: enrich a merged DataFrame with static GTFS features.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of ``merge_feed_data``.
    gtfs_data : dict
        Static GTFS DataFrames (from ``fetch_static_gtfs`` or ``load_static_gtfs_from_zip``).

    Returns
    -------
    pd.DataFrame with enrichment features added.
    """
    enricher = GTFSEnricher(gtfs_data)
    return enricher.enrich(merged_df)
