"""
GTFS Data Ingestion Module
==========================
Fetches and parses GTFS-RT (protobuf) and static GTFS data from:
  - Local parquet files (feed_data/ directory, downloaded from MinIO)
  - Live feed URLs (gtfs.ovapi.nl)
  - Static GTFS zip (gtfs.ovapi.nl/gtfs-nl.zip)

Produces merged DataFrames compatible with DisruptionFeatureBuilder.
"""
import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_FEED_URLS = {
    "alerts": "http://gtfs.ovapi.nl/nl/alerts.pb",
    "vehiclePositions": "http://gtfs.ovapi.nl/nl/vehiclePositions.pb",
    "tripUpdates": "http://gtfs.ovapi.nl/nl/tripUpdates.pb",
}
DEFAULT_STATIC_GTFS_URL = "http://gtfs.ovapi.nl/gtfs-nl.zip"
DEFAULT_LOCAL_DIR = "feed_data"

ALERT_CAUSE_MAP = {
    1: "UNKNOWN_CAUSE", 2: "OTHER_CAUSE", 3: "TECHNICAL_PROBLEM",
    4: "STRIKE", 5: "DEMONSTRATION", 6: "ACCIDENT", 7: "HOLIDAY",
    8: "WEATHER", 9: "MAINTENANCE", 10: "CONSTRUCTION",
    11: "POLICE_ACTIVITY", 12: "MEDICAL_EMERGENCY",
}

ALERT_EFFECT_MAP = {
    1: "NO_SERVICE", 2: "REDUCED_SERVICE", 3: "SIGNIFICANT_DELAYS",
    4: "DETOUR", 5: "ADDITIONAL_SERVICE", 6: "MODIFIED_SERVICE",
    7: "OTHER_EFFECT", 8: "UNKNOWN_EFFECT", 9: "STOP_MOVED",
    10: "NO_EFFECT", 11: "ACCESSIBILITY_ISSUE",
}


# ---------------------------------------------------------------------------
# Protobuf parsers
# ---------------------------------------------------------------------------

def _parse_vehicle_positions(feed: gtfs_realtime_pb2.FeedMessage) -> pd.DataFrame:
    """Parse a GTFS-RT VehiclePositions feed into a DataFrame."""
    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        trip = v.trip
        pos = v.position
        rows.append({
            "entity_id": entity.id,
            "trip_id": trip.trip_id or None,
            "route_id": trip.route_id or None,
            "direction_id": trip.direction_id if trip.direction_id else np.nan,
            "start_time": trip.start_time or None,
            "start_date": trip.start_date or None,
            "schedule_relationship": int(trip.schedule_relationship),
            "vehicle_id": v.vehicle.id or None,
            "vehicle_label": v.vehicle.label or None,
            "license_plate": v.vehicle.license_plate or None,
            "wheelchair_accessible": None,
            "latitude": pos.latitude if pos.latitude else np.nan,
            "longitude": pos.longitude if pos.longitude else np.nan,
            "bearing": pos.bearing if pos.bearing else np.nan,
            "odometer": pos.odometer if pos.odometer else np.nan,
            "speed": pos.speed if pos.speed else np.nan,
            "current_stop_sequence": float(v.current_stop_sequence) if v.current_stop_sequence else np.nan,
            "stop_id": v.stop_id or None,
            "current_status": int(v.current_status),
            "timestamp": int(v.timestamp) if v.timestamp else int(feed.header.timestamp),
            "congestion_level": str(v.congestion_level) if v.congestion_level else None,
            "occupancy_status": str(v.occupancy_status) if v.occupancy_status else None,
            "occupancy_percentage": str(v.occupancy_percentage) if v.occupancy_percentage else None,
            "multi_carriage_details": None,
            "retrieved_at": pd.Timestamp.now(),
        })
    return pd.DataFrame(rows)


def _parse_trip_updates(feed: gtfs_realtime_pb2.FeedMessage) -> pd.DataFrame:
    """Parse a GTFS-RT TripUpdates feed into a flattened DataFrame."""
    rows = []
    header_ts = feed.header.timestamp
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        trip = tu.trip
        vehicle = tu.vehicle

        # One row per stop_time_update (flattened)
        if tu.stop_time_update:
            for stu in tu.stop_time_update:
                rows.append({
                    "entity_id": entity.id,
                    "trip_id": trip.trip_id or None,
                    "route_id": trip.route_id or None,
                    "direction_id": trip.direction_id if trip.direction_id else np.nan,
                    "start_time": trip.start_time or None,
                    "start_date": trip.start_date or None,
                    "schedule_relationship": int(trip.schedule_relationship),
                    "vehicle_id": vehicle.id if vehicle.id else None,
                    "vehicle_label": vehicle.label or None,
                    "license_plate": None,
                    "trip_update_timestamp": pd.Timestamp.utcfromtimestamp(int(tu.timestamp) if tu.timestamp else header_ts),
                    "delay": tu.delay if tu.delay else 0,
                    "trip_properties_trip_id": trip.trip_id or None,
                    "trip_properties_start_date": trip.start_date or None,
                    "trip_properties_start_time": trip.start_time or None,
                    "trip_properties_shape_id": None,
                    "stop_sequence": float(stu.stop_sequence) if stu.stop_sequence else np.nan,
                    "stop_id": stu.stop_id or None,
                    "stop_time_schedule_relationship": int(stu.schedule_relationship),
                    "arrival_delay": stu.arrival.delay if stu.arrival.delay else 0,
                    "arrival_time": int(stu.arrival.time) if stu.arrival.time else np.nan,
                    "arrival_uncertainty": stu.arrival.uncertainty if stu.arrival.uncertainty else np.nan,
                    "departure_delay": stu.departure.delay if stu.departure.delay else 0,
                    "departure_time": int(stu.departure.time) if stu.departure.time else np.nan,
                    "departure_uncertainty": stu.departure.uncertainty if stu.departure.uncertainty else np.nan,
                    "assigned_stop_id": stu.stop_id or None,
                    "departure_occupancy_status": None,
                    "retrieved_at": pd.Timestamp.now(),
                })
        else:
            # Trip update with no stop_time_update entries
            rows.append({
                "entity_id": entity.id,
                "trip_id": trip.trip_id or None,
                "route_id": trip.route_id or None,
                "direction_id": trip.direction_id if trip.direction_id else np.nan,
                "start_time": trip.start_time or None,
                "start_date": trip.start_date or None,
                "schedule_relationship": int(trip.schedule_relationship),
                "vehicle_id": vehicle.id if vehicle.id else None,
                "vehicle_label": vehicle.label or None,
                "license_plate": None,
                "trip_update_timestamp": pd.Timestamp.utcfromtimestamp(int(tu.timestamp) if tu.timestamp else header_ts),
                "delay": tu.delay if tu.delay else 0,
                "trip_properties_trip_id": trip.trip_id or None,
                "trip_properties_start_date": trip.start_date or None,
                "trip_properties_start_time": trip.start_time or None,
                "trip_properties_shape_id": None,
                "stop_sequence": np.nan,
                "stop_id": None,
                "stop_time_schedule_relationship": np.nan,
                "arrival_delay": 0,
                "arrival_time": np.nan,
                "arrival_uncertainty": np.nan,
                "departure_delay": 0,
                "departure_time": np.nan,
                "departure_uncertainty": np.nan,
                "assigned_stop_id": None,
                "departure_occupancy_status": None,
                "retrieved_at": pd.Timestamp.now(),
            })
    return pd.DataFrame(rows)


def _parse_alerts(feed: gtfs_realtime_pb2.FeedMessage) -> pd.DataFrame:
    """Parse a GTFS-RT Alerts feed into a DataFrame."""
    rows = []
    for entity in feed.entity:
        if not entity.HasField("alert"):
            continue
        a = entity.alert

        header_text = None
        if a.header_text.translation:
            header_text = a.header_text.translation[0].text

        desc_text = None
        if a.description_text.translation:
            desc_text = a.description_text.translation[0].text

        url_text = None
        if a.url.translation:
            url_text = a.url.translation[0].text

        tts_header = None
        if a.tts_header_text.translation:
            tts_header = a.tts_header_text.translation[0].text

        tts_desc = None
        if a.tts_description_text.translation:
            tts_desc = a.tts_description_text.translation[0].text

        # Serialize active periods and informed entities as JSON strings
        active_periods = json.dumps([
            {"start": p.start, "end": p.end} for p in a.active_period
        ]) if a.active_period else None

        informed_entities = json.dumps([
            {
                "agency_id": ie.agency_id or None,
                "route_id": ie.route_id or None,
                "route_type": ie.route_type if ie.route_type else None,
                "trip_id": ie.trip.trip_id if ie.trip.trip_id else None,
                "stop_id": ie.stop_id or None,
            }
            for ie in a.informed_entity
        ]) if a.informed_entity else None

        rows.append({
            "entity_id": entity.id,
            "cause": ALERT_CAUSE_MAP.get(a.cause, "UNKNOWN_CAUSE"),
            "effect": ALERT_EFFECT_MAP.get(a.effect, "UNKNOWN_EFFECT"),
            "severity_level": str(a.severity_level) if getattr(a, 'severity_level', None) else None,
            "active_periods": active_periods,
            "informed_entities": informed_entities,
            "header_text": header_text,
            "description_text": desc_text,
            "url": url_text,
            "tts_header_text": tts_header,
            "tts_description_text": tts_desc,
            "image": getattr(a, 'image', None),
            "image_alternative_text": getattr(a, 'image_alternative_text', None),
            "cause_detail": str(a.cause_detail) if hasattr(a, 'cause_detail') and a.cause_detail else None,
            "effect_detail": str(a.effect_detail) if hasattr(a, 'effect_detail') and a.effect_detail else None,
            "retrieved_at": pd.Timestamp.now(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feed fetchers
# ---------------------------------------------------------------------------

def fetch_gtfs_rt(url: str, timeout: int = 30) -> gtfs_realtime_pb2.FeedMessage:
    """Fetch and parse a single GTFS-RT protobuf feed."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    return feed


def fetch_all_live_feeds(
    urls: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all three GTFS-RT feeds (live) and return parsed DataFrames.

    Returns
    -------
    Dict with keys 'vehicle_positions', 'trip_updates', 'alerts'.
    """
    if urls is None:
        urls = DEFAULT_FEED_URLS

    logger.info("Fetching live GTFS-RT feeds...")

    feed_vp = fetch_gtfs_rt(urls["vehiclePositions"], timeout=timeout)
    df_vp = _parse_vehicle_positions(feed_vp)
    logger.info(f"  Vehicle positions: {len(df_vp)} rows")

    feed_tu = fetch_gtfs_rt(urls["tripUpdates"], timeout=timeout)
    df_tu = _parse_trip_updates(feed_tu)
    logger.info(f"  Trip updates: {len(df_tu)} rows")

    feed_al = fetch_gtfs_rt(urls["alerts"], timeout=timeout)
    df_al = _parse_alerts(feed_al)
    logger.info(f"  Alerts: {len(df_al)} rows")

    return {
        "vehicle_positions": df_vp,
        "trip_updates": df_tu,
        "alerts": df_al,
    }


# ---------------------------------------------------------------------------
# Local parquet readers
# ---------------------------------------------------------------------------

def _read_parquet_from_zip(zip_path: str, max_files: Optional[int] = None) -> pd.DataFrame:
    """Read parquet files from a zip archive and concatenate them.

    Parameters
    ----------
    zip_path : str
        Path to the zip file.
    max_files : int, optional
        Maximum number of parquet files to read (for memory control).
        Files are sampled evenly across the sorted list to maximise
        temporal coverage.  If None, reads all files.
    """
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        parquet_names = sorted(n for n in z.namelist() if n.endswith(".parquet"))
        if max_files is not None and max_files < len(parquet_names):
            # Sample evenly across the full list for temporal spread
            indices = np.linspace(0, len(parquet_names) - 1, max_files, dtype=int)
            parquet_names = [parquet_names[i] for i in indices]
        for name in parquet_names:
            frames.append(pd.read_parquet(io.BytesIO(z.read(name))))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_local_feeds(
    local_dir: str = DEFAULT_LOCAL_DIR,
    max_files: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load GTFS data from local zip files in *local_dir*.

    Expected zip files (in filename-alphabetical order):
      1st zip -> vehicle positions
      2nd zip -> trip updates
      3rd zip -> alerts

    Parameters
    ----------
    local_dir : str
        Path to directory containing *_files_list.zip files.
    max_files : int, optional
        Max parquet files to read per zip (for memory control).

    Returns
    -------
    Dict with keys 'vehicle_positions', 'trip_updates', 'alerts'.
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local feed directory not found: {local_dir}")

    zip_files = sorted(local_path.glob("*_files_list.zip"))
    if len(zip_files) < 3:
        raise ValueError(
            f"Expected at least 3 *_files_list.zip files in {local_dir}, "
            f"found {len(zip_files)}"
        )

    labels = ["vehicle_positions", "trip_updates", "alerts"]
    result = {}
    for label, zf in zip(labels, zip_files[:3]):
        logger.info(f"Loading {label} from {zf.name}...")
        df = _read_parquet_from_zip(str(zf), max_files=max_files)
        logger.info(f"  Loaded {len(df)} rows x {len(df.columns)} cols")
        result[label] = df
    
    # Auto-load static GTFS data
    logger.info("Auto-loading static GTFS data...")
    try:
        # Try to load from cached zip or data directory
        import os
        static_path = "gtfs-nl.zip"
        if not os.path.exists(static_path):
            static_path = os.path.join("..", "data", "gtfs-nl.zip")
        if not os.path.exists(static_path):
            static_path = os.path.join("data", "gtfs-nl.zip")
        
        if os.path.exists(static_path):
            static_gtfs = load_static_gtfs_from_zip(static_path)
            result['static'] = static_gtfs
            logger.info(f"  Static GTFS loaded: {list(static_gtfs.keys())}")
        else:
            # Try fetching from URL
            try:
                static_gtfs = fetch_static_gtfs()
                result['static'] = static_gtfs
                logger.info(f"  Static GTFS loaded: {list(static_gtfs.keys())}")
            except Exception as e2:
                logger.warning(f"  Could not fetch static GTFS: {e2}")
                result['static'] = {}
    except Exception as e:
        logger.warning(f"  Could not load static GTFS: {e}")
        result['static'] = {}

    return result


def load_extracted_parquet(
    extracted_dir: str = "extracted_parquet",
    max_files_per_dir: int = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all GTFS-RT data from extracted parquet files.
    
    Automatically loads static GTFS as well.
    
    Parameters
    ----------
    extracted_dir : str
        Directory with extracted parquet files (subdirectories per feed)
    max_files_per_dir : int, optional
        Max files to read per subdirectory
    
    Returns
    -------
    Dict with 'vehicle_positions', 'trip_updates', 'alerts', 'static'
    """
    import glob
    
    extracted_path = Path(extracted_dir)
    if not extracted_path.exists():
        raise FileNotFoundError(f"Extracted directory not found: {extracted_dir}")
    
    result = {
        'vehicle_positions': pd.DataFrame(),
        'trip_updates': pd.DataFrame(),
        'alerts': pd.DataFrame()
    }
    
    # Load parquet files from each subdirectory
    for subdir in sorted(extracted_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        logger.info(f"Loading from {subdir.name}...")
        parquet_files = sorted(subdir.glob("*.parquet"))
        
        if max_files_per_dir:
            parquet_files = parquet_files[:max_files_per_dir]
        
        for pq in parquet_files:
            try:
                df = pd.read_parquet(pq)
                # Determine type from filename pattern
                fname = pq.stem
                if 'vehicle' in fname.lower() or 'position' in fname.lower():
                    result['vehicle_positions'] = pd.concat([result['vehicle_positions'], df], ignore_index=True)
                elif 'trip' in fname.lower() or 'update' in fname.lower():
                    result['trip_updates'] = pd.concat([result['trip_updates'], df], ignore_index=True)
                elif 'alert' in fname.lower():
                    result['alerts'] = pd.concat([result['alerts'], df], ignore_index=True)
                else:
                    # Default: try to infer from columns
                    result['vehicle_positions'] = pd.concat([result['vehicle_positions'], df], ignore_index=True)
            except Exception as e:
                logger.warning(f"  Error reading {pq.name}: {e}")
    
    logger.info(f"  vehicle_positions: {len(result['vehicle_positions'])} rows")
    logger.info(f"  trip_updates: {len(result['trip_updates'])} rows")
    logger.info(f"  alerts: {len(result['alerts'])} rows")
    
    # Auto-load static GTFS
    logger.info("Auto-loading static GTFS data...")
    try:
        import os
        static_path = "gtfs-nl.zip"
        if not os.path.exists(static_path):
            static_path = os.path.join("..", "data", "gtfs-nl.zip")
        if not os.path.exists(static_path):
            static_path = os.path.join("data", "gtfs-nl.zip")
        
        if os.path.exists(static_path):
            static_gtfs = load_static_gtfs_from_zip(static_path)
            result['static'] = static_gtfs
            logger.info(f"  Static GTFS loaded: {list(static_gtfs.keys())}")
        else:
            try:
                static_gtfs = fetch_static_gtfs()
                result['static'] = static_gtfs
                logger.info(f"  Static GTFS loaded: {list(static_gtfs.keys())}")
            except Exception as e2:
                logger.warning(f"  Could not fetch static GTFS: {e2}")
                result['static'] = {}
    except Exception as e:
        logger.warning(f"  Could not load static GTFS: {e}")
        result['static'] = {}
    
    return result


# ---------------------------------------------------------------------------
# Static GTFS
# ---------------------------------------------------------------------------

GTFS_FILES = {
    "agency": "agency.txt",
    "routes": "routes.txt",
    "trips": "trips.txt",
    "stops": "stops.txt",
    "stop_times": "stop_times.txt",
    "calendar": "calendar.txt",
    "calendar_dates": "calendar_dates.txt",
    "shapes": "shapes.txt",
    "transfers": "transfers.txt",
}

# Files too large to load in full (sample instead)
GTFS_LARGE_FILES = {"stop_times", "shapes"}

# Files to skip entirely (too large for memory)
GTFS_SKIP_FILES = {"stop_times"}


def fetch_static_gtfs(url: str = DEFAULT_STATIC_GTFS_URL,
                      timeout: int = 120) -> Dict[str, pd.DataFrame]:
    """Download and parse a static GTFS zip archive.

    Checks for a local cached copy (``gtfs-nl.zip`` in the current directory
    or ``data/`` directory) before downloading.
    """
    import os
    # Check for a cached copy first
    if os.path.exists("gtfs-nl.zip"):
        logger.info("Using cached static GTFS from gtfs-nl.zip")
        return load_static_gtfs_from_zip("gtfs-nl.zip")

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    cache_path = os.path.normpath(os.path.join(cache_dir, "gtfs-nl.zip"))
    if os.path.exists(cache_path):
        logger.info(f"Using cached static GTFS from {cache_path}")
        return load_static_gtfs_from_zip(cache_path)

    logger.info(f"Downloading static GTFS from {url}...")
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    content = resp.content
    # Cache for future use
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(content)
    logger.info(f"Cached static GTFS to {cache_path}")
    return _parse_static_gtfs_zip(io.BytesIO(content))


def load_static_gtfs_from_zip(zip_path: str) -> Dict[str, pd.DataFrame]:
    """Load static GTFS from a local zip file."""
    logger.info(f"Loading static GTFS from {zip_path}...")
    with open(zip_path, "rb") as f:
        return _parse_static_gtfs_zip(io.BytesIO(f.read()))


def _parse_static_gtfs_zip(zip_bytes: io.BytesIO,
                           sample_large: int = 500000) -> Dict[str, pd.DataFrame]:
    """Parse static GTFS data from a zip file.

    Handles zips where files are at the root or in a subdirectory.

    Parameters
    ----------
    zip_bytes : io.BytesIO
        The zip file contents.
    sample_large : int
        For files in ``GTFS_LARGE_FILES``, read at most this many rows
        (set to 0 to skip them entirely).
    """
    result = {}
    with zipfile.ZipFile(zip_bytes) as z:
        all_names = z.namelist()

        # Detect subdirectory prefix (e.g. "gtfs-nl/")
        prefix = ""
        for candidate in ("", "gtfs-nl/", "gtfs_nl/"):
            if any((candidate + fn) in all_names for fn in GTFS_FILES.values()):
                prefix = candidate
                break

        for key, filename in GTFS_FILES.items():
            full_name = prefix + filename
            if full_name not in all_names:
                logger.warning(f"  {filename} not found in GTFS zip")
                result[key] = pd.DataFrame()
                continue
            if key in GTFS_SKIP_FILES:
                logger.info(f"  {key}: skipped (too large for memory)")
                result[key] = pd.DataFrame()
                continue
            try:
                if key in GTFS_LARGE_FILES and sample_large > 0:
                    # For very large files, read chunked to avoid memory issues
                    chunks = pd.read_csv(
                        io.BytesIO(z.read(full_name)), dtype=str,
                        low_memory=False, chunksize=sample_large,
                    )
                    df = next(chunks, pd.DataFrame())
                    logger.info(f"  {key}: {len(df)} rows (sampled from large file)")
                elif key in GTFS_LARGE_FILES and sample_large == 0:
                    result[key] = pd.DataFrame()
                    continue
                else:
                    df = pd.read_csv(io.BytesIO(z.read(full_name)), dtype=str, low_memory=False)
                    logger.info(f"  {key}: {len(df)} rows")
                result[key] = df
            except Exception as e:
                logger.warning(f"  {key}: parse error - {e}")
                result[key] = pd.DataFrame()
    return result


# ---------------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------------

def merge_feed_data(
    vehicle_positions: pd.DataFrame,
    trip_updates: pd.DataFrame,
    alerts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge vehicle positions, trip updates, and alerts into a single DataFrame.

    The primary join key is (trip_id, route_id).  Trip-update stop-level
    columns are prefixed ``tu_`` to avoid collisions.  Alert columns are
    prefixed ``alert_`` where necessary.
    """
    logger.info("Merging feed data sources...")

    vp = vehicle_positions.copy()
    tu = trip_updates.copy()
    al = alerts.copy()

    # --- standardise key columns ---
    for df in (vp, tu):
        for col in ("trip_id", "route_id", "stop_id"):
            if col in df.columns:
                df[col] = df[col].astype(str)

    if "trip_id" in al.columns:
        al["trip_id"] = al["trip_id"].astype(str)

    # --- merge VP + TU on trip_id (one-to-many) ---
    tu_key_cols = ["trip_id"]
    tu_extra = [c for c in tu.columns if c not in vp.columns or c == "trip_id"]
    tu_sub = tu[tu_extra].drop_duplicates(subset=["trip_id"], keep="first") if "trip_id" in tu.columns else tu

    merged = vp.merge(tu_sub, on="trip_id", how="left", suffixes=("", "_tu"))

    # --- derive feed_timestamp ---
    if "retrieved_at" in merged.columns:
        merged["feed_timestamp"] = pd.to_datetime(merged["retrieved_at"])
    elif "timestamp" in merged.columns:
        merged["feed_timestamp"] = pd.to_datetime(merged["timestamp"], unit="s", errors="coerce")
    else:
        merged["feed_timestamp"] = pd.Timestamp.now()

    # --- compute delay_sec from trip updates ---
    if "arrival_delay" in merged.columns:
        merged["delay_sec"] = merged["arrival_delay"].astype(float)
    elif "delay" in merged.columns:
        merged["delay_sec"] = merged["delay"].astype(float)
    else:
        merged["delay_sec"] = np.nan

    # --- derive scheduled / actual time columns ---
    def _parse_gtfs_time(t):
        """Parse GTFS time string (HH:MM:SS) to seconds from midnight."""
        if pd.isna(t):
            return np.nan
        if isinstance(t, (int, float)):
            return float(t)
        try:
            parts = str(t).split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except:
            return np.nan
    
    if "arrival_time" in merged.columns:
        if merged["arrival_time"].dtype == object:
            merged["actual_time_sec"] = merged["arrival_time"].apply(_parse_gtfs_time)
        else:
            merged["actual_time_sec"] = merged["arrival_time"].astype(float)
    if "departure_time" in merged.columns:
        # use departure as actual if arrival missing
        if "actual_time_sec" not in merged.columns:
            if merged["departure_time"].dtype == object:
                merged["actual_time_sec"] = merged["departure_time"].apply(_parse_gtfs_time)
            else:
                merged["actual_time_sec"] = merged["departure_time"].astype(float)

    # --- merge alerts by route_id ---
    # Live alerts store route info in informed_entities JSON; local parquets may
    # have a direct route_id column.  Try both approaches.
    if "route_id" in al.columns and "route_id" in merged.columns:
        # Direct route_id join (local parquet alerts)
        alert_new = [c for c in al.columns if c not in merged.columns or c == "route_id"]
        al_sub = al[alert_new].drop_duplicates(subset=["route_id"], keep="first")
        merged = merged.merge(al_sub, on="route_id", how="left", suffixes=("", "_alert"))
    elif "informed_entities" in al.columns and "route_id" in merged.columns:
        # Parse informed_entities JSON to extract route_id
        alert_rows = []
        for _, row in al.iterrows():
            try:
                entities = json.loads(row["informed_entities"]) if row["informed_entities"] else []
            except (json.JSONDecodeError, TypeError):
                entities = []
            for ie in entities:
                if ie.get("route_id"):
                    alert_rows.append({
                        "route_id": str(ie["route_id"]),
                        "cause": row.get("cause"),
                        "effect": row.get("effect"),
                        "severity_level": row.get("severity_level"),
                        "header_text": row.get("header_text"),
                        "description_text": row.get("description_text"),
                    })
        if alert_rows:
            alert_route_df = pd.DataFrame(alert_rows).drop_duplicates(subset=["route_id"], keep="first")
            merged = merged.merge(alert_route_df, on="route_id", how="left", suffixes=("", "_alert"))

    # --- alert indicator columns ---
    cause_col = "cause" if "cause" in merged.columns else ("cause_alert" if "cause_alert" in merged.columns else None)
    if cause_col:
        merged["has_overlapping_alert"] = merged[cause_col].notna()
        merged["alert_cause"] = merged[cause_col]
    else:
        merged["has_overlapping_alert"] = False

    effect_col = "effect" if "effect" in merged.columns else ("effect_alert" if "effect_alert" in merged.columns else None)
    if effect_col:
        merged["alert_effect"] = merged[effect_col]

    desc_col = "description_text" if "description_text" in merged.columns else None
    if desc_col:
        merged["alert_text"] = merged[desc_col]

    logger.info(f"  Merged shape: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# High-level ingestion entry points
# ---------------------------------------------------------------------------

def ingest_local(
    local_dir: str = DEFAULT_LOCAL_DIR,
    static_gtfs_zip: Optional[str] = None,
    static_gtfs_url: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Ingest data from local parquet files and return merged data + static GTFS.

    Parameters
    ----------
    local_dir : str
        Path to the directory containing *_files_list.zip files.
    static_gtfs_zip : str, optional
        Path to a local static GTFS zip file.
    static_gtfs_url : str, optional
        URL to download static GTFS from (used if *static_gtfs_zip* is None).
    max_files : int, optional
        Max parquet files to read per zip (for memory control).

    Returns
    -------
    (merged_df, gtfs_data) tuple ready for DisruptionFeatureBuilder.
    """
    feeds = load_local_feeds(local_dir, max_files=max_files)
    merged = merge_feed_data(
        feeds["vehicle_positions"],
        feeds["trip_updates"],
        feeds["alerts"],
    )

    if static_gtfs_zip:
        gtfs_data = load_static_gtfs_from_zip(static_gtfs_zip)
    elif static_gtfs_url:
        gtfs_data = fetch_static_gtfs(static_gtfs_url)
    else:
        # Try the default URL
        gtfs_data = fetch_static_gtfs()

    return merged, gtfs_data


def ingest_live(
    feed_urls: Optional[Dict[str, str]] = None,
    static_gtfs_url: str = DEFAULT_STATIC_GTFS_URL,
    timeout: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Ingest live GTFS-RT feeds and static GTFS.

    Parameters
    ----------
    feed_urls : dict, optional
        Override the default GTFS-RT URLs.
    static_gtfs_url : str
        URL for the static GTFS zip.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    (merged_df, gtfs_data) tuple ready for DisruptionFeatureBuilder.
    """
    feeds = fetch_all_live_feeds(urls=feed_urls, timeout=timeout)
    merged = merge_feed_data(
        feeds["vehicle_positions"],
        feeds["trip_updates"],
        feeds["alerts"],
    )
    gtfs_data = fetch_static_gtfs(static_gtfs_url, timeout=timeout)
    return merged, gtfs_data


def ingest_combined(
    local_dir: str = DEFAULT_LOCAL_DIR,
    feed_urls: Optional[Dict[str, str]] = None,
    static_gtfs_zip: Optional[str] = None,
    static_gtfs_url: Optional[str] = None,
    timeout: int = 30,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Ingest data from both local parquet files AND live feeds, then combine.

    Local data is loaded first, then live feed data is appended.  Duplicates
    on (entity_id, retrieved_at) are dropped, keeping the first occurrence.

    Parameters
    ----------
    local_dir : str
        Path to local feed_data directory.
    feed_urls : dict, optional
        Override live GTFS-RT URLs.
    static_gtfs_zip : str, optional
        Path to a local static GTFS zip file.
    static_gtfs_url : str, optional
        URL for static GTFS download.
    timeout : int
        HTTP timeout for live feeds.
    max_files : int, optional
        Max parquet files to read per zip (for memory control).

    Returns
    -------
    (merged_df, gtfs_data) tuple ready for DisruptionFeatureBuilder.
    """
    # --- Local ---
    logger.info("=" * 60)
    logger.info("INGESTING LOCAL DATA")
    logger.info("=" * 60)
    local_feeds = load_local_feeds(local_dir, max_files=max_files)

    # --- Live ---
    logger.info("=" * 60)
    logger.info("INGESTING LIVE DATA")
    logger.info("=" * 60)
    live_feeds = fetch_all_live_feeds(urls=feed_urls, timeout=timeout)

    # --- Concatenate each source ---
    combined_vp = pd.concat(
        [local_feeds["vehicle_positions"], live_feeds["vehicle_positions"]],
        ignore_index=True,
    )
    combined_tu = pd.concat(
        [local_feeds["trip_updates"], live_feeds["trip_updates"]],
        ignore_index=True,
    )
    combined_al = pd.concat(
        [local_feeds["alerts"], live_feeds["alerts"]],
        ignore_index=True,
    )

    # Deduplicate by entity_id + retrieved_at where possible
    for name, df in [("VP", combined_vp), ("TU", combined_tu), ("AL", combined_al)]:
        before = len(df)
        if "entity_id" in df.columns and "retrieved_at" in df.columns:
            df.drop_duplicates(subset=["entity_id", "retrieved_at"], keep="first", inplace=True)
        logger.info(f"  {name}: {before} -> {len(df)} after dedup")

    logger.info("=" * 60)
    logger.info("MERGING COMBINED DATA")
    logger.info("=" * 60)
    merged = merge_feed_data(combined_vp, combined_tu, combined_al)

    # --- Static GTFS ---
    if static_gtfs_zip:
        gtfs_data = load_static_gtfs_from_zip(static_gtfs_zip)
    elif static_gtfs_url:
        gtfs_data = fetch_static_gtfs(static_gtfs_url)
    else:
        gtfs_data = fetch_static_gtfs()

    return merged, gtfs_data
