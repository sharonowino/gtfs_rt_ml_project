"""
Merge feed_data_2 zip archives into a single merged_with_alerts.parquet
for consumption by mm_pipeline.py.
"""
import zipfile
import pandas as pd
import numpy as np
import os
import json

BASE = r"C:\Users\Sharon\Documents\python_learning\code\feed_data_2"
OUT  = r"C:\Users\Sharon\Documents\python_learning\code\merged_with_alerts.parquet"

ZIP_TRIP_UPDATES  = os.path.join(BASE, "04-01-2026-20-51-53_files_list.zip")
ZIP_ALERTS        = os.path.join(BASE, "04-01-2026-20-53-58_files_list.zip")
ZIP_VEHICLE_POS   = os.path.join(BASE, "04-01-2026-21-04-06_files_list.zip")


def load_zip_parquets(zip_path, prefix=""):
    """Load all parquet files from a zip archive, concatenate them."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        print(f"  Loading {len(names)} files from {os.path.basename(zip_path)}...")
        for i, name in enumerate(names):
            with zf.open(name) as f:
                df = pd.read_parquet(f)
                frames.append(df)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(names)} loaded")
    result = pd.concat(frames, ignore_index=True)
    print(f"  -> {len(result):,} rows, {len(result.columns)} cols")
    return result


# ── Step 1: Load all three feeds ──────────────────────────────────────────
print("=" * 60)
print("LOADING FEED DATA")
print("=" * 60)

df_tu = load_zip_parquets(ZIP_TRIP_UPDATES)   # Trip updates
df_al = load_zip_parquets(ZIP_ALERTS)          # Service alerts
df_vp = load_zip_parquets(ZIP_VEHICLE_POS)     # Vehicle positions

# ── Step 2: Clean column names for alerts (flatten JSON fields) ───────────
print("\n" + "=" * 60)
print("PROCESSING ALERTS")
print("=" * 60)

# Extract route_id from informed_entities JSON
def extract_route_id(val):
    try:
        if isinstance(val, str):
            entities = json.loads(val)
        elif isinstance(val, list):
            entities = val
        else:
            return None
        for ent in entities:
            rid = ent.get("route_id") or ent.get("trip_route_id")
            if rid:
                return str(rid)
    except:
        pass
    return None

df_al["alert_route_id"] = df_al["informed_entities"].apply(extract_route_id)

# Extract header text
def extract_text(val):
    try:
        if isinstance(val, str):
            items = json.loads(val)
        elif isinstance(val, list):
            items = val
        else:
            return ""
        for item in items:
            if isinstance(item, dict) and "text" in item:
                return item["text"]
    except:
        pass
    return ""

df_al["alert_header_text"] = df_al["header_text"].apply(extract_text)
df_al["alert_description_text"] = df_al["description_text"].apply(extract_text)

# Map cause/effect enums to strings
CAUSE_MAP = {1: "UNKNOWN_CAUSE", 2: "OTHER_CAUSE", 3: "TECHNICAL_PROBLEM",
             4: "STRIKE", 5: "DEMONSTRATION", 6: "ACCIDENT", 7: "HOLIDAY",
             8: "WEATHER", 9: "MAINTENANCE", 10: "CONSTRUCTION", 11: "POLICE_ACTIVITY",
             12: "MEDICAL_EMERGENCY"}
EFFECT_MAP = {1: "NO_SERVICE", 2: "REDUCED_SERVICE", 3: "SIGNIFICANT_DELAYS",
              4: "DETOUR", 5: "ADDITIONAL_SERVICE", 6: "MODIFIED_SERVICE",
              7: "OTHER_EFFECT", 8: "UNKNOWN_EFFECT", 9: "STOP_MOVED"}

df_al["alert_cause"] = df_al["cause"].map(CAUSE_MAP).fillna("UNKNOWN_CAUSE")
df_al["alert_effect"] = df_al["effect"].map(EFFECT_MAP).fillna("UNKNOWN_EFFECT")
df_al["has_alert"] = True

# Deduplicate alerts
alert_cols_keep = ["alert_route_id", "alert_cause", "alert_effect", "has_alert",
                   "alert_header_text", "alert_description_text", "retrieved_at"]
df_al_clean = df_al[alert_cols_keep].drop_duplicates(subset=["alert_route_id", "retrieved_at"])
print(f"  Unique alert-route combinations: {len(df_al_clean):,}")

# ── Step 3: Clean vehicle positions ──────────────────────────────────────
print("\n" + "=" * 60)
print("PROCESSING VEHICLE POSITIONS")
print("=" * 60)

# Convert speed to numeric
df_vp["speed"] = pd.to_numeric(df_vp["speed"], errors="coerce")

# Rename columns to avoid conflicts
df_vp_renamed = df_vp.rename(columns={
    "latitude": "vehicle_lat",
    "longitude": "vehicle_lon",
})
vp_cols = ["trip_id", "vehicle_id", "vehicle_lat", "vehicle_lon", "speed",
           "current_status", "retrieved_at"]
df_vp_clean = df_vp_renamed[vp_cols].drop_duplicates(subset=["trip_id", "retrieved_at"])
print(f"  Unique vehicle-position records: {len(df_vp_clean):,}")

# ── Step 4: Clean trip updates ───────────────────────────────────────────
print("\n" + "=" * 60)
print("PROCESSING TRIP UPDATES")
print("=" * 60)

# Convert delay columns to numeric
for col in ["arrival_delay", "departure_delay", "arrival_time", "departure_time"]:
    df_tu[col] = pd.to_numeric(df_tu[col], errors="coerce")

# Use departure_delay as primary delay signal
df_tu["delay_sec"] = df_tu["departure_delay"].fillna(df_tu["arrival_delay"])
df_tu["delay_min"] = df_tu["delay_sec"] / 60.0

# Speed flag
def speed_flag(speed):
    if pd.isna(speed):
        return "unknown"
    if speed <= 2:
        return "stopped"
    if speed <= 10:
        return "slow"
    return "normal"

# Delay flag
def delay_flag(delay):
    if pd.isna(delay):
        return "unknown"
    if delay > 60:
        return "late"
    if delay < -60:
        return "early"
    return "on_time"

print(f"  Trip update rows: {len(df_tu):,}")

# ── Step 5: Merge trip updates + vehicle positions ───────────────────────
print("\n" + "=" * 60)
print("MERGING TRIP UPDATES + VEHICLE POSITIONS")
print("=" * 60)

df_merged = df_tu.merge(
    df_vp_clean,
    on=["trip_id", "retrieved_at"],
    how="left",
    suffixes=("", "_vp")
)

# Fill speed from vp if available
if "speed_vp" in df_merged.columns:
    df_merged["speed"] = df_merged["speed"].fillna(df_merged["speed_vp"])
    df_merged = df_merged.drop(columns=["speed_vp"])

df_merged["speed"] = pd.to_numeric(df_merged["speed"], errors="coerce")
df_merged["speed_flag"] = df_merged["speed"].apply(speed_flag)
df_merged["delay_flag"] = df_merged["delay_sec"].apply(delay_flag)

print(f"  Merged rows: {len(df_merged):,}")

# ── Step 6: Merge alerts by route_id ─────────────────────────────────────
print("\n" + "=" * 60)
print("MERGING ALERTS")
print("=" * 60)

# Get unique active alerts per route (latest snapshot per route)
df_al_route = df_al_clean.drop_duplicates(subset=["alert_route_id"], keep="last")
df_al_route = df_al_route.rename(columns={"alert_route_id": "route_id"})

# Drop conflicting columns from alerts before merge
alert_merge_cols = ["route_id", "alert_cause", "alert_effect", "has_alert",
                    "alert_header_text", "alert_description_text"]
df_al_route = df_al_route[alert_merge_cols].drop_duplicates(subset=["route_id"])

df_merged = df_merged.merge(df_al_route, on="route_id", how="left")
df_merged["has_alert"] = df_merged["has_alert"].fillna(False)
df_merged["alert_cause"] = df_merged["alert_cause"].fillna("NONE")
df_merged["alert_effect"] = df_merged["alert_effect"].fillna("NONE")

print(f"  Final merged rows: {len(df_merged):,}")

# ── Step 7: Create disruption targets ────────────────────────────────────
print("\n" + "=" * 60)
print("CREATING DISRUPTION TARGETS")
print("=" * 60)

def classify_disruption(row):
    """Priority-ordered disruption classifier."""
    # CANCELLED
    status = str(row.get("current_status", "")).upper()
    if status == "3":  # CANCELED = 3 in GTFS-RT enum
        return "CANCELLED"
    if "NO_SERVICE" in str(row.get("alert_effect", "")).upper():
        return "CANCELLED"
    # MAJOR_DELAY
    delay = row.get("delay_sec")
    if pd.notna(delay) and delay > 600:
        return "MAJOR_DELAY"
    # STOPPED_ON_ROUTE
    speed = row.get("speed")
    sf = row.get("speed_flag", "unknown")
    if sf == "stopped" and pd.notna(delay) and delay > 120:
        return "STOPPED_ON_ROUTE"
    # MINOR_DELAY
    if pd.notna(delay) and delay > 120:
        return "MINOR_DELAY"
    # SLOW_TRAFFIC
    if sf == "slow":
        return "SLOW_TRAFFIC"
    # EARLY
    if pd.notna(delay) and delay < -60:
        return "EARLY"
    # SERVICE_ALERT
    if row.get("has_alert", False):
        return "SERVICE_ALERT"
    return "ON_TIME"

df_merged["disruption_type"] = df_merged.apply(classify_disruption, axis=1)
df_merged["disruption_target"] = (df_merged["disruption_type"] != "ON_TIME").astype(int)

# Multi-class mapping
CLASS_MAP = {"EARLY": 0, "MAJOR_DELAY": 1, "MINOR_DELAY": 2, "ON_TIME": 3,
             "SERVICE_ALERT": 4, "CANCELLED": 5, "STOPPED_ON_ROUTE": 6, "SLOW_TRAFFIC": 7}
df_merged["disruption_class"] = df_merged["disruption_type"].map(CLASS_MAP).fillna(3).astype(int)

# Severity score
SEVERITY = {"CANCELLED": 10, "MAJOR_DELAY": 7, "STOPPED_ON_ROUTE": 5,
            "MINOR_DELAY": 5, "SLOW_TRAFFIC": 3, "EARLY": 3,
            "SERVICE_ALERT": 3, "ON_TIME": 0}
df_merged["severity_score"] = df_merged["disruption_type"].map(SEVERITY).fillna(0)

# Add feed_timestamp from retrieved_at
df_merged["feed_timestamp"] = df_merged["retrieved_at"]

# Print distribution
print("\n  Disruption type distribution:")
counts = df_merged["disruption_type"].value_counts()
for k, v in counts.items():
    print(f"    {k:<22} {v:>8,} ({v/len(df_merged)*100:.2f}%)")
print(f"\n  Binary disruption rate: {df_merged['disruption_target'].mean():.4%}")

# ── Step 8: Save ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING")
print("=" * 60)

df_merged.to_parquet(OUT, index=False)
print(f"  Saved: {OUT}")
print(f"  Rows: {len(df_merged):,}  Cols: {len(df_merged.columns)}")
print(f"  Size: {os.path.getsize(OUT)/1024/1024:.1f} MB")
print("\nDone.")
