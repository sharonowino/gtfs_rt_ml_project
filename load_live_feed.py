"""
Load Live GTFS-RT Feed Data
===========================
Fetches real-time data from ovapi.nl endpoints.
"""
import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2
import io

URLS = {
    "tripUpdates": "http://gtfs.ovapi.nl/nl/tripUpdates.pb",
    "vehiclePositions": "http://gtfs.ovapi.nl/nl/vehiclePositions.pb",
    "alerts": "http://gtfs.ovapi.nl/nl/alerts.pb",
}

def fetch_feed(url):
    """Fetch GTFS-RT feed from URL."""
    print(f"Fetching {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content

def parse_vehicle_positions(data):
    """Parse vehicle positions feed."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)
    
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
            "direction_id": trip.direction_id if trip.direction_id else None,
            "start_time": trip.start_time or None,
            "start_date": trip.start_date or None,
            "vehicle_id": v.vehicle.id or None,
            "vehicle_label": v.vehicle.label or None,
            "latitude": pos.latitude or None,
            "longitude": pos.longitude or None,
            "bearing": pos.bearing or None,
            "speed": pos.speed or None,
            "current_status": v.current_status or None,
            "congestion_level": v.congestion_level or None,
        })
    
    return pd.DataFrame(rows)

def parse_alerts(data):
    """Parse alerts feed."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)
    
    rows = []
    for entity in feed.entity:
        if not entity.HasField("alert"):
            continue
        alert = entity.alert
        cause = alert.Cause.Name(alert.cause) if alert.cause else None
        effect = alert.Effect.Name(alert.effect) if alert.effect else None
        rows.append({
            "entity_id": entity.id,
            "cause": cause,
            "effect": effect,
            "severity_level": alert.severity_level or None,
        })
    
    return pd.DataFrame(rows)

# Fetch and parse
print("Loading live GTFS-RT feed data...")

# Vehicle positions
vp_data = fetch_feed(URLS["vehiclePositions"])
df_vp = parse_vehicle_positions(vp_data)
print(f"Vehicle positions: {df_vp.shape}")

# Alerts
alerts_data = fetch_feed(URLS["alerts"])
df_alerts = parse_alerts(alerts_data)
print(f"Alerts: {df_alerts.shape}")

# Prepare for dashboard
df = df_vp.copy()
df['disruption_type'] = 'ON_TIME'
df['delay_min'] = 0.0
df['feed_timestamp'] = pd.Timestamp.now()

# Add alert columns
if not df_alerts.empty:
    df['alert_cause'] = df_alerts['cause'].iloc[0] if 'cause' in df_alerts.columns else 'UNKNOWN'
    df['alert_effect'] = df_alerts['effect'].iloc[0] if 'effect' in df_alerts.columns else 'UNKNOWN'
else:
    df['alert_cause'] = 'UNKNOWN'
    df['alert_effect'] = 'UNKNOWN'

# Add required columns
df['risk_level'] = 'low'
df['first_lat'] = df['latitude']
df['first_lon'] = df['longitude']
df['first_loc_text'] = df['route_id']
df['predicted_disruption'] = 0

# Save
df.to_csv('live_feed_data.csv', index=False)
print(f"\nSaved live_feed_data.csv with {len(df)} rows")
print("Columns:", list(df.columns)[:12])
