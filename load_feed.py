"""
Load Feed Data Directly from Parquet
====================================
Bypasses gtfs_disruption package to avoid torch import issues.
"""
import pandas as pd
import os

def read_parquet_from_zip(zip_path, max_files=None):
    """Read parquet files from a zip archive."""
    import zipfile
    import io
    
    result = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = sorted([f for f in zf.namelist() if f.endswith('.parquet')])
        if max_files:
            files = files[:max_files]
        
        for f in files:
            with zf.open(f) as bf:
                df = pd.read_parquet(io.BytesIO(bf.read()))
                result.append(df)
    
    if result:
        return pd.concat(result, ignore_index=True)
    return pd.DataFrame()

# Load from feed_data_4
feed_dir = "feed_data_4"
zip_files = sorted([f for f in os.listdir(feed_dir) if f.endswith('.zip')])
print("Found zip files:", zip_files[:3])

# Load vehicle positions (first zip)
print("\nLoading vehicle positions...")
df_vp = read_parquet_from_zip(os.path.join(feed_dir, zip_files[0]))
print(f"Vehicle positions: {df_vp.shape}")

# Load alerts (third zip)  
print("\nLoading alerts...")
df_alerts = read_parquet_from_zip(os.path.join(feed_dir, zip_files[2]))
print(f"Alerts: {df_alerts.shape}")

# Prepare dashboard data
df = df_vp.copy()
df['disruption_type'] = 'ON_TIME'
df['delay_min'] = 0.0

if 'retrieved_at' in df.columns:
    df['feed_timestamp'] = pd.to_datetime(df['retrieved_at'])
else:
    df['feed_timestamp'] = pd.Timestamp.now()

# Add alert columns
df['alert_cause'] = df_alerts['cause'].iloc[0] if len(df_alerts) > 0 and 'cause' in df_alerts.columns else 'UNKNOWN'
df['alert_effect'] = df_alerts['effect'].iloc[0] if len(df_alerts) > 0 and 'effect' in df_alerts.columns else 'UNKNOWN'

# Add risk level
df['risk_level'] = 'low'
df['first_lat'] = df['latitude']
df['first_lon'] = df['longitude']
df['first_loc_text'] = df['route_id']
df['predicted_disruption'] = 0

# Save
df.to_csv('feed_data_4_test.csv', index=False)
print(f"\nSaved feed_data_4_test.csv with {len(df)} rows")
print("Columns:", list(df.columns)[:15])
