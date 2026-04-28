import pandas as pd
import numpy as np

vp = pd.read_parquet('syn_feed/vehiclePositions.parquet')
alerts = pd.read_parquet('syn_feed/alerts.parquet')

df = vp.copy()
df['disruption_type'] = np.random.choice(['ON_TIME', 'DELAY', 'NO_SERVICE'], len(df), p=[0.6, 0.3, 0.1])

delay_vals = np.zeros(len(df))
delay_mask = df['disruption_type'] == 'DELAY'
no_svc_mask = df['disruption_type'] == 'NO_SERVICE'
delay_vals[delay_mask] = np.random.uniform(1, 30, delay_mask.sum())
delay_vals[no_svc_mask] = np.random.uniform(30, 60, no_svc_mask.sum())
df['delay_min'] = delay_vals

df['predicted_disruption'] = (df['disruption_type'] != 'ON_TIME').astype(int)
df['feed_timestamp'] = pd.Timestamp.now() - pd.to_timedelta(np.random.randint(0, 24, len(df)), unit='h')
df['alert_cause'] = np.random.choice(list(alerts['cause'].dropna().unique())[:6], len(df))
df['alert_effect'] = np.random.choice(list(alerts['effect'].dropna().unique())[:5], len(df))
df['first_lat'] = df['latitude']
df['first_lon'] = df['longitude']
df['first_loc_text'] = df['route_id']
df['risk_level'] = df['disruption_type'].map({'NO_SERVICE': 'high', 'DELAY': 'moderate', 'ON_TIME': 'low'})

df.to_csv('test_dashboard_data.csv', index=False)
print('Saved test_dashboard_data.csv with', len(df), 'rows')