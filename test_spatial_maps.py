"""Test spatial maps with synthetic data."""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
np.random.seed(42)

n = 500
df = pd.DataFrame({
    'stop_id': np.random.choice([f's{i}' for i in range(50)], n),
    'stop_lat': np.random.uniform(51.8, 52.4, n),
    'stop_lon': np.random.uniform(4.8, 5.2, n),
    'delay_sec': np.random.normal(0, 120, n),
    'disruption_type': np.random.choice(['ON_TIME','ON_TIME','ON_TIME','ON_TIME','EARLY','MINOR_DELAY','MAJOR_DELAY'], n),
    'severity_score': np.random.choice([0,0,0,0,3,5,7], n),
    'disruption_target': np.random.choice([0,0,0,0,0,0,1], n),
    'feed_timestamp': pd.date_range('2026-03-31 18:00', periods=n, freq='30s'),
})
df['spatial_lag_delay'] = np.random.normal(0, 60, n)

from gtfs_disruption.evaluation.spatial_maps import (
    plot_disruption_density_map,
    plot_severity_map,
    plot_spatial_lag_map,
    plot_interactive_map,
    plot_temporal_evolution_map,
    plot_hotspots_map,
)

out = 'visualizations'

print("Map 1: Disruption density...")
plot_disruption_density_map(df, out)
print("Map 2: Severity...")
plot_severity_map(df, out)
print("Map 3: Spatial lag...")
plot_spatial_lag_map(df, out)
print("Map 4: Interactive...")
plot_interactive_map(df, out)
print("Map 5: Temporal evolution...")
plot_temporal_evolution_map(df, out)
print("Map 6: Hot spots...")
plot_hotspots_map(df, out)
print("DONE")
