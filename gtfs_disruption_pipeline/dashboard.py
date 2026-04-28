"""
Transit Disruption Dashboard
====================
Production-grade Streamlit dashboard for transit operators.

Key Features (Industry-Aligned):
- Service Delivery metrics (Cal-ITP style)
- GTFS Data Quality monitoring
- Real-time disruption monitoring
- Early warning system (10/30/60 min predictions)
- NLP insights from service alerts
- Network analysis with Pyvis
- Model performance tracking
- SHAP interpretability

Usage:
------
# Run with live feed
streamlit run gtfs_disruption/dashboard.py

# Select "Live Feed" in sidebar for real-time data from ovapi.nl

pip install streamlit plotly pandas numpy pyvis networkx
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time

# Optional imports
try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# Weather integration
try:
    from gtfs_disruption.features.weather import add_weather_features, WeatherFeatures
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False

# Transit Dashboard Integration
DASHBOARD_INTEGRATION_AVAILABLE = False
try:
    from utils.dashboard_integration import (
        DashboardAPIClient,
        SentinelConnector,
        UnifiedPredictor,
        create_sidebar_controls,
    )
    DASHBOARD_INTEGRATION_AVAILABLE = True
except ImportError as e:
    # Dashboard integration optional
    pass

st.set_page_config(
    page_title="Transit Operations Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ====================================================
# INDUSTRY-ALIGNED KPI DEFINITIONS (Cal-ITP, Swiftly, Transit 360 style)
# ====================================================

KPI_DEFINITIONS = {
    'service_delivered': {
        'label': 'Service Delivered',
        'description': 'Percentage of scheduled trips that actually ran',
        'format': '{:.1f}%',
        'target': 98.0,
        'warning_threshold': 95.0
    },
    'on_time_performance': {
        'label': 'On-Time Performance',
        'description': 'Trips arriving within tolerance (default: <2min late)',
        'format': '{:.1f}%',
        'target': 90.0,
        'warning_threshold': 85.0
    },
    'data_quality_score': {
        'label': 'Data Quality Score',
        'description': 'GTFS-RT completeness and validity (Cal-ITP style)',
        'format': '{:.1f}%',
        'target': 95.0,
        'warning_threshold': 90.0
    },
    'avg_delay': {
        'label': 'Avg System Delay',
        'description': 'Mean delay across all active trips',
        'format': '{:.1f} min',
        'target': 2.0,
        'warning_threshold': 5.0
    },
    'active_disruptions': {
        'label': 'Active Disruptions',
        'description': 'Currently active service disruptions',
        'format': '{}',
        'target': 5,
        'warning_threshold': 15
    },
    'prediction_f1': {
        'label': 'Model F1-Score',
        'description': 'ML model prediction accuracy',
        'format': '{:.2f}',
        'target': 0.85,
        'warning_threshold': 0.75
    },
    'inference_latency': {
        'label': 'Inference Latency P95',
        'description': 'Model prediction time (ms)',
        'format': '{} ms',
        'target': 100,
        'warning_threshold': 300
    },
    'early_warning_accuracy': {
        'label': 'Early Warning Accuracy',
        'description': '10/30/60 min prediction accuracy',
        'format': '{:.2f}',
        'target': 0.80,
        'warning_threshold': 0.70
    },
    # == Traffic Management KPIs == (InetSoft industry standard)
    'throughput': {
        'label': 'Throughput',
        'description': 'Vehicles per hour through key corridors',
        'format': '{} veh/hr',
        'target': 1200,
        'warning_threshold': 800
    },
    'incident_response_time': {
        'label': 'Incident Response Time',
        'description': 'Time from detection to clearance (min)',
        'format': '{:.0f} min',
        'target': 15,
        'warning_threshold': 30
    },
    'travel_time_index': {
        'label': 'Travel Time Index',
        'description': 'Actual vs free-flow travel time ratio',
        'format': '{:.2f}',
        'target': 1.2,
        'warning_threshold': 1.5
    },
    'congestion_level': {
        'label': 'Congestion Level',
        'description': 'Average delay per vehicle (min)',
        'format': '{:.1f} min',
        'target': 3.0,
        'warning_threshold': 8.0
    },
    'route_efficiency': {
        'label': 'Route Efficiency',
        'description': 'Percentage of routes on-time',
        'format': '{:.1f}%',
        'target': 85.0,
        'warning_threshold': 70.0
    }
}


def load_live_feed_data() -> pd.DataFrame:
    """
    Load live GTFS-RT data from ovapi.nl feed URLs.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with live vehicle positions and alerts
    """
    try:
        import requests
        from google.transit import gtfs_realtime_pb2
        
        URLS = {
            "vehiclePositions": "http://gtfs.ovapi.nl/nl/vehiclePositions.pb",
            "alerts": "http://gtfs.ovapi.nl/nl/alerts.pb",
        }
        
        # Fetch vehicle positions
        resp = requests.get(URLS["vehiclePositions"], timeout=30)
        resp.raise_for_status()
        
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        
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
            })
        
        df = pd.DataFrame(rows)
        
        # Fetch alerts
        resp = requests.get(URLS["alerts"], timeout=30)
        if resp.status_code == 200:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(resp.content)
            
            alert_rows = []
            for entity in feed.entity:
                if not entity.HasField("alert"):
                    continue
                alert = entity.alert
                cause = alert.Cause.Name(alert.cause) if alert.cause else "UNKNOWN"
                effect = alert.Effect.Name(alert.effect) if alert.effect else "UNKNOWN"
                alert_rows.append({
                    "entity_id": entity.id,
                    "cause": cause,
                    "effect": effect,
                })
            
            if alert_rows:
                df_alerts = pd.DataFrame(alert_rows)
                df['alert_cause'] = df_alerts['cause'].iloc[0]
                df['alert_effect'] = df_alerts['effect'].iloc[0]
            else:
                df['alert_cause'] = 'UNKNOWN'
                df['alert_effect'] = 'UNKNOWN'
        else:
            df['alert_cause'] = 'UNKNOWN'
            df['alert_effect'] = 'UNKNOWN'
        
        # Add dashboard required columns
        df['disruption_type'] = 'ON_TIME'
        df['delay_min'] = 0.0
        df['feed_timestamp'] = pd.Timestamp.now()
        df['risk_level'] = 'low'
        df['first_lat'] = df['latitude']
        df['first_lon'] = df['longitude']
        df['first_loc_text'] = df['route_id']
        df['predicted_disruption'] = 0
        
        return df
        
    except Exception as e:
        st.warning(f"Could not load live feed: {e}")
        return pd.DataFrame()


def load_pipeline_data(
    config_path: str = "config.yaml",
    realtime_url: str = None,
    static_url: str = None
) -> pd.DataFrame:
    """
    Load and process data using gtfs_disruption modules properly.

    Returns
    -------
    pd.DataFrame
        Enriched feature DataFrame with proper column names
    """
    try:
        from gtfs_disruption.features import DisruptionFeatureBuilder
        from gtfs_disruption.ingestion import (
            ingest_combined, ingest_local, ingest_live,
            fetch_static_gtfs, load_static_gtfs_from_zip,
            DEFAULT_FEED_URLS, DEFAULT_STATIC_GTFS_URL, DEFAULT_LOCAL_DIR
        )

        merged_df = pd.DataFrame()
        gtfs_data = {}

        with st.spinner("Loading real-time feed data..."):
            if realtime_url:
                merged_df = ingest_live(realtime_url)
            elif DEFAULT_LOCAL_DIR:
                try:
                    merged_df = ingest_local(DEFAULT_LOCAL_DIR)
                except FileNotFoundError:
                    merged_df = ingest_live(list(DEFAULT_FEED_URLS.values()))
            else:
                merged_df = ingest_live(list(DEFAULT_FEED_URLS.values()))

        if merged_df.empty:
            st.warning("No real-time data loaded")
            return pd.DataFrame()

        st.success(f"Loaded {len(merged_df)} real-time records")

        with st.spinner("Loading static GTFS data..."):
            try:
                gtfs_data = load_static_gtfs_from_zip(static_url or DEFAULT_STATIC_GTFS_URL)
            except Exception:
                gtfs_data = fetch_static_gtfs()

        with st.spinner("Building features with DisruptionFeatureBuilder..."):
            builder = DisruptionFeatureBuilder(merged_df, gtfs_data)
            feature_df = builder.build()

        if feature_df.empty:
            st.warning("Feature building produced empty DataFrame")
            return pd.DataFrame()

        st.success(f"Built {len(feature_df)} feature records with {len(feature_df.columns)} columns")

        feature_df = _ensure_dashboard_columns(feature_df)
        return feature_df

    except ImportError as e:
        st.error(f"Missing module: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return pd.DataFrame()


def _ensure_dashboard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist for dashboard panels."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if 'disruption_type' not in df.columns:
        if 'delay_min' in df.columns:
            df['disruption_type'] = pd.cut(
                df['delay_min'],
                bins=[-float('inf'), 2, 5, float('inf')],
                labels=['ON_TIME', 'MINOR_DELAY', 'MAJOR_DELAY']
            )
        elif 'delay_sec' in df.columns:
            df['delay_min'] = df['delay_sec'] / 60
            df['disruption_type'] = pd.cut(
                df['delay_min'],
                bins=[-float('inf'), 2, 5, float('inf')],
                labels=['ON_TIME', 'MINOR_DELAY', 'MAJOR_DELAY']
            )
        else:
            df['disruption_type'] = 'ON_TIME'
            df['delay_min'] = 0.0

    if 'delay_min' not in df.columns:
        df['delay_min'] = 0.0

    if 'route_id' not in df.columns and 'consolidated_route' in df.columns:
        df['route_id'] = df['consolidated_route']

    return df


def prepare_dashboard_data(df: pd.DataFrame) -> Dict:
    """
    Prepare data for dashboard visualizations.
    
    Handles different column names from various GTFS-RT sources.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from pipeline
        
    Returns
    -------
    dict
        Dictionary of prepared data for each visualization
    """
    if df is None or df.empty:
        return _generate_sample_data()
    
    data = {}
    
    # == Column name mapping for different data sources ==
    # Try common column names
    timestamp_col = None
    for col in ['feed_timestamp', 'timestamp', 'timestamp_unix', 'arrival_time', 'departure_time', 'start_time']:
        if col in df.columns:
            timestamp_col = col
            break
    
    delay_col = None
    for col in ['delay_min', 'delay_sec', 'arrival_delay', 'departure_delay', 'delay']:
        if col in df.columns:
            delay_col = col
            break
    
    stop_col = None
    for col in ['stop_id', 'stop_sequence', 'stop_lat', 'stop_lon']:
        if col in df.columns:
            stop_col = col
            break
    
    disruption_col = None
    for col in ['disruption_type', 'disruption_target', 'target']:
        if col in df.columns:
            disruption_col = col
            break
    
    # == Active disruptions ==
    if disruption_col:
        active = df[df[disruption_col] != 'ON_TIME']
        data['active_disruptions'] = len(active)
        disruption_counts = df[disruption_col].value_counts()
        data['alert_distribution'] = disruption_counts.to_dict()
    else:
        # Try to infer from delay if no disruption column
        if delay_col:
            delayed = df[df[delay_col] > 0]
            data['active_disruptions'] = len(delayed)
            data['alert_distribution'] = {'Delayed': len(delayed), 'On-Time': len(df) - len(delayed)}
        else:
            data['active_disruptions'] = 0
            data['alert_distribution'] = {'Total Records': len(df)}
    
    # == Average delay ==
    if delay_col:
        data['avg_delay'] = df[delay_col].mean()
    else:
        data['avg_delay'] = 0.0
    
    # == Prediction F1 (if available) ==
    if 'predicted_disruption' in df.columns and disruption_col:
        actual = (df[disruption_col] != 'ON_TIME').astype(int)
        predicted = df['predicted_disruption']
        correct = (actual == predicted).sum()
        total = len(actual)
        data['prediction_f1'] = correct / total if total > 0 else 0
    
    # == Timeline data ==
    if timestamp_col:
        try:
            # Try to parse as datetime
            timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
            if timestamps.notna().any():
                df_copy = df.copy()
                df_copy['hour'] = timestamps.dt.floor('h')
                timeline = df_copy.groupby('hour').size().reset_index(name='disruptions')
                data['timeline'] = timeline
            else:
                # Use numeric timestamp
                data['timeline'] = pd.DataFrame({'hour': [], 'disruptions': []})
        except:
            data['timeline'] = pd.DataFrame({'hour': [], 'disruptions': []})
    else:
        data['timeline'] = pd.DataFrame({'hour': [], 'disruptions': []})
    
    # == Route performance ==
    if 'route_id' in df.columns:
        if disruption_col:
            route_perf = df.groupby('route_id').apply(
                lambda x: (x[disruption_col] == 'ON_TIME').mean()
            ).reset_index(name='On-Time %')
        else:
            # Just count trips per route
            route_perf = df.groupby('route_id').size().reset_index(name='Trip Count')
            route_perf.columns = ['route_id', 'Trip Count']
        data['route_performance'] = route_perf
    
    # == Active alerts table ==
    alert_cols = []
    if timestamp_col and timestamp_col in df.columns:
        alert_cols.append(timestamp_col)
    if 'route_id' in df.columns:
        alert_cols.append('route_id')
    if 'trip_id' in df.columns:
        alert_cols.append('trip_id')
    if 'vehicle_id' in df.columns:
        alert_cols.append('vehicle_id')
    if delay_col and delay_col in df.columns:
        alert_cols.append(delay_col)
    if alert_cols:
        alerts_df = df[alert_cols].head(20)
        data['active_alerts'] = alerts_df
    
    # == Stop hotspots ==
    if 'stop_id' in df.columns or 'latitude' in df.columns:
        if delay_col:
            if 'stop_id' in df.columns:
                hotspots = df.groupby('stop_id')[delay_col].mean().sort_values(ascending=False).head(10)
            else:
                # Use lat/lon for hotspots
                hotspots = df.groupby(['latitude', 'longitude'])[delay_col].mean().sum()
            data['hotspots'] = hotspots
    
    return data


def _generate_sample_data() -> Dict:
    """Generate sample data for demonstration."""
    hours = pd.date_range(end=datetime.now(), periods=24, freq='h')
    
    return {
        'active_disruptions': 12,
        'avg_delay': 8.2,
        'prediction_f1': 0.87,
        'timeline': pd.DataFrame({
            'hour': hours,
            'disruptions': np.random.poisson(15, 24)
        }),
        'alert_distribution': {
            'Technical': 35,
            'Weather': 25,
            'Construction': 20,
            'Strike': 10,
            'Other': 10
        },
        'route_performance': pd.DataFrame({
            'Route': [f'Route {i}' for i in 'ABCDE'],
            'On-Time %': [0.85, 0.72, 0.91, 0.68, 0.79]
        }),
        'active_alerts': pd.DataFrame({
            'Time': ['10:23', '09:45', '08:12', '07:30'],
            'Route': ['A12', 'B5', 'C3', 'R17'],
            'Type': ['Delay', 'Cancellation', 'Delay', 'Alert'],
            'Severity': ['Medium', 'High', 'Low', 'Medium'],
            'Expected Clear': ['11:30', 'TBD', '09:00', '10:15']
        }),
        'hotspots': pd.Series({
            f'Stop {i}': np.random.randint(5, 20) for i in range(1, 11)
        })
    }


class TransitDashboard:
    """
    Transit Operations Dashboard - Production-Grade.
    
    Industry-aligned dashboard following Cal-ITP, Swiftly, Transit 360 best practices.
    
    Parameters
    ----------
    data : dict, optional
        Pre-prepared data dictionary from prepare_dashboard_data()
    """
    
    def __init__(self, data: Optional[Dict] = None):
        self.title = "     Transit Operations Dashboard"
        self.layout = "wide"
        self.data = data or _generate_sample_data()
        self.last_refresh = datetime.now()
    
    def render_header(self):
        """Render industry-standard dashboard header."""
        # Top banner with agency info
        st.markdown("""
        <div style="background: linear-gradient(90deg, #0033A0, #0055A4); 
                    padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">     Netherlands Transit Operations Center</h2>
            <p style="color: #E0E0E0; margin: 5px 0 0 0;">
                Real-time disruption monitoring | Early warning system | Service delivery tracking
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prominent Traffic Management KPI Row (Inspired by InetSoft)
        st.subheader("Traffic Management Overview")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        
        # Extract metrics from data or calculate defaults
        data_for_kpis = self.data or {}
        df = data_for_kpis.get('df')
        
        # Calculate traffic management metrics
        if df is not None and not df.empty:
            # Throughput - vehicles per hour
            if 'feed_timestamp' in df.columns and 'trip_id' in df.columns:
                df_ts = pd.to_datetime(df['feed_timestamp'], errors='coerce')
                hour_counts = df_ts.dt.hour.value_counts()
                throughput = hour_counts.max() if not hour_counts.empty else len(df)
            else:
                throughput = len(df) if 'trip_id' in df.columns else 0
            
            # Incident Response Time
            if 'delay_min' in df.columns:
                disp_df = df[df['disruption_type'] != 'ON_TIME'] if 'disruption_type' in df.columns else df
                incident_response = disp_df['delay_min'].mean() if not disp_df.empty else 0
            else:
                incident_response = 15  # default
            
            # Travel Time Index
            if 'delay_min' in df.columns:
                free_flow = 15  # assumed free flow time
                avg_actual = free_flow + df['delay_min'].mean()
                travel_time_index = avg_actual / free_flow
            else:
                travel_time_index = 1.3  # default
            
            # Congestion Level
            congestion_level = df['delay_min'].mean() if 'delay_min' in df.columns else 2.5
            
            # Active Incidents
            if 'disruption_type' in df.columns:
                active_incidents = len(df[df['disruption_type'] != 'ON_TIME'])
            else:
                active_incidents = 0
        else:
            # Default values when no data
            throughput = 0
            incident_response = 0
            travel_time_index = 1.0
            congestion_level = 0.0
            active_incidents = 0
        
        with kpi_col1:
            st.metric(
                "     Throughput",
                f"{throughput:,.0f} veh/hr",
                delta=f"{throughput - 1200:+,}" if throughput > 0 else None,
                delta_color="normal" if throughput >= 1200 else "inverse"
            )
        with kpi_col2:
            st.metric(
                "    Incident Response",
                f"{incident_response:.0f} min",
                delta=f"{incident_response - 15:+.0f}",
                delta_color="normal" if incident_response <= 15 else "inverse"
            )
        with kpi_col3:
            st.metric(
                "     Travel Time Index",
                f"{travel_time_index:.2f}",
                delta=f"{travel_time_index - 1.2:+.2f}",
                delta_color="normal" if travel_time_index <= 1.2 else "inverse"
            )
        with kpi_col4:
            st.metric(
                "     Congestion Level",
                f"{congestion_level:.1f} min",
                delta=f"{congestion_level - 3.0:+.1f}",
                delta_color="normal" if congestion_level <= 3.0 else "inverse"
            )
        with kpi_col5:
            st.metric(
                "    Active Incidents",
                f"{active_incidents}",
                delta=f"{active_incidents - 5:+,}" if active_incidents > 0 else None,
                delta_color="inverse" if active_incidents > 5 else "normal"
            )
        
        # Status bar
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "* Last Update",
                datetime.now().strftime("%H:%M:%S")
            )
        with col2:
            # System health indicator
            status = self.data.get('system_status', 'operational')
            status_color = "green" if status == "operational" else "orange"
            st.markdown(f":{status_color}[   ] System Status: {status.title()}]")
        with col3:
            st.metric(
                "     Feed Status",
                "Connected" if self.data.get('live_feed_active', True) else "Disconnected"
            )
        with col4:
            st.metric(
                "     Model Version",
                "STARN-GAT v2.1"
            )
        with col5:
            # Auto-refresh indicator
            st.metric(
                "     Auto-Refresh",
                "30s"
            )
        
        st.markdown("---")
    
    # =================================================
    # SAP Digital Boardroom Panel Implementations
    # =================================================
    
    # Netherlands-aligned cost constants (2026 standards)
    NETHERLANDS_COST_PER_DISRUPTION = {
        'CANCELLED': 15,         # Full fare refund per passenger
        'MAJOR_DELAY': 10,       # Full compensation (60+ min)
        'MINOR_DELAY': 3,        # Half compensation (30-59 min)
        'STOPPED_ON_ROUTE': 20,  # Taxi/alternative transport
        'SERVICE_ALERT': 5,     # Passenger inconvenience
        'MAINTENANCE': 350,     # Average maintenance incident
        'TECHNICAL_PROBLEM': 500,  # Vehicle breakdown
        'ON_TIME': 0
    }
    
    # Average passengers per bus (Dutch urban)
    AVG_PASSENGERS_PER_VEHICLE = 40
    
    # Monthly budget (based on Dutch transit operator averages)
    MONTHLY_BUDGET = {
        'CANCELLED': 15000,
        'MAJOR_DELAY': 8000,
        'MINOR_DELAY': 3000,
        'MAINTENANCE': 12000,
        'TECHNICAL_PROBLEM': 8000,
        'total': 46000
    }
    
    # Caching for metrics (production optimization)
    _metrics_cache = {}
    _cache_ttl = 30  # seconds
    
    def _safe_render(self, panel_fn, data: Optional[Dict] = None):
        """Error boundary wrapper for panel rendering."""
        try:
            panel_fn(data)
        except Exception as e:
            st.error(f"Panel error in {panel_fn.__name__}: {str(e)}")
            st.caption("Please check data availability and format.")
    
    def _get_safe_copy(self, df) -> pd.DataFrame:
        """Create safe copy to prevent data mutation/leakage."""
        if df is None or not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        return df.copy()
    
    def _get_cached_metrics(self, key: str, compute_fn):
        """Cached computation for metrics."""
        import time
        now = time.time()
        
        if key in self._metrics_cache:
            cached_time, cached_val = self._metrics_cache[key]
            if now - cached_time < self._cache_ttl:
                return cached_val
        
        result = compute_fn()
        self._metrics_cache[key] = (now, result)
        return result
    
    def render_sap_kpi_row(self, data: Optional[Dict] = None):
        """SAP-style KPI metrics row aligned with Dutch transit standards."""
        import streamlit as st
        data = data or self.data
        
        df = data.get('df')
        st.subheader("SAP Digital Boardroom - KPI Overview")
        
        if df is None or df.empty:
            st.info("No data available for KPIs")
            return
        
        # Create copy to prevent data mutation (prevents leakage)
        df = df.copy()
        
        col1, col2, col3, col4 = st.columns(4)
        
        # 1. Service Delivered (NS target: 98%)
        total_trips = len(df)
        completed = len(df[~df.get('disruption_type', 'ON_TIME').isin(['CANCELLED', 'NO_SERVICE'])])
        service_delivered = (completed / total_trips * 100) if total_trips > 0 else 98.0
        
        with col1:
            delta = service_delivered - 98.0
            st.metric(
                "    Service Delivered",
                f"{service_delivered:.1f}%",
                delta=f"{delta:+.1f}%",
                delta_color="normal" if delta >= 0 else "inverse"
            )
        
        # 2. Maintenance Events (from alerts)
        maint_types = ['MAINTENANCE', 'TECHNICAL_PROBLEM']
        maint_count = len(df[df.get('alert_cause', '').isin(maint_types)]) if 'alert_cause' in df.columns else 0
        est_maintenance_cost = maint_count * self.NETHERLANDS_COST_PER_DISRUPTION['MAINTENANCE']
        
        with col2:
            st.metric(
                "     Maintenance Events",
                str(maint_count),
                delta=f"   {est_maintenance_cost:,}",
                delta_color="inverse"
            )
        
        # 3. Cost Impact (Dutch standards)
        dtype_col = 'disruption_type' if 'disruption_type' in df.columns else 'disruption_target'
        if dtype_col in df.columns:
            df['cost_impact'] = df[dtype_col].map(self.NETHERLANDS_COST_PER_DISRUPTION).fillna(0)
            total_cost = (df['cost_impact'] * self.AVG_PASSENGERS_PER_VEHICLE).sum()
        else:
            total_cost = 0
        
        with col3:
            st.metric(
                "     Est. Cost Impact",
                f"   {total_cost:,.0f}",
                delta_color="inverse"
            )
        
        # 4. On-Time Performance (NS target: 90%)
        on_time = len(df[df.get('disruption_type', 'ON_TIME') == 'ON_TIME']) / len(df) * 100 if len(df) > 0 else 90.0
        
        with col4:
            delta = on_time - 90.0
            st.metric(
                "* On-Time Performance",
                f"{on_time:.1f}%",
                delta=f"{delta:+.1f}%",
                delta_color="normal" if delta >= 0 else "inverse"
            )
    
    def render_incident_management_panel(self, data: Optional[Dict] = None):
        """Incident Management & Response Panel (Inspired by InetSoft Traffic Management Dashboard)"""
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        data = data or self.data
        
        df = data.get('df')
        st.subheader("Incident Management & Response")
        
        if df is None or df.empty:
            st.info("No disruption data for incident management")
            return
        
        # Incident summary metrics
        inc_col1, inc_col2, inc_col3, inc_col4 = st.columns(4)
        
        if 'disruption_type' in df.columns:
            active_incidents = len(df[df['disruption_type'] != 'ON_TIME'])
            major_incidents = len(df[df['disruption_type'].isin(['CANCELLED', 'MAJOR_DELAY'])])
            minor_incidents = len(df[df['disruption_type'] == 'MINOR_DELAY'])
            # Estimate response time from delay data
            if 'delay_min' in df.columns:
                avg_response = df[df['disruption_type'] != 'ON_TIME']['delay_min'].mean() if not df[df['disruption_type'] != 'ON_TIME'].empty else 0
            else:
                avg_response = 15  # default estimate
        else:
            active_incidents = 0
            major_incidents = 0
            minor_incidents = 0
            avg_response = 0
        
        with inc_col1:
            st.metric(
                "     Active Incidents",
                f"{active_incidents}",
                delta=f"{active_incidents - 5:+,}" if active_incidents > 0 else None,
                delta_color="inverse" if active_incidents > 5 else "normal"
            )
        with inc_col2:
            st.metric(
                "    Major Incidents",
                f"{major_incidents}",
                delta=f"{major_incidents - 2:+,}" if major_incidents > 0 else None,
                delta_color="inverse" if major_incidents > 2 else "normal"
            )
        with inc_col3:
            st.metric(
                "     Minor Incidents",
                f"{minor_incidents}",
                delta=f"{minor_incidents - 3:+,}" if minor_incidents > 0 else None,
                delta_color="inverse" if minor_incidents > 3 else "normal"
            )
        with inc_col4:
            st.metric(
                "    Avg Response Time",
                f"{avg_response:.0f} min",
                delta=f"{avg_response - 15:+.0f}",
                delta_color="normal" if avg_response <= 15 else "inverse"
            )
        
        st.markdown("---")
        
        # Incident details and response tracking
        if 'disruption_type' in df.columns and not df[df['disruption_type'] != 'ON_TIME'].empty:
            inc_df = df[df['disruption_type'] != 'ON_TIME'].copy()
            
            # Incident type breakdown
            inc_col_left, inc_col_right = st.columns(2)
            
            with inc_col_left:
                st.write("**Incident Type Distribution**")
                if 'disruption_type' in inc_df.columns:
                    incident_counts = inc_df['disruption_type'].value_counts()
                    fig_incident = px.pie(
                        values=incident_counts.values,
                        names=incident_counts.index,
                        title="Incidents by Type",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_incident.update_layout(height=300)
                    st.plotly_chart(fig_incident, use_container_width=True)
                else:
                    st.info("No disruption type data available")
            
            with inc_col_right:
                st.write("**Impact Assessment**")
                # Calculate impact scores
                if 'delay_min' in inc_df.columns:
                    inc_df['impact_score'] = inc_df['delay_min'] * (inc_df.get('vehicle_count', 1)).fillna(1)  # Simplified impact
                    total_impact = inc_df['impact_score'].sum()
                    avg_impact = inc_df['impact_score'].mean()
                    
                    impact_col1, impact_col2 = st.columns(2)
                    with impact_col1:
                        st.metric("Total Impact Score", f"{total_impact:.0f}")
                    with impact_col2:
                        st.metric("Avg Impact per Incident", f"{avg_impact:.1f}")
                    
                    # Impact by incident type
                    if 'disruption_type' in inc_df.columns:
                        impact_by_type = inc_df.groupby('disruption_type')['impact_score'].mean().reset_index()
                        fig_impact = px.bar(
                            impact_by_type,
                            x='disruption_type',
                            y='impact_score',
                            title="Average Impact by Incident Type",
                            color='impact_score',
                            color_continuous_scale='Reds'
                        )
                        fig_impact.update_layout(height=300)
                        st.plotly_chart(fig_impact, use_container_width=True)
                else:
                    # Fallback to delay-based impact
                    if 'delay_min' in inc_df.columns:
                        total_delay = inc_df['delay_min'].sum()
                        avg_delay = inc_df['delay_min'].mean()
                        
                        impact_col1, impact_col2 = st.columns(2)
                        with impact_col1:
                            st.metric("Total Delay Minutes", f"{total_delay:.0f}")
                        with impact_col2:
                            st.metric("Avg Delay per Incident", f"{avg_delay:.1f} min")
                        
                        # Delay by incident type
                        if 'disruption_type' in inc_df.columns:
                            delay_by_type = inc_df.groupby('disruption_type')['delay_min'].mean().reset_index()
                            fig_delay = px.bar(
                                delay_by_type,
                                x='disruption_type',
                                y='delay_min',
                                title="Average Delay by Incident Type",
                                color='delay_min',
                                color_continuous_scale='Reds'
                            )
                            fig_delay.update_layout(height=300)
                            st.plotly_chart(fig_delay, use_container_width=True)
                    else:
                        st.info("No delay or impact data available")
            
            # Incident timeline and response tracking
            st.write("**Incident Timeline & Response Tracking**")
            
            if 'feed_timestamp' in inc_df.columns:
                try:
                    inc_df['timestamp'] = pd.to_datetime(inc_df['feed_timestamp'], errors='coerce')
                    inc_df_sorted = inc_df.sort_values('timestamp')
                    
                    # Create timeline chart
                    fig_timeline = px.scatter(
                        inc_df_sorted,
                        x='timestamp',
                        y='delay_min' if 'delay_min' in inc_df.columns else 'disruption_type',
                        color='disruption_type' if 'disruption_type' in inc_df.columns else None,
                        size='delay_min' if 'delay_min' in inc_df.columns else None,
                        hover_data=['route_id', 'trip_id'] if 'route_id' in inc_df.columns else None,
                        title="Incident Timeline",
                        labels={
                            'timestamp': 'Time',
                            'delay_min': 'Delay (min)' if 'delay_min' in inc_df.columns else 'Disruption Type',
                            'disruption_type': 'Disruption Type'
                        }
                    )
                    fig_timeline.update_layout(height=350)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Response time analysis
                    if 'delay_min' in inc_df.columns:
                        st.write("**Response Time Analysis**")
                        resp_col1, resp_col2, resp_col3 = st.columns(3)
                        
                        with resp_col1:
                            # Fast responses (<10 min)
                            fast_responses = len(inc_df[inc_df['delay_min'] < 10])
                            fast_pct = (fast_responses / len(inc_df)) * 100 if len(inc_df) > 0 else 0
                            st.metric("Fast Responses (<10 min)", f"{fast_pct:.0f}%")
                        
                        with resp_col2:
                            # Medium responses (10-30 min)
                            medium_responses = len(inc_df[(inc_df['delay_min'] >= 10) & (inc_df['delay_min'] < 30)])
                            medium_pct = (medium_responses / len(inc_df)) * 100 if len(inc_df) > 0 else 0
                            st.metric("Medium Responses (10-30 min)", f"{medium_pct:.0f}%")
                        
                        with resp_col3:
                            # Slow responses (>30 min)
                            slow_responses = len(inc_df[inc_df['delay_min'] >= 30])
                            slow_pct = (slow_responses / len(inc_df)) * 100 if len(inc_df) > 0 else 0
                            st.metric("Slow Responses (>30 min)", f"{slow_pct:.0f}%")
                            
                except Exception as e:
                    st.warning(f"Could not process timeline data: {e}")
                    # Show incident table instead
                    self._render_incident_table(inc_df)
            else:
                # Show incident table when no timestamp
                self._render_incident_table(inc_df)
        else:
            st.info("No active incidents to display")
    
    def _render_incident_table(self, inc_df: pd.DataFrame):
        """Render detailed incident table when timeline data is not available."""
        st.write("**Active Incidents Details**")
        
        # Prepare incident table
        display_cols = []
        if 'route_id' in inc_df.columns:
            display_cols.append('route_id')
        if 'trip_id' in inc_df.columns:
            display_cols.append('trip_id')
        if 'disruption_type' in inc_df.columns:
            display_cols.append('disruption_type')
        if 'delay_min' in inc_df.columns:
            display_cols.append('delay_min')
        if 'alert_cause' in inc_df.columns:
            display_cols.append('alert_cause')
        if 'alert_effect' in inc_df.columns:
            display_cols.append('alert_effect')
        if 'feed_timestamp' in inc_df.columns:
            display_cols.append('feed_timestamp')
        
        if display_cols:
            table_df = inc_df[display_cols].head(10).copy()
            
            # Rename columns for display
            rename_map = {
                'route_id': 'Route',
                'trip_id': 'Trip ID',
                'disruption_type': 'Type',
                'delay_min': 'Delay (min)',
                'alert_cause': 'Cause',
                'alert_effect': 'Effect',
                'feed_timestamp': 'Time'
            }
            table_df = table_df.rename(columns={k: v for k, v in rename_map.items() if k in table_df.columns})
            
            st.dataframe(table_df, use_container_width=True)
        else:
            # Minimal table
            st.dataframe(inc_df.head(10), use_container_width=True)
    
    def render_simulation_panel(self, data: Optional[Dict] = None):
        """Simulation Panel - What-If analysis for closed-loop planning."""
        import streamlit as st
        data = data or self.data
        
        df = data.get('df')
        st.subheader("Simulation: What-If Analysis")
        
        if df is None or df.empty:
            st.info("No data for simulation")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delay_increase = st.slider(
                "Increase delays by (min)",
                min_value=0,
                max_value=30,
                value=5,
                help="Simulate if delays increase by X minutes",
                key="delay_increase_slider"
            )
        
        with col2:
            alert_multiplier = st.slider(
                "Alert frequency multiplier",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Simulate more/fewer service alerts",
                key="alert_multiplier_slider"
            )
        
        with col3:
            sim_horizon = st.selectbox(
                "Prediction horizon",
                options=['10 min', '30 min', '60 min'],
                index=1
            )
        
        # Base metrics
        dtype_col = 'disruption_type' if 'disruption_type' in df.columns else 'disruption_target'
        base_disruptions = len(df[df.get(dtype_col, 0) != 'ON_TIME']) if dtype_col in df.columns else 0
        
        # Simulated impact calculation
        simulated_factor = 1 + (delay_increase / 10) * alert_multiplier
        simulated_disruptions = int(base_disruptions * simulated_factor)
        
        # Display simulation results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(
                "Simulated Disruptions",
                str(simulated_disruptions),
                delta=str(simulated_disruptions - base_disruptions)
            )
        
        with res_col2:
            # Estimated cost impact after simulation
            est_cost = simulated_disruptions * 12 * self.AVG_PASSENGERS_PER_VEHICLE
            st.metric(
                "Est. Cost Impact",
                f"   {est_cost:,}",
                delta=f"   {est_cost - (base_disruptions * 12 * self.AVG_PASSENGERS_PER_VEHICLE):+,}"
            )
        
        # Predicted disruption distribution
        if simulated_disruptions > 0:
            pred_dist = {
                'ON_TIME': max(0, len(df) - simulated_disruptions),
                'MINOR_DELAY': int(simulated_disruptions * 0.4),
                'MAJOR_DELAY': int(simulated_disruptions * 0.3),
                'CANCELLED': int(simulated_disruptions * 0.1),
                'SERVICE_ALERT': int(simulated_disruptions * 0.2)
            }
            
            pred_df = pd.DataFrame(list(pred_dist.items()), columns=['Type', 'Count'])
            fig_dist = px.bar(pred_df, x='Count', y='Type', orientation='h', title="Predicted Disruption Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)
    
    def render_budget_actuals(self, data: Optional[Dict] = None):
        """Budget vs Actuals Panel - aligned with Dutch standards."""
        data = data or self.data
        
        df = data.get('df')
        st.subheader("Budget vs Actuals (Netherlands Standards)")
        
        if df is None or df.empty:
            st.info("No data for budget comparison")
            return
        
        # Calculate actuals from data
        actuals = {
            'CANCELLED': 0,
            'MAJOR_DELAY': 0,
            'MINOR_DELAY': 0,
            'MAINTENANCE': 0,
            'TECHNICAL_PROBLEM': 0
        }
        
        dtype_col = 'disruption_type' if 'disruption_type' in df.columns else None
        
        if dtype_col and dtype_col in df.columns:
            for dtype in ['CANCELLED', 'MAJOR_DELAY', 'MINOR_DELAY']:
                count = len(df[df[dtype_col] == dtype])
                cost_per_incident = self.NETHERLANDS_COST_PER_DISRUPTION.get(dtype, 0)
                actuals[dtype] = count * cost_per_incident * self.AVG_PASSENGERS_PER_VEHICLE
        
        # Maintenance from alerts
        if 'alert_cause' in df.columns:
            maint_count = len(df[df['alert_cause'].isin(['MAINTENANCE', 'TECHNICAL_PROBLEM'])])
            actuals['MAINTENANCE'] = maint_count * self.NETHERLANDS_COST_PER_DISRUPTION['MAINTENANCE']
            tech_count = len(df[df['alert_cause'] == 'TECHNICAL_PROBLEM'])
            actuals['TECHNICAL_PROBLEM'] = tech_count * self.NETHERLANDS_COST_PER_DISRUPTION['TECHNICAL_PROBLEM']
        
        # Total
        actuals['total'] = sum(actuals.values())
        
        # Create comparison DataFrame
        categories = ['CANCELLED', 'MAJOR_DELAY', 'MINOR_DELAY', 'MAINTENANCE', 'TECHNICAL_PROBLEM']
        comp_data = []
        
        for cat in categories:
            budget = self.MONTHLY_BUDGET.get(cat, 0)
            actual = actuals.get(cat, 0)
            variance = budget - actual
            variance_pct = (variance / budget * 100) if budget > 0 else 0
            
            comp_data.append({
                'Category': cat,
                'Budget (   )': budget,
                'Actual (   )': actual,
                'Variance (   )': variance,
                'Variance %': variance_pct
            })
        
        comp_df = pd.DataFrame(comp_data)
        
        # Add totals row
        comp_df = pd.concat([comp_df, pd.DataFrame([{
            'Category': 'TOTAL',
            'Budget (   )': self.MONTHLY_BUDGET['total'],
            'Actual (   )': actuals['total'],
            'Variance (   )': self.MONTHLY_BUDGET['total'] - actuals['total'],
            'Variance %': ((self.MONTHLY_BUDGET['total'] - actuals['total']) / self.MONTHLY_BUDGET['total'] * 100) if self.MONTHLY_BUDGET['total'] > 0 else 0
        }])], ignore_index=True)
        
        # Display as styled table
        def color_variance(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: green'
                elif val < 0:
                    return 'color: red'
            return ''
        
        st.dataframe(
            comp_df.style.format({
                'Budget (   )': '   {:,.0f}',
                'Actual (   )': '   {:,.0f}',
                'Variance (   )': '   {:+,.0f}',
                'Variance %': '{:+.1f}%'
            }),
            use_container_width=True
        )
        
        # Budget utilization gauge
        if self.MONTHLY_BUDGET['total'] > 0:
            utilization = actuals['total'] / self.MONTHLY_BUDGET['total'] * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=utilization,
                title={"text": "Budget Utilization (%)"},
                gauge={
                    "axis": {"range": [0, 150]},
                    "bar": {"color": "green"},
                    "steps": [
                        {"range": [0, 70], "color": "red"},
                        {"range": [70, 90], "color": "yellow"},
                        {"range": [90, 110], "color": "green"},
                        {"range": [110, 150], "color": "red"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 100
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    # =================================================
    # == Traffic Management KPIs == (InetSoft industry standard)
    # =================================================
    
    def render_traffic_kpis(self, data: Optional[Dict] = None):
        """Traffic Management KPIs aligned with InetSoft standards."""
        data = data or self.data
        
        df = data.get('df')
        st.subheader("Traffic Management KPIs")
        
        if df is None or df.empty:
            st.info("No data for traffic KPIs")
            return
        
        # Calculate traffic management metrics
        metrics = {}
        
        # 1. Throughput - vehicles per hour through key corridors
        if 'feed_timestamp' in df.columns and 'trip_id' in df.columns:
            df_ts = pd.to_datetime(df['feed_timestamp'], errors='coerce')
            hour_counts = df_ts.dt.hour.value_counts()
            metrics['throughput'] = hour_counts.max() if not hour_counts.empty else 0
        else:
            metrics['throughput'] = len(df) if 'trip_id' in df.columns else 0
        
        # 2. Incident Response Time - estimated from disruption duration
        # If we have delay_min, we can estimate response time
        if 'delay_min' in df.columns:
            disp_df = df[df['disruption_type'] != 'ON_TIME'] if 'disruption_type' in df.columns else df
            if not disp_df.empty:
                metrics['incident_response_time'] = disp_df['delay_min'].mean()
            else:
                metrics['incident_response_time'] = 0
        else:
            # Estimate from disruption severity
            if 'disruption_type' in df.columns:
                severity_map = {'CANCELLED': 25, 'MAJOR_DELAY': 20, 'MINOR_DELAY': 10, 'ON_TIME': 0}
                df['severity_est'] = df['disruption_type'].map(severity_map).fillna(0)
                metrics['incident_response_time'] = df['severity_est'].mean()
            else:
                metrics['incident_response_time'] = 15  # default
        
        # 3. Travel Time Index - actual vs free-flow ratio
        # Free-flow = scheduled time, actual = delay adjusted
        if 'delay_min' in df.columns:
            # TTI = 1 + (avg_delay / free_flow_time_estimate)
            # Assume free-flow is 15 min average
            free_flow = 15
            avg_actual = free_flow + df['delay_min'].mean()
            metrics['travel_time_index'] = avg_actual / free_flow
        else:
            metrics['travel_time_index'] = 1.3  # default
        
        # 4. Congestion Level - average delay per vehicle
        if 'delay_min' in df.columns:
            metrics['congestion_level'] = df['delay_min'].mean()
        else:
            metrics['congestion_level'] = 2.5  # default
        
        # 5. Route Efficiency - percentage of routes on-time
        if 'disruption_type' in df.columns:
            on_time_routes = len(df[df['disruption_type'] == 'ON_TIME'])
            total = len(df)
            metrics['route_efficiency'] = (on_time_routes / total * 100) if total > 0 else 85
        else:
            metrics['route_efficiency'] = 85
        
        # Display traffic KPIs in a row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            tti = metrics['throughput']
            st.metric(
                "     Throughput",
                f"{tti:,.0f} veh/hr",
                delta=f"{tti - 1200:+,}" if tti > 0 else None,
                delta_color="normal" if tti >= 1200 else "inverse"
            )
        
        with col2:
            irt = metrics['incident_response_time']
            st.metric(
                "    Incident Response",
                f"{irt:.0f} min",
                delta=f"{irt - 15:+.0f}",
                delta_color="normal" if irt <= 15 else "inverse"
            )
        
        with col3:
            tti_idx = metrics['travel_time_index']
            st.metric(
                "     Travel Time Index",
                f"{tti_idx:.2f}",
                delta=f"{tti_idx - 1.2:+.2f}",
                delta_color="normal" if tti_idx <= 1.2 else "inverse"
            )
        
        with col4:
            cl = metrics['congestion_level']
            st.metric(
                "     Congestion Level",
                f"{cl:.1f} min",
                delta=f"{cl - 3.0:+.1f}",
                delta_color="normal" if cl <= 3.0 else "inverse"
            )
        
        with col5:
            re = metrics['route_efficiency']
            st.metric(
                "    Route Efficiency",
                f"{re:.1f}%",
                delta=f"{re - 85:+.1f}%",
                delta_color="normal" if re >= 85 else "inverse"
            )
        
        # Add traffic visualization - congestion trend
        st.caption("== Traffic Management KPIs == (InetSoft standard)")
        
        # Throughput chart by hour
        if 'feed_timestamp' in df.columns:
            try:
                df_ts = pd.to_datetime(df['feed_timestamp'], errors='coerce')
                hourly = df_ts.dt.hour.value_counts().sort_index()
                fig = px.line(
                    x=hourly.index,
                    y=hourly.values,
                    title="Hourly Throughput",
                    labels={'x': 'Hour', 'y': 'Vehicles'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass
    
    # =================================================
    # == Temporal Comparison == & == Uncertainty Estimation ==
    # =================================================
    
    def render_temporal_comparison(self, data: Optional[Dict] = None):
        """Temporal comparison panel - trend analysis."""
        import streamlit as st
        data = data or self.data
        
        df = data.get('df')
        st.subheader("📈 Temporal Comparison - Trend Analysis")
        
        if df is None or df.empty:
            st.info("No data for temporal comparison")
            return
        
        df = self._get_safe_copy(df)
        
        # Get historical data for comparison
        if 'feed_timestamp' in df.columns:
            try:
                df_ts = pd.to_datetime(df['feed_timestamp'], errors='coerce')
                
                if len(df) >= 50:
                    # Current window (last 50%)
                    current = df.tail(len(df) // 2)
                    # Previous window
                    previous = df.iloc[:len(df) // 2]
                    
                    # Compare disruption rates
                    curr_rate = len(current[current['disruption_type'] != 'ON_TIME']) / len(current) if 'disruption_type' in current.columns else 0
                    prev_rate = len(previous[previous['disruption_type'] != 'ON_TIME']) / len(previous) if 'disruption_type' in previous.columns else 0
                    
                    curr_delay = current['delay_min'].mean() if 'delay_min' in current.columns else 0
                    prev_delay = previous['delay_min'].mean() if 'delay_min' in previous.columns else 0
                    
                    # Compare KPIs
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    
                    with comp_col1:
                        delta = curr_rate - prev_rate
                        st.metric(
                            "     Disruption Rate",
                            f"{curr_rate:.1%}",
                            delta=f"{delta:+.1%}",
                            delta_color="inverse" if delta > 0 else "normal"
                        )
                    
                    with comp_col2:
                        delta = curr_delay - prev_delay
                        st.metric(
                            "    Avg Delay",
                            f"{curr_delay:.1f} min",
                            delta=f"{delta:+.1f}",
                            delta_color="inverse" if delta > 0 else "normal"
                        )
                    
                    with comp_col3:
                        # Week-over-week
                        trend = "    " if curr_rate > prev_rate else "    "
                        trend_desc = "Improving" if curr_rate < prev_rate else "Degrading"
                        st.metric(
                            "     Trend",
                            trend_desc,
                            delta="WoW change"
                        )
                    
                    # Trend chart
                    hourly = df_ts.dt.hour.value_counts().sort_index()
                    fig = px.line(
                        x=hourly.index,
                        y=hourly.values,
                        title="Disruptions by Hour",
                        labels={'x': 'Hour', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for temporal comparison (need 50+ records)")
            except Exception as e:
                st.warning(f"Temporal analysis unavailable: {e}")
    
    def render_weather_impact_panel(self, data: Optional[Dict] = None):
        """Weather Impact Panel - Shows current weather conditions and their impact on transit operations."""
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime
        data = data or self.data
        
        st.subheader("Weather Impact Analysis")
        
        # Weather status metrics
        weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
        
        # Sample weather data (in production, this would come from weather API)
        sample_temp = 8.5  # Celsius
        sample_wind = 12.3  # m/s
        sample_precip = 2.1  # mm/h
        sample_visibility = 8000  # meters
        
        with weather_col1:
            st.metric(
                "     Temperature",
                f"{sample_temp:.1f}°C",
                delta=f"{sample_temp - 10.0:+.1f}°C",
                delta_color="inverse" if sample_temp < 0 or sample_temp > 25 else "normal"
            )
        with weather_col2:
            st.metric(
                "    Wind Speed",
                f"{sample_wind:.1f} m/s",
                delta=f"{sample_wind - 5.0:+.1f} m/s",
                delta_color="inverse" if sample_wind > 10 else "normal"
            )
        with weather_col3:
            st.metric(
                "     Precipitation",
                f"{sample_precip:.1f} mm/h",
                delta=f"{sample_precip:+.1f} mm/h",
                delta_color="inverse" if sample_precip > 2 else "normal"
            )
        with weather_col4:
            visibility_km = sample_visibility / 1000
            st.metric(
                "    Visibility",
                f"{visibility_km:.1f} km",
                delta=f"{visibility_km - 10.0:+.1f} km",
                delta_color="normal" if visibility_km > 5 else "inverse"
            )
        
        st.markdown("---")
        
        # Weather impact assessment
        if 'df' in data and data['df'] is not None and not data['df'].empty:
            df = data['df']
            
            # Calculate weather impact if weather features exist
            weather_impact_cols = [col for col in df.columns if col.startswith('weather_') and col.endswith('_impact')]
            
            if weather_impact_cols:
                df['weather_total_impact'] = df[weather_impact_cols].mean(axis=1)
                avg_weather_impact = df['weather_total_impact'].mean()
                max_weather_impact = df['weather_total_impact'].max()
                
                impact_col1, impact_col2 = st.columns(2)
                
                with impact_col1:
                    st.metric(
                        "     Avg Weather Impact",
                        f"{avg_weather_impact:.2f}",
                        delta=f"{avg_weather_impact - 0.3:+.2f}",
                        delta_color="inverse" if avg_weather_impact > 0.5 else "normal"
                    )
                with impact_col2:
                    st.metric(
                        "    Max Weather Impact",
                        f"{max_weather_impact:.2f}",
                        delta=f"{max_weather_impact - 0.5:+.2f}",
                        delta_color="inverse" if max_weather_impact > 0.7 else "normal"
                    )
                
                # Weather impact over time
                if 'feed_timestamp' in df.columns:
                    try:
                        df_ts = pd.to_datetime(df['feed_timestamp'], errors='coerce')
                        df_copy = df.copy()
                        df_copy['hour'] = df_ts.dt.hour
                        hourly_impact = df_copy.groupby('hour')['weather_total_impact'].mean().reset_index()
                        
                        fig_impact = px.line(
                            hourly_impact,
                            x='hour',
                            y='weather_total_impact',
                            title="Hourly Weather Impact Trend",
                            markers=True
                        )
                        fig_impact.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Low Impact")
                        fig_impact.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="High Impact")
                        fig_impact.update_layout(
                            xaxis_title="Hour of Day",
                            yaxis_title="Weather Impact Score",
                            height=350
                        )
                        st.plotly_chart(fig_impact, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not process weather timestamp data: {e}")
                        # Show sample weather impact chart
                        self._render_sample_weather_chart()
                else:
                    self._render_sample_weather_chart()
            else:
                # No weather features available - show placeholder
                st.info("Weather data not available in current dataset. Weather features would be calculated from external weather data sources.")
                self._render_sample_weather_chart()
        else:
            # No data available - show sample
            st.info("No transit data available for weather correlation analysis.")
            self._render_sample_weather_chart()
    
    def _render_sample_weather_chart(self):
        """Render sample weather impact chart when no real data is available."""
        st.write("**Sample Weather Impact Analysis**")
        
        # Create sample hourly weather impact data
        hours = list(range(0, 24))
        # Simulate Dutch weather pattern: worse in morning/evening, some rain periods
        base_impact = [0.1] * 24
        # Add morning rush hour impact
        for h in [6, 7, 8, 9]:
            base_impact[h] += 0.3
        # Add evening rush hour impact  
        for h in [16, 17, 18, 19]:
            base_impact[h] += 0.2
        # Add some random weather events
        import random
        for _ in range(3):
            h = random.randint(0, 23)
            base_impact[h] += random.uniform(0.2, 0.5)
        
        # Ensure values are in valid range
        base_impact = [min(1.0, max(0.0, x)) for x in base_impact]
        
        impact_df = pd.DataFrame({
            'Hour': [f"{h:02d}:00" for h in hours],
            'Weather Impact Score': base_impact
        })
        
        fig = px.line(
            impact_df,
            x='Hour',
            y='Weather Impact Score',
            title="Sample Hourly Weather Impact Pattern",
            markers=True
        )
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Low Impact Threshold")
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="High Impact Threshold")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather condition breakdown
        st.write("**Current Weather Conditions (Sample)**")
        weather_conditions = {
            'Temperature': '8.5°C (Cool)',
            'Wind': '12.3 m/s (Moderate)', 
            'Precipitation': '2.1 mm/h (Light Rain)',
            'Visibility': '8.0 km (Reduced)',
            'Humidity': '85% (High)'
        }
        
        for condition, value in weather_conditions.items():
            st.write(f"• {condition}: {value}")
    
    def render_uncertainty_estimation(self, data: Optional[Dict] = None):
        import streamlit as st
        data = data or self.data
        
        st.subheader("🎲 Uncertainty Estimation")
        
        prediction_proba = data.get('prediction_proba')
        
        if prediction_proba is None:
            # Generate sample uncertainty if no prediction available
            st.info("No prediction probabilities available - showing model uncertainty")
            
            # Sample from bootstrap
            n_samples = 1000
            base_pred = 0.65  # base prediction
            
            # Simulate uncertainty with different variance
            samples = np.random.normal(base_pred, 0.15, n_samples)
            samples = np.clip(samples, 0, 1)
            
            lower_95 = np.percentile(samples, 2.5)
            upper_95 = np.percentile(samples, 97.5)
            lower_80 = np.percentile(samples, 10)
            upper_80 = np.percentile(samples, 90)
            mean_pred = np.mean(samples)
        else:
            # Use actual predictions
            samples = prediction_proba if isinstance(prediction_proba, np.ndarray) else np.array(prediction_proba)
            
            lower_95 = np.percentile(samples, 2.5)
            upper_95 = np.percentile(samples, 97.5)
            lower_80 = np.percentile(samples, 10)
            upper_80 = np.percentile(samples, 90)
            mean_pred = np.mean(samples)
        
        # Display uncertainty
        unc_col1, unc_col2, unc_col3 = st.columns(3)
        
        with unc_col1:
            st.metric(
                "     Mean Prediction",
                f"{mean_pred:.2f}",
                delta=f"95% CI: [{lower_95:.2f}, {upper_95:.2f}]"
            )
        
        with unc_col2:
            st.metric(
                "     80% CI",
                f"[{lower_80:.2f}, {upper_80:.2f}]"
            )
        
        with unc_col3:
            uncertainty = (upper_95 - lower_95) / 2
            conf_label = "Low" if uncertainty < 0.1 else "Medium" if uncertainty < 0.2 else "High"
            st.metric(
                "    Uncertainty",
                conf_label,
                delta=f"  {uncertainty:.2f}",
                delta_color="normal" if uncertainty < 0.15 else "inverse"
            )
        
        # Uncertainty visualization
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=samples,
            name='Prediction Distribution',
            nbinsx=30,
            opacity=0.7,
            marker_color='steelblue'
        ))
        
        # Add confidence interval lines
        fig.add_vline(x=lower_95, line_dash="dash", line_color="red", annotation="95% CI")
        fig.add_vline(x=upper_95, line_dash="dash", line_color="red")
        fig.add_vline(x=mean_pred, line_dash="solid", line_color="green", annotation="Mean")
        
        fig.update_layout(
            title="Prediction Uncertainty Distribution",
            xaxis_title="Probability",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Uncertainty estimation for operator decision-making under confidence")
    
    # =================================================
    # NLP Alert Summary Panel
    # =================================================
    
    def render_alert_nlp_summary(self, data: Optional[Dict] = None):
        """NLP-extracted alert summary panel."""
        import streamlit as st
        data = data or self.data
        
        df = data.get('df')
        st.subheader("📝 Alert NLP Summary")
        
        if df is None or df.empty:
            st.info("No data for NLP summary")
            return
        
        df = self._get_safe_copy(df)
        
        # Extract NLP features if available
        alert_cause = data.get('alert_cause', [])
        alert_effect = data.get('alert_effect', [])
        
        if not alert_cause and 'alert_cause' in df.columns:
            alert_cause = df['alert_cause'].unique().tolist()
        
        if not alert_effect and 'alert_effect' in df.columns:
            alert_effect = df['alert_effect'].unique().tolist()
        
        # Create summary
        nlp_col1, nlp_col2 = st.columns(2)
        
        with nlp_col1:
            st.write("**Alert Causes (extracted)**")
            if alert_cause:
                cause_counts = df['alert_cause'].value_counts().head(5) if 'alert_cause' in df.columns else pd.Series()
                fig_cause = px.bar(
                    x=cause_counts.values,
                    y=cause_counts.index,
                    orientation='h',
                    title="Top Alert Causes",
                    labels={'x': 'Count', 'y': 'Cause'}
                )
                st.plotly_chart(fig_cause, use_container_width=True)
            else:
                st.info("No alert cause data available")
        
        with nlp_col2:
            st.write("**Alert Effects (extracted)**")
            if alert_effect:
                effect_counts = df['alert_effect'].value_counts().head(5) if 'alert_effect' in df.columns else pd.Series()
                fig_effect = px.bar(
                    x=effect_counts.values,
                    y=effect_counts.index,
                    orientation='h',
                    title="Top Alert Effects",
                    labels={'x': 'Count', 'y': 'Effect'}
                )
                st.plotly_chart(fig_effect, use_container_width=True)
            else:
                st.info("No alert effect data available")
        
        # Sentiment summary (if available)
        if 'alert_text' in df.columns:
            st.write("**Recent Alert Texts**")
            texts = df['alert_text'].dropna().head(5).tolist()
            for i, text in enumerate(texts):
                st.caption(f"{i+1}. {text[:100]}...")
    
    def render_kpi_cards(self, data: Optional[Dict] = None):
        """Render industry-aligned KPI cards (Transit 360 style)."""
        data = data or self.data
        
        # Extract metrics
        active_disruptions = data.get('active_disruptions', 0)
        avg_delay = data.get('avg_delay', 0.0)
        prediction_f1 = data.get('prediction_f1', 0.0)
        service_delivered = data.get('service_delivered_pct', 96.5)
        on_time_pct = data.get('on_time_pct', 88.2)
        data_quality = data.get('data_quality_score', 94.0)
        latency = data.get('inference_latency_ms', 145)
        early_warning_acc = data.get('early_warning_accuracy', 0.78)
        
        # Row 1: Service Delivery KPIs (Cal-ITP style)
        st.subheader("Service Delivery Metrics")
        
        kpi_row1_col1, kpi_row1_col2, kpi_row1_col3, kpi_row1_col4 = st.columns(4)
        
        with kpi_row1_col1:
            delta_color = "normal" if service_delivered >= 98 else "inverse"
            st.metric(
                "    Service Delivered",
                f"{service_delivered:.1f}%",
                delta=f"{(service_delivered - 98):.1f}%" if service_delivered < 98 else None,
                delta_color=delta_color
            )
        with kpi_row1_col2:
            delta_color = "normal" if on_time_pct >= 90 else "inverse"
            st.metric(
                "    On-Time Performance",
                f"{on_time_pct:.1f}%",
                delta=f"{(on_time_pct - 90):.1f}%" if on_time_pct < 90 else None,
                delta_color=delta_color
            )
        with kpi_row1_col3:
            delta_color = "normal" if data_quality >= 95 else "inverse"
            st.metric(
                "     Data Quality Score",
                f"{data_quality:.1f}%",
                delta=f"{(data_quality - 95):.1f}%" if data_quality < 95 else None,
                delta_color=delta_color
            )
        with kpi_row1_col4:
            st.metric(
                "     Active Disruptions",
                str(active_disruptions),
                delta=str(-3 if active_disruptions > 10 else 0),
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Row 2: ML Performance KPIs
        st.subheader("🤖 ML Model Performance")
        
        kpi_row2_col1, kpi_row2_col2, kpi_row2_col3, kpi_row2_col4 = st.columns(4)
        
        with kpi_row2_col1:
            delta_color = "normal" if prediction_f1 >= 0.85 else "inverse"
            st.metric(
                "     Prediction F1-Score",
                f"{prediction_f1:.2f}",
                delta=f"{prediction_f1 - 0.85:.2f}" if prediction_f1 < 0.85 else None,
                delta_color=delta_color
            )
        with kpi_row2_col2:
            delta_color = "normal" if latency < 100 else "inverse"
            st.metric(
                "    Latency P95",
                f"{latency} ms",
                delta=f"{latency - 100}" if latency > 100 else None,
                delta_color=delta_color
            )
        with kpi_row2_col3:
            delta_color = "normal" if early_warning_acc >= 0.80 else "inverse"
            st.metric(
                "     Early Warning Accuracy",
                f"{early_warning_acc:.2f}",
                delta=f"{early_warning_acc - 0.80:.2f}" if early_warning_acc < 0.80 else None,
                delta_color=delta_color
            )
        with kpi_row2_col4:
            st.metric(
                "     Avg System Delay",
                f"{avg_delay:.1f} min",
                delta=f"{avg_delay - 2.0:.1f}" if avg_delay > 2.0 else None,
                delta_color="inverse" if avg_delay > 5 else "normal"
            )
        
        st.markdown("---")
    
    def render_timeline_chart(
        self,
        data: Optional[Dict] = None
    ):
        """Render disruption timeline chart."""
        data = data or self.data
        st.subheader("📈 Disruption Timeline (24h)")
        
        df_timeline = data.get('timeline')
        
        if df_timeline is None or df_timeline.empty:
            # Generate sample data
            hours = pd.date_range(
                end=datetime.now(),
                periods=24,
                freq='h'
            )
            df_timeline = pd.DataFrame({
                'hour': hours,
                'disruptions': np.random.poisson(15, 24)
            })
        
        fig = px.line(
            df_timeline,
            x='hour',
            y='disruptions',
            markers=True
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Disruptions",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_time_distributions(
        self,
        data: Optional[Dict] = None
    ):
        """Render time-based distributions (hourly, daily, peak windows)."""
        import plotly.graph_objects as go
        data = data or self.data
        
        st.subheader("⏰ Time Distributions")
        
        df = data.get('df')
        
        if df is None or df.empty:
            st.info("No data for time distribution")
            return
        
        # Build time features
        timestamp_col = None
        for col in ['timestamp', 'feed_timestamp', 'active_period_start']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            st.info("No timestamp column available")
            return
        
        # Parse timestamps
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Hourly distribution
        hourly = timestamps.dt.hour.value_counts().sort_index().reindex(range(24), fill_value=0)
        
        # Create figure with peak windows
        fig = go.Figure()
        
        # Bar chart for hours
        fig.add_trace(go.Bar(
            x=list(range(24)),
            y=hourly.values,
            marker_color='steelblue',
            name='Alerts'
        ))
        
        # Add peak windows (AM: 6-9, PM: 16-19)
        fig.add_vrect(x0=6, x1=9, fillcolor="orange", opacity=0.2, layer="below", line_width=0, annotation_text="AM Peak")
        fig.add_vrect(x0=16, x1=19, fillcolor="red", opacity=0.2, layer="below", line_width=0, annotation_text="PM Peak")
        
        fig.update_layout(
            title="Alerts by Hour (with Peak Windows)",
            xaxis_title="Hour (0-23)",
            yaxis_title="Count",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily distribution
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily = timestamps.dt.day_name().value_counts().reindex(day_order, fill_value=0)
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily.index,
            y=daily.values,
            marker_color='forestgreen',
            name='Alerts'
        ))
        
        fig_daily.update_layout(
            title="Alerts by Day of Week",
            xaxis_title="Day",
            yaxis_title="Count",
            height=250,
            template="plotly_white"
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Peak statistics
        col1, col2, col3 = st.columns(3)
        
        # Calculate peak percentages
        hour = timestamps.dt.hour
        is_peak = ((hour >= 6) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))
        is_weekend = timestamps.dt.dayofweek >= 5
        
        peak_pct = is_peak.mean() * 100
        weekend_pct = is_weekend.mean() * 100
        
        with col1:
            st.metric("📈 Peak Hour %", f"{peak_pct:.1f}%")
        with col2:
            st.metric("Weekend %", f"{weekend_pct:.1f}%")
        with col3:
            # Calculate avg alerts per hour
            avg_per_hour = hourly.mean()
            st.metric("Avg/Hour", f"{avg_per_hour:.1f}")
    
    def render_cause_effect_analysis(
        self,
        data: Optional[Dict] = None
    ):
        """Render alert cause and effect distribution."""
        import plotly.graph_objects as go
        data = data or self.data
        
        st.subheader("📋 Alert Cause & Effect Analysis")
        
        df = data.get('df')
        
        if df is None or df.empty:
            st.info("No data for cause/effect analysis")
            return
        
        col_cause, col_effect = st.columns(2)
        
        # Cause distribution
        if 'cause' in df.columns:
            cause_counts = df['cause'].value_counts().head(10)
            
            fig_cause = go.Figure()
            fig_cause.add_trace(go.Bar(
                x=cause_counts.values,
                y=cause_counts.index,
                orientation='h',
                marker_color='coral',
                name='Cause'
            ))
            fig_cause.update_layout(
                title="Top Alert Causes",
                xaxis_title="Count",
                yaxis_title="Cause",
                height=300,
                template="plotly_white"
            )
            col_cause.plotly_chart(fig_cause, use_container_width=True)
        else:
            col_cause.info("Cause data not available")
        
        # Effect distribution
        if 'effect' in df.columns:
            effect_counts = df['effect'].value_counts().head(10)
            
            fig_effect = go.Figure()
            fig_effect.add_trace(go.Bar(
                x=effect_counts.values,
                y=effect_counts.index,
                orientation='h',
                marker_color='teal',
                name='Effect'
            ))
            fig_effect.update_layout(
                title="Top Alert Effects",
                xaxis_title="Count",
                yaxis_title="Effect",
                height=300,
                template="plotly_white"
            )
            col_effect.plotly_chart(fig_effect, use_container_width=True)
        else:
            col_effect.info("Effect data not available")
        
        # Alert duration if available
        col_dur, col_severity = st.columns(2)
        
        if 'alert_duration_min' in df.columns:
            duration = df['alert_duration_min'].dropna()
            if len(duration) > 0:
                fig_dur = go.Figure()
                fig_dur.add_trace(go.Histogram(
                    x=duration,
                    nbinsx=30,
                    marker_color='steelblue',
                    name='Duration'
                ))
                fig_dur.update_layout(
                    title="Alert Duration Distribution",
                    xaxis_title="Minutes",
                    yaxis_title="Frequency",
                    height=250
                )
                col_dur.plotly_chart(fig_dur, use_container_width=True)
                
                # Stats
                col_dur.metric("Avg Duration", f"{duration.mean():.1f} min")
                col_dur.metric("Max Duration", f"{duration.max():.1f} min")
        else:
            col_dur.info("Duration data not available")
        
        # Severity if available
        if 'severity_score' in df.columns or 'severity' in df.columns:
            sev_col = 'severity_score' if 'severity_score' in df.columns else 'severity'
            severity = df[sev_col].dropna()
            if len(severity) > 0:
                fig_sev = go.Figure()
                fig_sev.add_trace(go.Histogram(
                    x=severity,
                    nbinsx=10,
                    marker_color='indianred',
                    name='Severity'
                ))
                fig_sev.update_layout(
                    title="Severity Distribution",
                    xaxis_title="Severity",
                    yaxis_title="Count",
                    height=250
                )
                col_severity.plotly_chart(fig_sev, use_container_width=True)
        else:
            col_severity.info("Severity data not available")
    
    def render_alert_distribution(
        self,
        data: Optional[Dict] = None
    ):
        """Render alert type distribution."""
        data = data or self.data
        st.subheader("Alert Type Distribution")
        
        alert_dist = data.get('alert_distribution')
        
        if alert_dist is None:
            alert_dist = {
                'Technical': 35,
                'Weather': 25,
                'Construction': 20,
                'Strike': 10,
                'Other': 10
            }
        
        fig = px.pie(
            values=list(alert_dist.values()),
            names=list(alert_dist.keys()),
            hole=0.4
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_route_performance(
        self,
        data: Optional[Dict] = None
    ):
        """Render route performance chart."""
        data = data or self.data
        st.subheader("🚏 Route Performance")
        
        df_routes = data.get('route_performance')
        
        if df_routes is None or df_routes.empty:
            df_routes = pd.DataFrame({
                'Route': [f'Route {i}' for i in 'ABCDE'],
                'On-Time %': [0.85, 0.72, 0.91, 0.68, 0.79]
            })
        
        fig = px.bar(
            df_routes,
            x='Route',
            y='On-Time %',
            color='On-Time %',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_active_alerts(
        self,
        data: Optional[Dict] = None
    ):
        """Render active alerts table."""
        data = data or self.data
        st.subheader("Active Alerts")
        
        df_alerts = data.get('active_alerts')
        
        if df_alerts is None or df_alerts.empty:
            df_alerts = pd.DataFrame({
                'Time': ['10:23', '09:45', '08:12', '07:30'],
                'Route': ['A12', 'B5', 'C3', 'R17'],
                'Type': ['Delay', 'Cancellation', 'Delay', 'Alert'],
                'Severity': ['Medium', 'High', 'Low', 'Medium'],
                'Expected Clear': ['11:30', 'TBD', '09:00', '10:15']
            })
        
        st.dataframe(df_alerts, use_container_width=True)
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("Controls")
        
        refresh_rate = st.sidebar.slider(
            "Refresh Rate (seconds)",
            min_value=30,
            max_value=300,
            value=60,
            step=30,
            key="refresh_rate_slider"
        )
        
        time_range = st.sidebar.selectbox(
            "Time Range",
            options=["1h", "6h", "24h", "7d", "30d"],
            index=2
        )
        
        route_filter = st.sidebar.multiselect(
            "Route Filter",
            options=['All Routes'] + [f'Route {i}' for i in range(1, 20)],
            default=['All Routes']
        )
        
        show_predictions = st.sidebar.checkbox(
            "Show Predictions",
            value=True
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Weather Integration")
        
        show_weather = st.sidebar.checkbox(
            "Show Weather Data",
            value=False,
            help="Display weather conditions and impacts"
        )
        
        weather_source = st.sidebar.selectbox(
            "Weather Source",
            ["None (Placeholder)", "KNMI Netherlands", "OpenWeatherMap", "Custom Upload"],
            disabled=not show_weather
        )
        
        if show_weather and weather_source != "None (Placeholder)":
            weather_update = st.sidebar.slider(
                "Weather Update (min)",
                min_value=5,
                max_value=60,
                value=15,
                step=5,
                disabled=not show_weather,
                key="weather_update_slider"
            )
        else:
            weather_update = 15
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Display Options")
        
        public_mode = st.sidebar.checkbox(
            "Public-Facing Mode",
            value=False,
            help="Simplified view for public consumption with travel advisories"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Filter Settings")
        
        min_severity = st.sidebar.slider(
            "Minimum Severity",
            min_value=1,
            max_value=10,
            value=3,
            key="min_severity_slider"
        )
        
        return {
            'refresh_rate': refresh_rate,
            'time_range': time_range,
            'route_filter': route_filter,
            'show_predictions': show_predictions,
            'min_severity': min_severity,
            'show_weather': show_weather,
            'weather_source': weather_source if show_weather else "None",
            'weather_update': weather_update
        }
    
    def render_heatmap(
        self,
        data: Optional[Dict] = None
    ):
        """Render enhanced congestion visualization with color-coded corridors and heatmaps."""
        data = data or self.data
        st.subheader("Congestion Analysis & Heatmaps")
        
        # Create tabs for different congestion views
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Corridor Speed Map", "Stop Hotspots", "Congestion Trends"])
        
        with viz_tab1:
            self._render_corridor_speed_map(data)
            
        with viz_tab2:
            self._render_enhanced_stop_heatmap(data)
            
        with viz_tab3:
            self._render_congestion_trends(data)
    
    def _render_corridor_speed_map(self, data: Optional[Dict] = None):
        """Render color-coded corridor speed visualization."""
        st.write("**Corridor Speed Analysis**")
        
        df = data.get('df') if data else None
        if df is None or df.empty:
            # Sample corridor data
            corridors = ['Corridor A', 'Corridor B', 'Corridor C', 'Corridor D', 'Corridor E']
            speeds = [25, 18, 32, 22, 28]  # mph
            status = []
            for speed in speeds:
                if speed >= 25:
                    status.append("🟢 Free Flow")
                elif speed >= 20:
                    status.append("🟡 Moderate")
                else:
                    status.append("🔴 Congested")
            
            corridor_df = pd.DataFrame({
                'Corridor': corridors,
                'Speed (mph)': speeds,
                'Status': status
            })
            
            fig = px.bar(
                corridor_df,
                x='Corridor',
                y='Speed (mph)',
                color='Speed (mph)',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="Average Speed by Corridor",
                text='Speed (mph)'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Status summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Free Flow Corridors", "2", "0")
            with col2:
                st.metric("Moderate Corridors", "2", "+1")
            with col3:
                st.metric("Congested Corridors", "1", "-1")
        else:
            # Real corridor analysis from data
            if 'route_id' in df.columns and 'delay_min' in df.columns:
                # Calculate average delay by route
                route_delay = df.groupby('route_id')['delay_min'].mean().reset_index()
                route_delay.columns = ['Route', 'Avg Delay (min)']
                
                # Convert delay to speed estimate (simplified)
                # Base speed 30 mph, reduce by delay factor
                route_delay['Speed Estimate (mph)'] = 30 / (1 + route_delay['Avg Delay (min)'] / 10)
                route_delay['Speed Estimate (mph)'] = route_delay['Speed Estimate (mph)'].clip(5, 30)
                
                # Color coding
                def get_speed_color(speed):
                    if speed >= 25:
                        return '🟢 Free Flow'
                    elif speed >= 20:
                        return '🟡 Moderate'
                    else:
                        return '🔴 Congested'
                
                route_delay['Status'] = route_delay['Speed Estimate (mph)'].apply(get_speed_color)
                
                fig = px.bar(
                    route_delay,
                    x='Route',
                    y='Speed Estimate (mph)',
                    color='Speed Estimate (mph)',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="Estimated Speed by Route",
                    text='Speed Estimate (mph)'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Route status summary
                status_counts = route_delay['Status'].value_counts()
                col1, col2, col3 = st.columns(3)
                with col1:
                    free_flow = status_counts.get('🟢 Free Flow', 0)
                    st.metric("Free Flow Routes", free_flow)
                with col2:
                    moderate = status_counts.get('🟡 Moderate', 0)
                    st.metric("Moderate Routes", moderate)
                with col3:
                    congested = status_counts.get('🔴 Congested', 0)
                    st.metric("Congested Routes", congested)
            else:
                st.info("Route or delay data not available for corridor analysis")
    
    def _render_enhanced_stop_heatmap(self, data: Optional[Dict] = None):
        """Render enhanced stop hotspot heatmap with congestion coloring."""
        st.write("**Stop Congestion Hotspots**")
        
        hotspots = data.get('hotspots') if data else None
        
        if hotspots is None or (isinstance(hotspots, pd.Series) and hotspots.empty) or (isinstance(hotspots, pd.DataFrame) and hotspots.empty):
            # Sample stop data with congestion levels
            stops = [f'Stop {i}' for i in range(1, 16)]
            hours = list(range(6, 22))  # 6 AM to 10 PM
            
            # Create congestion pattern: higher during peak hours
            values = np.random.poisson(5, (len(stops), len(hours)))
            # Add peak hour patterns
            for i, hour in enumerate(hours):
                if hour in [7, 8, 9, 16, 17, 18]:  # Peak hours
                    values[:, i] = values[:, i] * 3
            
            df_heatmap = pd.DataFrame(values, index=stops, columns=[f"{h:02d}:00" for h in hours])
        elif isinstance(hotspots, pd.Series):
            # Convert Series to matrix format for single metric
            stops = hotspots.index.tolist()
            hours = list(range(6, 22))
            # Create dummy hourly data based on the series values
            values = np.outer(hotspots.values, np.ones(len(hours)))
            df_heatmap = pd.DataFrame(values, index=stops, columns=[f"{h:02d}:00" for h in hours])
        else:
            df_heatmap = hotspots if isinstance(hotspots, pd.DataFrame) else pd.DataFrame()
        
        if df_heatmap.empty or not isinstance(df_heatmap, pd.DataFrame):
            st.info("No hotspot data available")
        else:
            # Create heatmap with congestion coloring (yellow to red)
            fig = px.imshow(
                df_heatmap.values,
                labels=dict(x="Time of Day", y="Stop", color="Delay (min)"),
                x=df_heatmap.columns,
                y=df_heatmap.index,
                color_continuous_scale=['yellow', 'orange', 'red', 'darkred']
            )
            fig.update_layout(
                height=500,
                title="Stop Delay Heatmap (6:00 - 22:00)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top congested stops
            if isinstance(hotspots, pd.Series) or (isinstance(hotspots, pd.DataFrame) and len(hotspots.columns) == 1):
                # Already have aggregated data
                top_stops = hotspots.head(10) if isinstance(hotspots, pd.Series) else hotspots.iloc[:, 0].head(10)
            else:
                # Calculate average delay per stop
                top_stops = df_heatmap.mean(axis=1).sort_values(ascending=False).head(10)
            
            st.write("**Top 10 Most Congested Stops**")
            for i, (stop, delay) in enumerate(top_stops.items(), 1):
                congestion_level = "🔴 Severe" if delay > 15 else "🟡 High" if delay > 8 else "🟢 Moderate"
                st.write(f"{i}. {stop}: {delay:.1f} min delay {congestion_level}")
    
    def _render_congestion_trends(self, data: Optional[Dict] = None):
        """Render congestion trends and analytics."""
        st.write("**Congestion Trends & Analytics**")
        
        df = data.get('df') if data else None
        if df is None or df.empty:
            # Sample trend data
            hours = list(range(6, 22))
            congestion_level = [2, 3, 5, 8, 12, 18, 20, 19, 16, 12, 8, 5, 4, 3, 2, 2]  # 6AM-10PM (16 values)
            
            trend_df = pd.DataFrame({
                'Hour': [f"{h:02d}:00" for h in hours],
                'Congestion Level (min delay)': congestion_level
            })
            
            fig = px.line(
                trend_df,
                x='Hour',
                y='Congestion Level (min delay)',
                title="Daily Congestion Pattern",
                markers=True
            )
            fig.add_hline(y=3, line_dash="dash", line_color="green", annotation_text="Target (<3 min)")
            fig.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="Heavy Congestion (>8 min)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak congestion info
            peak_hour_idx = congestion_level.index(max(congestion_level))
            peak_hour = f"{hours[peak_hour_idx]:02d}:00"
            peak_value = max(congestion_level)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Congestion Hour", peak_hour)
            with col2:
                st.metric("Max Delay", f"{peak_value:.1f} min")
            with col3:
                st.metric("Duration >5 min", f"{sum(1 for c in congestion_level if c > 5)} hours")
        else:
            # Real trend analysis from data
            if 'feed_timestamp' in df.columns and 'delay_min' in df.columns:
                try:
                    df_ts = pd.to_datetime(df['feed_timestamp'], errors='coerce')
                    df_copy = df.copy()
                    df_copy['hour'] = df_ts.dt.hour
                    hourly_delay = df_copy.groupby('hour')['delay_min'].mean().reset_index()
                    
                    fig = px.line(
                        hourly_delay,
                        x='hour',
                        y='delay_min',
                        title="Hourly Average Delay Trend",
                        markers=True
                    )
                    fig.add_hline(y=3, line_dash="dash", line_color="green", annotation_text="Target (<3 min)")
                    fig.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="Heavy Congestion (>8 min)")
                    fig.update_layout(
                        xaxis_title="Hour of Day",
                        yaxis_title="Average Delay (min)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend statistics
                    peak_hour = hourly_delay.loc[hourly_delay['delay_min'].idxmax()]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Peak Hour", f"{int(peak_hour['hour']):02d}:00")
                    with col2:
                        st.metric("Max Delay", f"{peak_hour['delay_min']:.1f} min")
                    with col3:
                        hours_above_target = sum(1 for d in hourly_delay['delay_min'] if d > 3)
                        st.metric("Hours Above Target", f"{hours_above_target}/24")
                        
                except Exception as e:
                    st.warning(f"Could not process timestamp data for trends: {e}")
                    # Fallback to sample
                    self._render_sample_congestion_trends()
            else:
                st.info("Timestamp or delay data not available for trend analysis")
                # Show sample
                self._render_sample_congestion_trends()
    
    def render_network_tab(
        self,
        G=None,
        stops_df: Optional[pd.DataFrame] = None
    ):
        """
        Render interactive network visualization tab.
        
        Uses Pyvis for interactive graph visualization.
        Requires pyvis package.
        """
        st.subheader("🔗 Interactive Stop Network")
        
        try:
            from pyvis.network import Network
            import networkx as nx
        except ImportError:
            st.warning("Pyvis not installed. Install with: pip install pyvis")
            st.info("Sample network visualization:")
            
            # Show sample static network
            sample_nodes = ['Stop A', 'Stop B', 'Stop C', 'Stop D', 'Stop E']
            sample_edges = [
                ('Stop A', 'Stop B', 10),
                ('Stop B', 'Stop C', 15),
                ('Stop C', 'Stop A', 8),
                ('Stop A', 'Stop D', 5),
                ('Stop D', 'Stop E', 12),
            ]
            
            fig = go.Figure()
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=[0, 1, 2, 0, 1],
                y=[0, 0, 0, 1, 1],
                mode='markers+text',
                marker=dict(size=30, color='#7F77DD'),
                text=sample_nodes,
                textposition='top center'
            ))
            
            # Add edges
            for src, dst, weight in sample_edges:
                x_coords = [sample_nodes.index(src), sample_nodes.index(dst)]
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=[0.5, 0.5],
                    mode='lines',
                    line=dict(width=weight/3),
                    showlegend=False
                ))
            
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            st.plotly_chart(fig, use_container_width=True)
            return
        
        if G is None or G.number_of_nodes() == 0:
            st.info("No network data available. Generate from stop_times.")
            
            # Create sample network
            import networkx as nx
            G = nx.DiGraph()
            sample_edges = [
                ('0', '1'), ('1', '2'), ('2', '0'), ('0', '3'), ('3', '4')
            ]
            G.add_edges_from(sample_edges)
        
        # Remove isolated nodes
        SG = G.copy()
        SG.remove_nodes_from(list(nx.isolates(SG)))
        
        if SG.number_of_nodes() == 0:
            st.warning("No connected nodes in network")
            return
        
        # Build Pyvis network
        net = Network(
            height='500px',
            width='100%',
            bgcolor='#0f1117',
            font_color='#c2c0b6',
            directed=True,
            notebook=True,
            cdn_resources='remote'
        )
        
        # Physics configuration
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 120,
              "springConstant": 0.04,
              "damping": 0.09
            }
          },
          "edges": {
            "color": {"color": "#444444"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "width": 0.5
          },
          "nodes": {
            "borderWidth": 1,
            "font": {"size": 12}
          }
        }
        """)
        
        # Node degree metrics
        in_deg = dict(SG.in_degree())
        out_deg = dict(SG.out_degree())
        deg = dict(SG.degree())
        max_in = max(in_deg.values()) if in_deg else 1
        
        # Color by centrality
        def get_color(node):
            d = in_deg.get(node, 0)
            ratio = d / max_in if max_in > 0 else 0
            if ratio > 0.66:
                return '#7F77DD'  # purple - hub
            elif ratio > 0.33:
                return '#1D9E75'  # teal - medium
            return '#888780'  # gray - minor
        
        def get_size(node):
            return max(10, min(40, deg.get(node, 1) * 4))
        
        # Get stop names if available
        name_map = {}
        if stops_df is not None and 'stop_name' in stops_df.columns:
            name_map = stops_df.set_index(
                stops_df['stop_id'].astype(str)
            )['stop_name'].to_dict()
        
        # Add nodes
        for node in SG.nodes():
            name = name_map.get(str(node), str(node))
            in_d = in_deg.get(node, 0)
            out_d = out_deg.get(node, 0)
            title = f"<b>{name}</b><br>In: {in_d}<br>Out: {out_d}"
            
            net.add_node(
                str(node),
                label=name[:15] if len(name) > 15 else name,
                title=title,
                color=get_color(node),
                size=get_size(node)
            )
        
        # Add edges with weights
        edge_weights = {}
        for u, v in SG.edges():
            key = (str(u), str(v))
            edge_weights[key] = edge_weights.get(key, 0) + 1
        
        for (u, v), w in edge_weights.items():
            net.add_edge(u, v, width=min(0.5 + w * 0.3, 4))
        
        # Save and display
        try:
            html_path = "/tmp/transit_network.html"
            net.save_graph(html_path)
            
            # Read and display
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            st.components.html(html_content, height=520, scrolling=True)
        except Exception as e:
            st.warning(f"Could not render network: {e}")
    
    def render_prediction_gauge(
        self,
        accuracy: float = 0.85
    ):
        """Render prediction accuracy gauge."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,
            title={"text": "Model Accuracy (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green"},
                "steps": [
                    {"range": [0, 70], "color": "red"},
                    {"range": [70, 85], "color": "yellow"},
                    {"range": [85, 100], "color": "green"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 85
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_dashboard(
        self,
        df: Optional[pd.DataFrame] = None,
        G=None,
        stops_df: Optional[pd.DataFrame] = None
    ):
        """
        Render the complete dashboard with tabs.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            Feature DataFrame from pipeline
        G : networkx.DiGraph, optional
            Pre-built stop sequence graph
        stops_df : pd.DataFrame, optional
            Stop metadata for labels
        """
        data = self.data
        
        if df is not None and not df.empty:
            data = prepare_dashboard_data(df)
        
        st.title(self.title)
        st.markdown(
            "Real-time disruption monitoring for transit operators"
        )
        
        # Header
        self.render_header()
        st.markdown("---")
        
        # SAP Digital Boardroom Panels (with error boundaries)
        try:
            self.render_sap_kpi_row(data)
        except Exception as e:
            st.error(f"SAP KPI error: {e}")
        
        try:
            self.render_incident_management_panel(data)
        except Exception as e:
            st.error(f"Incident Management error: {e}")
        
        try:
            self.render_simulation_panel(data)
        except Exception as e:
            st.error(f"Simulation error: {e}")
        
        try:
            self.render_budget_actuals(data)
        except Exception as e:
            st.error(f"Budget error: {e}")
        
        # == Traffic Management KPIs == (InetSoft standard)
        try:
            self.render_traffic_kpis(data)
        except Exception as e:
            st.error(f"Traffic KPIs error: {e}")
        
        # == Temporal Comparison == & Uncertainty
        try:
            self.render_temporal_comparison(data)
        except Exception as e:
            st.error(f"Temporal comparison error: {e}")
        
        try:
            self.render_uncertainty_estimation(data)
        except Exception as e:
            st.error(f"Uncertainty estimation error: {e}")
        
        # NLP Summary
        try:
            self.render_alert_nlp_summary(data)
        except Exception as e:
            st.error(f"NLP summary error: {e}")
        
        # Weather Impact Panel
        try:
            self.render_weather_impact_panel(data)
        except Exception as e:
            st.error(f"Weather panel error: {e}")
        
        st.markdown("---")
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.info(
            "Transit Disruption Dashboard\n\n"
            "Real-time monitoring for transit operators.\n\n"
            "Version: 2.0.0\n"
            "Model: LightGBM + STARN-GAT"
        )


def load_saved_artifacts() -> pd.DataFrame:
    """
    Load saved model artifacts and visualizations.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with artifact information
    """
    import os
    
    artifacts = {
        'models': [],
        'visualizations': [],
        'configs': []
    }
    
    # Models
    models_dir = "models"
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith('.pkl'):
                path = os.path.join(models_dir, f)
                size = os.path.getsize(path) / 1024
                artifacts['models'].append({
                    'name': f,
                    'type': 'model',
                    'size_kb': size,
                    'path': path
                })
    
    # Visualizations
    viz_dir = "visualizations"
    if os.path.exists(viz_dir):
        for f in os.listdir(viz_dir):
            ext = os.path.splitext(f)[1]
            path = os.path.join(viz_dir, f)
            size = os.path.getsize(path) / 1024
            artifacts['visualizations'].append({
                'name': f,
                'type': ext.replace('.', ''),
                'size_kb': size,
                'path': path
            })
    
    # Configs
    configs = ['config.yaml', 'config_test.yaml']
    for cfg in configs:
        if os.path.exists(cfg):
            size = os.path.getsize(cfg) / 1024
            artifacts['configs'].append({
                'name': cfg,
                'type': 'yaml',
                'size_kb': size,
                'path': cfg
            })
    
    # Create summary DataFrame
    all_items = (
        artifacts['models'] + 
        artifacts['visualizations'] + 
        artifacts['configs']
    )
    
    if all_items:
        return pd.DataFrame(all_items)
    return pd.DataFrame()


def main():
    """Main entry point with data loading."""
    
    # Sidebar - Data Source Selection
    st.sidebar.header("Data Source")
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Sample Data", "Live Feed", "Saved Artifacts", "Zip File (parquet)"]
    )
    
    df = pd.DataFrame()
    
    if data_source == "Live Feed":
        with st.spinner("Fetching live feed from ovapi.nl..."):
            df = load_live_feed_data()
            if df.empty:
                st.warning("No live feed data available, using sample data")
                data_source = "Sample Data"
    elif data_source == "Saved Artifacts":
        with st.spinner("Loading saved artifacts..."):
            artifacts_df = load_saved_artifacts()
            if not artifacts_df.empty:
                st.success(f"Found {len(artifacts_df)} artifacts")
                
                # Build dropdown options from artifact name and type
                options = []
                for _, row in artifacts_df.iterrows():
                    name = row.get('name', 'Unknown')
                    art_type = row.get('type', '?')
                    options.append(f"{name} ({art_type})")
                
                # Dropdown selection
                selected = st.selectbox(
                    "Select Artifact",
                    options,
                    key="saved_artifact_select",
                    help="Choose a saved model, visualization, or config"
                )
                
                # Show details for selected artifact
                sel_name = selected.split(' (')[0]
                selected_row = artifacts_df[artifacts_df['name'] == sel_name].iloc[0]
                
                st.write("**Artifact Details:**")
                # Display key fields
                detail_cols = ['name', 'type', 'size_kb', 'path']
                for col in detail_cols:
                    if col in selected_row:
                        st.write(f"**{col.replace('_', ' ').title()}:** {selected_row[col]}")
            else:
                st.info("No saved artifacts found in models/, visualizations/, or configs/ directories.")
    elif data_source == "Zip File (parquet)":
        import io
        import zipfile
        MAX_UPLOAD_MB = 300
        uploaded_zip = st.sidebar.file_uploader(
            f"Upload ZIP with Parquet Files (max {MAX_UPLOAD_MB}MB)",
            type=['zip'],
            accept_multiple_files=False,
            help=f"Maximum file size: {MAX_UPLOAD_MB}MB"
        )
        if uploaded_zip:
            file_size_mb = uploaded_zip.size / (1024 * 1024)
            if file_size_mb > MAX_UPLOAD_MB:
                st.error(f"File size ({file_size_mb:.1f}MB) exceeds {MAX_UPLOAD_MB}MB limit")
                uploaded_zip = None
        if uploaded_zip:
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as z:
                    parquet_files = [f for f in z.namelist() if f.endswith('.parquet')]
                    if not parquet_files:
                        st.warning("No parquet files found in ZIP")
                    else:
                        frames = []
                        for pf in parquet_files:
                            with z.open(pf) as f:
                                frames.append(pd.read_parquet(io.BytesIO(f.read())))
                        df_raw = pd.concat(frames, ignore_index=True)
                        st.success(f"Loaded {len(df_raw)} rows from {len(parquet_files)} parquet files in ZIP")

                        with st.spinner("Building features..."):
                            from gtfs_disruption.features import DisruptionFeatureBuilder
                            try:
                                gtfs_data = {}
                                builder = DisruptionFeatureBuilder(df_raw, gtfs_data)
                                df = builder.build()
                                if df.empty:
                                    df = df_raw
                            except Exception:
                                df = df_raw

                        st.success(f"Processed {len(df)} feature records")
            except Exception as e:
                st.error(f"Error loading ZIP: {e}")

    # Ensure dashboard columns before prepare
    if data_source not in ["Sample Data", "Saved Artifacts"] and not df.empty:
        df = _ensure_dashboard_columns(df)

    # Prepare data
    if data_source == "Sample Data":
        data = _generate_sample_data()
    elif data_source == "Saved Artifacts":
        data = _generate_sample_data()
    else:
        data = prepare_dashboard_data(df)
    
    # Render dashboard
    dashboard = TransitDashboard(data)
    dashboard.render_dashboard()


# =========================================================================
# EXTERNAL API INTEGRATION TAB
# =========================================================================

    def render_public_facing_view(self, data: Optional[Dict] = None):
        """Public-facing view - Simplified dashboard for commuters and general public."""
        import streamlit as st
        data = data or self.data
        
        st.header("🚇 Public Transit Information")
        st.caption("Real-time service updates for commuters")
        
        # Service status banner
        df = data.get('df') if data else None
        if df is not None and not df.empty and 'disruption_type' in df.columns:
            on_time_pct = len(df[df['disruption_type'] == 'ON_TIME']) / len(df) * 100 if len(df) > 0 else 100
            if on_time_pct >= 95:
                service_status = "🟢 GOOD SERVICE"
                service_msg = "Most routes operating normally"
            elif on_time_pct >= 85:
                service_status = "🟡 MODERATE DELAYS"
                service_msg = "Some routes experiencing delays"
            else:
                service_status = "🔴 SERVICE DISRUPTIONS"
                service_msg = "Significant delays or cancellations reported"
        else:
            service_status = "⚪ SERVICE STATUS UNKNOWN"
            service_msg = "No service data available"
        
        st.markdown(f"""
        <div style="background-color: {'#d4edda' if 'GOOD' in service_status else '#fff3cd' if 'MODERATE' in service_status else '#f8d7da'}; 
                    padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: {'#155724' if 'GOOD' in service_status else '#856404' if 'MODERATE' in service_status else '#721c24'}; margin: 0;">
                {service_status}
            </h3>
            <p style="color: {'#155724' if 'GOOD' in service_status else '#856404' if 'MODERATE' in service_status else '#721c24'}; margin: 5px 0 0 0;">
                {service_msg}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Travel advisories
        st.subheader("📢 Travel Advisories")
        
        if df is not None and not df.empty and 'disruption_type' in df.columns:
            disruptions = df[df['disruption_type'] != 'ON_TIME']
            if not disruptions.empty:
                # Group disruptions by type and route
                advisory_data = []
                for _, disruption in disruptions.head(10).iterrows():  # Show top 10
                    route = disruption.get('route_id', 'Unknown')
                    disruption_type = disruption.get('disruption_type', 'Unknown')
                    delay = disruption.get('delay_min', 0)
                    
                    if disruption_type == 'CANCELLED':
                        advisory = f"Route {route}: Service cancelled"
                    elif delay > 15:
                        advisory = f"Route {route}: Major delays ({delay:.0f} min)"
                    elif delay > 5:
                        advisory = f"Route {route}: Moderate delays ({delay:.0f} min)"
                    else:
                        advisory = f"Route {route}: Minor delays"
                    
                    advisory_data.append(advisory)
                
                if advisory_data:
                    for advisory in advisory_data:
                        st.write(f"• {advisory}")
                else:
                    st.info("No specific advisories available")
            else:
                st.success("No service disruptions reported")
        else:
            st.info("Service information not available")
        
        # Recommended actions
        st.subheader("💡 Travel Recommendations")
        
        if df is not None and not df.empty and 'disruption_type' in df.columns:
            on_time_pct = len(df[df['disruption_type'] == 'ON_TIME']) / len(df) * 100 if len(df) > 0 else 100
            
            if on_time_pct >= 95:
                st.write("• Normal travel times expected")
                st.write("• All routes operating on or near schedule")
            elif on_time_pct >= 85:
                st.write("• Allow extra 5-10 minutes for your journey")
                st.write("• Check specific route status before traveling")
                st.write("• Consider alternative routes if available")
            else:
                st.write("• Allow extra 15-20 minutes for your journey")
                st.write("• Significant delays expected on multiple routes")
                st.write("• Strongly consider alternative transportation if possible")
                st.write("• Check for service updates frequently")
        else:
            st.write("• Check back closer to your travel time for updates")
            st.write("• Allow normal travel time unless notified otherwise")
        
        # Simple service metrics
        st.subheader("Service Overview")
        
        if df is not None and not df.empty:
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                total_routes = len(df['route_id'].unique()) if 'route_id' in df.columns else 0
                st.metric("Active Routes", total_routes)
            
            with metric_col2:
                if 'disruption_type' in df.columns:
                    on_time_count = len(df[df['disruption_type'] == 'ON_TIME'])
                    on_time_pct = (on_time_count / len(df)) * 100 if len(df) > 0 else 0
                    st.metric("On-Time Service", f"{on_time_pct:.0f}%")
                else:
                    st.metric("On-Time Service", "N/A")
            
            with metric_col3:
                if 'disruption_type' in df.columns:
                    disrupted_count = len(df[df['disruption_type'] != 'ON_TIME'])
                    st.metric("Routes with Issues", disrupted_count)
                else:
                    st.metric("Routes with Issues", "N/A")
        else:
            st.info("Service data not available")
        
        # Next update
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M')} | Next update in {sidebar_settings.get('refresh_rate', 60)}s")
    
    def render_adaptive_control_panel(self, data: Optional[Dict] = None):
        """Adaptive Transit Control Panel - Holding strategies, headway regulation, transfer protection."""
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        data = data or self.data
        
        st.subheader("Adaptive Transit Control")
        
        # Control strategy metrics
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)
        
        with control_col1:
            st.metric(
                "     Holding Actions",
                "12",
                delta="+3",
                delta_color="inverse"
            )
        with control_col2:
            st.metric(
                "    Headway Regularity",
                "78%",
                delta="-5%",
                delta_color="inverse"
            )
        with control_col3:
            st.metric(
                "     Transfer Protection",
                "92%",
                delta="+2%",
                delta_color="normal"
            )
        with control_col4:
            st.metric(
                "    Schedule Adherence",
                "85%",
                delta="+1%",
                delta_color="normal"
            )
        
        st.markdown("---")
        
        # Control strategy explanations
        st.write("**Adaptive Control Strategies**")
        
        ctrl_col1, ctrl_col2 = st.columns(2)
        
        with ctrl_col1:
            st.info("""
            **🟢 Holding Strategies**
            - Hold vehicles at key stops to maintain headways
            - Prevent bunching and improve regularity
            - Applied when downstream delay > threshold
            - Maximizes passenger throughput
            """)
            
            st.info("""
            **🟡 Headway Regulation**
            - Maintain consistent time between vehicles
            - Real-time adjustments to dwell time
            - Based on actual vs scheduled headways
            - Reduces passenger waiting time variance
            """)
        
        with ctrl_col2:
            st.info("""
            **🔵 Transfer Protection**
            - Hold connecting vehicles for transferring passengers
            - Based on real-time arrival predictions
            - Balances transfer vs. downstream delay
            - Improves network reliability
            """)
            
            st.info("""
            **⚪ Schedule Adherence**
            - Overall conformity to timetable
            - Combined measure of punctuality and reliability
            - Key performance indicator for transit agencies
            - Affected by all control strategies
            """)
        
        st.markdown("---")
        
        # Control effectiveness visualization
        st.write("**Control Effectiveness Over Time**")
        
        # Sample data for control effectiveness
        hours = list(range(0, 24))
        holding_effectiveness = [0.6 + 0.2 * (abs(h - 12) / 12) for h in hours]  # Better during peak
        headway_regularity = [0.7 - 0.1 * (abs(h - 12) / 12) for h in hours]  # Worse during peak
        transfer_protection = [0.85 + 0.1 * (1 - abs(h - 12) / 12) for h in hours]  # Better during peak
        
        control_df = pd.DataFrame({
            'Hour': [f"{h:02d}:00" for h in hours],
            'Holding Effectiveness': holding_effectiveness,
            'Headway Regularity': headway_regularity,
            'Transfer Protection': transfer_protection
        })
        
        fig = px.line(
            control_df,
            x='Hour',
            y=['Holding Effectiveness', 'Headway Regularity', 'Transfer Protection'],
            title="Adaptive Control Effectiveness by Time of Day",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Effectiveness Score (0-1)",
            height=400,
            legend_title="Control Strategy"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Control recommendations based on current conditions
        st.write("**Current Control Recommendations**")
        
        df = data.get('df') if data else None
        if df is not None and not df.empty:
            # Analyze current conditions for control recommendations
            if 'delay_min' in df.columns:
                avg_delay = df['delay_min'].mean()
                delay_std = df['delay_min'].std()
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    if avg_delay > 10:
                        st.warning("""
                        **High Delay Conditions**
                        - Increase holding at terminal stations
                        - Implement headway-based control
                        - Consider short-turning vehicles
                        """)
                    elif avg_delay > 5:
                        st.info("""
                        **Moderate Delay Conditions**
                        - Standard holding strategies
                        - Monitor for bunching formation
                        - Prepare transfer protection
                        """)
                    else:
                        st.success("""
                        **Normal Conditions**
                        - Regular schedule adherence
                        - Minimal intervention needed
                        - Standard transfer protection
                        """)
                
                with rec_col2:
                    if delay_std > 8:
                        st.warning("""
                        **High Delay Variability**
                        - Strong bunching likely
                        - Aggressive holding recommended
                        - Consider skip-stop patterns
                        """)
                    elif delay_std > 4:
                        st.info("""
                        **Moderate Delay Variability**
                        - Standard regulation approaches
                        - Watch for emerging platoons
                        """)
                    else:
                        st.success("""
                        **Low Delay Variability**
                        - Regular operations
                        - Standard headway maintenance
                        """)
        else:
            # Sample recommendations
            st.info("""
            **Current Conditions (Sample)**
            - Moderate delays detected on Corridor B
            - Headway bunching observed on Route 12
            - Recommend: Holding at key transfer points
            """)
    
def render_external_apis():
    """Render external API integration tab."""
    import streamlit as st
    from utils.dashboard_integration import (
        DashboardAPIClient,
        create_sidebar_controls,
    )
    
    st.header("🔗 External API Integration")
    
    # Check if integration is available
    if not DASHBOARD_INTEGRATION_AVAILABLE:
        st.warning("Dashboard integration module not available")
        return
    
    # Create sidebar controls
    controls = create_sidebar_controls()
    
    st.markdown("---")
    
    # API Connection section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Transit Dashboard API")
        
        api_url = controls.get("api_url", "http://localhost:8000")
        
        # Test connection
        if st.button("Test API Connection"):
            with st.spinner("Testing connection..."):
                try:
                    client = DashboardAPIClient(api_url)
                    health = client.health_check()
                    
                    if "error" not in health:
                        st.success(f"Connected: {health}")
                    else:
                        st.error(f"Connection failed: {health}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Get model info
        if st.button("Get Model Info"):
            try:
                client = DashboardAPIClient(api_url)
                info = client.get_model_info()
                st.json(info)
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Metrics
        st.markdown("##### Service Metrics")
        if st.button("Get Metrics", key="get_metrics"):
            try:
                client = DashboardAPIClient(api_url)
                metrics = client.get_metrics()
                st.json(metrics)
            except Exception as e:
                st.warning(f"API not available: {e}")
    
    with col2:
        st.subheader("Transit Sentinel")
        
        st.info("Transit Sentinel provides live data feeds:")
        st.markdown("""
        - AVL Feed (vehicle positions)
        - Service Alerts
        - Trip Updates
        - GTFS-RT feeds
        """)
        
        st.markdown("##### Connection Status")
        if st.button("Check AVL Feed", key="check_avl"):
            try:
                sentinel = SentinelConnector()
                avl = sentinel.get_avl_feed(limit=10)
                st.success(f"AVL Feed: {len(avl)} records")
            except Exception as e:
                st.warning(f"Sentinel not connected: {e}")
    
    st.markdown("---")
    
    # Unified Prediction
    st.subheader("🎯 Unified Predictions")
    
    # Sample features for testing
    test_features = st.text_area(
        "Test Features (comma-separated)",
        "120, 25, 5, 14, 3",
        help="delay_sec, speed_kmh, headway_min, hour, day_of_week"
    )
    
    if st.button("Get Prediction"):
        try:
            features = [float(x) for x in test_features.split(",")]
            predictor = UnifiedPredictor(api_url=controls.get("api_url"))
            result = predictor.predict(features)
            st.json(result)
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()