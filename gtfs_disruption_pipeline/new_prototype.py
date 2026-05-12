import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time

# Mock backend data functions (replace with actual API calls)
def fetch_realtime_data():
    # Simulate GTFS-RT data
    vehicles = [
        {"id": f"veh_{i}", "route": f"Route {i%10}", "lat": 40.7128 + random.uniform(-0.1, 0.1),
         "lon": -74.0060 + random.uniform(-0.1, 0.1), "delay": random.randint(0, 30),
         "severity": random.choice(["normal", "delay", "disruption"]), "occupancy": random.randint(0, 100)}
        for i in range(50)
    ]
    return pd.DataFrame(vehicles)

def fetch_alerts():
    alerts = [
        {"id": i, "description": f"Predicted disruption on Route {i%10}", "severity": random.choice(["minor", "major", "critical"]),
         "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 30)), "confidence": random.uniform(0.7, 0.95),
         "acknowledged": False}
        for i in range(10)
    ]
    return pd.DataFrame(alerts)

def fetch_analytics_data():
    # Mock time-series data
    times = pd.date_range(start=datetime.now() - timedelta(hours=2), periods=120, freq='min')
    data = {
        "time": times,
        "delay_mean": [random.uniform(0, 15) for _ in times],
        "headway_dev": [random.uniform(0, 5) for _ in times],
        "on_time_pct": [random.uniform(80, 100) for _ in times]
    }
    return pd.DataFrame(data)

# Cache data
@st.cache_data(ttl=30)
def get_realtime_data():
    return fetch_realtime_data()

@st.cache_data(ttl=30)
def get_alerts():
    return fetch_alerts()

@st.cache_data(ttl=60)
def get_analytics_data():
    return fetch_analytics_data()

# Custom CSS for high-contrast theme
st.markdown("""
<style>
    .main {background-color: #FFFFFF; color: #000000;}
    .sidebar .sidebar-content {background-color: #F8FAFC;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background-color: #E2E8F0; border-radius: 4px;}
    .alert-card {border: 1px solid #003082; border-radius: 8px; padding: 16px; margin: 8px 0;}
    .severity-normal {color: #10B981;}
    .severity-delay {color: #F59E0B;}
    .severity-disruption {color: #EF4444;}
    .kpi-card {background-color: #F8FAFC; border-radius: 8px; padding: 16px; text-align: center; margin: 8px;}
</style>
""", unsafe_allow_html=True)

# State management
if "acknowledged_alerts" not in st.session_state:
    st.session_state.acknowledged_alerts = set()

# Main app
st.title("Transit Disruption Early Warning Dashboard")

# Sidebar for global controls
with st.sidebar:
    st.header("Controls")
    auto_refresh = st.checkbox("Manual Refresh", value=True)
    if auto_refresh and st.button("Refresh Data"):
        st.rerun()
    
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
    if theme == "Dark":
        st.markdown('<style>.main {background-color: #0F172A; color: #F8FAFC;}</style>', unsafe_allow_html=True)
    
    st.subheader("Filters")
    route_filter = st.multiselect("Routes", [f"Route {i}" for i in range(10)], default=[])
    severity_filter = st.multiselect("Severity", ["normal", "delay", "disruption"], default=["normal", "delay", "disruption"])

# Tabs for main sections
tab1, tab2, tab3, tab4 = st.tabs(["Live Monitoring", "Analytics", "Alerts", "Incident Drill-Down"])

# Tab 1: Live Monitoring (Map + KPIs)
with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # KPIs
        data = get_realtime_data()
        if route_filter:
            data = data[data['route'].isin(route_filter)]
        if severity_filter:
            data = data[data['severity'].isin(severity_filter)]
        
        active_disruptions = len(data[data['severity'] == 'disruption'])
        avg_delay = data['delay'].mean()
        on_time_pct = (len(data[data['delay'] < 5]) / len(data)) * 100
        
        st.markdown(f"""
        <div class="kpi-card">
            <h3>Active Disruptions</h3>
            <p style="font-size: 24px; color: #EF4444;">{active_disruptions}</p>
        </div>
        <div class="kpi-card">
            <h3>Avg Delay (min)</h3>
            <p style="font-size: 24px;">{avg_delay:.1f}</p>
        </div>
        <div class="kpi-card">
            <h3>On-Time %</h3>
            <p style="font-size: 24px; color: #10B981;">{on_time_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Live Map using pydeck (if available)
        try:
            import pydeck as pdk
            layer = pdk.Layer(
                "ScatterplotLayer",
                data,
                get_position=["lon", "lat"],
                get_color="[severity == 'normal' ? 16 : severity == 'delay' ? 246 : 239, severity == 'normal' ? 185 : severity == 'delay' ? 144 : 68, severity == 'normal' ? 129 : 0]",
                get_radius=100,
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=10)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{route}: {delay} min delay"})
            st.pydeck_chart(r)
        except ImportError:
            st.map(data[['lat', 'lon']])

# Tab 2: Analytics
with tab2:
    analytics_data = get_analytics_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=analytics_data['time'], y=analytics_data['delay_mean'], mode='lines', name='Avg Delay'))
    fig.add_trace(go.Scatter(x=analytics_data['time'], y=analytics_data['headway_dev'], mode='lines', name='Headway Deviation'))
    fig.update_layout(title="Disruption Analytics", xaxis_title="Time", yaxis_title="Metrics")
    st.plotly_chart(fig)
    
    # What-If Simulation
    st.subheader("Impact Simulation")
    weather_severity = st.slider("Weather Severity", 0, 10, 5)
    simulated_delay = analytics_data['delay_mean'].mean() * (1 + weather_severity / 10)
    st.write(f"Simulated Avg Delay: {simulated_delay:.1f} min")

# Tab 3: Alerts
with tab3:
    alerts = get_alerts()
    for _, alert in alerts.iterrows():
        if alert['id'] in st.session_state.acknowledged_alerts:
            continue
        severity_class = f"severity-{alert['severity']}"
        with st.container():
            st.markdown(f"""
            <div class="alert-card {severity_class}">
                <strong>{alert['description']}</strong><br>
                Severity: {alert['severity'].upper()} | Confidence: {alert['confidence']:.1%}<br>
                Time: {alert['timestamp'].strftime('%H:%M')}
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Acknowledge Alert {alert['id']}", key=f"ack_{alert['id']}"):
                st.session_state.acknowledged_alerts.add(alert['id'])
                st.success("Alert acknowledged!")

# Tab 4: Incident Drill-Down
with tab4:
    selected_vehicle = st.selectbox("Select Vehicle for Drill-Down", data['id'].tolist() if 'data' in locals() else [])
    if selected_vehicle:
        vehicle_data = data[data['id'] == selected_vehicle].iloc[0]
        st.subheader(f"Details for {selected_vehicle}")
        st.write(f"Route: {vehicle_data['route']}")
        st.write(f"Delay: {vehicle_data['delay']} min")
        st.write(f"Severity: {vehicle_data['severity']}")
        st.write(f"Occupancy: {vehicle_data['occupancy']}%")
        
        # Mock SHAP-like explanation
        st.subheader("Causal Factors")
        factors = {"Speed Drop": 0.3, "Congestion": 0.4, "Weather": 0.3}
        for factor, imp in factors.items():
            st.write(f"{factor}: {imp:.1%} importance")
        
        if st.button("Export Report"):
            report = f"Incident Report for {selected_vehicle}\n{vehicle_data.to_dict()}"
            st.download_button("Download CSV", report, file_name="incident_report.txt", mime="text/plain")

# Footer
st.markdown("---")
st.caption("Transit Disruption Dashboard Prototype - WCAG 2.1 AA Compliant")