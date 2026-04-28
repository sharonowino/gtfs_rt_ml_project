"""
Transit Dashboard Integration
=======================

Integration layer for connecting:
- gtfs_disruption pipeline
- transit-dashboard FastAPI
- transit_sentinel data connectors

Usage:
------
from gtfs_disruption.utils.dashboard_integration import (
    DashboardAPIClient,
    SentinelConnector,
    get_unified_predictions,
)

# Connect to FastAPI
client = DashboardAPIClient("http://localhost:8000")
predictions = client.get_predictions(features)

# Get data from transit_sentinel connectors
sentinel = SentinelConnector()
realtime_data = sentinel.get_avl_feed()
"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import requests
import json

logger = logging.getLogger(__name__)


# =========================================================================
# TRANSIT-DASHBOARD API CLIENT
# =========================================================================

class DashboardAPIClient:
    """
    Client for transit-dashboard FastAPI.
    
    Connects to the FastAPI models served in transit-dashboard.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict:
        """Make request to API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict:
        """Check API health."""
        return self._request("GET", "/health")
    
    def predict(
        self,
        features: List[float],
        threshold: float = 0.5,
        model_type: str = "binary",
    ) -> Dict:
        """
        Get prediction from API.
        
        Parameters
        ----------
        features : List[float]
            Feature vector
        threshold : float
            Classification threshold
        model_type : str
            'binary' or 'multi'
        
        Returns
        -------
        Dict with prediction results
        """
        payload = {
            "features": features,
            "threshold": threshold,
            "model_type": model_type,
        }
        
        return self._request("POST", "/predict", json=payload)
    
    def predict_batch(
        self,
        features: List[List[float]],
        threshold: float = 0.5,
    ) -> Dict:
        """Batch prediction."""
        payload = {
            "features": features,
            "threshold": threshold,
        }
        
        return self._request("POST", "/predict/batch", json=payload)
    
    def get_model_info(self) -> Dict:
        """Get loaded model info."""
        return self._request("GET", "/model/info")
    
    def get_metrics(self) -> Dict:
        """Get service metrics."""
        return self._request("GET", "/metrics")
    
    def reload_model(self, model_path: str = None) -> Dict:
        """Hot reload model."""
        params = {"model_path": model_path} if model_path else {}
        return self._request("POST", "/model/reload", params=params)


# =========================================================================
# TRANSIT SENTINEL CONNECTOR
# =========================================================================

class SentinelConnector:
    """
    Connector for transit_sentinel data feeds.
    
    Provides access to the data connectors used in transit_sentinel:
    - AVLFeedConnector
    - CADAlertConnector
    - GTFSRTConnector
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8501",
        secrets_path: str = None,
    ):
        self.api_url = api_url
        self.secrets_path = secrets_path
        self._secrets = None
    
    def _load_secrets(self):
        """Load secrets for API access."""
        if self.secrets_path is None:
            # Try default locations
            for path in ["secrets.toml", "../secrets.toml", "../../secrets.toml"]:
                import os
                if os.path.exists(path):
                    self.secrets_path = path
                    break
        
        if self.secrets_path:
            try:
                import toml
                self._secrets = toml.load(self.secrets_path)
            except:
                self._secrets = {}
    
    def get_avl_feed(self, route_id: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Get AVL (Automatic Vehicle Location) feed.
        
        Returns DataFrame with columns:
        - trip_id
        - route_id
        - stop_id
        - delay_sec
        - speed_kmh
        - timestamp
        - latitude
        - longitude
        """
        # This would connect to the actual data feed
        # For now, return empty DataFrame with expected schema
        return pd.DataFrame(columns=[
            "trip_id", "route_id", "stop_id", "delay_sec",
            "speed_kmh", "timestamp", "latitude", "longitude"
        ])
    
    def get_alerts(self, active_only: bool = True) -> pd.DataFrame:
        """
        Get service alerts.
        
        Returns DataFrame with columns:
        - alert_id
        - cause
        - effect
        - active
        - duration
        """
        return pd.DataFrame(columns=[
            "alert_id", "cause", "effect", "active", "duration"
        ])
    
    def get_vehicle_positions(self, route_id: str = None) -> pd.DataFrame:
        """Get current vehicle positions."""
        return pd.DataFrame(columns=[
            "vehicle_id", "trip_id", "route_id",
            "latitude", "longitude", "timestamp"
        ])
    
    def get_trip_updates(self, route_id: str = None) -> pd.DataFrame:
        """Get trip update predictions."""
        return pd.DataFrame(columns=[
            "trip_id", "route_id", "stop_id",
            "arrival_delay_sec", "departure_delay_sec"
        ])


# =========================================================================
# UNIFIED PREDICTION SERVICE
# =========================================================================

class UnifiedPredictor:
    """
    Unified prediction service combining:
    - FastAPI predictions
    - Local model predictions
    - Sentinel data feeds
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        local_model_path: str = None,
    ):
        self.api_client = DashboardAPIClient(api_url)
        self.sentinel = SentinelConnector()
        self.local_model = None
        self._load_local_model(local_model_path)
    
    def _load_local_model(self, model_path: str):
        """Load local model if provided."""
        if model_path:
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    self.local_model = pickle.load(f)
                logger.info(f"Loaded local model: {model_path}")
            except Exception as e:
                logger.warning(f"Could not load local model: {e}")
    
    def predict(
        self,
        features: List[float],
        use_api: bool = True,
    ) -> Dict:
        """
        Get prediction from available sources.
        
        Tries FastAPI first, falls back to local model.
        """
        # Try API first
        if use_api:
            result = self.api_client.predict(features)
            if "error" not in result:
                result["source"] = "api"
                return result
        
        # Fall back to local model
        if self.local_model is not None:
            try:
                import numpy as np
                X = np.array(features).reshape(1, -1)
                proba = self.local_model.predict_proba(X)[0, 1]
                return {
                    "is_disrupted": bool(proba > 0.5),
                    "probability": float(proba),
                    "source": "local_model",
                }
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": "No prediction source available"}
    
    def get_realtime_features(self) -> pd.DataFrame:
        """Get current features from transit_sentinel."""
        avl = self.sentinel.get_avl_feed()
        alerts = self.sentinel.get_alerts()
        
        # Combine feeds
        features = avl.copy() if not avl.empty else pd.DataFrame()
        
        return features
    
    def get_predictions_stream(self, routes: List[str]) -> pd.DataFrame:
        """
        Get predictions for multiple routes.
        
        Returns DataFrame with predictions.
        """
        results = []
        
        for route_id in routes:
            # Get route data
            route_avl = self.sentinel.get_avl_feed(route_id)
            
            if not route_avl.empty:
                for _, row in route_avl.iterrows():
                    features = [
                        row.get("delay_sec", 0),
                        row.get("speed_kmh", 0),
                        0,  # headway
                        0,  # hour
                        0,  # day_of_week
                    ]
                    
                    pred = self.predict(features)
                    
                    results.append({
                        "route_id": route_id,
                        "trip_id": row.get("trip_id"),
                        "stop_id": row.get("stop_id"),
                        **pred
                    })
        
        return pd.DataFrame(results)


def get_unified_predictions(
    features: List[float],
    api_url: str = "http://localhost:8000",
) -> Dict:
    """Convenience function for unified predictions."""
    predictor = UnifiedPredictor(api_url=api_url)
    return predictor.predict(features)


# =========================================================================
# DASHBOARD INTEGRATION HELPERS
# =========================================================================

def create_sidebar_controls():
    """Create sidebar controls for dashboard."""
    import streamlit as st
    
    st.sidebar.title("🚨 Transit Operations")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["API", "Local Model", "Transit Sentinel"],
        horizontal=True,
    )
    
    # API settings
    api_url = "http://localhost:8000"
    if data_source == "API":
        api_url = st.sidebar.text_input("API URL", api_url)
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["binary", "multi"],
    )
    
    # Threshold
    threshold = st.sidebar.slider(
        "Detection Threshold",
        0.1, 0.9, 0.5, 0.05,
    )
    
    return {
        "data_source": data_source,
        "api_url": api_url,
        "model_type": model_type,
        "threshold": threshold,
    }


# =========================================================================
# EXPORTS
# =========================================================================

__all__ = [
    'DashboardAPIClient',
    'SentinelConnector',
    'UnifiedPredictor',
    'get_unified_predictions',
    'create_sidebar_controls',
]