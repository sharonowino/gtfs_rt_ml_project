"""
GTFS Geocoding and Mapping Module
================================
Geocode addresses and create interactive maps for disruption visualization.

Features:
- NER location geocoding using Nominatim
- Folium map with risk-level markers
- Netherlands bounding box filtering
- Interactive popup with disruption details

Usage:
------
from gtfs_disruption.utils.mapping import Geocoder, DisruptionMap

# Geocode locations
geocoder = Geocoder()
df = geocoder.geocode_locations(df, 'location_text')

# Create map
mapper = DisruptionMap()
mapper.create_map(df, save_path='map.html')
"""
import logging
import os
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Netherlands bounding box
NETHERLANDS_BOUNDS = {
    'lat_min': 50.75,
    'lat_max': 53.55,
    'lon_min': 3.30,
    'lon_max': 7.20
}

RISK_COLORS = {
    'critical': '#ff4757',
    'high': '#ff6b35',
    'moderate': '#ffa502',
    'low': '#2ed573',
    'unknown': '#808080'
}

try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    logger.warning("geopy not installed - geocoding disabled")
    GEOPY_AVAILABLE = False

try:
    import folium
    from folium.plugins import MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    logger.warning("folium not installed - mapping disabled")
    FOLIUM_AVAILABLE = False


class Geocoder:
    """
    Geocode NER-extracted locations to lat/lon coordinates.
    
    Parameters
    ----------
    user_agent : str
        User agent for Nominatim (required)
    """
    
    def __init__(self, user_agent: str = 'gtfs_disruption'):
        self.user_agent = user_agent
        self.geolocator = None
        self._cache = {}
        
        if GEOPY_AVAILABLE:
            self.geolocator = Nominatim(user_agent=user_agent)
    
    def geocode(
        self,
        location: str,
        country: str = 'Netherlands'
    ) -> Optional[Tuple[float, float]]:
        """Geocode a single location."""
        if not location or not GEOPY_AVAILABLE:
            return None
        
        # Check cache
        if location in self._cache:
            return self._cache[location]
        
        # Geocode
        try:
            query = f"{location}, {country}"
            result = self.geolocator.geocode(query)
            
            if result:
                coords = (result.latitude, result.longitude)
                self._cache[location] = coords
                return coords
        except Exception as e:
            logger.debug(f"Geocode failed for {location}: {e}")
        
        return None
    
    def geocode_column(
        self,
        df: pd.DataFrame,
        location_column: str = 'location_text'
    ) -> pd.DataFrame:
        """Geocode all locations in a column."""
        if location_column not in df.columns:
            return df
        
        logger.info(f"Geocoding {len(df)} locations...")
        
        coords = []
        for loc in df[location_column].fillna(''):
            if pd.notna(loc) and str(loc).strip():
                result = self.geocode(str(loc).strip())
                coords.append(result)
            else:
                coords.append(None)
        
        df['geocoded_lat'] = [c[0] if c else None for c in coords]
        df['geocoded_lon'] = [c[1] if c else None for c in coords]
        
        geocoded_count = pd.notna(df['geocoded_lat']).sum()
        logger.info(f"  Geocoded {geocoded_count}/{len(df)} locations")
        
        return df
    
    def _nl_bounds_check(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Netherlands."""
        if pd.isna(lat) or pd.isna(lon):
            return False
        
        return (
            NETHERLANDS_BOUNDS['lat_min'] <= lat <= NETHERLANDS_BOUNDS['lat_max'] and
            NETHERLANDS_BOUNDS['lon_min'] <= lon <= NETHERLANDS_BOUNDS['lon_max']
        )
    
    def filter_valid_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to valid Netherlands coordinates."""
        lat_col = 'geocoded_lat' if 'geocoded_lat' in df.columns else 'first_lat'
        lon_col = 'geocoded_lon' if 'geocoded_lon' in df.columns else 'first_lon'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            return df
        
        valid_mask = df.apply(
            lambda row: self._nl_bounds_check(row[lat_col], row[lon_col]),
            axis=1
        )
        
        n_removed = len(df) - valid_mask.sum()
        logger.info(f"  Filtered {n_removed} out-of-bounds coordinates")
        
        return df[valid_mask].copy()


class DisruptionMap:
    """
    Create interactive Folium map of disruptions.
    
    Parameters
    ----------
    risk_colors : dict
        Color mapping for risk levels
    """
    
    def __init__(
        self,
        risk_colors: Dict[str, str] = None,
        bounds: Dict[str, float] = None
    ):
        self.risk_colors = risk_colors or RISK_COLORS
        self.bounds = bounds or NETHERLANDS_BOUNDS
    
    def create_map(
        self,
        df: pd.DataFrame,
        lat_column: str = 'first_lat',
        lon_column: str = 'first_lon',
        risk_column: str = 'risk_level',
        popup_columns: List[str] = None,
        save_path: Optional[str] = None
    ):
        """Create interactive disruption map."""
        if not FOLIUM_AVAILABLE:
            raise ImportError("folium required: pip install folium")
        
        if df.empty or lat_column not in df.columns or lon_column not in df.columns:
            logger.warning("No valid coordinates for mapping")
            return None
        
        # Filter valid bounds
        df_clean = df[
            (df[lat_column] >= self.bounds['lat_min']) &
            (df[lat_column] <= self.bounds['lat_max']) &
            (df[lon_column] >= self.bounds['lon_min']) &
            (df[lon_column] <= self.bounds['lon_max'])
        ].copy()
        
        if df_clean.empty:
            logger.warning("No coordinates within Netherlands bounds")
            return None
        
        # Create map center
        center_lat = df_clean[lat_column].mean()
        center_lon = df_clean[lon_column].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles='CartoDB positron'
        )
        
        # Add risk clusters
        risk_levels = df_clean[risk_column].fillna('unknown').unique().tolist()
        clusters = {}
        
        for level in risk_levels:
            if level not in clusters:
                clusters[level] = MarkerCluster(
                    name=f"Risk: {level}"
                ).add_to(m)
        
        # Popup columns default
        if popup_columns is None:
            popup_columns = [
                'risk_level',
                'disruption_class',
                'route_id',
                'cause',
                'effect'
            ]
            popup_columns = [c for c in popup_columns if c in df_clean.columns]
        
        # Add markers
        for _, row in df_clean.iterrows():
            level = str(row.get(risk_column, 'unknown')).lower()
            color = self.risk_colors.get(level, '#808080')
            
            # Build popup
            popup_lines = [f"<b>Risk:</b> {level}"]
            for col in popup_columns:
                if col in row.index and pd.notna(row[col]):
                    popup_lines.append(f"<b>{col}:</b> {row[col]}")
            
            popup = folium.Popup("<br>".join(popup_lines), max_width=300)
            
            folium.CircleMarker(
                location=[row[lat_column], row[lon_column]],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=popup,
                tooltip=level
            ).add_to(clusters.get(level, m))
        
        # Layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Legend
        legend = self._create_legend()
        m.get_root().html.add_child(folium.Element(legend))
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            m.save(save_path)
            logger.info(f"Map saved to {save_path}")
        
        return m
    
    def _create_legend(self) -> str:
        """Create HTML legend."""
        legend = """<div style="position:fixed; bottom:50px; left:50px;
            z-index:9999; background:white; border:2px solid #ccc;
            border-radius:6px; padding:12px; font-size:12px;">
        <b>Risk Level</b><br>"""
        
        for level, color in self.risk_colors.items():
            legend += f"""<i style="background:{color};width:12px;
                height:12px;display:inline-block;margin-right:6px;
                border-radius:50%;"></i>{level}<br>"""
        
        legend += "</div>"
        return legend
    
    def add_to_dashboard(self, st, df: pd.DataFrame):
        """Add map to Streamlit dashboard."""
        if not FOLIUM_AVAILABLE:
            st.warning("Folium not installed")
            return
        
        m = self.create_map(df)
        
        if m:
            # Convert to HTML
            from streamlit.components.v1 import components
            folium_static = components.html(m._repr_html_(), height=500)
            st.plotly_chart(folium_static, use_container_width=True)
        else:
            st.info("No valid geocoded data for mapping")


def create_disruption_map(
    df: pd.DataFrame,
    save_path: str = 'visualizations/disruption_map.html'
):
    """
    Convenience function to create a disruption map.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lat, lon, risk_level columns
    save_path : str
        Path to save HTML map
        
    Returns
    -------
    folium.Map
    """
    mapper = DisruptionMap()
    return mapper.create_map(df, save_path=save_path)