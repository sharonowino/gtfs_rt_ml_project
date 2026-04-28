"""
GTFS Disruption Detection - Spatial Map Visualizations
======================================================
Generates geographic maps showing stop-level disruption patterns,
spatial lag, and temporal evolution across the Netherlands.

Maps produced:
1. Disruption density heatmap (static, geopandas + contextily)
2. Severity choropleth by stop (static)
3. Spatial lag delay map (static)
4. Interactive folium map with popups
5. Temporal hourly evolution (static, faceted)
6. Hot spots map with hub overlay (static)
"""
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Netherlands bounding box
NL_BOUNDS = {
    "lat_min": 50.75, "lat_max": 53.55,
    "lon_min": 3.30, "lon_max": 7.20,
}

# Major hubs for overlay
MAJOR_HUBS = {
    "Amsterdam CS":  (52.3791, 4.9003),
    "Utrecht CS":    (52.0890, 5.1093),
    "Rotterdam CS":  (51.9249, 4.4689),
    "Den Haag CS":   (52.0807, 4.3247),
    "Leiden CS":     (52.1663, 4.4814),
    "Schiphol":      (52.3094, 4.7625),
    "Eindhoven CS":  (51.4431, 5.4813),
    "Arnhem CS":     (51.9846, 5.9010),
}


def _validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows with valid NL coordinates."""
    for col in ["stop_lat", "stop_lon"]:
        if col not in df.columns:
            logger.warning(f"  Missing column: {col}")
            return pd.DataFrame()
    sub = df.dropna(subset=["stop_lat", "stop_lon"]).copy()
    sub["stop_lat"] = pd.to_numeric(sub["stop_lat"], errors="coerce")
    sub["stop_lon"] = pd.to_numeric(sub["stop_lon"], errors="coerce")
    sub = sub.dropna(subset=["stop_lat", "stop_lon"])
    sub = sub[
        (sub["stop_lat"] >= NL_BOUNDS["lat_min"]) &
        (sub["stop_lat"] <= NL_BOUNDS["lat_max"]) &
        (sub["stop_lon"] >= NL_BOUNDS["lon_min"]) &
        (sub["stop_lon"] <= NL_BOUNDS["lon_max"])
    ]
    return sub


def _make_gdf(df: pd.DataFrame) -> "gpd.GeoDataFrame":
    """Convert DataFrame with stop_lat/stop_lon to GeoDataFrame (EPSG:4326)."""
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["stop_lon"], df["stop_lat"]),
        crs="EPSG:4326",
    )
    return gdf


def _get_basemap_ax(ax, crs="EPSG:3857"):
    """Add a contextily basemap to an existing axes. Returns True on success."""
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.CartoDB.Positron,
                        zoom=8, alpha=0.8)
        return True
    except Exception as e:
        logger.warning(f"  Basemap unavailable ({e}). Using plain axes.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Map 1: Disruption Density Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_disruption_density_map(
    classified_df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 14),
) -> None:
    """
    Static map: stops sized/colored by disruption density.

    Each unique stop is aggregated (mean disruption rate), then plotted
    on a CartoDB basemap. Stops with higher disruption rates appear larger
    and redder.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd

    logger.info("  Generating disruption density map...")
    out = Path(output_dir)

    sub = _validate_coordinates(classified_df)
    if sub.empty:
        logger.warning("  No valid coordinates for density map.")
        return

    # Aggregate per stop
    if "disruption_target" in sub.columns:
        agg = sub.groupby(["stop_id", "stop_lat", "stop_lon"]).agg(
            disruption_rate=("disruption_target", "mean"),
            n_events=("disruption_target", "count"),
        ).reset_index()
    elif "disruption_type" in sub.columns:
        sub["_is_disrupted"] = (sub["disruption_type"] != "ON_TIME").astype(int)
        agg = sub.groupby(["stop_id", "stop_lat", "stop_lon"]).agg(
            disruption_rate=("_is_disrupted", "mean"),
            n_events=("_is_disrupted", "count"),
        ).reset_index()
    else:
        logger.warning("  No disruption label column found.")
        return

    # Filter to stops with enough observations for reliability
    agg = agg[agg["n_events"] >= 3].copy()
    if agg.empty:
        logger.warning("  No stops with >= 3 observations for density map.")
        return

    gdf = _make_gdf(agg)
    gdf_web = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Size proportional to log(event count), color by disruption rate
    sizes = np.log1p(gdf_web["n_events"]) * 8
    sc = ax.scatter(
        gdf_web.geometry.x, gdf_web.geometry.y,
        c=gdf_web["disruption_rate"],
        s=sizes,
        cmap="YlOrRd",
        vmin=0, vmax=max(0.5, gdf_web["disruption_rate"].quantile(0.95)),
        alpha=0.65,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="Disruption Rate", shrink=0.6, pad=0.02)

    # Hub overlay
    for name, (lat, lon) in MAJOR_HUBS.items():
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_hub, y_hub = transformer.transform(lon, lat)
        ax.scatter(x_hub, y_hub, marker="*", s=180, c="black", zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(name, (x_hub, y_hub), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    _get_basemap_ax(ax)

    ax.set_title(
        "Disruption Density Map — Netherlands GTFS-RT\n"
        "(stop size ∝ log(event count), color = disruption rate)",
        fontsize=11, fontweight="bold",
    )
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out / "map01_disruption_density.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "map01_disruption_density.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ✓ map01_disruption_density")


# ─────────────────────────────────────────────────────────────────────────────
# Map 2: Severity Choropleth by Stop
# ─────────────────────────────────────────────────────────────────────────────

def plot_severity_map(
    classified_df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 14),
) -> None:
    """
    Static map: stops colored by mean severity score (0–10).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd

    logger.info("  Generating severity map...")
    out = Path(output_dir)

    sub = _validate_coordinates(classified_df)
    if sub.empty or "severity_score" not in sub.columns:
        logger.warning("  Missing severity_score or coordinates.")
        return

    agg = sub.groupby(["stop_id", "stop_lat", "stop_lon"]).agg(
        mean_severity=("severity_score", "mean"),
        max_severity=("severity_score", "max"),
        n_events=("severity_score", "count"),
    ).reset_index()
    agg = agg[agg["n_events"] >= 3]
    if agg.empty:
        return

    gdf = _make_gdf(agg).to_crs(epsg=3857)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))

    for ax, col, title, cmap in [
        (axes[0], "mean_severity", "Mean Severity", "RdYlGn_r"),
        (axes[1], "max_severity", "Max Severity", "RdYlGn_r"),
    ]:
        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=gdf[col],
            s=12,
            cmap=cmap,
            vmin=0, vmax=10,
            alpha=0.6,
            edgecolors="none",
            zorder=3,
        )
        plt.colorbar(sc, ax=ax, label="Severity (0–10)", shrink=0.6)
        _get_basemap_ax(ax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle(
        "Stop Severity Map — Netherlands GTFS-RT\n"
        "(severity: 0=ON_TIME … 10=CANCELLED)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out / "map02_severity.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "map02_severity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ✓ map02_severity")


# ─────────────────────────────────────────────────────────────────────────────
# Map 3: Spatial Lag Delay Map
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_lag_map(
    classified_df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 14),
) -> None:
    """
    Static map: stops colored by spatial_lag_delay (average delay at
    neighboring stops within ~1 km). Captures spatial propagation of delays.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd

    logger.info("  Generating spatial lag map...")
    out = Path(output_dir)

    sub = _validate_coordinates(classified_df)
    if sub.empty:
        return

    # Check for spatial_lag_delay; if missing, compute it
    if "spatial_lag_delay" not in sub.columns:
        if "delay_sec" in sub.columns:
            logger.info("    Computing spatial_lag_delay on the fly...")
            sub["spatial_lag_delay"] = np.nan
            coords = sub[["stop_lat", "stop_lon"]].values
            delays = sub["delay_sec"].values
            for i in range(len(sub)):
                dist = np.sqrt(
                    (coords[:, 0] - coords[i, 0]) ** 2 +
                    (coords[:, 1] - coords[i, 1]) ** 2
                )
                mask = (dist < 0.01) & (np.arange(len(sub)) != i)
                if mask.any():
                    sub.iloc[i, sub.columns.get_loc("spatial_lag_delay")] = np.nanmean(delays[mask])
        else:
            logger.warning("  No spatial_lag_delay or delay_sec column.")
            return

    agg = sub.groupby(["stop_id", "stop_lat", "stop_lon"]).agg(
        spatial_lag=("spatial_lag_delay", "mean"),
        own_delay=("delay_sec", "mean") if "delay_sec" in sub.columns else ("spatial_lag_delay", "mean"),
        n_events=("spatial_lag_delay", "count"),
    ).reset_index()
    agg = agg.dropna(subset=["spatial_lag"])
    if agg.empty:
        return

    gdf = _make_gdf(agg).to_crs(epsg=3857)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))

    # Left: spatial lag (neighbor avg delay)
    vmax_lag = np.nanpercentile(np.abs(gdf["spatial_lag"]), 95)
    sc1 = axes[0].scatter(
        gdf.geometry.x, gdf.geometry.y,
        c=gdf["spatial_lag"],
        s=14, cmap="RdBu_r", vmin=-vmax_lag, vmax=vmax_lag,
        alpha=0.6, edgecolors="none", zorder=3,
    )
    plt.colorbar(sc1, ax=axes[0], label="Spatial Lag Delay (sec)", shrink=0.6)
    _get_basemap_ax(axes[0])
    axes[0].set_title("Spatial Lag Delay\n(neighbor avg delay ± 1 km)", fontsize=10, fontweight="bold")
    axes[0].set_axis_off()

    # Right: own delay for comparison
    if "own_delay" in gdf.columns:
        vmax_own = np.nanpercentile(np.abs(gdf["own_delay"]), 95)
        sc2 = axes[1].scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=gdf["own_delay"],
            s=14, cmap="RdBu_r", vmin=-vmax_own, vmax=vmax_own,
            alpha=0.6, edgecolors="none", zorder=3,
        )
        plt.colorbar(sc2, ax=axes[1], label="Own Delay (sec)", shrink=0.6)
        _get_basemap_ax(axes[1])
        axes[1].set_title("Own Delay\n(per-stop mean)", fontsize=10, fontweight="bold")
        axes[1].set_axis_off()

    fig.suptitle(
        "Spatial Lag vs Own Delay — Netherlands GTFS-RT\n"
        "(spatial lag = avg delay at nearby stops; captures delay propagation)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out / "map03_spatial_lag.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "map03_spatial_lag.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ✓ map03_spatial_lag")


# ─────────────────────────────────────────────────────────────────────────────
# Map 4: Interactive Folium Map
# ─────────────────────────────────────────────────────────────────────────────

def plot_interactive_map(
    classified_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Interactive HTML map (folium): each stop is a circle marker colored by
    disruption rate, with popups showing stop name, severity, delay, and
    event count. Includes a layer control and a heat map overlay.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        logger.warning("  folium not installed. Skipping interactive map.")
        return

    logger.info("  Generating interactive folium map...")
    out = Path(output_dir)

    sub = _validate_coordinates(classified_df)
    if sub.empty:
        return

    # Aggregate per stop
    agg_cols = {"stop_id": "first", "stop_lat": "mean", "stop_lon": "mean"}
    if "stop_name" in sub.columns:
        agg_cols["stop_name"] = "first"
    if "disruption_target" in sub.columns:
        agg_cols["disruption_rate"] = ("disruption_target", "mean")
    elif "disruption_type" in sub.columns:
        sub["_is_disrupted"] = (sub["disruption_type"] != "ON_TIME").astype(int)
        agg_cols["disruption_rate"] = ("_is_disrupted", "mean")
    if "severity_score" in sub.columns:
        agg_cols["mean_severity"] = ("severity_score", "mean")
    if "delay_sec" in sub.columns:
        agg_cols["mean_delay_sec"] = ("delay_sec", "mean")
    if "spatial_lag_delay" in sub.columns:
        agg_cols["spatial_lag"] = ("spatial_lag_delay", "mean")

    agg_dict = {}
    for k, v in agg_cols.items():
        if isinstance(v, tuple):
            agg_dict[k] = v
        else:
            agg_dict[k] = (v, "first") if k != "stop_id" else ("stop_id", "first")

    grouped = sub.groupby("stop_id").agg(
        stop_lat=("stop_lat", "mean"),
        stop_lon=("stop_lon", "mean"),
        **{k: v for k, v in agg_dict.items() if k != "stop_id"},
    ).reset_index()

    if "stop_name" in sub.columns:
        names = sub.groupby("stop_id")["stop_name"].first().reset_index()
        grouped = grouped.merge(names, on="stop_id", how="left")

    # Limit to top 2000 stops by event count to keep map responsive
    if len(grouped) > 2000:
        event_counts = sub.groupby("stop_id").size().reset_index(name="n")
        grouped = grouped.merge(event_counts, on="stop_id")
        grouped = grouped.nlargest(2000, "n")

    # Center map on Netherlands
    center_lat = grouped["stop_lat"].mean()
    center_lon = grouped["stop_lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8,
                   tiles="CartoDB positron")

    # Circle markers colored by disruption rate
    def _color(rate):
        if pd.isna(rate):
            return "gray"
        if rate < 0.05:
            return "#1a9850"   # green
        elif rate < 0.15:
            return "#fee08b"   # yellow
        elif rate < 0.30:
            return "#fdae61"   # orange
        else:
            return "#d73027"   # red

    for _, row in grouped.iterrows():
        rate = row.get("disruption_rate", 0)
        name = row.get("stop_name", row["stop_id"])
        popup_html = f"<b>{name}</b><br>Stop: {row['stop_id']}<br>"
        popup_html += f"Disruption rate: {rate:.1%}<br>" if not pd.isna(rate) else ""
        if "mean_severity" in row and not pd.isna(row.get("mean_severity")):
            popup_html += f"Mean severity: {row['mean_severity']:.1f}/10<br>"
        if "mean_delay_sec" in row and not pd.isna(row.get("mean_delay_sec")):
            popup_html += f"Mean delay: {row['mean_delay_sec']:.0f}s<br>"
        if "spatial_lag" in row and not pd.isna(row.get("spatial_lag")):
            popup_html += f"Spatial lag delay: {row['spatial_lag']:.0f}s<br>"

        folium.CircleMarker(
            location=[row["stop_lat"], row["stop_lon"]],
            radius=max(3, min(12, rate * 40)) if not pd.isna(rate) else 3,
            color=_color(rate),
            fill=True,
            fill_color=_color(rate),
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{name}: {rate:.1%}" if not pd.isna(rate) else name,
        ).add_to(m)

    # Heat map layer
    heat_data = grouped.dropna(subset=["stop_lat", "stop_lon", "disruption_rate"])
    if not heat_data.empty:
        heat_points = heat_data[["stop_lat", "stop_lon", "disruption_rate"]].values.tolist()
        HeatMap(heat_points, radius=15, blur=10, max_zoom=12,
                name="Disruption Heatmap").add_to(m)

    # Major hubs
    hub_group = folium.FeatureGroup(name="Major Hubs")
    for name, (lat, lon) in MAJOR_HUBS.items():
        folium.Marker(
            location=[lat, lon],
            popup=name,
            icon=folium.Icon(color="black", icon="train", prefix="fa"),
        ).add_to(hub_group)
    hub_group.add_to(m)

    folium.LayerControl().add_to(m)

    m.save(str(out / "map04_interactive.html"))
    logger.info("  ✓ map04_interactive.html")


# ─────────────────────────────────────────────────────────────────────────────
# Map 5: Temporal Hourly Evolution
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_evolution_map(
    classified_df: pd.DataFrame,
    output_dir: str,
    n_hours: int = 6,
    figsize: Tuple[int, int] = (18, 12),
) -> None:
    """
    Faceted static map showing disruption rate by stop across key hours
    of the day. Captures the temporal component of spatial disruption patterns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd

    logger.info("  Generating temporal evolution map...")
    out = Path(output_dir)

    sub = _validate_coordinates(classified_df)
    if sub.empty:
        return

    # Determine hour column
    hour_col = None
    if "feed_timestamp" in sub.columns:
        sub["_hour"] = pd.to_datetime(sub["feed_timestamp"]).dt.hour
        hour_col = "_hour"
    elif "timestamp" in sub.columns:
        sub["_hour"] = pd.to_datetime(sub["timestamp"], unit="s", utc=True).dt.hour
        hour_col = "_hour"
    else:
        logger.warning("  No timestamp column for temporal map.")
        return

    # Disruption label
    if "disruption_target" in sub.columns:
        disp_col = "disruption_target"
    elif "disruption_type" in sub.columns:
        sub["_is_disrupted"] = (sub["disruption_type"] != "ON_TIME").astype(int)
        disp_col = "_is_disrupted"
    else:
        logger.warning("  No disruption label for temporal map.")
        return

    # Select representative hours spanning the day
    all_hours = sorted(sub[hour_col].unique())
    if len(all_hours) < 2:
        logger.warning("  Only one hour value; skipping temporal map.")
        return
    # Pick n_hours evenly spaced hours that exist in data
    target_hours = np.linspace(0, 23, n_hours + 2).astype(int)[1:-1]
    selected_hours = []
    for h in target_hours:
        closest = min(all_hours, key=lambda x: abs(x - h))
        if closest not in selected_hours:
            selected_hours.append(closest)
    if len(selected_hours) < 2:
        selected_hours = all_hours[:min(n_hours, len(all_hours))]

    n_panels = len(selected_hours)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows / 2))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Global color scale
    vmax_global = 0.5

    for idx, hour in enumerate(selected_hours):
        ax = axes[idx]
        hour_data = sub[sub[hour_col] == hour]
        if hour_data.empty:
            ax.set_axis_off()
            continue

        agg = hour_data.groupby(["stop_id", "stop_lat", "stop_lon"]).agg(
            rate=(disp_col, "mean"),
            n=(disp_col, "count"),
        ).reset_index()
        agg = agg[agg["n"] >= 2]
        if agg.empty:
            ax.set_axis_off()
            continue

        gdf = _make_gdf(agg).to_crs(epsg=3857)
        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=gdf["rate"],
            s=8 + np.log1p(gdf["n"]) * 3,
            cmap="YlOrRd",
            vmin=0, vmax=vmax_global,
            alpha=0.6,
            edgecolors="none",
            zorder=3,
        )
        _get_basemap_ax(ax)
        label = f"{hour:02d}:00"
        n_stops = len(agg)
        mean_rate = agg["rate"].mean()
        ax.set_title(f"{label}\n{n_stops} stops, mean rate {mean_rate:.1%}",
                      fontsize=9, fontweight="bold")
        ax.set_axis_off()

    # Shared colorbar
    if n_panels > 0:
        plt.colorbar(sc, ax=axes[:n_panels].tolist(), label="Disruption Rate",
                     shrink=0.6, pad=0.02)

    # Hide unused axes
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Temporal Evolution of Disruption Density — Netherlands GTFS-RT\n"
        "(faceted by hour-of-day; captures diurnal spatial patterns)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out / "map05_temporal_evolution.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "map05_temporal_evolution.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ✓ map05_temporal_evolution")


# ─────────────────────────────────────────────────────────────────────────────
# Map 6: Hot Spots with Hub Overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_hotspots_map(
    classified_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 50,
    figsize: Tuple[int, int] = (12, 14),
) -> None:
    """
    Static map highlighting the top-N most disrupted stops with hub stations
    overlaid. Useful for identifying operational focus areas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import geopandas as gpd

    logger.info("  Generating hot spots map...")
    out = Path(output_dir)

    sub = _validate_coordinates(classified_df)
    if sub.empty:
        return

    # Severity or disruption-based ranking
    if "severity_score" in sub.columns:
        agg = sub.groupby(["stop_id", "stop_lat", "stop_lon"]).agg(
            score=("severity_score", "mean"),
            n_events=("severity_score", "count"),
        ).reset_index()
    elif "disruption_target" in sub.columns:
        agg = sub.groupby(["stop_id", "stop_lat", "stop_lon"]).agg(
            score=("disruption_target", "mean"),
            n_events=("disruption_target", "count"),
        ).reset_index()
    else:
        return

    agg = agg[agg["n_events"] >= 5]
    if agg.empty:
        return

    # Top-N hot spots
    hotspots = agg.nlargest(top_n, "score")
    background = agg[~agg["stop_id"].isin(hotspots["stop_id"])]

    gdf_hot = _make_gdf(hotspots).to_crs(epsg=3857)
    gdf_bg = _make_gdf(background).to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Background stops (gray)
    if not gdf_bg.empty:
        ax.scatter(gdf_bg.geometry.x, gdf_bg.geometry.y,
                   s=4, c="#cccccc", alpha=0.3, zorder=2, label="Other stops")

    # Hot spots (red, sized by score)
    sc = ax.scatter(
        gdf_hot.geometry.x, gdf_hot.geometry.y,
        c=gdf_hot["score"],
        s=30 + gdf_hot["score"] * 15,
        cmap="Reds",
        vmin=0,
        vmax=max(gdf_hot["score"].max(), 1),
        alpha=0.8,
        edgecolors="darkred",
        linewidths=0.5,
        zorder=4,
    )
    plt.colorbar(sc, ax=ax, label="Severity / Disruption Rate", shrink=0.6)

    # Hub overlay
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    for name, (lat, lon) in MAJOR_HUBS.items():
        x_hub, y_hub = transformer.transform(lon, lat)
        ax.scatter(x_hub, y_hub, marker="*", s=250, c="navy", zorder=6,
                   edgecolors="white", linewidths=1.0)
        ax.annotate(name, (x_hub, y_hub), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    _get_basemap_ax(ax)

    # Legend
    hotspot_patch = mpatches.Patch(color="darkred", label=f"Top-{top_n} hot spots")
    hub_patch = mpatches.Patch(color="navy", label="Major hub stations")
    ax.legend(handles=[hotspot_patch, hub_patch], loc="lower left", fontsize=9)

    ax.set_title(
        f"Disruption Hot Spots — Top {top_n} Stops\n"
        "(sized/colored by severity; ★ = major hub stations)",
        fontsize=11, fontweight="bold",
    )
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out / "map06_hotspots.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "map06_hotspots.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  ✓ map06_hotspots")


# ─────────────────────────────────────────────────────────────────────────────
# Map 7: Interactive Spatio-Temporal Map (TimestampedGeoJson)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatiotemporal_map(
    classified_df: pd.DataFrame,
    output_dir: str,
    interval_minutes: int = 30,
    max_stops: int = 1500,
) -> None:
    """
    Interactive spatio-temporal map with a time slider.

    Uses folium's TimestampedGeoJson plugin to animate stop-level disruption
    patterns over time. Each time step shows disruption rates at stops,
    allowing the user to scrub through time and observe how disruptions
    propagate geographically.

    Parameters
    ----------
    classified_df : pd.DataFrame
        Classified DataFrame with stop_lat, stop_lon, feed_timestamp,
        disruption_target (or disruption_type), and severity_score.
    output_dir : str
        Directory to save the HTML map.
    interval_minutes : int
        Temporal resolution of each animation frame (default: 30 min).
    max_stops : int
        Maximum unique stops to include (for performance).
    """
    try:
        import folium
        from folium.plugins import TimestampedGeoJson
    except ImportError:
        logger.warning("  folium not installed. Skipping spatio-temporal map.")
        return

    logger.info("  Generating interactive spatio-temporal map...")
    out = Path(output_dir)

    sub = _validate_coordinates(classified_df)
    if sub.empty:
        return

    # Require timestamp
    ts_col = None
    if "feed_timestamp" in sub.columns:
        ts_col = "feed_timestamp"
    elif "timestamp" in sub.columns:
        sub["feed_timestamp"] = pd.to_datetime(sub["timestamp"], unit="s", utc=True, errors="coerce")
        ts_col = "feed_timestamp"
    else:
        logger.warning("  No timestamp column for spatio-temporal map.")
        return

    sub["_ts"] = pd.to_datetime(sub[ts_col], utc=True, errors="coerce")
    sub = sub.dropna(subset=["_ts"])

    # Disruption label
    if "disruption_target" in sub.columns:
        disp_col = "disruption_target"
    elif "disruption_type" in sub.columns:
        sub["_is_disrupted"] = (sub["disruption_type"] != "ON_TIME").astype(int)
        disp_col = "_is_disrupted"
    else:
        logger.warning("  No disruption label for spatio-temporal map.")
        return

    # Bin timestamps into intervals
    sub["_time_bin"] = sub["_ts"].dt.floor(f"{interval_minutes}min")
    sub["_time_bin_str"] = sub["_time_bin"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Aggregate per stop per time bin
    agg = sub.groupby(["stop_id", "stop_lat", "stop_lon", "_time_bin_str"]).agg(
        disruption_rate=(disp_col, "mean"),
        severity=("severity_score", "mean") if "severity_score" in sub.columns else (disp_col, "count"),
        n_events=(disp_col, "count"),
    ).reset_index()

    # Filter stops with enough events overall
    stop_counts = agg.groupby("stop_id")["n_events"].sum()
    valid_stops = stop_counts[stop_counts >= 3].index
    agg = agg[agg["stop_id"].isin(valid_stops)]

    # Limit to top stops by total events
    if len(valid_stops) > max_stops:
        top_stops = stop_counts.nlargest(max_stops).index
        agg = agg[agg["stop_id"].isin(top_stops)]

    if agg.empty:
        logger.warning("  No data for spatio-temporal map.")
        return

    logger.info(f"    {len(agg)} time-stop observations across {agg['_time_bin_str'].nunique()} time bins")

    # Build GeoJson features for TimestampedGeoJson
    def _color(rate):
        if pd.isna(rate):
            return "#808080"
        if rate < 0.05:
            return "#1a9850"
        elif rate < 0.15:
            return "#fee08b"
        elif rate < 0.30:
            return "#fdae61"
        else:
            return "#d73027"

    features = []
    for _, row in agg.iterrows():
        rate = row["disruption_rate"]
        name = row["stop_id"]
        color = _color(rate)
        radius = max(3, min(12, rate * 40)) if not pd.isna(rate) else 3
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["stop_lon"], row["stop_lat"]],
            },
            "properties": {
                "time": row["_time_bin_str"],
                "popup": (
                    f"<b>Stop: {name}</b><br>"
                    f"Time: {row['_time_bin_str']}<br>"
                    f"Disruption rate: {rate:.1%}<br>"
                    f"Events: {int(row['n_events'])}"
                ),
                "icon": "circle",
                "iconstyle": {
                    "fillColor": color,
                    "fillOpacity": 0.75,
                    "stroke": True,
                    "weight": 0.5,
                    "color": "white",
                    "radius": radius,
                },
            },
        })

    # Build folium map
    center_lat = agg["stop_lat"].mean()
    center_lon = agg["stop_lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8,
                   tiles="CartoDB positron")

    # Add TimestampedGeoJson layer
    tdg = TimestampedGeoJson(
        data={"type": "FeatureCollection", "features": features},
        period=f"PT{interval_minutes}M",
        duration=f"PT{interval_minutes}M",
        add_last_point=False,
        auto_play=False,
        loop=True,
        max_speed=3,
        loop_button=True,
        date_options="YYYY-MM-DD HH:mm",
        time_slider_drag_update=True,
    )
    tdg.add_to(m)

    # Major hubs
    hub_group = folium.FeatureGroup(name="Major Hubs")
    for name, (lat, lon) in MAJOR_HUBS.items():
        folium.Marker(
            location=[lat, lon],
            popup=name,
            icon=folium.Icon(color="black", icon="train", prefix="fa"),
        ).add_to(hub_group)
    hub_group.add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
         background:white; padding:10px 14px; border-radius:6px;
         box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:12px;">
      <b>Disruption Rate</b><br>
      <span style="color:#1a9850;">&#9679;</span> &lt; 5%<br>
      <span style="color:#fee08b;">&#9679;</span> 5–15%<br>
      <span style="color:#fdae61;">&#9679;</span> 15–30%<br>
      <span style="color:#d73027;">&#9679;</span> &gt; 30%<br>
      <span style="color:#808080;">&#9679;</span> No data
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    m.save(str(out / "map07_spatiotemporal.html"))
    logger.info("  ✓ map07_spatiotemporal.html")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_spatial_maps(
    classified_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Generate all spatial map visualizations.

    Parameters
    ----------
    classified_df : pd.DataFrame
        Output of DisruptionClassifier.classify(), with stop_lat, stop_lon,
        disruption_type, severity_score, delay_sec, and optionally
        spatial_lag_delay.
    output_dir : str
        Directory to save map files (PNG, PDF, HTML).
    """
    logger.info("=" * 60)
    logger.info("GENERATING SPATIAL MAPS")
    logger.info("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if geopandas is available (required for all static maps)
    try:
        import geopandas
    except ImportError:
        logger.error("  geopandas is required for spatial maps. "
                      "Install with: pip install geopandas contextily folium")
        return

    plot_disruption_density_map(classified_df, output_dir)
    plot_severity_map(classified_df, output_dir)
    plot_spatial_lag_map(classified_df, output_dir)
    plot_interactive_map(classified_df, output_dir)
    plot_temporal_evolution_map(classified_df, output_dir)
    plot_hotspots_map(classified_df, output_dir)
    plot_spatiotemporal_map(classified_df, output_dir)

    logger.info("  All spatial maps generated.\n")
