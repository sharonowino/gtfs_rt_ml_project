"""
GTFS Network Graph & Headway Features
===================================
This module provides:

1. Headway Features - inter-vehicle spacing in time
2. Stop Sequence Graph - NetworkX graph from stop_times
3. Betweenness Centrality - network importance metrics
4. Network Disruption Load - 2-hop neighborhood features
5. Dutch Temporal Calendar - school days, holidays, peak hours

Usage:
------
from gtfs_disruption.features.network_graph import (
    StopSequenceGraph,
    HeadwayFeatures,
    DutchCalendarFeatures,
    add_network_features
)
"""
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class StopSequenceGraph:
    """
    Build directed graph from GTFS stop_times.
    
    For every trip, the stops visited IN ORDER become consecutive
    directed edges in the graph. This represents the transit
    network topology.
    
    Parameters
    ----------
    stop_times_df : pd.DataFrame
        GTFS stop_times with trip_id, stop_id, stop_sequence
    max_trips : int
        Maximum trips to include (default 50000)
    seed : int
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        stop_times_df: pd.DataFrame,
        max_trips: int = 50000,
        seed: int = 42
    ):
        self.stop_times_df = stop_times_df.copy()
        self.max_trips = max_trips
        self.seed = seed
        self.graph = None
        self._nbr_cache = None
    
    def build_graph(self) -> nx.DiGraph:
        """Build directed graph from stop sequence."""
        logger.info("Building stop sequence graph...")
        
        _st = (
            self.stop_times_df[["trip_id", "stop_id", "stop_sequence"]]
            .copy()
            .assign(
                trip_id=self.stop_times_df["trip_id"].astype(str),
                stop_id=self.stop_times_df["stop_id"].astype(str),
                stop_sequence=pd.to_numeric(
                    self.stop_times_df["stop_sequence"], errors="coerce"
                ),
            )
            .dropna(subset=["stop_sequence"])
        )
        
        logger.info(f"  Working rows after type cast: {len(_st):,}")
        
        _unique_trips = _st["trip_id"].unique()
        logger.info(f"  Total unique trips: {len(_unique_trips):,}")
        
        if len(_unique_trips) > self.max_trips:
            _rng = np.random.default_rng(self.seed)
            _sampled = _rng.choice(
                _unique_trips, size=self.max_trips, replace=False
            )
            _st = _st[_st["trip_id"].isin(_sampled)]
            logger.info(f"  Sampled to {self.max_trips:,} trips")
        else:
            logger.info(f"  Using all {len(_unique_trips):,} trips")
        
        _st_sorted = _st.sort_values(["trip_id", "stop_sequence"])
        
        _grouped = _st_sorted.groupby("trip_id")["stop_id"].apply(list)
        logger.info(f"  Trips after grouping: {len(_grouped):,}")
        
        edges = []
        for _stop_list in _grouped:
            for _j in range(len(_stop_list) - 1):
                edges.append((_stop_list[_j], _stop_list[_j + 1]))
        
        logger.info(f"  Edges generated: {len(edges):,}")
        
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        self.graph = G
        logger.info(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def get_graph(self) -> nx.DiGraph:
        """Get graph, building if needed."""
        if self.graph is None:
            return self.build_graph()
        return self.graph
    
    def compute_betweenness_centrality(
        self,
        normalized: bool = True,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """Compute betweenness centrality."""
        G = self.get_graph()
        
        if G.number_of_nodes() == 0:
            return {}
        
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        G_lcc = G.subgraph(largest_wcc).to_undirected()
        
        k_val = min(k, G_lcc.number_of_nodes()) if k else min(2000, G_lcc.number_of_nodes())
        
        bc = nx.betweenness_centrality(
            G_lcc,
            normalized=normalized,
            k=k_val,
            seed=self.seed
        )
        
        return bc
    
    def compute_pagerank(self) -> Dict[str, float]:
        """Compute PageRank for nodes."""
        G = self.get_graph()
        if G.number_of_nodes() == 0:
            return {}
        
        return nx.pagerank(G, alpha=0.85)
    
    def get_2hop_neighbors(self, node: str) -> Set[str]:
        """Get 2-hop neighborhood of a node."""
        G = self.get_graph()
        
        if node not in G:
            return set()
        
        h1 = set(G.successors(node))
        h2 = set()
        for _n in h1:
            h2.update(G.successors(_n))
        
        return (h1 | h2) - {node}
    
    def build_neighbor_cache(self, nodes: List[str]) -> Dict[str, Set[str]]:
        """Build 2-hop neighbor cache for a list of nodes."""
        logger.info("Building 2-hop neighbor cache...")
        
        self._nbr_cache = {
            str(node): self.get_2hop_neighbors(str(node))
            for node in nodes
        }
        
        logger.info(f"  Cache built for {len(self._nbr_cache)} nodes")
        return self._nbr_cache
    
    def get_neighbor_cache(self) -> Dict[str, Set[str]]:
        """Get neighbor cache, building if needed."""
        if self._nbr_cache is None:
            if self.graph:
                nodes = list(self.graph.nodes())
                return self.build_neighbor_cache(nodes)
        return self._nbr_cache


class HeadwayFeatures:
    """
    Headway = inter-vehicle spacing in time.
    
    Primary passenger-visible service quality metric and primary
    propagation mechanism for delay cascades.
    
    Parameters
    ----------
    timestamp_col : str
        Timestamp column name
    route_col : str
        Route identifier column
    stop_col : str
        Stop identifier column
    """
    
    def __init__(
        self,
        timestamp_col: str = 'feed_timestamp',
        route_col: str = 'route_id',
        stop_col: str = 'stop_id'
    ):
        self.timestamp_col = timestamp_col
        self.route_col = route_col
        self.stop_col = stop_col
    
    def compute_headway(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute headway features."""
        logger.info("Computing headway features...")
        
        if self.timestamp_col not in df.columns:
            logger.warning(f"  Column {self.timestamp_col} not found")
            return df
        
        sort_cols = [c for c in [self.route_col, self.stop_col, self.timestamp_col] if c in df.columns]
        
        if sort_cols:
            df = df.sort_values(sort_cols)
        
        if self.route_col in df.columns and self.stop_col in df.columns:
            df['actual_headway'] = df.groupby(
                [self.route_col, self.stop_col]
            )[self.timestamp_col].diff().dt.total_seconds()
        
        df['headway_minutes'] = df['actual_headway'] / 60
        
        scheduled_col = 'scheduled_headway_sec'
        if scheduled_col in df.columns:
            df['headway_ratio'] = (
                df['actual_headway'] / df[scheduled_col].replace(0, np.nan)
            )
            df['headway_deviation'] = df['actual_headway'] - df[scheduled_col]
        
        if 'actual_headway' in df.columns:
            df['headway_variability'] = df.groupby(
                [self.route_col, self.stop_col]
            )['actual_headway'].transform(
                lambda x: x.rolling(5, min_periods=1).std()
            )
            
            df['headway_mean'] = df.groupby(
                [self.route_col, self.stop_col]
            )['actual_headway'].transform('mean')
            
            df['headway_is_bunched'] = (
                df['actual_headway'] < df['headway_mean'] * 0.5
            ).astype(int)
            
            df['headway_is_gap'] = (
                df['actual_headway'] > df['headway_mean'] * 2
            ).astype(int)
        
        return df


class NetworkDisruptionLoad:
    """
    Network disruption load features.
    
    Computes disruption severity aggregated over 2-hop neighborhood.
    
    Parameters
    ----------
    graph : StopSequenceGraph
        Pre-built stop sequence graph
    severity_col : str
        Column containing disruption severity
    """
    
    def __init__(
        self,
        graph: StopSequenceGraph,
        severity_col: str = 'arrival_delay_seconds'
    ):
        self.graph = graph
        self.severity_col = severity_col
    
    def compute_network_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute network disruption load features."""
        logger.info("Computing network disruption load...")
        
        if self.severity_col not in df.columns:
            logger.warning(f"  Column {self.severity_col} not found")
            return df
        
        if self.graph._nbr_cache is None:
            if self.graph.graph:
                nodes = list(self.graph.graph.nodes())
                self.graph.build_neighbor_cache(nodes)
        
        if self.graph._nbr_cache is None:
            logger.warning("  No neighbor cache available")
            return df
        
        severity_by_stop = (
            df.sort_values('feed_timestamp')
            .groupby('stop_id')[self.severity_col]
            .last()
            .to_dict()
        )
        
        def _compute_load(node):
            nbrs = self.graph._nbr_cache.get(str(node), set())
            return sum(
                severity_by_stop.get(str(n), 0.0)
                for n in nbrs
            )
        
        df['network_disruption_load'] = df['stop_id'].apply(_compute_load)
        
        df['network_load_normalized'] = (
            df['network_disruption_load'] / 
            df[self.severity_col].replace(0, np.nan)
        )
        
        return df


class DutchCalendarFeatures:
    """
    Dutch temporal calendar features.
    
    Dutch OV disruption patterns are strongly modulated by:
    - School days
    - Public holidays
    - Peak hours
    
    Parameters
    ----------
    timestamp_col : str
        Timestamp column name
    """
    
    def __init__(
        self,
        timestamp_col: str = 'feed_timestamp'
    ):
        self.timestamp_col = timestamp_col
        
        self._dutch_holidays_2025 = [
            '2025-01-01',  # Nieuwjaarsdag
            '2025-04-18',  # Goede Vrijdag
            '2025-04-20',  # Eerste paasdag
            '2025-04-21',  # Tweede paasdag
            '2025-04-27',  # Koningsdag
            '2025-05-01',  # Dag van de Arbeid
            '2025-05-29',  # Hemelvaartdag
            '2025-06-08',  # Eerste pinksterdag
            '2025-06-09',  # Tweede pinkesterdag
            '2025-12-25',  # Eerste kerstdag
            '2025-12-26',  # Tweede kerstdag
        ]
        
        self._dutch_holidays_2026 = [
            '2026-01-01',
            '2026-04-03',
            '2026-04-05',
            '2026-04-06',
            '2026-04-26',
            '2026-05-01',
            '2026-05-14',
            '2026-05-24',
            '2026-05-25',
            '2026-12-25',
            '2026-12-26',
        ]
        
        self._school_holidays_2025 = {
            'winter': [('2025-02-15', '2025-02-23')],
            'spring': [('2025-04-25', '2025-05-03')],
            'summer': [('2025-07-01', '2025-08-31')],
            'autumn': [('2025-10-11', '2025-10-19')],
        }
        
        self._school_holidays_2026 = {
            'winter': [('2026-02-14', '2026-02-22')],
            'spring': [('2026-04-17', '2026-04-25')],
            'summer': [('2026-07-01', '2026-08-31')],
            'autumn': [('2026-10-10', '2026-10-18')],
        }
    
    def _is_dutch_holiday(self, date) -> bool:
        """Check if date is a Dutch public holiday."""
        date_str = date.strftime('%Y-%m-%d')
        return (
            date_str in self._dutch_holidays_2025 or
            date_str in self._dutch_holidays_2026
        )
    
    def _is_school_holiday(self, date) -> bool:
        """Check if date is in Dutch school holiday."""
        for year, holidays in [(2025, self._school_holidays_2025), 
                           (2026, self._school_holidays_2026)]:
            for period, ranges in holidays.items():
                for start, end in ranges:
                    start_date = pd.to_datetime(start)
                    end_date = pd.to_datetime(end)
                    if start_date <= date <= end_date:
                        return True
        return False
    
    def _is_school_day(self, date) -> bool:
        """Check if it's a school day (not holiday, not weekend)."""
        weekday = date.dayofweek
        return weekday < 5 and not self._is_school_holiday(date)
    
    def compute_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Dutch calendar features."""
        logger.info("Computing Dutch calendar features...")
        
        ts_col = self.timestamp_col
        if ts_col not in df.columns:
            for alt in ['timestamp', 'event_time']:
                if alt in df.columns:
                    ts_col = alt
                    break
        
        if ts_col is None or ts_col not in df.columns:
            logger.warning(f"  No timestamp column found")
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        
        if df[ts_col].empty:
            return df
        
        dt = df[ts_col]
        
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['day_of_month'] = dt.dt.day
        df['week_of_year'] = dt.dt.isocalendar().week
        
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df['is_morning_peak'] = (
            (df['hour'] >= 7) & (df['hour'] <= 9)
        ).astype(int)
        
        df['is_evening_peak'] = (
            (df['hour'] >= 16) & (df['hour'] <= 19)
        ).astype(int)
        
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        df['is_night'] = (df['hour'] >= 22).astype(int) | (df['hour'] <= 5).astype(int)
        
        df['is_dutch_holiday'] = dt.dt.date.apply(self._is_dutch_holiday).astype(int)
        
        df['is_school_holiday'] = dt.dt.date.apply(self._is_school_holiday).astype(int)
        
        df['is_school_day'] = dt.dt.date.apply(self._is_school_day).astype(int)
        
        df['is_holiday_period'] = (
            df['is_dutch_holiday'] | df['is_school_holiday']
        ).astype(int)
        
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df


def add_network_features(
    df: pd.DataFrame,
    stop_times_df: Optional[pd.DataFrame] = None,
    gtfs_data: Optional[Dict[str, pd.DataFrame]] = None,
    timestamp_col: str = 'feed_timestamp',
    severity_col: str = 'arrival_delay_seconds'
) -> pd.DataFrame:
    """
    Convenience function to add all network and calendar features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    stop_times_df : pd.DataFrame
        GTFS stop_times for graph building
    gtfs_data : dict
        Static GTFS data
    timestamp_col : str
        Timestamp column
    severity_col : str
        Severity column for network load
        
    Returns
    -------
    pd.DataFrame with network features added
    """
    logger.info("=" * 60)
    logger.info("NETWORK & CALENDAR FEATURES")
    logger.info("=" * 60)
    
    out = df.copy()
    
    calendar = DutchCalendarFeatures(timestamp_col=timestamp_col)
    out = calendar.compute_calendar_features(out)
    logger.info("  Dutch calendar features added")
    
    if 'actual_headway' not in out.columns:
        headway = HeadwayFeatures(timestamp_col=timestamp_col)
        out = headway.compute_headway(out)
        logger.info("  Headway features added")
    
    if stop_times_df is not None and not stop_times_df.empty:
        graph_builder = StopSequenceGraph(stop_times_df)
        G = graph_builder.build_graph()
        
        bc = graph_builder.compute_betweenness_centrality()
        if bc:
            bc_df = pd.DataFrame({
                'stop_id': list(bc.keys()),
                'betweenness_centrality': list(bc.values())
            })
            bc_df['stop_id'] = bc_df['stop_id'].astype(str)
            out['stop_id'] = out['stop_id'].astype(str)
            out = out.merge(bc_df, on='stop_id', how='left')
            out['betweenness_centrality'] = out['betweenness_centrality'].fillna(0.0)
            
            out['betweenness_centrality_log'] = np.log1p(
                out['betweenness_centrality'] * 1000
            )
            logger.info("  Betweenness centrality added")
        
        pr = graph_builder.compute_pagerank()
        if pr:
            pr_df = pd.DataFrame({
                'stop_id': list(pr.keys()),
                'pagerank': list(pr.values())
            })
            pr_df['stop_id'] = pr_df['stop_id'].astype(str)
            out = out.merge(pr_df, on='stop_id', how='left')
            out['pagerank'] = out['pagerank'].fillna(0.0)
            logger.info("  PageRank added")
        
        if severity_col in out.columns and G.number_of_nodes() > 0:
            graph_builder.build_neighbor_cache(list(G.nodes()))
            net_load = NetworkDisruptionLoad(graph_builder, severity_col)
            out = net_load.compute_network_load(out)
            logger.info("  Network disruption load added")
    
    new_cols = [c for c in out.columns if c not in df.columns]
    logger.info(f"  Added {len(new_cols)} features: {new_cols}")
    
    return out


def build_stop_graph(
    stop_times_df: pd.DataFrame,
    max_trips: int = 50000,
    seed: int = 42
) -> nx.DiGraph:
    """
    Convenience function to build stop sequence graph.
    
    Parameters
    ----------
    stop_times_df : pd.DataFrame
        GTFS stop_times DataFrame
    max_trips : int
        Maximum trips to include
    seed : int
        Random seed
        
    Returns
    -------
    nx.DiGraph
    """
    builder = StopSequenceGraph(stop_times_df, max_trips, seed)
    return builder.build_graph()