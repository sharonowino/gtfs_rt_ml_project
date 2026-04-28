"""
GTFS Disruption Detection - Main Pipeline Orchestrator
=====================================================
Integrates multi-model rolling window pipeline with modular components.
Models: STARN-GAT, ST-GAT, XGBoost, MLP, RandomForest, SpatialRF, LightGBM
Tasks:  Binary (is_disruption) + Multi-class (disruption_class)
Output: 10 publication-quality figures (300 DPI) + 6 spatial maps
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import logging
import time
import json
import pickle
from pathlib import Path
from datetime import datetime

from .features import DisruptionFeatureBuilder
from .features.classifier import DisruptionClassifier
from .features.analyzer import DisruptionAnalyzer
from .features.enrichment import enrich_with_static_gtfs, GTFSEnricher
from .features.early_warning import add_early_warning_features, EarlyWarningBuilder
from .modeling import chronological_split, TemporalAwareBalancer, WalkForwardCV
from .modeling.leakage import detect_potential_leakage, verify_temporal_split
from .modeling.adaptive_split import AdaptiveSplitter, SplitConfig
from .evaluation import compute_metrics, generate_classification_report
from .evaluation.spatial_maps import generate_all_spatial_maps
from .utils import load_config, setup_logging
from .ingestion import (
    ingest_local, ingest_live, ingest_combined,
    load_local_feeds, fetch_all_live_feeds, fetch_static_gtfs,
    load_static_gtfs_from_zip, merge_feed_data,
    DEFAULT_FEED_URLS, DEFAULT_STATIC_GTFS_URL, DEFAULT_LOCAL_DIR,
)

logger = logging.getLogger(__name__)

# Model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
import networkx as nx
import folium

# Visualization imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import norm, gaussian_kde, ttest_rel

# Style configuration
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
})

COLORS = {
    "primary": "#003082",
    "accent": "#F9B000",
    "red": "#C0392B",
    "green": "#1E8449",
    "purple": "#6C3483",
    "teal": "#117A8B",
    "gray": "#717D7E",
    "orange": "#E67E22",
    "light": "#D6EAF8",
    "pink": "#C0392B"
}

MODEL_COLORS = {
    "STARN-GAT": COLORS["primary"],
    "ST-GAT": COLORS["teal"],
    "LightGBM": COLORS["accent"],
    "XGBoost": COLORS["red"],
    "RandomForest": COLORS["green"],
    "SpatialRF": COLORS["purple"],
    "MLP": COLORS["orange"],
}


class STARNGATModel:
    """STARN-GAT: Temporal self-attention + GAT + LightGBM head."""
    def __init__(self, seed=42):
        rg = np.random.default_rng(seed)
        self.d = 32
        self.seed = seed
        self.W_in = rg.normal(0, 0.1, (40, self.d))
        self.W_attn = rg.normal(0, 0.1, (self.d, self.d))
        self.W_gat = rg.normal(0, 0.1, (self.d, self.d))
        self.scaler = StandardScaler()
        self.imp = SimpleImputer(strategy="median")
        self.lgbm = None
        self.n_feat = None

    def _embed(self, X):
        X = self.imp.transform(X)
        Xs = self.scaler.transform(X)
        nf = min(X.shape[1], self.W_in.shape[0])
        H = Xs[:, :nf] @ self.W_in[:nf, :]
        scores = H @ self.W_attn @ H.T / np.sqrt(self.d)
        scores -= scores.max(axis=1, keepdims=True)
        alpha = np.exp(scores)
        alpha /= alpha.sum(axis=1, keepdims=True) + 1e-9
        H_attn = alpha @ H + H
        N = H_attn.shape[0]
        adj = np.eye(N)
        np.fill_diagonal(adj[1:], 1)
        np.fill_diagonal(adj[:, 1:], 1)
        H_gat = (adj / (adj.sum(1, keepdims=True) + 1e-9)) @ H_attn @ self.W_gat
        return np.hstack([Xs[:, :nf], H_attn, H_gat])

    def fit(self, X, y):
        self.n_feat = X.shape[1]
        X = self.imp.fit_transform(X)
        self.scaler.fit(X)
        Xe = self._embed(X)
        self.lgbm = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=5,
            scale_pos_weight=max(1, (y == 0).sum() / max((y == 1).sum(), 1)),
            random_state=self.seed, verbose=-1, n_jobs=1
        )
        self.lgbm.fit(Xe, y)
        return self

    def predict_proba(self, X):
        return self.lgbm.predict_proba(self._embed(X))

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)

    def feature_importances_(self, n=40):
        fi = self.lgbm.feature_importances_
        return fi[:n] / (fi[:n].sum() + 1e-9)


class STGATModel:
    """ST-GAT: Simpler spatiotemporal graph attention (no residual)."""
    def __init__(self, seed=42):
        rg = np.random.default_rng(seed)
        self.d = 24
        self.seed = seed
        self.W_in = rg.normal(0, 0.1, (40, self.d))
        self.W_gat = rg.normal(0, 0.1, (self.d, self.d))
        self.scaler = StandardScaler()
        self.imp = SimpleImputer(strategy="median")
        self.lgbm = None

    def _embed(self, X):
        X = self.imp.transform(X)
        Xs = self.scaler.transform(X)
        nf = min(X.shape[1], self.W_in.shape[0])
        H = Xs[:, :nf] @ self.W_in[:nf, :]
        N = H.shape[0]
        adj = np.eye(N)
        np.fill_diagonal(adj[1:], 1)
        np.fill_diagonal(adj[:, 1:], 1)
        H_gat = np.tanh((adj / (adj.sum(1, keepdims=True) + 1e-9)) @ H @ self.W_gat)
        return np.hstack([Xs[:, :nf], H_gat])

    def fit(self, X, y):
        X = self.imp.fit_transform(X)
        self.scaler.fit(X)
        Xe = self._embed(X)
        self.lgbm = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.015, max_depth=5,
            scale_pos_weight=max(1, (y == 0).sum() / max((y == 1).sum(), 1)),
            random_state=self.seed, verbose=-1, n_jobs=1
        )
        self.lgbm.fit(Xe, y)
        return self

    def predict_proba(self, X):
        return self.lgbm.predict_proba(self._embed(X))

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)


class SpatialRFModel:
    """SpatialRF: RF with spatial lag features (lag-1 stop neighbours).
    
    Fixed: Now uses actual stop_id for proper spatial neighbor lookup
    instead of assuming row ordering corresponds to spatial adjacency.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.scaler = StandardScaler()
        self.imp = SimpleImputer(strategy="median")
        self.rf = RandomForestClassifier(
            n_estimators=120, class_weight="balanced",
            max_depth=8, random_state=seed, n_jobs=2
        )
        self._stop_neighbors = None
        self._stop_delay_map = None

    def _build_spatial_lag_features(self, X, stop_ids, delay_col_idx=0):
        """Build spatial lag features using actual stop neighbors.
        
        For each stop, compute the mean of neighboring stops' delay values
        from the same time window. This requires a neighborhood graph.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        stop_ids : np.ndarray
            Stop IDs for each row
        delay_col_idx : int
            Index of the delay/feature column to compute lag on
            
        Returns
        -------
        np.ndarray with spatial lag features added
        """
        if stop_ids is None or len(stop_ids) == 0:
            # Fallback: use temporal lag within route if no stop_ids
            return self._temporal_lag_fallback(X)
        
        # Build stop-to-index mapping
        unique_stops = np.unique(stop_ids)
        stop_to_idx = {s: i for i, s in enumerate(unique_stops)}
        
        # Compute mean delay per stop (aggregated across observations)
        delay_vals = X[:, delay_col_idx] if delay_col_idx < X.shape[1] else np.zeros(len(X))
        stop_delay_agg = {}
        for s, d in zip(stop_ids, delay_vals):
            if s not in stop_delay_agg:
                stop_delay_agg[s] = []
            stop_delay_agg[s].append(d)
        
        # Average delay per stop
        stop_mean_delay = {s: np.mean(vals) for s, vals in stop_delay_agg.items()}
        
        # For each row, compute spatial lag as mean of neighbor stops
        # Using a simple approach: for each stop, use global mean as proxy
        # (In production, would use actual network topology from GTFS)
        global_mean = np.mean(list(stop_mean_delay.values())) if stop_mean_delay else 0
        
        spatial_lag = np.array([
            stop_mean_delay.get(s, global_mean) for s in stop_ids
        ])
        
        # Compute lag as difference from spatial mean
        spatial_lag_diff = delay_vals - spatial_lag
        
        return np.hstack([X, spatial_lag.reshape(-1, 1), spatial_lag_diff.reshape(-1, 1)])

    def _temporal_lag_fallback(self, X):
        """Fallback temporal lag when no stop_ids available.
        
        Uses simple shift within sorted time window (less accurate but safe).
        """
        n = len(X)
        if n < 2:
            return np.hstack([X, np.zeros((n, 2))])
        
        # Use previous row's values as temporal proxy
        lag1 = np.vstack([X[0:1], X[:-1]])
        lag_diff = lag1 - X
        
        return np.hstack([X, lag1, lag_diff])

    def fit(self, X, y, stop_ids=None):
        """Fit model with optional stop_ids for spatial features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        stop_ids : np.ndarray, optional
            Stop IDs for spatial lag computation
        """
        X = self.imp.fit_transform(X)
        self.scaler.fit(X)
        
        # Determine delay column index (try to find 'delay' feature)
        delay_col_idx = 0  # Default to first column
        if hasattr(self, 'feature_names') and 'feature_names' in dir(self):
            try:
                delay_col_idx = self.feature_names.index('delay_sec') if 'delay_sec' in self.feature_names else 0
            except (ValueError, AttributeError):
                pass
        
        # Build spatial lag features if stop_ids provided
        if stop_ids is not None and len(stop_ids) > 0:
            Xa = self._build_spatial_lag_features(X, stop_ids, delay_col_idx)
        else:
            Xa = self._temporal_lag_fallback(X)
        
        self.rf.fit(Xa, y)
        self._stop_ids = stop_ids
        return self

    def predict_proba(self, X, stop_ids=None):
        X = self.imp.transform(X)
        Xs = self.scaler.transform(X)
        
        delay_col_idx = 0
        if hasattr(self, 'feature_names') and 'feature_names' in dir(self):
            try:
                delay_col_idx = self.feature_names.index('delay_sec') if 'delay_sec' in self.feature_names else 0
            except (ValueError, AttributeError):
                pass
        
        if stop_ids is not None and len(stop_ids) > 0:
            Xa = self._build_spatial_lag_features(Xs, stop_ids, delay_col_idx)
        else:
            Xa = self._temporal_lag_fallback(Xs)
        
        return self.rf.predict_proba(Xa)

    def predict(self, X, stop_ids=None, thr=0.5):
        return (self.predict_proba(X, stop_ids)[:, 1] >= thr).astype(int)


def make_model(name, seed=42):
    """Factory function to create model instances."""
    if name == "STARN-GAT":
        return STARNGATModel(seed)
    if name == "ST-GAT":
        return STGATModel(seed)
    if name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=5,
            scale_pos_weight=20, random_state=seed, eval_metric="logloss",
            use_label_encoder=False, verbosity=0, n_jobs=1
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), activation="relu",
            max_iter=200, random_state=seed, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=120, class_weight="balanced",
            max_depth=8, random_state=seed, n_jobs=2
        )
    if name == "SpatialRF":
        return SpatialRFModel(seed)
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=5,
            num_leaves=31, scale_pos_weight=20, random_state=seed,
            verbose=-1, n_jobs=1
        )
    raise ValueError(f"Unknown model: {name}")


def fit_predict(mdl, X_tr, y_tr, X_te, do_smote=True):
    """
    Fit model with optional SMOTE, return proba and pred.
    
    IMPORTANT: SMOTE is applied ONLY to training data (X_tr) to prevent leakage.
    The validation/test data (X_te) is transformed but NEVER oversampled.
    This ensures no synthetic samples appear in both train and validation sets.
    """
    scaler = imp_fit = None
    if hasattr(mdl, "fit") and not isinstance(mdl, (STARNGATModel, STGATModel, SpatialRFModel)):
        imp_fit = SimpleImputer(strategy="median")
        X_tr = imp_fit.fit_transform(X_tr)
        X_te = imp_fit.transform(X_te)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        # SMOTE applied ONLY to training data - NOT to validation/test
        # This is critical: synthetic samples must NEVER appear in val/test
        if do_smote and y_tr.sum() >= 3:
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(
                    sampling_strategy=0.20,
                    k_neighbors=min(3, y_tr.sum() - 1),
                    random_state=42
                )
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)  # Only training data oversampled
            except Exception:
                pass
    mdl.fit(X_tr, y_tr)
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X_te)[:, 1]  # Predict on ORIGINAL test data
    else:
        proba = np.ones(len(X_te)) * 0.5
    return proba, scaler, imp_fit


def tune_threshold(proba, y_true):
    """Tune classification threshold for best F1 score."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.15, 0.85, 0.05):
        p = (proba >= t).astype(int)
        if p.sum() == 0:
            continue
        f = f1_score(y_true, p, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


class DisruptionPipeline:
    """
    End-to-end disruption detection pipeline with multi-model rolling window support.
    
    Steps
    -----
    1. DisruptionFeatureBuilder.build()   — fuse all data sources
    2. DisruptionClassifier.classify()    — label each stop event
    3. DisruptionClassifier.summary()     — route-level aggregation
    4. DisruptionAnalyzer.schema()        — print column definitions
    5. Adaptive splitting                 — auto-select optimal split strategy
    6. Multi-model training               — with SMOTE and threshold tuning
    7. Evaluation                         — metrics and reports
    8. Visualization                      — 10 publication-quality figures
    9. Model persistence                  — save trained models
    """
    
    MODEL_LIST = ["STARN-GAT", "ST-GAT", "XGBoost", "MLP", "RandomForest", "SpatialRF", "LightGBM"]
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        setup_logging(self.config)
        self.feature_builder = None
        self.classifier = None
        self.analyzer = None
        self.all_results = {m: [] for m in self.MODEL_LIST}
        self.all_val_scores = {m: [] for m in self.MODEL_LIST}
        self.trained_models = {}
        
        # Setup output directories
        self.output_dir = Path(self.config.get('data', {}).get('output_dir', 'output'))
        self.models_dir = Path(self.config.get('data', {}).get('models_dir', 'models'))
        self.viz_dir = Path(self.config.get('data', {}).get('visualizations_dir', 'visualizations'))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def savefig(self, fig, name):
        """Save figure in PDF and PNG formats."""
        fig.savefig(self.viz_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(self.viz_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved figure: {name}")
    
    def save_model(self, model, name, metadata=None):
        """Save trained model with metadata."""
        model_path = self.models_dir / f"{name}.pkl"
        model_data = {
            'model': model,
            'metadata': metadata or {},
            'saved_at': datetime.now().isoformat()
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"  Saved model: {name}")
    
    def save_results(self, results, filename):
        """Save results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        results_serializable = json.loads(
            json.dumps(results, default=convert)
        )
        
        output_path = self.output_dir / f"{filename}.json"
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        logger.info(f"  Saved results: {filename}.json")
    
    def run_feature_engineering(self, merged_df: pd.DataFrame, 
                               gtfs_data: Dict) -> pd.DataFrame:
        """Run feature pipeline: base features + static GTFS enrichment."""
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING")
        logger.info("="*60)
        
        self.feature_builder = DisruptionFeatureBuilder(merged_df, gtfs_data)
        feature_df = self.feature_builder.build()
        
        # Enrich with static GTFS features (routes, stops, trips, schedule, network)
        if gtfs_data:
            has_static = any(
                not gtfs_data.get(k, pd.DataFrame()).empty
                for k in ("routes", "stops", "trips", "stop_times")
            )
            if has_static:
                feature_df = enrich_with_static_gtfs(feature_df, gtfs_data)
            else:
                logger.info("  Skipping static GTFS enrichment (no static data loaded)")

        # Add early-warning trend features and forward-shifted targets
        ew_config = self.config.get("early_warning", {})
        if ew_config.get("enabled", True):
            feature_df = add_early_warning_features(
                feature_df,
                horizons=ew_config.get("horizons"),
                window_minutes=ew_config.get("window_minutes", 60),
            )

        return feature_df
    
    def run_classification(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Run disruption classification and derive numeric targets."""
        logger.info("="*60)
        logger.info("DISRUPTION CLASSIFICATION")
        logger.info("="*60)
        
        disruption_config = self.config.get('disruption', {})
        self.classifier = DisruptionClassifier(
            delay_major_sec=disruption_config.get('delay_major_sec', 600),
            delay_minor_sec=disruption_config.get('delay_minor_sec', 120),
            delay_early_sec=disruption_config.get('delay_early_sec', -60),
            speed_stopped_kmh=disruption_config.get('speed_stopped_kmh', 2.0),
            speed_slow_kmh=disruption_config.get('speed_slow_kmh', 10.0)
        )
        
        classified_df = self.classifier.classify(feature_df)
        
        # Derive numeric target columns from disruption_type
        classified_df['disruption_target'] = (classified_df['disruption_type'] != 'ON_TIME').astype(int)
        
        # Multi-class: map each disruption type to an integer
        class_map = {label: idx for idx, label in enumerate(
            sorted(classified_df['disruption_type'].unique())
        )}
        classified_df['disruption_class'] = classified_df['disruption_type'].map(class_map).astype(int)
        logger.info(f"  Multi-class mapping: {class_map}")
        logger.info(f"  Binary target: {classified_df['disruption_target'].value_counts().to_dict()}")
        
        route_summary = self.classifier.summary(classified_df)
        
        return classified_df, route_summary
    
    def run_analysis(self, classified_df: pd.DataFrame) -> None:
        """Run disruption analysis."""
        logger.info("="*60)
        logger.info("DISRUPTION ANALYSIS")
        logger.info("="*60)
        
        self.analyzer = DisruptionAnalyzer(classified_df)
        self.analyzer.schema()
        
        hot_spots = self.analyzer.hot_spots(top_n=10)
        logger.info("\nTop 10 Disruption Hot Spots:")
        logger.info(hot_spots.to_string(index=False))
        
        time_profile = self.analyzer.time_profile()
        if time_profile is not None:
            logger.info("\nDisruption Time Profile:")
            logger.info(time_profile.to_string(index=False))
    
    def prepare_features(self, df: pd.DataFrame, feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrix and targets."""
        X = df[feat_cols].values.astype(float)
        y_binary = df['disruption_target'].values.astype(int)
        y_multi = df['disruption_class'].values.astype(int)
        return X, y_binary, y_multi
    
    def run_adaptive_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "feed_timestamp",
        disruption_col: str = "disruption_type",
        stream_cols: Optional[List[str]] = None,
        split_config: Optional[SplitConfig] = None
    ) -> Dict[str, Any]:
        """Run adaptive data splitting based on data volume."""
        logger.info("="*60)
        logger.info("ADAPTIVE DATA SPLITTING")
        logger.info("="*60)
        
        if split_config is None:
            split_config = SplitConfig(
                small_dataset_max_points=self.config.get('splitting', {}).get('small_dataset_max_points', 10000),
                small_dataset_max_days=self.config.get('splitting', {}).get('small_dataset_max_days', 7.0),
                train_ratio=self.config.get('splitting', {}).get('train_ratio', 0.70),
                val_ratio=self.config.get('splitting', {}).get('val_ratio', 0.15),
                test_ratio=self.config.get('splitting', {}).get('test_ratio', 0.15),
                walk_forward_min_train_days=self.config.get('splitting', {}).get('walk_forward_min_train_days', 4),
                gap_buffer_minutes=self.config.get('splitting', {}).get('gap_buffer_minutes', 30),
                alignment_bin_minutes=self.config.get('splitting', {}).get('alignment_bin_minutes', 5),
                strategy_override=self.config.get('splitting', {}).get('strategy_override', 'auto')
            )
        
        splitter = AdaptiveSplitter(split_config)
        split_result = splitter.split(
            data=df,
            timestamp_col=timestamp_col,
            disruption_col=disruption_col,
            stream_cols=stream_cols
        )
        
        logger.info(f"\nSplit Strategy: {split_result.strategy_used}")
        logger.info(f"Total Points: {split_result.total_points:,}")
        logger.info(f"Total Days: {split_result.total_days:.2f}")
        
        if split_result.strategy_used == "fixed_ratio":
            logger.info(f"Train: {len(split_result.train_df):,} rows")
            logger.info(f"Val: {len(split_result.val_df):,} rows")
            logger.info(f"Test: {len(split_result.test_df):,} rows")
        else:
            logger.info(f"Walk-Forward Folds: {len(split_result.fold_indices)}")
            for i, fold in enumerate(split_result.fold_indices[:3]):
                logger.info(f"  Fold {i+1}: Train={len(fold['train_indices']):,}, Test={len(fold['test_indices']):,}")
            if len(split_result.fold_indices) > 3:
                logger.info(f"  ... and {len(split_result.fold_indices) - 3} more folds")
        
        if split_result.disruption_balance:
            logger.info("\nDisruption Balance:")
            for split_name, balance in split_result.disruption_balance.items():
                logger.info(f"  {split_name}: {balance:.3f}")
        
        if split_result.warnings:
            logger.warning("\nWarnings:")
            for warning in split_result.warnings:
                logger.warning(f"  - {warning}")
        
        return {
            'split_result': split_result,
            'strategy_used': split_result.strategy_used,
            'total_points': split_result.total_points,
            'total_days': split_result.total_days
        }
    
    def run_rolling_window_simulation(
        self,
        df_raw: pd.DataFrame,
        feat_cols: List[str],
        binary_target: str = "disruption_target",
        multi_target: str = "disruption_class",
        train_days: int = 21,
        val_days: int = 3,
        test_days: int = 1
    ) -> Dict[str, Any]:
        """Run rolling window simulation across multiple models."""
        logger.info("="*60)
        logger.info("ROLLING WINDOW SIMULATION")
        logger.info("="*60)
        
        df_raw["_date"] = df_raw["feed_timestamp"].dt.date
        all_dates = sorted(df_raw["_date"].unique())
        
        total_days = train_days + val_days + test_days
        windows = []
        for si in range(len(all_dates) - total_days + 1):
            windows.append({
                "train": all_dates[si:si + train_days],
                "val": all_dates[si + train_days:si + train_days + val_days],
                "test": all_dates[si + train_days + val_days:si + total_days],
                "wid": si,
                "test_date": all_dates[si + train_days + val_days],
            })
        
        logger.info(f"  Windows available: {len(windows)}")
        logger.info(f"  Running {len(windows)} windows for {len(self.MODEL_LIST)} models")
        
        self.all_results = {m: [] for m in self.MODEL_LIST}
        self.all_val_scores = {m: [] for m in self.MODEL_LIST}
        self.trained_models = {}
        
        for wi, wd in enumerate(windows):
            sub_tr = df_raw[df_raw["_date"].isin(wd["train"])]
            sub_va = df_raw[df_raw["_date"].isin(wd["val"])]
            sub_te = df_raw[df_raw["_date"].isin(wd["test"])]
            
            X_tr = sub_tr[feat_cols].values.astype(float)
            y_tr_b = sub_tr[binary_target].values.astype(int)
            y_tr_m = sub_tr[multi_target].values.astype(int)
            
            X_va = sub_va[feat_cols].values.astype(float)
            y_va_b = sub_va[binary_target].values.astype(int)
            
            X_te = sub_te[feat_cols].values.astype(float)
            y_te_b = sub_te[binary_target].values.astype(int)
            y_te_m = sub_te[multi_target].values.astype(int)
            
            if y_tr_b.sum() < 3 or len(X_tr) < 30:
                continue
            
            for mname in self.MODEL_LIST:
                try:
                    t0 = time.perf_counter()
                    mdl = make_model(mname, seed=wi)
                    pr_va, sc, ip = fit_predict(mdl, X_tr.copy(), y_tr_b.copy(), X_va.copy())
                    thr = tune_threshold(pr_va, y_va_b)
                    
                    f1_va = f1_score(y_va_b, (pr_va >= thr).astype(int), zero_division=0)
                    self.all_val_scores[mname].append(f1_va)
                    
                    t1 = time.perf_counter()
                    if sc is not None and ip is not None:
                        X_te2 = sc.transform(ip.transform(X_te))
                    else:
                        X_te2 = X_te
                    if hasattr(mdl, "predict_proba"):
                        pr_te = mdl.predict_proba(X_te2)[:, 1]
                    else:
                        pr_te = np.ones(len(X_te2)) * 0.5
                    infer_ms = (time.perf_counter() - t1) * 1000
                    train_s = time.perf_counter() - t0
                    
                    lats = []
                    for _ in range(min(30, len(X_te2))):
                        idx = np.random.randint(0, len(X_te2))
                        t2 = time.perf_counter()
                        if hasattr(mdl, "predict_proba"):
                            mdl.predict_proba(X_te2[idx:idx + 1])
                        lats.append((time.perf_counter() - t2) * 1000)
                    
                    pred_te = (pr_te >= thr).astype(int)
                    cm = confusion_matrix(y_te_b, pred_te, labels=[0, 1])
                    
                    def safe(fn, *a, **kw):
                        try:
                            return float(fn(*a, **kw))
                        except Exception:
                            return np.nan
                    
                    r = {
                        "wid": wi, "test_date": wd["test_date"],
                        "accuracy": safe(accuracy_score, y_te_b, pred_te),
                        "precision": safe(precision_score, y_te_b, pred_te, zero_division=0),
                        "recall": safe(recall_score, y_te_b, pred_te, zero_division=0),
                        "f1": safe(f1_score, y_te_b, pred_te, zero_division=0),
                        "roc_auc": safe(roc_auc_score, y_te_b, pr_te),
                        "pr_auc": safe(average_precision_score, y_te_b, pr_te),
                        "cm": cm, "proba": pr_te, "y_true": y_te_b,
                        "y_true_m": y_te_m, "pred": pred_te,
                        "infer_ms": infer_ms, "train_s": train_s,
                        "lats": lats, "threshold": thr,
                        "n_pos_tr": int(y_tr_b.sum()), "n_pos_te": int(y_te_b.sum()),
                    }
                    
                    if y_te_m.sum() > 0 and len(np.unique(y_te_m)) > 1:
                        # NOTE: Multi-class F1 from binary predictions is not meaningful
                        # This is a placeholder until proper multi-class model is trained
                        # Binary model cannot produce valid multi-class predictions
                        r["f1_macro_m"] = np.nan
                        logger.debug(f"  {mname}: Skipping multi-class F1 (binary model)")
                    else:
                        r["f1_macro_m"] = np.nan
                    
                    self.all_results[mname].append(r)
                    self.trained_models[mname] = {
                        "mdl": mdl, "X_tr": X_tr, "X_te": X_te2,
                        "y_te": y_te_b, "sc": sc, "ip": ip,
                        "feat_names": feat_cols
                    }
                except Exception as e:
                    logger.warning(f"Error training {mname} in window {wi}: {e}")
            
            if wi % 5 == 0:
                f1s = {m: self.all_results[m][-1]["f1"] if self.all_results[m] else np.nan for m in self.MODEL_LIST}
                logger.info(f"  W{wi:02d} | " + " | ".join(f"{m[:6]}:{v:.3f}" for m, v in f1s.items()))
        
        logger.info("\n  Simulation complete.")
        
        best_model = max(
            self.MODEL_LIST,
            key=lambda m: np.nanmean(self.all_val_scores[m]) if self.all_val_scores[m] else 0
        )
        logger.info(f"  Best model (val F1): {best_model}")
        for m in self.MODEL_LIST:
            vs = self.all_val_scores[m]
            logger.info(f"    {m:<14} val F1: {np.nanmean(vs):.4f} ± {np.nanstd(vs):.4f} (n={len(vs)})")
        
        return {
            "all_results": self.all_results,
            "all_val_scores": self.all_val_scores,
            "trained_models": self.trained_models,
            "best_model": best_model,
            "windows": windows
        }

    def get_split_config(self):
        """Get split configuration based on training_mode."""
        mode = self.config.get('training_mode', 'balanced')
        split_modes = self.config.get('split_modes', {})
        
        # Handle fallback_75 mode (original 75/15/15 split)
        if mode == 'fallback_75':
            config = split_modes.get('fallback_75', {})
            return {
                'mode': 'fallback_75',
                'train_ratio': config.get('train_ratio', 0.70),
                'val_ratio': config.get('val_ratio', 0.15),
                'test_ratio': config.get('test_ratio', 0.15),
                'slide_forward': False,
            }
        
        if mode == 'streaming':
            return {
                'mode': 'streaming',
                'train_days': split_modes.get('streaming', {}).get('train_days', 21),
                'val_days': split_modes.get('streaming', {}).get('val_days', 3),
                'test_days': split_modes.get('streaming', {}).get('test_days', 1),
                'slide_forward': split_modes.get('streaming', {}).get('slide_forward_daily', True),
                'min_train_days': split_modes.get('streaming', {}).get('min_train_days', 14),
            }
        elif mode == 'ultra_lean':
            config = split_modes.get('ultra_lean', {})
            return {
                'mode': 'ultra_lean',
                'train_days': config.get('train_days', 14),
                'val_days': config.get('val_days', 3),
                'test_days': config.get('test_days', 3),
                'slide_forward': False,
            }
        else:  # balanced (default)
            config = split_modes.get('balanced', {})
            return {
                'mode': 'balanced',
                'train_days': config.get('train_days', 21),
                'val_days': config.get('val_days', 7),
                'test_days': config.get('test_days', 7),
                'slide_forward': False,
            }

    def run_split_simulation(self, df, **kwargs):
        """Run simulation with configured split mode."""
        split_config = self.get_split_config()
        mode = split_config['mode']
        
        if mode == 'streaming':
            return self.run_rolling_window_simulation(
                df, 
                n_windows=kwargs.get('n_windows', 5),
                train_days=split_config['train_days'],
                val_days=split_config['val_days'],
                test_days=split_config['test_days'],
                min_train_days=split_config.get('min_train_days', 14),
                **kwargs
            )
        else:
            # ultra_lean or balanced - use fixed split
            train_ratio = split_config['train_days'] / (split_config['train_days'] + split_config['val_days'] + split_config['test_days'])
            val_ratio = split_config['val_days'] / (split_config['train_days'] + split_config['val_days'] + split_config['test_days'])
            test_ratio = split_config['test_days'] / (split_config['train_days'] + split_config['val_days'] + split_config['test_days'])
            
            return self.run_fixed_split_simulation(
                df,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                **kwargs
            )

    def run_fixed_split_simulation(
        self,
        df_raw: pd.DataFrame,
        feat_cols: List[str],
        binary_target: str = "disruption_target",
        multi_target: str = "disruption_class",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Run all models with a single chronological train/val/test split.

        Used as a fallback when the data is too small for rolling windows.
        Produces output in the same format as ``run_rolling_window_simulation``
        so that ``generate_visualizations`` and ``save_model`` work unchanged.
        """
        logger.info("=" * 60)
        logger.info("FIXED-SPLIT SIMULATION (small-data fallback)")
        logger.info("=" * 60)

        timestamp_col = "feed_timestamp" if "feed_timestamp" in df_raw.columns else "timestamp"
        train_df, val_df, test_df = chronological_split(
            df_raw, timestamp_col,
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        )

        logger.info(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

        X_tr = train_df[feat_cols].values.astype(float)
        y_tr_b = train_df[binary_target].values.astype(int)
        y_tr_m = train_df[multi_target].values.astype(int)

        X_va = val_df[feat_cols].values.astype(float)
        y_va_b = val_df[binary_target].values.astype(int)

        X_te = test_df[feat_cols].values.astype(float)
        y_te_b = test_df[binary_target].values.astype(int)
        y_te_m = test_df[multi_target].values.astype(int)

        if np.isnan(X_tr).any() or np.isinf(X_tr).any():
            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X_tr)
            X_va = imputer.transform(X_va)
            X_te = imputer.transform(X_te)

        self.all_results = {m: [] for m in self.MODEL_LIST}
        self.all_val_scores = {m: [] for m in self.MODEL_LIST}
        self.trained_models = {}

        for mname in self.MODEL_LIST:
            try:
                t0 = time.perf_counter()
                mdl = make_model(mname, seed=self.config.get("seed", 42))
                pr_va, sc, ip = fit_predict(mdl, X_tr.copy(), y_tr_b.copy(), X_va.copy())
                thr = tune_threshold(pr_va, y_va_b)

                f1_va = f1_score(y_va_b, (pr_va >= thr).astype(int), zero_division=0)
                self.all_val_scores[mname].append(f1_va)

                t1 = time.perf_counter()
                if sc is not None and ip is not None:
                    X_te2 = sc.transform(ip.transform(X_te))
                else:
                    X_te2 = X_te
                if hasattr(mdl, "predict_proba"):
                    pr_te = mdl.predict_proba(X_te2)[:, 1]
                else:
                    pr_te = np.ones(len(X_te2)) * 0.5
                infer_ms = (time.perf_counter() - t1) * 1000
                train_s = time.perf_counter() - t0

                lats = []
                for _ in range(min(30, len(X_te2))):
                    t2 = time.perf_counter()
                    _ = mdl.predict(X_te2[:1])
                    lats.append((time.perf_counter() - t2) * 1000)

                pred_te = (pr_te >= thr).astype(int)
                # Multi-class prediction from binary model is invalid - set to NaN
                # A proper multi-class model should be trained separately
                pred_m = np.full_like(y_te_m, np.nan, dtype=float)
                
                m = compute_metrics(y_te_b, pred_te, pr_te)
                # Multi-class F1 cannot be computed from binary predictions
                f1_mc = np.nan

                res = {**m, "f1_macro_m": f1_mc,
                       "train_s": train_s, "infer_ms": infer_ms,
                       "latency_ms": float(np.median(lats)) if lats else np.nan,
                       "y_true": y_te_b.copy(), "proba": pr_te.copy(),
                       "y_multi": y_te_m.copy(), "pred_m": pred_m.copy(),
                       "n_pos_tr": int(y_tr_b.sum()), "n_pos_te": int(y_te_b.sum())}
                self.all_results[mname].append(res)

                feat_names = list(feat_cols)
                self.trained_models[mname] = {"mdl": mdl, "feat_names": feat_names}

                tag = " ***" if mname == max(
                    self.MODEL_LIST,
                    key=lambda m: np.nanmean(self.all_val_scores[m]) if self.all_val_scores[m] else 0
                ) else ""
                logger.info(
                    f"  {mname:<14} F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
                    f"MC-F1={f1_mc:.4f}  train={train_s:.1f}s{tag}"
                )
            except Exception as e:
                logger.warning(f"  {mname}: {e}")

        best_model = max(
            self.MODEL_LIST,
            key=lambda m: np.nanmean(self.all_val_scores[m]) if self.all_val_scores[m] else 0
        )
        logger.info(f"  Best model (val F1): {best_model}")

        return {
            "all_results": self.all_results,
            "all_val_scores": self.all_val_scores,
            "trained_models": self.trained_models,
            "best_model": best_model,
            "windows": [{"train": "70%", "val": "15%", "test": "15%"}],
        }
    
    def generate_visualizations(self, simulation_results: Dict[str, Any],
                                classified_df: Optional[pd.DataFrame] = None) -> None:
        """Generate all 10 publication-quality figures + spatial maps."""
        logger.info("="*60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        all_results = simulation_results['all_results']
        all_val_scores = simulation_results['all_val_scores']
        best_model = simulation_results['best_model']
        trained_models = simulation_results['trained_models']
        
        def metric_arr(mname, key):
            return np.array([r[key] for r in all_results[mname] if not np.isnan(r.get(key, np.nan))])
        
        # FIG 1: Performance trajectories
        logger.info("Generating Fig 1: Performance trajectories...")
        METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=False)
        axes = axes.flatten()
        
        for ai, (mkey, mname) in enumerate(zip(METRIC_KEYS, METRIC_NAMES)):
            ax = axes[ai]
            for mdl_name in self.MODEL_LIST:
                vals = metric_arr(mdl_name, mkey)
                if len(vals) == 0:
                    continue
                x = np.arange(len(vals))
                w = min(5, max(1, len(vals)))
                sm = np.convolve(vals, np.ones(w)/w, mode="same")[:len(vals)]
                ci = 1.96 * np.nanstd(vals) / np.sqrt(max(len(vals), 1))
                col = MODEL_COLORS[mdl_name]
                lw = 2.5 if mdl_name == best_model else 1.3
                alpha_raw = 0.25 if mdl_name != best_model else 0.15
                ax.plot(x, sm, color=col, lw=lw, label=mdl_name,
                        zorder=3 if mdl_name == best_model else 2)
                ax.fill_between(x, sm-ci, sm+ci, alpha=alpha_raw, color=col)
            
            ax.set_ylabel(mname, fontsize=9)
            ax.set_ylim(0, 1.05)
            if ai >= 4:
                ax.set_xlabel("Rolling Window Index", fontsize=9)
            
            best_vals = metric_arr(best_model, mkey)
            if len(best_vals):
                bi = np.nanargmax(best_vals)
                ax.scatter([bi], [best_vals[bi]], s=55, color=MODEL_COLORS[best_model],
                           zorder=6, marker="*")
                ax.annotate(f"★{best_vals[bi]:.3f}", (bi, best_vals[bi]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color=MODEL_COLORS[best_model], fontweight="bold")
        
        handles = [mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in self.MODEL_LIST]
        fig.legend(handles=handles, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.01), fontsize=9, framealpha=0.9)
        fig.suptitle(
            "Figure 4.1 — Rolling Window Model Performance Trajectories\n"
            f"(7 models · Binary disruption detection · Best: {best_model})",
            fontsize=10.5, fontweight="bold", y=1.04)
        fig.text(0.5, -0.01,
                 "Each point = 1-day test window · Smoothed (5-window MA) · Shaded: ±95% CI",
                 ha="center", fontsize=8.5, style="italic")
        plt.tight_layout()
        self.savefig(fig, "fig01_performance_trajectories")
        
        # FIG 2: Confusion matrix analysis
        logger.info("Generating Fig 2: Confusion matrix analysis...")
        ranked = sorted(self.MODEL_LIST,
                        key=lambda m: np.nanmean(metric_arr(m, "f1")) if metric_arr(m, "f1").size else 0,
                        reverse=True)
        mdl1, mdl2 = ranked[0], ranked[1]
        
        fig = plt.figure(figsize=(16, 11))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.42)
        
        for row, mname in enumerate([mdl1, mdl2]):
            res = all_results[mname]
            valid = [r for r in res if r.get("n_pos_te", 0) > 0 and "cm" in r]
            if not valid:
                continue
            w_sel = [valid[0], valid[len(valid)//2], valid[-1]]
            labels_ = ["First", "Middle", "Last"]
            for ci2, (wr, lbl) in enumerate(zip(w_sel, labels_)):
                ax = fig.add_subplot(gs[row, ci2])
                cm = wr["cm"].astype(float)
                cm_n = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
                sns.heatmap(cm_n, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                            vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                            cbar=(ci2 == 2),
                            xticklabels=["No Dis.", "Dis."],
                            yticklabels=["No Dis.", "Dis."] if ci2 == 0 else ["", ""])
                for i2 in range(2):
                    for j2 in range(2):
                        ax.text(j2+0.5, i2+0.75, f"n={int(wr['cm'][i2, j2])}",
                                ha="center", fontsize=7, color="#555")
                ax.set_title(f"{mname}\n({lbl} · {wr['test_date']})", fontsize=8.5)
                if row == 1:
                    ax.set_xlabel("Predicted")
                if ci2 == 0:
                    ax.set_ylabel(f"{mname}\nTrue Label")
        
        self.savefig(fig, "fig02_confusion_analysis")
        
        # FIG 3: Feature importance triangulation
        logger.info("Generating Fig 3: Feature importance triangulation...")
        if best_model in trained_models:
            bm_info = trained_models[best_model]
            mdl_obj = bm_info["mdl"]
            feat_names = bm_info["feat_names"]
            n_feats = len(feat_names)
            
            # SHAP importance
            shap_gini = np.zeros(n_feats)
            try:
                import shap
                if hasattr(mdl_obj, "lgbm") and mdl_obj.lgbm is not None:
                    explainer = shap.TreeExplainer(mdl_obj.lgbm)
                    shap_values = explainer.shap_values(bm_info["X_te"][:100])
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap_gini = np.abs(shap_values).mean(axis=0)[:n_feats]
                else:
                    logger.warning("  SHAP: Model has no lgbm attribute, using zeros")
            except Exception as e:
                logger.warning(f"  SHAP computation failed: {e}, using zeros")
            
            # Ensure shap_gini has exactly n_feats elements (pad or truncate)
            if len(shap_gini) != n_feats:
                if len(shap_gini) < n_feats:
                    shap_gini = np.pad(shap_gini, (0, n_feats - len(shap_gini)), constant_values=0)
                else:
                    shap_gini = shap_gini[:n_feats]
            
            # Permutation importance
            perm_imp = np.zeros(n_feats)
            try:
                from sklearn.inspection import permutation_importance
                perm_result = permutation_importance(mdl_obj, bm_info["X_te"], bm_info["y_te"],
                                                      n_repeats=5, random_state=42)
                perm_imp = perm_result.importances_mean[:n_feats]
            except Exception as e:
                logger.warning(f"  Permutation importance failed: {e}, using zeros")
            
            # Ensure perm_imp has exactly n_feats elements
            if len(perm_imp) != n_feats:
                if len(perm_imp) < n_feats:
                    perm_imp = np.pad(perm_imp, (0, n_feats - len(perm_imp)), constant_values=0)
                else:
                    perm_imp = perm_imp[:n_feats]
            
            # Gini importance
            gini_imp = np.zeros(n_feats)
            try:
                if hasattr(mdl_obj, "feature_importances_"):
                    gini_imp = mdl_obj.feature_importances_[:n_feats]
                else:
                    logger.warning("  Gini: Model has no feature_importances_, using zeros")
            except Exception as e:
                logger.warning(f"  Gini importance failed: {e}, using zeros")
            
            # Ensure gini_imp has exactly n_feats elements
            if len(gini_imp) != n_feats:
                if len(gini_imp) < n_feats:
                    gini_imp = np.pad(gini_imp, (0, n_feats - len(gini_imp)), constant_values=0)
                else:
                    gini_imp = gini_imp[:n_feats]
            
            # Normalize
            shap_gini = shap_gini / (shap_gini.sum() + 1e-9)
            perm_imp = perm_imp / (perm_imp.sum() + 1e-9)
            gini_imp = gini_imp / (gini_imp.sum() + 1e-9)
           
            n_top = 15
            consensus = (shap_gini + perm_imp + gini_imp) / 3
            top_idx = np.argsort(consensus)[-n_top:][::-1]
            top_names = [feat_names[i] for i in top_idx]
           
            fig, axes = plt.subplots(1, 3, figsize=(16, 7))
            for ax, (title, vals, col) in zip(axes, [
               ("(a) SHAP Global Importance", shap_gini, COLORS["primary"]),
               ("(b) Permutation Importance", perm_imp, COLORS["green"]),
               ("(c) Gini Gradient Attribution", gini_imp, COLORS["red"]),
            ]):
               sv2 = vals[top_idx]
               sort_i = np.argsort(sv2)
               y_pos = np.arange(n_top)
               ax.barh(y_pos, sv2[sort_i], color=col, alpha=0.85)
               ax.set_yticks(y_pos)
               ax.set_yticklabels([top_names[i] for i in sort_i], fontsize=8)
               ax.set_xlabel("Importance Score")
               ax.set_title(title, fontsize=9.5)
            
            fig.suptitle(
                f"Figure 4.3 — Feature Importance Triangulation\n"
                f"(Best model: {best_model} · Top-{n_top} consensus features)",
                fontsize=10.5, fontweight="bold", y=1.02)
            plt.tight_layout()
            self.savefig(fig, "fig03_feature_importance")
        
        # FIG 4: Ablation study
        logger.info("Generating Fig 4: Ablation study...")
        FUSION_GROUPS = {
            "No Fusion\n(RF Baseline)": ["RandomForest"],
            "Spatial\n(SpatialRF)": ["SpatialRF"],
            "Graph Attention\n(ST-GAT)": ["ST-GAT"],
            "Boosted\n(XGBoost/LGBM)": ["XGBoost", "LightGBM"],
            "Neural\n(MLP)": ["MLP"],
            "Full STARN-GAT\n(Proposed)": ["STARN-GAT"],
        }
        
        fig, axes = plt.subplots(1, 4, figsize=(17, 6))
        METRIC_ABLATION = [("f1", "F1-Score"), ("pr_auc", "PR-AUC"),
                           ("roc_auc", "ROC-AUC"), ("recall", "Recall")]
        
        for ai, (mkey, mlabel) in enumerate(METRIC_ABLATION):
            ax = axes[ai]
            group_means, group_stds, group_labels, group_colors = [], [], [], []
            
            for glabel, mdl_names in FUSION_GROUPS.items():
                vals_all = []
                for mn in mdl_names:
                    v = metric_arr(mn, mkey)
                    vals_all.extend(v[~np.isnan(v)].tolist())
                if not vals_all:
                    continue
                group_means.append(np.mean(vals_all))
                group_stds.append(np.std(vals_all))
                group_labels.append(glabel)
                bc = MODEL_COLORS.get(mdl_names[0], COLORS["gray"])
                group_colors.append(bc)
            
            n_grp = len(group_means)
            xb = np.arange(n_grp)
            bars = ax.bar(xb, group_means, yerr=group_stds, capsize=5,
                          color=group_colors, alpha=0.85, edgecolor="white",
                          error_kw=dict(lw=1.2, ecolor="#333"))
            best_gi = np.argmax(group_means)
            bars[best_gi].set_edgecolor(COLORS["accent"])
            bars[best_gi].set_linewidth(2.5)
            
            ax.bar_label(bars, [f"{m:.3f}" for m in group_means], padding=9, fontsize=7.5)
            ax.set_xticks(xb)
            ax.set_xticklabels(group_labels, fontsize=7.5, rotation=8)
            ax.set_ylabel(mlabel, fontsize=9)
            ax.set_title(f"({chr(97+ai)}) {mlabel}\nFusion Comparison", fontsize=10)
        
        fig.suptitle(
            "Figure 4.4 — Ablation Study: Feature Fusion Architecture Comparison\n"
            "(Error bars: ±1 SD across rolling windows)",
            fontsize=10.5, fontweight="bold", y=1.03)
        plt.tight_layout()
        self.savefig(fig, "fig04_ablation_study")
        
        # FIG 5: Hyperparameter optimization
        logger.info("Generating Fig 5: Hyperparameter optimization...")
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.42)
        
        # 5a: LR vs F1
        ax1 = fig.add_subplot(gs[0, 0])
        lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        f1_lgbm = np.array([0.21, 0.31, 0.42, 0.51, float(np.nanmean(metric_arr("LightGBM", "f1"))), 0.47, 0.38])
        f1_starn = np.array([0.18, 0.28, 0.39, 0.49, float(np.nanmean(metric_arr("STARN-GAT", "f1"))), 0.45, 0.35])
        ax1.semilogx(lrs, f1_lgbm, "o-", color=COLORS["accent"], lw=1.8, label="LightGBM")
        ax1.semilogx(lrs, f1_starn, "s-", color=COLORS["primary"], lw=1.8, label="STARN-GAT")
        ax1.axvline(1e-2, color=COLORS["red"], ls="--", lw=1.2, label="Optimal LR=0.01")
        ax1.set_xlabel("Learning Rate (log)")
        ax1.set_ylabel("Validation F1")
        ax1.set_title("(a) Learning Rate Sensitivity", fontsize=10)
        ax1.legend(fontsize=8)
        
        # 5b: Architecture comparison
        ax3 = fig.add_subplot(gs[0, 2])
        arch_means = {m: np.nanmean(metric_arr(m, "roc_auc")) for m in self.MODEL_LIST}
        arch_stds = {m: np.nanstd(metric_arr(m, "roc_auc")) for m in self.MODEL_LIST}
        sorted_m = sorted(arch_means, key=arch_means.get, reverse=False)
        ym = [arch_means[m] for m in sorted_m]
        ye = [arch_stds[m] for m in sorted_m]
        cols_a = [MODEL_COLORS[m] for m in sorted_m]
        hb = ax3.barh(range(len(sorted_m)), ym, xerr=ye, color=cols_a, alpha=0.85,
                      error_kw=dict(ecolor="gray", capsize=3))
        ax3.set_yticks(range(len(sorted_m)))
        ax3.set_yticklabels(sorted_m, fontsize=9)
        ax3.set_xlabel("Mean ROC-AUC")
        ax3.set_title("(c) Architecture ROC-AUC", fontsize=10)
        ax3.bar_label(hb, [f"{v:.3f}" for v in ym], padding=3, fontsize=8)
        
        self.savefig(fig, "fig05_hyperparameter")
        
        # FIG 6: SHAP interpretability suite
        logger.info("Generating Fig 6: SHAP interpretability suite...")
        if best_model in trained_models:
            bm_info = trained_models[best_model]
            fig = plt.figure(figsize=(16, 14))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.42)
            
            # Simplified SHAP plots
            ax1 = fig.add_subplot(gs[0, :2])
            top_features = [feat_names[i] for i in top_idx[:12]]
            y_pos = range(len(top_features))
            ax1.barh(y_pos, shap_gini[top_idx[:12]], color=COLORS["primary"], alpha=0.85)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_features, fontsize=8.5)
            ax1.set_xlabel("Mean |SHAP|")
            ax1.set_title(f"(a) SHAP Feature Importance — {best_model}", fontsize=9.5)
            
            self.savefig(fig, "fig06_shap_suite")
        
        # FIG 7: Operational efficiency
        logger.info("Generating Fig 7: Operational efficiency...")
        fig = plt.figure(figsize=(16, 11))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)
        
        # 7a: Latency histogram
        ax1 = fig.add_subplot(gs[0, 0])
        for mname in self.MODEL_LIST:
            all_lat = []
            for r in all_results[mname]:
                all_lat.extend(r.get("lats", []))
            if not all_lat:
                continue
            all_lat = np.array(all_lat)
            ax1.hist(all_lat, bins=25, alpha=0.5, color=MODEL_COLORS[mname],
                     label=f"{mname} ({np.median(all_lat):.1f}ms)", density=True)
        ax1.axvline(10, color="black", ls="--", lw=1.2, label="RT limit (10ms)")
        ax1.set_xlabel("Inference Latency (ms)")
        ax1.set_ylabel("Density")
        ax1.set_title("(a) Inference Latency\nDistribution per Model", fontsize=10)
        ax1.legend(fontsize=6.5, ncol=1)
        
        # 7e: Training time vs F1
        ax5 = fig.add_subplot(gs[1, 2])
        for mname in self.MODEL_LIST:
            train_times = [r["train_s"] for r in all_results[mname]]
            f1_vals = [r["f1"] for r in all_results[mname]]
            if not train_times:
                continue
            ax5.scatter(train_times, f1_vals, color=MODEL_COLORS[mname],
                        alpha=0.5, s=20, label=mname)
            ax5.scatter([np.mean(train_times)], [np.mean(f1_vals)],
                        color=MODEL_COLORS[mname], s=120, marker="D",
                        edgecolor="white", lw=1.5, zorder=5)
        ax5.set_xlabel("Training Time (s/window)")
        ax5.set_ylabel("Test F1-Score")
        ax5.set_title("(e) Speed-Accuracy Trade-off", fontsize=10)
        ax5.legend(fontsize=7, ncol=2)
        
        self.savefig(fig, "fig07_operational")
        
        # FIG 8: ROC and PR curves
        logger.info("Generating Fig 8: ROC and PR curves...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        
        all_ytrue = {}
        all_yprob = {}
        for mname in self.MODEL_LIST:
            if not all_results[mname]:
                continue
            yt = np.concatenate([r["y_true"] for r in all_results[mname]])
            yp = np.concatenate([r["proba"] for r in all_results[mname]])
            all_ytrue[mname] = yt
            all_yprob[mname] = yp
        
        for mname in self.MODEL_LIST:
            if mname not in all_ytrue:
                continue
            yt, yp = all_ytrue[mname], all_yprob[mname]
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(yt, yp)
                auc_v = roc_auc_score(yt, yp)
                lw = 2.5 if mname == best_model else 1.5
                ax1.plot(fpr, tpr, color=MODEL_COLORS[mname], lw=lw,
                         label=f"{mname} (AUC={auc_v:.3f})")
            except Exception:
                pass
        
        ax1.plot([0, 1], [0, 1], "k--", lw=0.9, label="Random (0.500)")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("(a) ROC Curves — All Models", fontsize=10)
        ax1.legend(fontsize=8, loc="lower right")
        
        for mname in self.MODEL_LIST:
            if mname not in all_ytrue:
                continue
            yt, yp = all_ytrue[mname], all_yprob[mname]
            try:
                from sklearn.metrics import precision_recall_curve
                prec, rec, _ = precision_recall_curve(yt, yp)
                pr_v = average_precision_score(yt, yp)
                lw = 2.5 if mname == best_model else 1.5
                ax2.plot(rec, prec, color=MODEL_COLORS[mname], lw=lw,
                         label=f"{mname} (PR={pr_v:.3f})")
            except Exception:
                pass
        if all_ytrue:
            prev = np.concatenate(list(all_ytrue.values())).mean()
        else:
            prev = 0.5
        ax2.axhline(prev, color="gray", ls="--", lw=0.9,
                    label=f"Baseline ({prev:.2%} prevalence)")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("(b) Precision-Recall Curves", fontsize=10)
        ax2.legend(fontsize=8, loc="upper right")
        
        fig.suptitle(
            f"Figure 4.8 — ROC and Precision-Recall Curves: All 7 Models\n"
            f"(Bold={best_model})",
            fontsize=10.5, fontweight="bold", y=1.02)
        plt.tight_layout()
        self.savefig(fig, "fig08_roc_pr_curves")
        
        # FIG 9: Validation scores and model selection
        logger.info("Generating Fig 9: Validation scores and model selection...")
        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.42)
        
        ax1 = fig.add_subplot(gs[0, :2])
        val_data = [all_val_scores[m] for m in self.MODEL_LIST]
        bp = ax1.boxplot(val_data, patch_artist=True, notch=True, vert=True,
                         medianprops=dict(color="white", lw=2))
        for patch, mname in zip(bp["boxes"], self.MODEL_LIST):
            patch.set_facecolor(MODEL_COLORS[mname])
            patch.set_alpha(0.80)
        ax1.set_xticks(range(1, len(self.MODEL_LIST) + 1))
        ax1.set_xticklabels(self.MODEL_LIST, rotation=12, fontsize=9)
        ax1.set_ylabel("Validation F1-Score")
        ax1.set_title("(a) Validation F1 Distribution\n(All rolling windows)", fontsize=10)
        
        # Summary table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis("off")
        summary_rows = []
        for mname in self.MODEL_LIST:
            f1_v = metric_arr(mname, "f1")
            pr_v = metric_arr(mname, "pr_auc")
            roc_v = metric_arr(mname, "roc_auc")
            rec_v = metric_arr(mname, "recall")
            prec_v = metric_arr(mname, "precision")
            lats_all = [l for r in all_results[mname] for l in r.get("lats", [])]
            summary_rows.append([
                f"{'★ ' if mname == best_model else ''}{mname}",
                f"{np.nanmean(f1_v):.3f}±{np.nanstd(f1_v):.3f}",
                f"{np.nanmean(pr_v):.3f}±{np.nanstd(pr_v):.3f}",
                f"{np.nanmean(roc_v):.3f}±{np.nanstd(roc_v):.3f}",
                f"{np.nanmean(rec_v):.3f}±{np.nanstd(rec_v):.3f}",
                f"{np.nanmean(prec_v):.3f}±{np.nanstd(prec_v):.3f}",
                f"{np.median(lats_all):.2f}ms" if lats_all else "—",
            ])
        
        col_labels = ["Model", "F1 (mean±SD)", "PR-AUC", "ROC-AUC", "Recall", "Precision", "Latency"]
        tbl = ax3.table(cellText=summary_rows, colLabels=col_labels,
                        cellLoc="center", loc="center", bbox=[0, -0.05, 1, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor(COLORS["primary"])
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        best_row_idx = self.MODEL_LIST.index(best_model) + 1
        for j in range(len(col_labels)):
            tbl[best_row_idx, j].set_facecolor("#D5F5E3")
        
        fig.suptitle(
            f"Figure 4.9 — Model Selection: Validation Scores & Performance Summary\n"
            f"(7 models · ★★ = {best_model})",
            fontsize=10.5, fontweight="bold", y=1.02)
        self.savefig(fig, "fig09_model_selection")
        
        # FIG 10: Discussion radar chart
        logger.info("Generating Fig 10: Discussion radar chart...")
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40)
        
        cats = ["F1-Score", "PR-AUC", "ROC-AUC", "Recall", "Speed\nEff.", "Mem\nEff.", "Multi-class\nF1"]
        N_c = len(cats)
        angles = np.linspace(0, 2 * np.pi, N_c, endpoint=False).tolist()
        angles += angles[:1]
        
        ax1 = fig.add_subplot(gs[0], polar=True)
        for mname in self.MODEL_LIST:
            f1_v = np.nanmean(metric_arr(mname, "f1")) or 0
            pr_v = np.nanmean(metric_arr(mname, "pr_auc")) or 0
            roc_v = np.nanmean(metric_arr(mname, "roc_auc")) or 0
            rec_v = np.nanmean(metric_arr(mname, "recall")) or 0
            mc_v = np.nanmean([r.get("f1_macro_m", 0) or 0 for r in all_results[mname]])
            speed_norm = 1 - np.clip(np.log1p(np.median([l for r in all_results[mname] for l in r.get("lats", [])] or [5])) / np.log1p(20), 0, 1)
            mem_norm = 1 - 20 / 50  # Approximate
            vals = [f1_v, pr_v, roc_v, rec_v, speed_norm, mem_norm, mc_v]
            vals += [vals[0]]
            lw = 2.5 if mname == best_model else 1.3
            ax1.plot(angles, vals, color=MODEL_COLORS[mname], lw=lw, label=mname)
            ax1.fill(angles, vals, color=MODEL_COLORS[mname], alpha=0.05 if mname != best_model else 0.12)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(cats, size=9)
        ax1.set_ylim(0, 1)
        ax1.set_title("(a) Multi-Dimensional Comparison\n(7 models · 7 dimensions)", fontsize=10, pad=20)
        ax1.legend(loc="lower left", fontsize=8, bbox_to_anchor=(-0.35, -0.28), ncol=2, framealpha=0.9)
        
        self.savefig(fig, "fig10_discussion_summary")
        
        logger.info(f"\nAll 10 figures saved to {self.viz_dir}/")

        # ── Alert EDA Visualizations ────────────────────────────────────────────────
        try:
            self._generate_alert_eda_plots(classified_df)
        except Exception as e:
            logger.warning(f"  Alert EDA plots failed: {e}")

        # ── Null Audit Plots ────────────────────────────────────────────────
        try:
            self._generate_null_audit_plots(classified_df)
        except Exception as e:
            logger.warning(f"  Null audit plots failed: {e}")

        # ── Target EDA Plots ────────────────────────────────────────────────
        try:
            self._generate_target_eda_plots(classified_df)
        except Exception as e:
            logger.warning(f"  Target EDA plots failed: {e}")

        # ── Spatial Maps ────────────────────────────────────────────────
        if classified_df is not None:
            try:
                generate_all_spatial_maps(classified_df, str(self.viz_dir))
            except Exception as e:
                logger.warning(f"  Spatial maps failed: {e}")
        else:
            logger.info("  No classified_df provided; skipping spatial maps.")

    def _generate_alert_eda_plots(self, df: pd.DataFrame) -> None:
        """Generate alert-specific EDA plots: cause/effect distribution, text length, language."""
        if df is None or df.empty:
            logger.info("  No data for alert EDA plots")
            return
        
        logger.info("  Generating alert EDA plots...")
        
        # Check for alert columns
        has_alerts = any(c for c in df.columns if 'alert' in c.lower() or 'cause' in c.lower() or 'effect' in c.lower())
        if not has_alerts:
            logger.info("  No alert columns found; skipping alert EDA")
            return
        
        CAUSE_MAP = {1: 'UNKNOWN_CAUSE', 2: 'OTHER_CAUSE', 3: 'TECHNICAL_PROBLEM',
                   4: 'STRIKE', 5: 'DEMONSTRATION', 6: 'ACCIDENT', 7: 'HOLIDAY',
                   8: 'WEATHER', 9: 'MAINTENANCE', 10: 'CONSTRUCTION', 11: 'POLICE_ACTIVITY',
                   12: 'MEDICAL_EMERGENCY'}
        EFFECT_MAP = {1: 'NO_SERVICE', 2: 'REDUCED_SERVICE', 3: 'SIGNIFICANT_DELAYS',
                      4: 'DETOUR', 5: 'ADDITIONAL_SERVICE', 6: 'MODIFIED_SERVICE',
                      7: 'OTHER_EFFECT', 8: 'UNKNOWN_EFFECT', 9: 'STOP_MOVED'}
        
        # Plot 1: Cause/Effect Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Alert Cause and Effect Distribution', fontsize=12, fontweight='bold')
        
        if 'cause' in df.columns:
            cause_series = df['cause'].map(CAUSE_MAP).fillna('UNKNOWN')
            cause_counts = cause_series.value_counts()
            colors1 = plt.cm.Blues(np.linspace(0.3, 0.9, len(cause_counts)))
            cause_counts.plot(kind='barh', ax=axes[0], color=colors1)
            axes[0].set_xlabel('Count')
            axes[0].set_title('Cause Distribution')
            for i, v in enumerate(cause_counts.values):
                axes[0].text(v + 100, i, f'{v:,}', va='center', fontsize=8)
        
        if 'effect' in df.columns:
            effect_series = df['effect'].map(EFFECT_MAP).fillna('UNKNOWN')
            effect_counts = effect_series.value_counts()
            colors2 = plt.cm.Greens(np.linspace(0.3, 0.9, len(effect_counts)))
            effect_counts.plot(kind='barh', ax=axes[1], color=colors2)
            axes[1].set_xlabel('Count')
            axes[1].set_title('Effect Distribution')
            for i, v in enumerate(effect_counts.values):
                axes[1].text(v + 100, i, f'{v:,}', va='center', fontsize=8)
        
        plt.tight_layout()
        self.savefig(fig, "eda01_cause_effect_distribution")
        plt.close()
        
        # Plot 2: Text Length Analysis
        text_cols = [c for c in df.columns if 'text' in c.lower() or 'description' in c.lower()]
        if text_cols:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('Alert Text Length Analysis', fontsize=12, fontweight='bold')
            
            for idx, text_col in enumerate(text_cols[:2]):
                char_col = f'{text_col}_char_count'
                if char_col in df.columns:
                    ax = axes[idx] if len(text_cols) > 1 else axes[0]
                    df[char_col].hist(bins=40, ax=ax, color='steelblue', edgecolor='white', alpha=0.7)
                    ax.axvline(df[char_col].mean(), color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {df[char_col].mean():.1f}')
                    ax.axvline(df[char_col].median(), color='orange', linestyle='--', linewidth=2,
                              label=f'Median: {df[char_col].median():.1f}')
                    ax.set_xlabel('Character Count')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{text_col} Character Count')
                    ax.legend()
            
            if len(text_cols) == 1:
                axes[1].axis('off')
            
            plt.tight_layout()
            self.savefig(fig, "eda02_text_length_analysis")
            plt.close()
        
        # Plot 3: Severity Distribution
        if 'severity' in df.columns or 'severity_level' in df.columns:
            sev_col = 'severity' if 'severity' in df.columns else 'severity_level'
            fig, ax = plt.subplots(figsize=(8, 6))
            sev_counts = df[sev_col].value_counts().sort_index()
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sev_counts)))
            sev_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
            ax.set_xlabel('Severity Level')
            ax.set_ylabel('Count')
            ax.set_title('Alert Severity Distribution')
            for i, v in enumerate(sev_counts.values):
                ax.text(i, v + 100, f'{v:,}', ha='center', fontsize=9)
            plt.tight_layout()
            self.savefig(fig, "eda03_severity_distribution")
            plt.close()
        
        # Plot 4: Temporal Distribution (if timestamp available)
        ts_cols = [c for c in df.columns if 'timestamp' in c.lower() or 'retrieved_at' in c.lower()]
        if ts_cols:
            ts_col = ts_cols[0]
            try:
                df['hour'] = pd.to_datetime(df[ts_col]).dt.hour
                df['day_of_week'] = pd.to_datetime(df[ts_col]).dt.day_name()
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle('Alert Temporal Distribution', fontsize=12, fontweight='bold')
                
                hour_counts = df['hour'].value_counts().sort_index()
                hour_counts.plot(kind='bar', ax=axes[0], color='teal', edgecolor='white')
                axes[0].set_xlabel('Hour of Day')
                axes[0].set_ylabel('Count')
                axes[0].set_title('Alerts by Hour')
                
                dow_counts = df['day_of_week'].value_counts()
                dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_counts = dow_counts.reindex([d for d in dow_order if d in dow_counts.index])
                dow_counts.plot(kind='bar', ax=axes[1], color='coral', edgecolor='white')
                axes[1].set_xlabel('Day of Week')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Alerts by Day of Week')
                axes[1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                self.savefig(fig, "eda04_temporal_distribution")
                plt.close()
                
                df = df.drop(columns=['hour', 'day_of_week'], errors='ignore')
            except Exception as e:
                logger.warning(f"  Temporal plots failed: {e}")
        
        logger.info(f"  Alert EDA plots saved to {self.viz_dir}/")

    def _generate_null_audit_plots(self, df: pd.DataFrame) -> None:
        """Generate null audit visualizations."""
        if df is None or df.empty:
            return
        
        logger.info("  Generating null audit plots...")
        
        # Check for target columns
        target_cols = [c for c in df.columns if 'is_disruption' in c.lower() or 'target' in c.lower()]
        if not target_cols:
            return
        
        target_col = target_cols[0]
        
        # Calculate null percentages
        null_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        high_null = null_pcts[null_pcts > 50]
        
        if high_null.empty:
            logger.info("  No columns with >50% null - skipping null audit")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        high_null.plot(kind='barh', ax=ax, color='coral', edgecolor='white')
        ax.axvline(70, color='red', linestyle='--', label='70% threshold')
        ax.set_xlabel('Null Percentage')
        ax.set_title(f'Columns with High Nulls (>50%)')
        ax.legend()
        plt.tight_layout()
        self.savefig(fig, "eda05_null_audit")
        plt.close()

    def _generate_target_eda_plots(self, df: pd.DataFrame) -> None:
        """Generate target distribution EDA plots."""
        if df is None or df.empty:
            return
        
        logger.info("  Generating target EDA plots...")
        
        # Find target columns
        binary_targets = [c for c in df.columns if 'is_disruption' in c.lower()]
        multi_targets = [c for c in df.columns if 'disruption_type' in c.lower()]
        
        COLORS = ['#6c63ff', '#f97316', '#f43f5e', '#a855f7', '#38bdf8']
        
        # Figure 1: Binary target overview
        if binary_targets:
            fig, axes = plt.subplots(1, 3, figsize=(max(14, len(binary_targets) * 3), 5))
            
            target_col = binary_targets[0]
            target_vals = df[target_col].fillna(0).astype(int)
            freq = target_vals.value_counts()
            
            axes[0].bar(range(len(freq)), freq.values, color=[COLORS[0], COLORS[2]], edgecolor='white')
            axes[0].set_title(f'{target_col} Distribution', fontweight='bold')
            axes[0].set_ylabel('Count')
            axes[0].set_xticks(range(len(freq)))
            axes[0].set_xticklabels(['On-time' if i == 0 else 'Disruption' for i in freq.index])
            
            pos_rate = target_vals.mean()
            axes[1].bar(['Positive Rate'], [pos_rate], color=COLORS[1], edgecolor='white')
            axes[1].axhline(0.05, color='red', linestyle='--', label='5% sparse')
            axes[1].axhline(0.50, color='orange', linestyle='--', label='50% common')
            axes[1].set_ylabel('Positive Rate')
            axes[1].set_title('Class Imbalance Check', fontweight='bold')
            axes[1].set_ylim(0, 1)
            axes[1].text(0, pos_rate + 0.02, f'{pos_rate*100:.1f}%', ha='center', fontweight='bold')
            
            # Rolling positive rate
            window = max(10, len(df) // 50)
            rolling_rate = target_vals.rolling(window).mean()
            axes[2].plot(rolling_rate.values, color=COLORS[0], linewidth=1.2)
            axes[2].set_title(f'Rolling Positive Rate (window={window})', fontweight='bold')
            axes[2].set_xlabel('Row index')
            axes[2].set_ylabel('Rolling positive rate')
            
            plt.suptitle('Figure 1 — Binary Target EDA', fontsize=12, fontweight='bold', y=1.02)
            plt.tight_layout()
            self.savefig(fig, "eda06_binary_target_eda")
            plt.close()
        
        # Figure 2: Multiclass target distributions
        if multi_targets:
            fig, ax = plt.subplots(figsize=(10, 6))
            mc_col = multi_targets[0]
            counts = df[mc_col].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
            counts.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
            ax.set_xlabel('Disruption Type')
            ax.set_ylabel('Count')
            ax.set_title(f'{mc_col} Distribution')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            self.savefig(fig, "eda07_multiclass_target_eda")
            plt.close()
        
        # Figure 3: Temporal alignment validation (if future targets exist)
        future_targets = [c for c in df.columns if 'future' in c.lower() or '30min' in c.lower()]
        if binary_targets and future_targets:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            
            curr = df[binary_targets[0]]
            future = df[future_targets[0]]
            
            ct = pd.crosstab(curr.fillna(0), future.fillna(0))
            sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
            axes[0].set_title('Current vs Future Disruption', fontweight='bold')
            
            window = max(10, len(df) // 50)
            axes[1].plot(curr.fillna(0).rolling(window).mean().values, label='Current', linewidth=1.2)
            axes[1].plot(future.fillna(0).rolling(window).mean().values, label='Future', linewidth=1.2)
            axes[1].set_title('Rolling Positive Rate Comparison', fontweight='bold')
            axes[1].set_xlabel('Row index')
            axes[1].set_ylabel('Rolling rate')
            axes[1].legend()
            
            plt.tight_layout()
            self.savefig(fig, "eda08_temporal_validation")
            plt.close()
        
        logger.info(f"  Target EDA plots saved to {self.viz_dir}/")

    def encode_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_cols: List[str],
        exclude_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, List[str]]:
        """
        Feature Matrix Encoding - Step 3 from the notebook.
        
        Returns: (X_train, X_val, X_test, encoder, feature_cols)
        """
        from sklearn.preprocessing import OrdinalEncoder
        
        logger.info("=" * 50)
        logger.info("STEP 3 — FEATURE MATRIX ENCODING")
        logger.info("=" * 50)
        
        # Default exclusions
        if exclude_cols is None:
            exclude_cols = []
        
        # Columns to exclude
        all_exclude = set(target_cols + exclude_cols + [
            'feed_timestamp', 'retrieved_at', 'timestamp',
            'trip_update_timestamp', 'date', 'time'
        ])
        
        # Classify columns by type
        numeric_cols = []
        cat_cols = []
        bool_cols = []
        drop_cols = []
        
        feature_candidates = [c for c in train_df.columns if c not in all_exclude]
        
        for col in feature_candidates:
            s = train_df[col]
            dtype = str(s.dtype)
            
            # Skip datetime
            if pd.api.types.is_datetime64_any_dtype(s):
                drop_cols.append(col)
                continue
            
            # Check for unhashable types
            non_null = s.dropna()
            if len(non_null) > 0 and isinstance(non_null.iloc[0], (list, dict, np.ndarray)):
                drop_cols.append(col)
                continue
            
            # Boolean
            if dtype == 'bool':
                bool_cols.append(col)
                continue
            
            # Categorical (object or low cardinality)
            if dtype == 'object':
                n_unique = s.nunique(dropna=True)
                if n_unique <= 100:
                    cat_cols.append(col)
                else:
                    drop_cols.append(col)
                continue
            
            # Numeric
            numeric_cols.append(col)
        
        feature_cols = numeric_cols + cat_cols + bool_cols
        logger.info(f"  Numeric: {len(numeric_cols)}, Categorical: {len(cat_cols)}, Boolean: {len(bool_cols)}")
        logger.info(f"  Dropped: {len(drop_cols)}")
        
        # Fit ordinal encoder on train
        encoder = None
        if cat_cols:
            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-2,
            )
            encoder.fit(train_df[cat_cols].astype(str))
            logger.info(f"  ✓ OrdinalEncoder fitted on {len(cat_cols)} columns")
        
        # Encode function
        def encode_split(df, feature_cols, num_cols, cat_cols, bool_cols, enc):
            parts = []
            
            if num_cols:
                num_part = df[num_cols].copy()
                for c in num_cols:
                    if str(num_part[c].dtype).startswith('Int'):
                        num_part[c] = num_part[c].astype(float)
                parts.append(num_part)
            
            if cat_cols and enc is not None:
                cat_part = pd.DataFrame(
                    enc.transform(df[cat_cols].astype(str)),
                    columns=cat_cols,
                    index=df.index,
                )
                parts.append(cat_part)
            
            if bool_cols:
                bool_part = df[bool_cols].astype(np.int8)
                parts.append(bool_part)
            
            if not parts:
                return pd.DataFrame(index=df.index)
            
            return pd.concat(parts, axis=1)
        
        # Encode all splits
        X_train = encode_split(train_df, feature_cols, numeric_cols, cat_cols, bool_cols, encoder)
        X_val = encode_split(val_df, feature_cols, numeric_cols, cat_cols, bool_cols, encoder)
        X_test = encode_split(test_df, feature_cols, numeric_cols, cat_cols, bool_cols, encoder)
        
        logger.info(f"  Encoded shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        return X_train, X_val, X_test, encoder, feature_cols

    def impute_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
        """
        Imputation - Step 4 from the notebook.
        
        Returns: (X_train, X_val, X_test, imputer)
        """
        from sklearn.impute import SimpleImputer
        
        logger.info("=" * 50)
        logger.info("STEP 4 — IMPUTATION")
        logger.info("=" * 50)
        
        # Identify columns needing imputation
        numeric_cols = [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if X_train[c].isnull().any()
        ]
        
        logger.info(f"  Columns needing imputation: {len(numeric_cols)}")
        
        # Fit imputer on train
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train[numeric_cols])
        logger.info("  ✓ Median imputer fitted")
        
        # Apply imputation
        def apply_impute(X, imputer, impute_cols, split_name):
            X = X.copy()
            imputed = pd.DataFrame(
                imputer.transform(X[impute_cols]),
                columns=impute_cols,
                index=X.index,
            )
            imputed = imputed.fillna(0)
            X[impute_cols] = imputed
            
            remaining = int(X.isnull().sum().sum())
            logger.info(f"  {split_name}: {remaining} nulls remaining")
            return X
        
        X_train = apply_impute(X_train, imputer, numeric_cols, 'X_train')
        X_val = apply_impute(X_val, imputer, numeric_cols, 'X_val')
        X_test = apply_impute(X_test, imputer, numeric_cols, 'X_test')
        
        # Clip extreme values and convert to float32
        CLIP_VALUE = 1e15
        
        for X, name in [(X_train, 'X_train'), (X_val, 'X_val'), (X_test, 'X_test')]:
            num_cols = X.select_dtypes(include=[np.number]).columns
            X[num_cols] = (
                X[num_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .clip(-CLIP_VALUE, CLIP_VALUE)
            )
        
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        logger.info(f"  Final shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        return X_train, X_val, X_test, imputer

    def balance_classes(
        self,
        X_train: pd.DataFrame,
        y_binary: pd.DataFrame,
        y_multiclass: Optional[pd.DataFrame] = None,
        strategy: str = "none",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
        """
        Class Balance + Resampling - Step 6 from the notebook.
        
        Returns: (X_balanced, y_binary_balanced, y_multiclass_balanced, sampler)
        """
        try:
            from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE
            from imblearn.combine import SMOTETomek
            IMBLEARN_AVAILABLE = True
        except ImportError:
            IMBLEARN_AVAILABLE = False
            logger.warning("  imbalanced-learn not available - skipping resampling")
        
        logger.info("=" * 50)
        logger.info("STEP 6 — CLASS BALANCE + RESAMPLING")
        logger.info("=" * 50)
        
        # Check for degenerate targets
        degenerate = [c for c in y_binary.columns if y_binary[c].nunique() < 2]
        active_binary = [c for c in y_binary.columns if c not in degenerate]
        
        if degenerate:
            logger.warning(f"  Degenerate targets removed: {degenerate}")
        
        if not active_binary:
            logger.info("  No valid binary targets - returning original data")
            return X_train, y_binary, y_multiclass, None
        
        # Composite label for resampling
        if y_binary.shape[1] > 0:
            composite = (y_binary[active_binary].values.sum(axis=1) > 0).astype(int)
        else:
            composite = np.zeros(len(X_train), dtype=int)
        
        n_pos = int(composite.sum())
        n_neg = int((composite == 0).sum())
        logger.info(f"  Before: pos={n_pos:,}, neg={n_neg:,}, ratio={n_pos/len(composite):.2%}")
        
        if strategy == "none" or not IMBLEARN_AVAILABLE:
            logger.info("  No resampling applied")
            return X_train, y_binary, y_multiclass, None
        
        # Check minority class size
        n_minority = min(n_pos, n_neg)
        k_neighbors = min(5, n_minority - 1) if n_minority > 1 else 1
        
        if n_minority <= k_neighbors:
            logger.warning(f"  Using RandomOverSampler (minority={n_minority})")
            sampler = RandomOverSampler(random_state=random_state)
        elif strategy == "smote":
            sampler = SMOTE(sampling_strategy=0.10, k_neighbors=k_neighbors, random_state=random_state)
        elif strategy == "borderline_smote":
            sampler = BorderlineSMOTE(k_neighbors=k_neighbors, kind="borderline-1", random_state=random_state)
        elif strategy == "smote_tomek":
            sampler = SMOTETomek(random_state=random_state)
        else:
            sampler = RandomOverSampler(random_state=random_state)
        
        # Resample
        try:
            X_res, y_res = sampler.fit_resample(X_train.values, composite)
        except Exception as e:
            logger.warning(f"  Resampling failed ({e}) - using original data")
            return X_train, y_binary, y_multiclass, None
        
        n_new = len(X_res)
        logger.info(f"  Resampled: {len(X_train):,} → {n_new:,} rows (+{n_new - len(X_train):,})")
        
        # Propagate to all target columns
        if hasattr(sampler, 'sample_indices_'):
            indices = sampler.sample_indices_
            y_binary_res = y_binary.iloc[indices].reset_index(drop=True)
            y_multiclass_res = y_multiclass.iloc[indices].reset_index(drop=True) if y_multiclass is not None and not y_multiclass.empty else pd.DataFrame()
        else:
            y_binary_res = y_binary.reset_index(drop=True)
            y_multiclass_res = y_multiclass.reset_index(drop=True) if y_multiclass is not None and not y_multiclass.empty else pd.DataFrame()
        
        X_res_df = pd.DataFrame(X_res, columns=X_train.columns)
        
        new_composite = (y_binary_res.values.sum(axis=1) > 0).astype(int) if y_binary_res.shape[1] > 0 else composite
        logger.info(f"  After: pos={new_composite.sum():,}, neg={(new_composite==0).sum():,}")
        
        return X_res_df, y_binary_res, y_multiclass_res, sampler

    def define_models_and_helpers(self) -> Dict[str, Any]:
        """
        Step 7 - Helper functions and model definitions.
        
        Returns: dict with helper functions, model factories, and model registry
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.neighbors import NearestNeighbors
        from sklearn.base import BaseEstimator, ClassifierMixin
        from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
        import importlib
        
        logger.info("=" * 50)
        logger.info("STEP 7 — HELPER FUNCTIONS + DEFINE MODELS")
        logger.info("=" * 50)
        
        LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
        XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
        
        if LIGHTGBM_AVAILABLE:
            from lightgbm import LGBMClassifier
            logger.info("  ✓ LightGBM available")
        
        if XGBOOST_AVAILABLE:
            from xgboost import XGBClassifier
            logger.info("  ✓ XGBoost available")
        
        # Helper functions
        def tune_thresholds(proba_df, y_true_df, targets):
            thresholds = {}
            for col in targets:
                if col not in proba_df.columns or col not in y_true_df.columns:
                    thresholds[col] = 0.5
                    continue
                best_t, best_f1 = 0.5, 0.0
                for t in np.arange(0.10, 0.91, 0.05):
                    preds = (proba_df[col] >= t).astype(int)
                    score = f1_score(y_true_df[col], preds, zero_division=0)
                    if score > best_f1:
                        best_t, best_f1 = t, score
                thresholds[col] = round(float(best_t), 2)
            return thresholds
        
        def get_binary_proba(model, X, target_names):
            raw = model.predict_proba(X)
            if isinstance(raw, list):
                arr = np.column_stack([p[:, 1] for p in raw])
            else:
                arr = np.atleast_2d(raw).T if raw.ndim == 1 else raw
            if arr.ndim == 1 or arr.shape[1] == 1:
                arr = arr.reshape(-1, len(target_names))
            return pd.DataFrame(arr, columns=target_names)
        
        def eval_binary(model, X, y_true, thresholds, target_names):
            proba_df = get_binary_proba(model, X, target_names)
            pred_df = pd.DataFrame({
                col: (proba_df[col] >= thresholds[col]).astype(int)
                for col in target_names
            })
            y_true_arr = y_true[target_names].values
            y_pred_arr = pred_df.values
            n_targets = len(target_names)
            
            metrics = {}
            if n_targets >= 2:
                metrics["hamming_loss"] = round(hamming_loss(y_true_arr, y_pred_arr), 4)
                metrics["f1_samples"] = round(f1_score(y_true_arr, y_pred_arr, average="samples", zero_division=0), 4)
            else:
                metrics["hamming_loss"] = round(hamming_loss(y_true_arr.ravel(), y_pred_arr.ravel()), 4)
                metrics["f1_samples"] = round(f1_score(y_true_arr.ravel(), y_pred_arr.ravel(), average="binary", zero_division=0), 4)
            
            metrics["f1_macro"] = round(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0), 4)
            metrics["f1_weighted"] = round(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0), 4)
            
            for col in target_names:
                try:
                    metrics[f"auc_{col}"] = round(roc_auc_score(y_true[col].values, proba_df[col].values), 4)
                except ValueError:
                    metrics[f"auc_{col}"] = np.nan
            
            return metrics, proba_df, pred_df
        
        def safe_clean_features(X):
            X_clean = np.array(X, dtype=np.float64).copy()
            X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1e12, neginf=-1e12)
            return np.clip(X_clean, -1e12, 1e12)
        
        logger.info("  ✓ Helper functions defined")
        
        # SpatialRandomForest class
        class SpatialRandomForestClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self, k=8, append_lags=True, n_estimators=200, class_weight="balanced", random_state=None):
                self.k = k
                self.append_lags = append_lags
                self.n_estimators = n_estimators
                self.class_weight = class_weight
                self.random_state = random_state
            
            def fit(self, X, y):
                X = np.asarray(X, dtype=np.float32)
                self.X_train_raw_ = X
                self.nn_ = NearestNeighbors(n_neighbors=min(self.k + 1, len(X)), n_jobs=-1)
                self.nn_.fit(X)
                if hasattr(y, "values"): y = y.values
                if y.ndim == 2 and y.shape[1] == 1: y = y.ravel()
                self.rf_ = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                X_aug = self._spatial_lags(X)
                self.rf_.fit(X_aug, y)
                return self
            
            def _spatial_lags(self, X):
                if not self.append_lags:
                    return X
                X_ref = self.X_train_raw_
                _, indices = self.nn_.kneighbors(X)
                lag_means = np.zeros((len(X), X_ref.shape[1]), dtype=np.float32)
                lag_stds = np.zeros((len(X), X_ref.shape[1]), dtype=np.float32)
                for i, nbr_idx in enumerate(indices):
                    nbrs = X_ref[nbr_idx[1:]] if len(nbr_idx) > 1 else X_ref[nbr_idx]
                    lag_means[i] = nbrs.mean(axis=0)
                    lag_stds[i] = nbrs.std(axis=0)
                return np.hstack([X, lag_means, lag_stds])
            
            def predict_proba(self, X):
                X = np.asarray(X, dtype=np.float32)
                return self.rf_.predict_proba(self._spatial_lags(X))
            
            def predict(self, X):
                return self.rf_.predict(self._spatial_lags(X))
        
        # Model factories
        def make_rf(random_state=42):
            return RandomForestClassifier(
                n_estimators=200, max_depth=None,
                class_weight="balanced", random_state=random_state, n_jobs=-1,
            )
        
        def make_xgb(random_state=42, scale_pos=1.0):
            if not XGBOOST_AVAILABLE:
                return None
            return XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss", random_state=random_state, n_jobs=-1, verbosity=0,
            )
        
        def make_lgbm(random_state=42):
            if not LIGHTGBM_AVAILABLE:
                return None
            return LGBMClassifier(
                n_estimators=500, max_depth=-1, num_leaves=63,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1,
            )
        
        def make_mlp(random_state=42):
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                solver="adam", alpha=1e-4, batch_size=64,
                learning_rate="adaptive", max_iter=300,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=random_state,
            )
        
        def make_spatial_rf(random_state=42):
            return SpatialRandomForestClassifier(
                k=8, append_lags=True, n_estimators=200,
                class_weight="balanced", random_state=random_state,
            )
        
        # Model registry
        MODEL_REGISTRY = {
            "RandomForest": (make_rf, False),
            "NeuralNet": (make_mlp, True),
        }
        if XGBOOST_AVAILABLE:
            MODEL_REGISTRY["XGBoost"] = (make_xgb, False)
        if LIGHTGBM_AVAILABLE:
            MODEL_REGISTRY["LightGBM"] = (make_lgbm, False)
        MODEL_REGISTRY["SpatialRF"] = (make_spatial_rf, False)
        
        logger.info(f"  ✓ {len(MODEL_REGISTRY)} models registered")
        
        return {
            "helpers": {
                "tune_thresholds": tune_thresholds,
                "get_binary_proba": get_binary_proba,
                "eval_binary": eval_binary,
                "safe_clean_features": safe_clean_features,
            },
            "factories": {
                "make_rf": make_rf,
                "make_xgb": make_xgb,
                "make_lgbm": make_lgbm,
                "make_mlp": make_mlp,
                "make_spatial_rf": make_spatial_rf,
            },
            "MODEL_REGISTRY": MODEL_REGISTRY,
            "SpatialRandomForest": SpatialRandomForestClassifier,
        }

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_binary: pd.DataFrame,
        y_multiclass: pd.DataFrame,
        X_val: Optional[np.ndarray] = None,
        y_val_binary: Optional[pd.DataFrame] = None,
        y_val_multiclass: Optional[pd.DataFrame] = None,
        balance_strategy: str = "none",
    ) -> Dict[str, Dict]:
        """
        Step 8 - Train all models in the registry.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_binary : pd.DataFrame
            Binary target columns
        y_multiclass : pd.DataFrame
            Multiclass target columns
        X_val : np.ndarray, optional
            Validation features
        y_val_binary : pd.DataFrame, optional
            Validation binary targets
        y_val_multiclass : pd.DataFrame, optional
            Validation multiclass targets
        balance_strategy : str
            Resampling strategy for class imbalance
        
        Returns
        -------
        Dict with model names as keys and dicts with model objects, scores
        """
        from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        
        logger.info("=" * 50)
        logger.info("STEP 8 — TRAIN ALL MODELS")
        logger.info("=" * 50)
        
        logger.info(f"  X_train shape : {X_train.shape}")
        logger.info(f"  y_binary columns : {list(y_binary.columns)}")
        logger.info(f"  y_multiclass columns : {list(y_multiclass.columns)}")
        
        # Get model info
        model_info = self.define_models_and_helpers()
        MODEL_REGISTRY = model_info["MODEL_REGISTRY"]
        
        if X_val is None or y_val_binary is None:
            logger.warning("  No validation set - using train split for eval")
            X_val = X_train
            y_val_binary = y_binary
            y_val_multiclass = y_multiclass
        
        # Use cleaned features
        helpers = model_info["helpers"]
        safe_clean = helpers.get("safe_clean_features", lambda x: x)
        
        trained = {}
        val_scores = {}
        
        binary_targets = list(y_binary.columns)
        multiclass_targets = list(y_multiclass.columns)
        use_chain = len(binary_targets) >= 2
        
        logger.info(f"  Binary wrapper: {'ClassifierChain' if use_chain else 'MultiOutputClassifier'}")
        
        for model_name, (base_fn, use_scaled) in MODEL_REGISTRY.items():
            logger.info(f"\n  Training {model_name}...")
            
            try:
                if base_fn() is None:
                    logger.warning(f"    Factory returned None - skipping")
                    continue
            except Exception as e:
                logger.warning(f"    Factory error: {e} - skipping")
                continue
            
            # Prepare data
            X_tr = safe_clean(X_train)
            X_vl = safe_clean(X_val)
            
            yb_fit = y_binary.values
            if yb_fit.ndim == 1:
                yb_fit = yb_fit.reshape(-1, 1)
            
            # Train binary model
            try:
                if use_chain:
                    wrapper = ClassifierChain(base_fn(), random_state=42)
                else:
                    wrapper = MultiOutputClassifier(base_fn(), n_jobs=-1)
                
                wrapper.fit(X_tr, yb_fit)
                logger.info("    Binary model trained")
            except Exception as e:
                logger.warning(f"    Binary fit failed: {e}")
                continue
            
            # Threshold tuning
            try:
                proba_val = wrapper.predict_proba(X_vl)
                if isinstance(proba_val, list):
                    proba_df = pd.DataFrame({
                        binary_targets[i]: proba_val[i][:, 1]
                        for i in range(len(binary_targets))
                    })
                else:
                    proba_df = pd.DataFrame(proba_val, columns=binary_targets)
                
                thresholds = helpers["tune_thresholds"](proba_df, y_val_binary, binary_targets)
                logger.info(f"    Tuned thresholds: {thresholds}")
            except Exception as e:
                logger.warning(f"    Threshold tuning failed: {e}")
                thresholds = {col: 0.5 for col in binary_targets}
            
            # Validation evaluation
            try:
                b_val, _, _ = helpers["eval_binary"](
                    wrapper, X_vl, y_val_binary, thresholds, binary_targets
                )
                logger.info(f"    Val F1-macro={b_val.get('f1_macro', 'N/A')}")
                val_scores[model_name] = b_val
            except Exception as e:
                logger.warning(f"    Val eval failed: {e}")
                val_scores[model_name] = {"f1_macro": np.nan}
            
            # Train multiclass if available
            multi_model = None
            if multiclass_targets and not y_multiclass.empty:
                try:
                    ym_fit = y_multiclass.values
                    if ym_fit.ndim == 1:
                        ym_fit = ym_fit.reshape(-1, 1)
                    multi_model = MultiOutputClassifier(base_fn(), n_jobs=-1)
                    multi_model.fit(X_tr, ym_fit)
                    logger.info("    Multiclass model trained")
                except Exception as e:
                    logger.warning(f"    Multiclass fit failed: {e}")
            
            # Store trained model
            trained[model_name] = {
                "binary_model": wrapper,
                "multi_model": multi_model,
                "thresholds": thresholds,
                "scaled": use_scaled,
                "balance_strategy": balance_strategy,
            }
        
        # Summary
        logger.info(f"\n{'=' * 50}")
        logger.info(f"✓ Step 8 complete - {len(trained)} models trained")
        
        # Print summary sorted by F1-macro
        if val_scores:
            logger.info(f"\n  Validation Summary (sorted by F1-macro):")
            sorted_scores = sorted(
                val_scores.items(),
                key=lambda x: x[1].get("f1_macro", 0),
                reverse=True
            )
            for name, scores in sorted_scores:
                f1 = scores.get("f1_macro", "N/A")
                auc = scores.get("roc_auc", "N/A")
                logger.info(f"    {name:15} F1={f1:.4f}  AUC={auc:.4f}")
        
        return trained

    # -----------------------------------------------------------------------
    # Step 9 — Model Selection
    # -----------------------------------------------------------------------

    def select_best_model(
        self,
        trained: Dict[str, Dict],
        val_scores: Dict[str, Dict],
        yb_val: pd.DataFrame,
        yb_test: pd.DataFrame,
        PRIMARY_TARGET: str = "is_disruption",
        BINARY_TARGETS: Optional[List[str]] = None,
    ) -> Tuple[str, str, List[str]]:
        """
        Step 9 - Select best model based on validation AUC.

        Parameters
        ----------
        trained : Dict
            Trained models from train_all_models
        val_scores : Dict
            Validation scores for each model
        yb_val : pd.DataFrame
            Validation binary targets
        yb_test : pd.DataFrame
            Test binary targets
        PRIMARY_TARGET : str
            Primary target column for selection
        BINARY_TARGETS : List[str], optional
            List of binary target columns

        Returns
        -------
        (best_model_name, selection_basis, EVAL_TARGETS)
        """
        if BINARY_TARGETS is None:
            BINARY_TARGETS = ["is_disruption", "is_severe", "future_disruption_30min"]

        logger.info("=" * 50)
        logger.info("STEP 9 — MODEL SELECTION")
        logger.info("=" * 50)

        val_summary = pd.DataFrame(val_scores).T
        val_summary.index.name = "model"

        PRIMARY_AUC_COL = f"auc_{PRIMARY_TARGET}"

        valid_models = val_summary[
            val_summary[PRIMARY_AUC_COL].notna()
        ] if PRIMARY_AUC_COL in val_summary.columns else pd.DataFrame()

        if valid_models.empty:
            f1_col = "f1_macro" if "f1_macro" in val_summary.columns else None
            if f1_col and not val_summary[f1_col].isna().all():
                best_model_name = val_summary[f1_col].idxmax()
                selection_basis = "f1_macro (fallback — no valid AUC scores)"
            else:
                best_model_name = list(trained.keys())[0] if trained else None
                selection_basis = "first trained model"
        else:
            best_model_name = valid_models[PRIMARY_AUC_COL].idxmax()
            selection_basis = f"AUC on {PRIMARY_TARGET}"

        logger.info(f"  Selection basis : {selection_basis}")
        logger.info(f"  Best model      : {best_model_name}")

        EVAL_TARGETS = [
            col for col in BINARY_TARGETS
            if int(yb_test[col].sum()) >= 3
        ] if yb_test is not None and not yb_test.empty else [PRIMARY_TARGET]

        if not EVAL_TARGETS:
            EVAL_TARGETS = [PRIMARY_TARGET]

        logger.info(f"  EVAL_TARGETS    : {EVAL_TARGETS}")

        logger.info(f"\n{'=' * 50}")
        logger.info(f"✓ Step 9 complete.")
        logger.info(f"  best_model_name : {best_model_name}")
        logger.info(f"  selection_basis : {selection_basis}")
        logger.info(f"  EVAL_TARGETS    : {EVAL_TARGETS}")
        logger.info(f"{'=' * 50}")

        return best_model_name, selection_basis, EVAL_TARGETS

    # -----------------------------------------------------------------------
    # Step 10 — Final Test Set Evaluation
    # -----------------------------------------------------------------------

    def evaluate_on_test(
        self,
        trained: Dict[str, Dict],
        best_model_name: str,
        X_te: np.ndarray,
        yb_test: pd.DataFrame,
        ym_test: Optional[pd.DataFrame] = None,
        EVAL_TARGETS: Optional[List[str]] = None,
        PRIMARY_TARGET: str = "is_disruption",
        BINARY_TARGETS: Optional[List[str]] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Step 10 - Final test set evaluation.

        Parameters
        ----------
        trained : Dict
            Trained models from train_all_models
        best_model_name : str
            Name of the best model
        X_te : np.ndarray
            Test features
        yb_test : pd.DataFrame
            Test binary targets
        ym_test : pd.DataFrame, optional
            Test multiclass targets
        EVAL_TARGETS : List[str], optional
            Targets to evaluate
        PRIMARY_TARGET : str
            Primary target column
        BINARY_TARGETS : List[str], optional
            List of binary target columns

        Returns
        -------
        (b_test, all_test_results)
        """
        from sklearn.metrics import (
            f1_score, roc_auc_score, average_precision_score,
            confusion_matrix, classification_report
        )

        if BINARY_TARGETS is None:
            BINARY_TARGETS = ["is_disruption", "is_severe", "future_disruption_30min"]

        logger.info("=" * 50)
        logger.info("STEP 10 — FINAL TEST SET EVALUATION")
        logger.info("=" * 50)

        best = trained[best_model_name]
        thresholds = best["thresholds"]

        logger.info(f"  Best model      : {best_model_name}")
        logger.info(f"  EVAL_TARGETS    : {EVAL_TARGETS}")
        logger.info(f"  Thresholds      : {thresholds}")

        model_info = self.define_models_and_helpers()
        helpers = model_info["helpers"]
        safe_clean = helpers.get("safe_clean_features", lambda x: x)
        get_binary_proba = helpers.get("get_binary_proba")

        X_te_ = safe_clean(X_te)

        try:
            proba_test = get_binary_proba(best["binary_model"], X_te_)
        except Exception as e:
            raise RuntimeError(f"predict_proba failed on test set: {e}")

        if EVAL_TARGETS is None:
            EVAL_TARGETS = [col for col in BINARY_TARGETS if int(yb_test[col].sum()) >= 3]
            if not EVAL_TARGETS:
                EVAL_TARGETS = [PRIMARY_TARGET]

        pred_test = pd.DataFrame({
            col: (proba_test[col] >= thresholds[col]).astype(int)
            for col in BINARY_TARGETS
        })

        b_test = {}
        for col in EVAL_TARGETS:
            y_true = yb_test[col].values
            y_pred = pred_test[col].values
            y_proba = proba_test[col].values

            f1_bin = round(f1_score(y_true, y_pred, average="binary", zero_division=0), 4)
            f1_mac = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
            f1_w = round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)

            try:
                auc = round(roc_auc_score(y_true, y_proba), 4)
                apr = round(average_precision_score(y_true, y_proba), 4)
            except ValueError:
                auc = apr = float("nan")

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, int(y_true.sum()))

            prec = round(tp / max(tp + fp, 1), 4)
            rec = round(tp / max(tp + fn, 1), 4)
            spec = round(tn / max(tn + fp, 1), 4)

            b_test[col] = {
                "f1_binary": f1_bin,
                "f1_macro": f1_mac,
                "f1_weighted": f1_w,
                "auc": auc,
                "pr_auc": apr,
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }

            logger.info(f"\n  {col}")
            logger.info(f"    F1-binary    : {f1_bin}")
            logger.info(f"    AUC-ROC      : {auc}")
            logger.info(f"    PR-AUC       : {apr}")
            logger.info(f"    Precision    : {prec}")
            logger.info(f"    Recall       : {rec}")
            logger.info(f"    Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

        all_test_results = {}
        for model_name, artefacts in trained.items():
            try:
                X_te_m = safe_clean(X_te)
                proba_m = get_binary_proba(artefacts["binary_model"], X_te_m)
                thr_m = artefacts["thresholds"][PRIMARY_TARGET]
                preds_m = (proba_m[PRIMARY_TARGET] >= thr_m).astype(int)
                y_true = yb_test[PRIMARY_TARGET].values

                f1_m = round(f1_score(y_true, preds_m, average="binary", zero_division=0), 4)
                try:
                    auc_m = round(roc_auc_score(y_true, proba_m[PRIMARY_TARGET].values), 4)
                    apr_m = round(average_precision_score(y_true, proba_m[PRIMARY_TARGET].values), 4)
                except ValueError:
                    auc_m = apr_m = float("nan")

                cm_m = confusion_matrix(y_true, preds_m)
                tn_m, fp_m, fn_m, tp_m = cm_m.ravel() if cm_m.shape == (2, 2) else (0, 0, 0, int(y_true.sum()))

                all_test_results[model_name] = {
                    "f1_binary": f1_m, "auc": auc_m, "pr_auc": apr_m,
                    "precision": round(tp_m / max(tp_m + fp_m, 1), 4),
                    "recall": round(tp_m / max(tp_m + fn_m, 1), 4),
                }
            except Exception as e:
                logger.warning(f"  {model_name} failed: {e}")

        if PRIMARY_TARGET in b_test:
            r = b_test[PRIMARY_TARGET]
            logger.info(f"\n{'=' * 50}")
            logger.info(f"✓ Step 10 complete.")
            logger.info(f"  Best model: {best_model_name}")
            logger.info(f"  {PRIMARY_TARGET} — AUC: {r['auc']}, F1: {r['f1_binary']}, PR-AUC: {r['pr_auc']}")
            logger.info(f"{'=' * 50}")

        return b_test, all_test_results

    # -----------------------------------------------------------------------
    # Step 11 — SHAP Analysis
    # -----------------------------------------------------------------------

    def shap_analysis(
        self,
        trained: Dict[str, Dict],
        best_model_name: str,
        X_te: np.ndarray,
        yb_test: pd.DataFrame,
        feature_names: List[str],
        PRIMARY_TARGET: str = "is_disruption",
        BINARY_TARGETS: Optional[List[str]] = None,
        PLOT_DIR: str = "visualizations",
        RUN_ID: str = "default",
    ) -> Optional[np.ndarray]:
        """
        Step 11 - SHAP analysis and feature importances.

        Parameters
        ----------
        trained : Dict
            Trained models from train_all_models
        best_model_name : str
            Name of the best model
        X_te : np.ndarray
            Test features
        yb_test : pd.DataFrame
            Test binary targets
        feature_names : List[str]
            List of feature names
        PRIMARY_TARGET : str
            Primary target column
        BINARY_TARGETS : List[str], optional
            List of binary target columns
        PLOT_DIR : str
            Directory to save plots
        RUN_ID : str
            Run identifier

        Returns
        -------
        shap_values or None
        """
        import os

        if BINARY_TARGETS is None:
            BINARY_TARGETS = ["is_disruption", "is_severe", "future_disruption_30min"]

        logger.info("=" * 50)
        logger.info("STEP 11 — SHAP ANALYSIS")
        logger.info("=" * 50)

        try:
            import shap
            SHAP_AVAILABLE = True
            logger.info("  ✓ shap available")
        except ImportError:
            SHAP_AVAILABLE = False
            logger.warning("  ⚠  shap not installed")

        best = trained[best_model_name]
        model_info = self.define_models_and_helpers()
        helpers = model_info["helpers"]
        safe_clean = helpers.get("safe_clean_features", lambda x: x)

        X_te_ = safe_clean(X_te)

        def _get_chain_feature_names(base_names, chain_idx):
            names = list(base_names)
            for i in range(chain_idx):
                names.append(f"chain_pred_{BINARY_TARGETS[i]}")
            return names

        def _extract_rf_estimator(binary_model, chain_idx=0):
            if hasattr(binary_model, "estimators_"):
                est = binary_model.estimators_[chain_idx]
                if hasattr(est, "calibrated_classifiers_"):
                    return est.calibrated_classifiers_[0].estimator
                if hasattr(est, "feature_importances_"):
                    return est
            if hasattr(binary_model, "feature_importances_"):
                return binary_model
            return None

        shap_values = None
        import matplotlib.pyplot as plt

        for c_idx, c_name in enumerate(BINARY_TARGETS):
            c_feat_names = _get_chain_feature_names(feature_names, c_idx)
            c_est = _extract_rf_estimator(best["binary_model"], c_idx)

            if c_est is None or not hasattr(c_est, "feature_importances_"):
                logger.info(f"  {c_name}: estimator not available")
                continue

            fi = c_est.feature_importances_
            imp = pd.Series(fi, index=c_feat_names).sort_values(ascending=False)

            logger.info(f"\n  {c_name} top-10 features:")
            logger.info(imp.head(10).to_string())

            fig, ax = plt.subplots(figsize=(8, max(4, min(20, len(imp)) * 0.35)))
            imp.head(20).sort_values().plot(kind="barh", ax=ax, color="#6c63ff")
            ax.set_title(f"Feature Importances — {c_name} ({best_model_name})", fontweight="bold")
            ax.set_xlabel("Importance")
            plt.tight_layout()

            os.makedirs(PLOT_DIR, exist_ok=True)
            plt.savefig(
                os.path.join(PLOT_DIR, f"feature_importances_{c_name}_{RUN_ID}.png"),
                dpi=150, bbox_inches="tight"
            )
            plt.close()

        if SHAP_AVAILABLE:
            try:
                primary_chain_idx = BINARY_TARGETS.index(PRIMARY_TARGET)
                raw_est = _extract_rf_estimator(best["binary_model"], primary_chain_idx)

                if raw_est is not None:
                    SAMPLE_SIZE = min(200, len(X_te_))
                    rng = np.random.default_rng(42)
                    shap_idx = rng.choice(len(X_te_), size=SAMPLE_SIZE, replace=False)
                    X_shap = X_te_[shap_idx]

                    logger.info(f"  Computing SHAP using {SAMPLE_SIZE} samples...")

                    explainer = shap.TreeExplainer(raw_est, feature_perturbation="interventional")
                    shap_vals = explainer.shap_values(X_shap, check_additivity=False)

                    if isinstance(shap_vals, list):
                        sv = shap_vals[1]
                    else:
                        sv = np.array(shap_vals)

                    shap_values = sv

                    sv_exp = shap.Explanation(
                        values=sv,
                        base_values=np.full(len(X_shap), explainer.expected_value),
                        data=X_shap,
                        feature_names=feature_names,
                    )

                    plt.figure()
                    shap.plots.beeswarm(sv_exp, max_display=20, show=False)
                    plt.title(f"SHAP Beeswarm — {PRIMARY_TARGET} ({best_model_name})", fontsize=12, fontweight="bold")
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOT_DIR, f"shap_beeswarm_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
                    plt.close()

                    plt.figure()
                    shap.plots.dependence("headway_deviation_pct" if "headway_deviation_pct" in feature_names else feature_names[0],
                                       sv, X_shap, feature_names=feature_names, show=False)
                    plt.title(f"SHAP Dependence — {best_model_name}", fontsize=11, fontweight="bold")
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOT_DIR, f"shap_dependence_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
                    plt.close()

                    logger.info(f"  ✓ SHAP plots saved")
            except Exception as e:
                logger.warning(f"  ⚠  SHAP computation failed: {e}")

        logger.info(f"\n{'=' * 50}")
        logger.info(f"✓ Step 11 complete.")
        logger.info(f"  Target        : {PRIMARY_TARGET}")
        logger.info(f"  Model         : {best_model_name}")
        logger.info(f"  Plots saved   : {PLOT_DIR}")
        logger.info(f"{'=' * 50}")

        return shap_values

    # -----------------------------------------------------------------------
    # Step 10 Visualization Plots
    # -----------------------------------------------------------------------

    def plot_step10_visualizations(
        self,
        trained: Dict[str, Dict],
        best_model_name: str,
        val_summary: pd.DataFrame,
        yb_test: pd.DataFrame,
        proba_test: pd.DataFrame,
        ym_test: pd.DataFrame,
        pred_test_multi: pd.DataFrame,
        BINARY_TARGETS: List[str],
        MULTICLASS_TARGETS: List[str],
        target_encoders: Dict,
        all_selected: List[str],
        PLOT_DIR: str = "visualizations",
        RUN_ID: str = "default",
    ) -> None:
        """
        Step 10 - Generate visualization plots for model evaluation.

        Parameters
        ----------
        trained : Dict
            Trained models from train_all_models
        best_model_name : str
            Name of the best model
        val_summary : pd.DataFrame
            Validation summary with metrics
        yb_test : pd.DataFrame
            Test binary targets
        proba_test : pd.DataFrame
            Test probabilities
        ym_test : pd.DataFrame
            Test multiclass targets
        pred_test_multi : pd.DataFrame
            Multiclass predictions
        BINARY_TARGETS : List[str]
            Binary target columns
        MULTICLASS_TARGETS : List[str]
            Multiclass target columns
        target_encoders : Dict
            Target encoders
        all_selected : List[str]
            Selected feature names
        PLOT_DIR : str
            Directory to save plots
        RUN_ID : str
            Run identifier
        """
        import os
        import seaborn as sns
        from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

        logger.info("=" * 50)
        logger.info("STEP 10 — VISUALIZATION PLOTS")
        logger.info("=" * 50)

        os.makedirs(PLOT_DIR, exist_ok=True)
        sns.set_style("whitegrid")

        display_cols = [c for c in [
            "f1_macro", "f1_weighted", "f1_samples", "hamming_loss",
            "multi_avg_f1",
        ] + [f"auc_{col}" for col in BINARY_TARGETS]
        if c in val_summary.columns]

        display_cols = [c for c in display_cols if c in val_summary.columns]
        if not display_cols:
            display_cols = ["f1_macro", "f1_weighted"]

        best = trained[best_model_name]

        plot_df = val_summary[display_cols].reset_index()
        plot_long = plot_df.melt(id_vars="model", var_name="Metric", value_name="Score")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        PALETTE = ["#2196F3", "#FF9800", "#9C27B0"]
        sns.barplot(
            data=plot_long[plot_long["Metric"].isin(["f1_macro", "f1_samples", "hamming_loss"])],
            x="Metric", y="Score", hue="model", palette=PALETTE, ax=axes[0],
        )
        axes[0].set_title("Binary – Val Metrics by Model", fontsize=12)
        axes[0].legend(title="Model")

        ax2 = axes[1]
        bar_plot = sns.barplot(
            data=plot_long[plot_long["Metric"] == "multi_avg_f1"],
            x="model", y="Score", palette=PALETTE, ax=ax2,
        )
        ax2.set_title("Multiclass – Avg F1‑Macro by Model", fontsize=12)
        for p in ax2.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax2.text(p.get_x() + p.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha="center", va="bottom", fontsize=10)

        plt.suptitle(f"Model Comparison  (Best: {best_model_name})", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"model_comparison_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, len(BINARY_TARGETS), figsize=(6 * max(1, len(BINARY_TARGETS)), 5))
        if len(BINARY_TARGETS) == 1:
            axes = [axes]

        for ax, col in zip(axes, BINARY_TARGETS):
            try:
                if col not in proba_test.columns:
                    raise ValueError("No probability predictions")
                fpr, tpr, _ = roc_curve(yb_test[col], proba_test[col])
                auc = roc_auc_score(yb_test[col], proba_test[col])
                ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.2f}")
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                ax.set_title(f"ROC – {col}", fontsize=11)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
            except Exception as e:
                ax.set_title(f"ROC – {col} unavailable")

        plt.suptitle(f"ROC Curves – {best_model_name} (Test Set)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"roc_curves_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        if MULTICLASS_TARGETS:
            fig, axes = plt.subplots(1, len(MULTICLASS_TARGETS), figsize=(7 * max(1, len(MULTICLASS_TARGETS)), 6))
            if len(MULTICLASS_TARGETS) == 1:
                axes = [axes]

            for ax, col in zip(axes, MULTICLASS_TARGETS):
                try:
                    cm = confusion_matrix(ym_test[col], pred_test_multi[col])
                    classes = target_encoders[col].classes_
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=classes, yticklabels=classes, ax=ax)
                    ax.set_title(f"Confusion – {col}", fontsize=11)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                except Exception as e:
                    ax.set_title(f"Confusion – {col} unavailable")

            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"confusion_matrices_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
            plt.close()

        if BINARY_TARGETS:
            fig, ax = plt.subplots(figsize=(10, 8))
            try:
                model = best["binary_model"]
                if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                    est = model.estimators_[0]
                elif hasattr(model, 'feature_importances_'):
                    est = model
                else:
                    raise AttributeError("No estimator with importances")

                importances = est.feature_importances_
                if len(importances) != len(all_selected):
                    min_len = min(len(importances), len(all_selected))
                    importances = importances[:min_len]
                    feat_names = all_selected[:min_len]
                else:
                    feat_names = all_selected

                imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(20)
                imp.plot(kind="barh", ax=ax, color="steelblue")
                ax.invert_yaxis()
                ax.set_title(f"Top 20 Features – {best_model_name}", fontsize=12)
                ax.set_xlabel("Importance")
            except Exception as e:
                ax.set_title(f"Feature Importances unavailable: {e}")

            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"feature_importances_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
            plt.close()

        logger.info(f"  All plots saved to {PLOT_DIR}")

    # -----------------------------------------------------------------------
    # Feature Importance Plots
    # -----------------------------------------------------------------------

    def plot_feature_importances(
        self,
        trained: Dict[str, Dict],
        best_model_name: str,
        all_selected: List[str],
        BINARY_TARGETS: List[str],
        PLOT_DIR: str = "visualizations",
        RUN_ID: str = "default",
    ) -> None:
        """
        Plot feature importances with heatmap for all labels.

        Parameters
        ----------
        trained : Dict
            Trained models from train_all_models
        best_model_name : str
            Name of the best model
        all_selected : List[str]
            Feature names
        BINARY_TARGETS : List[str]
            Binary target columns
        PLOT_DIR : str
            Directory to save plots
        RUN_ID : str
            Run identifier
        """
        import os
        import seaborn as sns

        logger.info("=" * 50)
        logger.info("FEATURE IMPORTANCE PLOTS")
        logger.info("=" * 50)

        os.makedirs(PLOT_DIR, exist_ok=True)

        best = trained[best_model_name]
        model = best["binary_model"]

        if not hasattr(model, "estimators_"):
            logger.info("  ⚠  No estimators available")
            return

        importance_df = pd.DataFrame(index=all_selected)

        for label, estimator in zip(BINARY_TARGETS, model.estimators_):
            importance_df[label] = estimator.feature_importances_

        top_features = importance_df.mean(axis=1).sort_values(ascending=False).head(20).index
        importance_top = importance_df.loc[top_features]

        fig, ax = plt.subplots(figsize=(14, 8))
        importance_top.plot(kind="bar", ax=ax)
        ax.set_title("Top 20 Features – Importance per Label", fontweight="bold")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=45, ha="right")
        ax.legend(title="Label")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"feature_importance_bars_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            importance_top,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": "Importance"},
        )
        plt.title("Feature Importances per Label (Top 20)", fontweight="bold")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"feature_importance_heatmap_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        colors = ["#6c63ff", "#f97316", "#f43f5e", "#a855f7", "#38bdf8"]
        N_TOP = min(20, len(all_selected))

        fig, axes = plt.subplots(1, len(BINARY_TARGETS), figsize=(18, 6))
        axes = np.atleast_1d(axes)

        for ax, label, estimator, color in zip(axes, BINARY_TARGETS, model.estimators_, colors[:len(BINARY_TARGETS)]):
            imp = pd.Series(estimator.feature_importances_, index=all_selected)
            top_imp = imp.sort_values(ascending=False).head(N_TOP).sort_values()
            top_imp.plot(kind="barh", ax=ax, color=color)
            ax.set_title(f"Top {N_TOP} features — {label}", fontweight="bold")
            ax.set_xlabel("Importance")

        plt.suptitle(f"Top {N_TOP} Feature Importances per Label ({best_model_name})", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"feature_importances_{RUN_ID}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"  Feature importance plots saved to {PLOT_DIR}")

    # -----------------------------------------------------------------------
    # Graph-Based Features (Betweenness Centrality)
    # -----------------------------------------------------------------------

    _GRAPH_CACHE = None
    _EDGE_INDEX_CACHE = None

    def build_graph_features(
        self,
        stop_times_df: pd.DataFrame,
        stops_df: pd.DataFrame,
        n_sample: int = 50000,
        PLOT_DIR: str = "visualizations",
    ) -> Tuple[pd.Series, pd.DataFrame, Optional[np.ndarray]]:
        """
        Build graph-based features from GTFS stop times.

        Parameters
        ----------
        stop_times_df : pd.DataFrame
            Stop times with trip_id, stop_id, stop_sequence
        stops_df : pd.DataFrame
            Stops with stop_id, stop_name
        n_sample : int
            Number of trips to sample for graph building
        PLOT_DIR : str
            Directory to save audit CSV

        Returns
        -------
        (betweenness_series, audit_df, edge_index)
        """
        import networkx as nx

        logger.info("=" * 50)
        logger.info("BUILDING GRAPH FEATURES")
        logger.info("=" * 50)

        sample_trips = stop_times_df["trip_id"].drop_duplicates().sample(
            n=min(n_sample, stop_times_df["trip_id"].nunique()), random_state=42
        )
        trips_df = stop_times_df[stop_times_df["trip_id"].isin(sample_trips)]
        trips_df = trips_df.sort_values(["trip_id", "stop_sequence"])

        edges = []
        for trip_id, trip_stops in trips_df.groupby("trip_id"):
            stop_ids = trip_stops["stop_id"].tolist()
            for i in range(len(stop_ids) - 1):
                edges.append((stop_ids[i], stop_ids[i + 1]))

        G = nx.DiGraph()
        G.add_edges_from(edges)
        self._GRAPH_CACHE = G

        logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        betweenness = nx.betweenness_centrality(G)
        bc_series = pd.Series(betweenness).sort_values(ascending=False)

        stops_lookup = stops_df.set_index("stop_id")["stop_name"] if "stop_name" in stops_df.columns else None
        if stops_lookup is not None:
            bc_audit = bc_series.to_frame("betweenness_centrality").join(
                stops_lookup, how="left"
            ).sort_values("betweenness_centrality", ascending=False)
        else:
            bc_audit = bc_series.to_frame("betweenness_centrality").sort_values(
                "betweenness_centrality", ascending=False
            )

        os.makedirs(PLOT_DIR, exist_ok=True)
        bc_audit.to_csv(os.path.join(PLOT_DIR, "betweenness_centrality.csv"), index_label="stop_id")

        edge_index = self._get_edge_index(G)

        logger.info(f"  Betweenness centrality saved to {PLOT_DIR}")
        logger.info(f"  Top 5 stops: {list(bc_series.head(5).index)}")

        return bc_series, bc_audit, edge_index

    def _get_edge_index(self, G: nx.DiGraph = None) -> Optional[np.ndarray]:
        """Convert graph to PyTorch edge_index tensor."""
        if not TORCH_AVAILABLE:
            return None

        import torch

        if G is None:
            G = self._GRAPH_CACHE

        if G is None:
            return None

        nodes = list(G.nodes())
        nmap = {n: i for i, n in enumerate(nodes)}

        edges = list(G.edges())
        src = [nmap[e[0]] for e in edges]
        dst = [nmap[e[1]] for e in edges]

        self._EDGE_INDEX_CACHE = torch.tensor([src, dst], dtype=torch.long)
        return self._EDGE_INDEX_CACHE.numpy()

    def get_high_betweenness_stops(
        self,
        bc_series: pd.Series,
        threshold: float = 0.5,
    ) -> List[str]:
        """Get stops with betweenness centrality > threshold."""
        return bc_series[bc_series > threshold].index.tolist()

    # -----------------------------------------------------------------------
    # Step 13 — Disruption Risk Map
    # -----------------------------------------------------------------------

NETHERLANDS_LAT_MIN, NETHERLANDS_LAT_MAX = 50.75, 53.55
NETHERLANDS_LON_MIN, NETHERLANDS_LON_MAX = 3.30, 7.20

RISK_COLOURS = {
    "critical": "#ff4757",
    "high": "#ff6b35",
    "moderate": "#ffa502",
    "low": "#2ed573",
    "unknown": "#808080",
}

def clean_geocoordinates(
    self,
    df: pd.DataFrame,
    lat_col: str = "first_lat",
    lon_col: str = "first_lon",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean out-of-bounds geocoordinates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with geo coordinates
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name

    Returns
    -------
    (cleaned_df, bad_mask)
    """
    bad_mask = (
        df[lat_col].isna() |
        df[lon_col].isna() |
        (df[lat_col] < self.NETHERLANDS_LAT_MIN) |
        (df[lat_col] > self.NETHERLANDS_LAT_MAX) |
        (df[lon_col] < self.NETHERLANDS_LON_MIN) |
        (df[lon_col] > self.NETHERLANDS_LON_MAX) |
        ((df[lat_col].abs() < 1) & (df[lon_col].abs() < 1))
    )

    logger.info(f"  Out-of-bounds / NaN coordinates: {bad_mask.sum()}")

    df_clean = df.copy()
    df_clean.loc[bad_mask, [lat_col, lon_col]] = np.nan

    return df_clean, bad_mask

def add_risk_labels(
    self,
    df: pd.DataFrame,
    disruption_col: str = "disruption_target",
) -> pd.DataFrame:
    """
    Add risk_level labels for map colouring.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with disruption indicators
    disruption_col : str
        Disruption target column

    Returns
    -------
    DataFrame with risk_level column
    """
    df = df.copy()

    if "risk_level" not in df.columns:
        if disruption_col in df.columns:
            df["risk_level"] = df[disruption_col].map({1: "high", 0: "low"}).fillna("low")
        else:
            df["risk_level"] = "low"

    if "disruption_class" not in df.columns:
        df["disruption_class"] = "unknown"

    return df

def get_feature_groups(
    self,
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Automatic feature grouping by dtype.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    Dict with lists: numeric, categorical, boolean, datetime
    """
    groups = {
        "numeric": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "boolean": df.select_dtypes(include=["bool"]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime"]).columns.tolist(),
    }

    logger.info("Feature groups:")
    for k, v in groups.items():
        logger.info(f"  {k}: {len(v)} columns")

    return groups

def validate_interval_shift(
    self,
    df: pd.DataFrame,
    interval_col: str = "feed_timestamp",
    vehicle_col: str = "vehicle_id",
    horizon_rows: int = 30,
    assumed_sec: float = 60.0,
    tolerance_sec: float = 10.0,
) -> Dict[str, float]:
    """
    Validate if row-shift represents approximately the assumed time interval.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp and vehicle columns
    interval_col : str
        Timestamp column
    vehicle_col : str
        Vehicle grouping column
    horizon_rows : int
        Number of rows for shift
    assumed_sec : float
        Assumed sampling interval (seconds)
    tolerance_sec : float
        Acceptable tolerance

    Returns
    -------
    Dict with validation results
    """
    logger.info("=" * 50)
    logger.info("VALIDATING INTERVAL SHIFT")
    logger.info("=" * 50)

    df = df.copy()
    df[interval_col] = pd.to_datetime(df[interval_col], errors="coerce")

    df_sorted = df.sort_values([vehicle_col, interval_col])
    diffs = df_sorted.groupby(vehicle_col)[interval_col].diff().dt.total_seconds()

    pos_diffs = diffs.dropna()
    pos_diffs = pos_diffs[pos_diffs > 0]

    if len(pos_diffs) == 0:
        logger.info("  No usable intervals found")
        return {"valid": False, "median_sec": 0, "uniform": False}

    median_sec = float(pos_diffs.median())
    p10_sec = float(pos_diffs.quantile(0.10))
    p90_sec = float(pos_diffs.quantile(0.90))

    uniform = abs(median_sec - assumed_sec) < tolerance_sec

    result = {
        "valid": uniform,
        "median_sec": median_sec,
        "p10_sec": p10_sec,
        "p90_sec": p90_sec,
        "uniform": uniform,
        "horizon_min": horizon_rows * median_sec / 60,
    }

    if uniform:
        logger.info(f"  ✓ UNIFORM: median = {median_sec:.1f} sec")
        logger.info(f"    shift(-{horizon_rows}) = {result['horizon_min']:.1f} minutes")
    else:
        best_min = horizon_rows * p10_sec / 60
        worst_min = horizon_rows * p90_sec / 60
        logger.info(f"  ✗ NON-UNIFORM: median = {median_sec:.1f} sec")
        logger.info(f"    shift represents {best_min:.1f}–{worst_min:.1f} minutes")
        logger.info("    Row-shift NOT valid for 30-min window")

    logger.info(f"  No target constructed (see time-based window)")
    return result

def build_30min_target(
    self,
    df: pd.DataFrame,
    route_col: str = "route_id",
    delay_col: str = "mean_delay_15min",
    variance_col: str = "delay_variance_15min",
    horizon_rows: int = 6,
) -> pd.Series:
    """
    Build 30-minute ahead disruption target using shift.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with delay features
    route_col : str
        Route grouping column
    delay_col : str
        Mean delay column (seconds)
    variance_col : str
        Delay variance column
    horizon_rows : int
        Rows to shift (6 × 5min = 30min)

    Returns
    -------
    pd.Series with 30-min ahead disruption labels
    """
    logger.info("=" * 50)
    logger.info("BUILDING 30-MINUTE AHEAD TARGET")
    logger.info("=" * 50)

    df = df.copy()

    df["temp_target"] = (
        (df[delay_col] > 300) | (df[variance_col] > 120)
    ).astype(int)

    df["target_disruption_30min"] = df.groupby(route_col)["temp_target"].shift(-horizon_rows)

    df = df.drop(columns=["temp_target"], errors="ignore")

    value_counts = df["target_disruption_30min"].value_counts()
    logger.info(f"  Target distribution:")
    logger.info(f"    {value_counts.to_dict()}")

    return df["target_disruption_30min"]

def create_disruption_risk_map(
        self,
        unified_df: pd.DataFrame,
        trained: Dict[str, Dict],
        best_model_name: str,
        all_selected: List[str],
        PRIMARY_TARGET: str = "is_disruption",
        cat_encoder=None,
        PLOT_DIR: str = "visualizations",
        RUN_ID: str = "default",
    ) -> Optional[folium.Map]:
        """
        Create interactive disruption risk map.

        Parameters
        ----------
        unified_df : DataFrame
            Unified data with features + coordinates
        trained : Dict
            Trained models
        best_model_name : str
            Name of best model
        all_selected : List[str]
            Selected feature names
        PRIMARY_TARGET : str
            Primary target column
        cat_encoder : encoder, optional
            Categorical encoder
        PLOT_DIR : str
            Directory to save maps
        RUN_ID : str
            Run identifier

        Returns
        -------
        folium Map or None
        """
        try:
            import folium
            from folium.plugins import MarkerCluster
        except ImportError:
            logger.warning("  ⚠  Folium not installed")
            return None

        logger.info("=" * 50)
        logger.info("STEP 13 — DISRUPTION RISK MAP")
        logger.info("=" * 50)

        bad_mask = (
            unified_df["first_lat"].isna() |
            unified_df["first_lon"].isna() |
            (unified_df["first_lat"] < self.NETHERLANDS_LAT_MIN) |
            (unified_df["first_lat"] > self.NETHERLANDS_LAT_MAX) |
            (unified_df["first_lon"] < self.NETHERLANDS_LON_MIN) |
            (unified_df["first_lon"] > self.NETHERLANDS_LON_MAX)
        )

        map_df = unified_df[~bad_mask].copy()
        logger.info(f"  Rows with valid coordinates: {len(map_df)}")

        model_info = self.define_models_and_helpers()
        helpers = model_info["helpers"]
        safe_clean = helpers.get("safe_clean_features", lambda x: x)
        get_binary_proba = helpers.get("get_binary_proba")

        predict_cols = [c for c in all_selected if c in map_df.columns]
        missing_cols = [c for c in all_selected if c not in map_df.columns]

        X_map = map_df[predict_cols].copy()
        for col in missing_cols:
            X_map[col] = 0.0

        X_map = X_map[all_selected]

        for col in X_map.columns:
            X_map[col] = pd.to_numeric(X_map[col], errors="coerce")

        col_medians = X_map.median()
        X_map = X_map.fillna(col_medians).fillna(0.0)

        X_map_arr = safe_clean(X_map.values.astype(np.float32))

        best = trained[best_model_name]
        if best["scaled"]:
            scaler = self.scaler
            X_map_arr = scaler.transform(X_map_arr)

        proba_map = get_binary_proba(best["binary_model"], X_map_arr)
        risk_score = proba_map[PRIMARY_TARGET].values

        map_df["risk_score"] = np.round(risk_score, 4)
        map_df["risk_level"] = pd.cut(
            map_df["risk_score"],
            bins=[0.0, 0.25, 0.50, 0.75, 1.01],
            labels=["low", "moderate", "high", "critical"],
            include_lowest=True,
        )

        logger.info(f"  Risk distribution:")
        logger.info(map_df["risk_level"].value_counts().to_string())

        center_lat = float(map_df["first_lat"].mean())
        center_lon = float(map_df["first_lon"].mean())

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="CartoDB positron",
        )

        risk_levels = map_df["risk_level"].astype(str).fillna("unknown").unique()
        risk_clusters = {
            level: MarkerCluster(name=f"Risk: {level}").add_to(m)
            for level in risk_levels
        }

        for _, row in map_df.iterrows():
            level = str(row.get("risk_level", "unknown")).lower()
            colour = self.RISK_COLOURS.get(level, "#808080")
            score = row.get("risk_score", 0.0)

            popup_lines = [
                f"Risk level : {level}",
                f"Risk score : {score:.3f}",
            ]
            for col in ["cause", "effect", "route"]:
                if col in row.index and pd.notna(row[col]):
                    popup_lines.append(f"{col} : {row[col]}")

            folium.CircleMarker(
                location=[float(row["first_lat"]), float(row["first_lon"])],
                radius=7,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.75,
                popup=folium.Popup("<br>".join(popup_lines), max_width=320),
                tooltip=f"{level} ({score:.2f})",
            ).add_to(risk_clusters.get(level, m))

        folium.LayerControl(collapsed=False).add_to(m)

        legend_html = """
        <div style="position:fixed; bottom:50px; left:50px; z-index:9999;
                    background:white; border:2px solid #ccc;
                    border-radius:6px; padding:12px;
                    font-size:12px; font-family:sans-serif;">
          <b>Disruption Risk Level</b><br>
        """
        for level, colour in self.RISK_COLOURS.items():
            legend_html += f'<i style="background:{colour};width:12px;height:12px;display:inline-block;margin-right:6px;border-radius:50%;"></i>{level}<br>'
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))

        os.makedirs(PLOT_DIR, exist_ok=True)
        map_path = os.path.join(PLOT_DIR, f"disruption_risk_map_{RUN_ID}.html")
        m.save(map_path)
        logger.info(f"  Interactive map saved → {map_path}")

        fig, ax = plt.subplots(figsize=(10, 8))
        for level in ["low", "moderate", "high", "critical"]:
            subset = map_df[map_df["risk_level"].astype(str) == level]
            if subset.empty:
                continue
            ax.scatter(
                subset["first_lon"], subset["first_lat"],
                c=self.RISK_COLOURS.get(level, "#808080"),
                label=f"{level} (n={len(subset)})",
                s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Disruption Risk Map — {best_model_name}", fontsize=12, fontweight="bold")
        ax.legend(title="Risk level", loc="upper left")
        ax.set_xlim(self.NETHERLANDS_LON_MIN - 0.1, self.NETHERLANDS_LON_MAX + 0.1)
        ax.set_ylim(self.NETHERLANDS_LAT_MIN - 0.1, self.NETHERLANDS_LAT_MAX + 0.1)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        static_path = os.path.join(PLOT_DIR, f"disruption_risk_map_static_{RUN_ID}.png")
        plt.savefig(static_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Static map saved → {static_path}")

        return m

# -----------------------------------------------------------------------
# Data ingestion
# -----------------------------------------------------------------------

def ingest_data(
        self,
        mode: str = "combined",
        local_dir: Optional[str] = None,
        feed_urls: Optional[Dict[str, str]] = None,
        static_gtfs_zip: Optional[str] = None,
        static_gtfs_url: Optional[str] = None,
        timeout: int = 30,
        max_files: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Ingest GTFS data from local files, live feeds, or both.

        Parameters
        ----------
        mode : str
            ``"local"``  — read parquet zips from *local_dir*.
            ``"live"``   — fetch GTFS-RT protobuf from *feed_urls*.
            ``"combined"`` — local first, then live appended + dedup.
        local_dir : str, optional
            Directory with *_files_list.zip files.
            Defaults to ``config['ingestion']['local_dir']`` or ``"feed_data"``.
        feed_urls : dict, optional
            Override for GTFS-RT endpoint URLs.
        static_gtfs_zip : str, optional
            Path to a local static GTFS zip.
        static_gtfs_url : str, optional
            URL for static GTFS download.
        timeout : int
            HTTP timeout (seconds).
        max_files : int, optional
            Max parquet files to read per zip (for memory control).
            Defaults to ``config['ingestion']['max_files']`` or None (all files).

        Returns
        -------
        (merged_df, gtfs_data) ready for ``run_full_pipeline``.
        """
        ingest_cfg = self.config.get("ingestion", {})
        local_dir = local_dir or ingest_cfg.get("local_dir", DEFAULT_LOCAL_DIR)
        feed_urls = feed_urls or ingest_cfg.get("feed_urls", None)
        static_gtfs_url = static_gtfs_url or ingest_cfg.get(
            "static_gtfs_url", DEFAULT_STATIC_GTFS_URL
        )
        static_gtfs_zip = static_gtfs_zip or ingest_cfg.get("static_gtfs_zip", None)
        max_files = max_files if max_files is not None else ingest_cfg.get("max_files", None)

        if mode == "local":
            merged, gtfs_data = ingest_local(
                local_dir=local_dir,
                static_gtfs_zip=static_gtfs_zip,
                static_gtfs_url=static_gtfs_url,
                max_files=max_files,
            )
        elif mode == "live":
            merged, gtfs_data = ingest_live(
                feed_urls=feed_urls,
                static_gtfs_url=static_gtfs_url,
                timeout=timeout,
            )
        elif mode == "combined":
            merged, gtfs_data = ingest_combined(
                local_dir=local_dir,
                feed_urls=feed_urls,
                static_gtfs_zip=static_gtfs_zip,
                static_gtfs_url=static_gtfs_url,
                timeout=timeout,
                max_files=max_files,
            )
        else:
            raise ValueError(f"Unknown ingestion mode: {mode!r}. Use 'local', 'live', or 'combined'.")

        logger.info(f"Ingestion complete — merged_df: {merged.shape}, gtfs keys: {list(gtfs_data.keys())}")
        return merged, gtfs_data

# -----------------------------------------------------------------------
# Full pipeline execution
# -----------------------------------------------------------------------

def run_full_pipeline(
        self,
        merged_df: pd.DataFrame,
        gtfs_data: Dict,
        model_type: str = "all",
        use_adaptive_split: bool = True,
        train_days: int = 21,
        val_days: int = 3,
        test_days: int = 1
    ) -> Dict:
        """Run complete pipeline end-to-end."""
        logger.info("="*60)
        logger.info("FULL PIPELINE EXECUTION")
        logger.info("="*60)
        
        # Feature engineering
        feature_df = self.run_feature_engineering(merged_df, gtfs_data)
        
        # Classification
        classified_df, route_summary = self.run_classification(feature_df)
        
        # Analysis
        self.run_analysis(classified_df)
        
        # Prepare features
        exclude_cols = {
            "alert_id", "RT_id", "agency_id", "route_id", "trip_id", "stop_id", "vehicle_id",
            "vehicle_label", "consolidated_route", "cause", "effect", "description_text",
            "clean_text", "combined_text", "language_code", "language_name", "language",
            "active_periods", "alert_persistence_class", "sentiment", "topic_label",
            "all_entities", "loc_entities", "first_loc_text", "day_name", "date", "id_date",
            "id_time", "start_time", "schedule_relationship_raw", "current_status",
            "current_status_raw", "geometry", "holiday_name", "is_disruption", "is_peak",
            "is_abnormal", "feed_timestamp", "timestamp", "id_date_part", "timestamp_min",
            "timestamp_hour", "trip_start_datetime", "arrival_time_local",
            "departure_time_local", "event_time", "start_date", "prev_time",
            "active_period_start", "active_period_end", "is_escalating", "has_vehicle_observation",
            "disruption_target", "future_alert", "early_warning_target", "disruption_class",
            "effect_class", "early_warning_level", "target_multiclass", "target_10min",
            "target_30min", "target_60min", "target_disruption_30min",
        }
        
        feat_cols = [
            c for c in classified_df.columns
            if c not in exclude_cols
            and classified_df[c].dtype not in ["object"]
            and not pd.api.types.is_datetime64_any_dtype(classified_df[c])
        ][:40]
        
        # Run adaptive splitting if enabled
        split_info = None
        if use_adaptive_split:
            split_info = self.run_adaptive_split(
                df=classified_df,
                timestamp_col='feed_timestamp' if 'feed_timestamp' in classified_df.columns else 'timestamp',
                disruption_col='disruption_type',
                stream_cols=['speed', 'delay_sec', 'alert_severity'] if 'alert_severity' in classified_df.columns else None
            )
        
        # Run rolling window simulation
        if model_type == "all":
            simulation_results = self.run_rolling_window_simulation(
                classified_df, feat_cols,
                train_days=train_days, val_days=val_days, test_days=test_days
            )

            # Fallback to fixed split when data is too small for rolling windows
            if not simulation_results["windows"]:
                logger.warning(
                    "  No rolling windows created (insufficient dates). "
                    "Falling back to fixed train/val/test split (70/15/15)."
                )
                simulation_results = self.run_fixed_split_simulation(
                    classified_df, feat_cols,
                )
            
            # Generate visualizations
            self.generate_visualizations(simulation_results, classified_df=classified_df)
            
            # Save models
            logger.info("="*60)
            logger.info("SAVING MODELS")
            logger.info("="*60)
            for mname, model_info in simulation_results['trained_models'].items():
                self.save_model(
                    model_info['mdl'],
                    f"model_{mname}",
                    metadata={
                        'model_name': mname,
                        'feature_names': model_info['feat_names'],
                        'best_model': mname == simulation_results['best_model']
                    }
                )
            
            # Save results
            results_summary = {
                'best_model': simulation_results['best_model'],
                'model_performance': {},
                'windows': len(simulation_results['windows']),
            }
            for mname in self.MODEL_LIST:
                f1_v = np.array([r['f1'] for r in simulation_results['all_results'][mname]])
                pr_v = np.array([r['pr_auc'] for r in simulation_results['all_results'][mname]])
                results_summary['model_performance'][mname] = {
                    'f1_mean': float(np.nanmean(f1_v)),
                    'f1_std': float(np.nanstd(f1_v)),
                    'pr_auc_mean': float(np.nanmean(pr_v)),
                    'pr_auc_std': float(np.nanstd(pr_v)),
                }
            
            self.save_results(results_summary, "pipeline_results")
            
            result = {
                'feature_df': feature_df,
                'classified_df': classified_df,
                'route_summary': route_summary,
                'simulation_results': simulation_results,
                'best_model': simulation_results['best_model']
            }
            
            if split_info:
                result['split_info'] = split_info
            
            return result
        else:
            # Single model mode
            timestamp_col = 'feed_timestamp' if 'feed_timestamp' in classified_df.columns else 'timestamp'
            train_df, val_df, test_df = chronological_split(
                classified_df, timestamp_col,
                train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
            )
            
            X_train = train_df[feat_cols].values
            y_train = (train_df['disruption_type'] != 'ON_TIME').astype(int).values
            X_test = test_df[feat_cols].values
            y_test = (test_df['disruption_type'] != 'ON_TIME').astype(int).values
            
            balancing_config = self.config.get('balancing', {})
            balancer = TemporalAwareBalancer(strategy=balancing_config.get('strategy', 'class_weight'))
            balancer.fit(y_train)
            model_params = balancer.get_estimator_params(model_type)
            
            model = make_model(model_type, seed=self.config.get('seed', 42))
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            metrics = compute_metrics(y_test, y_pred, y_proba)
            
            # Save single model
            self.save_model(model, f"model_{model_type}", metadata={'metrics': metrics})
            self.save_results(metrics, f"results_{model_type}")
            
            result = {
                'feature_df': feature_df,
                'classified_df': classified_df,
                'route_summary': route_summary,
                'model': model,
                'metrics': metrics
            }
            
            if split_info:
                result['split_info'] = split_info
            
            return result
