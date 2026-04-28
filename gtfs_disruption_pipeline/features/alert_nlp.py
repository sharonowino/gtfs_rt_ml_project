"""
GTFS Alert NLP Enrichment Module
==============================
Multilingual NLP Pipeline for GTFS service alerts.

This module provides:
1. Language Detection - papluca/xlm-roberta-base-language-detection
2. Named Entity Recognition (NER) - multilingual NER
3. Sentiment Analysis - cardiffnlp/twitter-xlm-roberta-base-sentiment
4. Topic Modeling - BERTopic + sentence-transformers
5. Duration-aware feature engineering
6. Geocoding - Nominatim for location entities

Usage:
------
from gtfs_disruption.features.alert_nlp import AlertNLPEnricher, add_alert_nlp_features

enricher = AlertNLPEnricher()
df = enricher.enrich(merged_df)
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NETHERLANDS_LAT_MIN, NETHERLANDS_LAT_MAX = 50.75, 53.55
NETHERLANDS_LON_MIN, NETHERLANDS_LON_MAX = 3.30, 7.20

NLP_MODELS = {
    'language_detection': 'papluca/xlm-roberta-base-language-detection',
    'ner': 'Davlan/xlm-roberta-base-ner-hrl',
    'sentiment': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
    'embedding': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
}

CAUSE_MAP = {
    'UNKNOWN_CAUSE': 0, 'OTHER_CAUSE': 1, 'TECHNICAL_PROBLEM': 2,
    'STRIKE': 3, 'DEMONSTRATION': 4, 'ACCIDENT': 5,
    'HOLIDAY': 6, 'WEATHER': 7, 'MAINTENANCE': 8,
    'CONSTRUCTION': 9, 'POLICE_ACTIVITY': 10, 'MEDICAL_EMERGENCY': 11
}

EFFECT_MAP = {
    'NO_SERVICE': 0, 'REDUCED_SERVICE': 1, 'SIGNIFICANT_DELAYS': 2,
    'DETOUR': 3, 'ADDITIONAL_SERVICE': 4, 'MODIFIED_SERVICE': 5,
    'OTHER_EFFECT': 6, 'UNKNOWN_EFFECT': 7, 'STOP_MOVED': 8
}


def _lazy_import_transformers():
    """Lazy import for transformers library."""
    try:
        from transformers import pipeline
        return pipeline
    except ImportError:
        logger.warning("transformers not installed - NLP features disabled")
        return None


def _lazy_import_bertopic():
    """Lazy import for BERTopic library."""
    try:
        from bertopic import BERTopic
        return BERTopic
    except ImportError:
        logger.warning("bertopic not installed - topic modeling disabled")
        return None


def _lazy_import_sentence_transformers():
    """Lazy import for sentence-transformers library."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed - embeddings disabled")
        return None


class AlertNLPEnricher:
    """
    Multilingual NLP enrichment for GTFS service alerts.
    
    Pipeline stages:
    1. Language Detection - identify alert language
    2. NER - extract entities (locations, routes, times)
    3. Sentiment - estimate urgency/severity
    4. Topic - discover semantic clusters
    
    Parameters
    ----------
    enable_language_detection : bool
        Enable language detection (default True)
    enable_ner : bool
        Enable named entity recognition (default True)
    enable_sentiment : bool
        Enable sentiment analysis (default True)
    enable_topic : bool
        Enable topic modeling (default False - requires more resources)
    batch_size : int
        Batch size for inference (default 8)
    device : str
        Device for inference ('cpu', 'cuda', default 'cpu')
    language_detection_model : str
        Model for language detection (default 'papluca/xlm-roberta-base-language-detection')
    ner_model : str
        Model for NER (default 'Davlan/xlm-roberta-base-ner-hrl')
    sentiment_model : str
        Model for sentiment (default 'cardiffnlp/twitter-xlm-roberta-base-sentiment')
    """
    
    def __init__(
        self,
        enable_language_detection: bool = True,
        enable_ner: bool = True,
        enable_sentiment: bool = True,
        enable_topic: bool = False,
        batch_size: int = 8,
        device: str = 'cpu',
        language_detection_model: str = 'papluca/xlm-roberta-base-language-detection',
        ner_model: str = 'Davlan/xlm-roberta-base-ner-hrl',
        sentiment_model: str = 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
    ):
        self.enable_language_detection = enable_language_detection
        self.enable_ner = enable_ner
        self.enable_sentiment = enable_sentiment
        self.enable_topic = enable_topic
        self.batch_size = batch_size
        self.device = device
        
        self.language_detection_model = language_detection_model
        self.ner_model = ner_model
        self.sentiment_model = sentiment_model
        
        self._lang_detector = None
        self._ner_pipeline = None
        self._sentiment_pipeline = None
        self._embedding_model = None
        self._topic_model = None
        
        self._models_loaded = False
    
    def _load_models(self):
        """Load NLP models on demand."""
        if self._models_loaded:
            return
        
        logger.info("Loading NLP models...")
        
        if self.enable_language_detection:
            try:
                pipeline = _lazy_import_transformers()
                if pipeline:
                    self._lang_detector = pipeline(
                        "text-classification",
                        model=self.language_detection_model,
                        device=self.device if self.device == 'cpu' else -1
                    )
                    logger.info(f"  Loaded: {self.language_detection_model}")
            except Exception as e:
                logger.warning(f"  Language detector failed: {e}")
                self.enable_language_detection = False
        
        if self.enable_ner:
            try:
                pipeline = _lazy_import_transformers()
                if pipeline:
                    self._ner_pipeline = pipeline(
                        "ner",
                        model=self.ner_model,
                        device=self.device if self.device == 'cpu' else -1,
                        aggregation_strategy="simple"
                    )
                    logger.info(f"  Loaded: {self.ner_model}")
            except Exception as e:
                logger.warning(f"  NER pipeline failed: {e}")
                self.enable_ner = False
        
        if self.enable_sentiment:
            try:
                pipeline = _lazy_import_transformers()
                if pipeline:
                    self._sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=self.sentiment_model,
                        device=self.device if self.device == 'cpu' else -1
                    )
                    logger.info(f"  Loaded: {self.sentiment_model}")
            except Exception as e:
                logger.warning(f"  Sentiment pipeline failed: {e}")
                self.enable_sentiment = False
        
        self._models_loaded = True
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text."""
        if not self._lang_detector or pd.isna(text):
            return 'unknown', 0.0
        
        try:
            result = self._lang_detector(text[:512])
            if result:
                return result[0]['label'], result[0]['score']
        except Exception:
            pass
        return 'unknown', 0.0
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""
        if not self._ner_pipeline or pd.isna(text):
            return []
        
        try:
            results = self._ner_pipeline(text[:512])
            entities = []
            for r in results:
                entities.append({
                    'entity_text': r['word'],
                    'entity_type': r['entity_group'],
                    'confidence': r['score']
                })
            return entities
        except Exception:
            return []
    
    def _analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment of text."""
        if not self._sentiment_pipeline or pd.isna(text):
            return 'neutral', 0.0
        
        try:
            result = self._sentiment_pipeline(text[:512])
            if result:
                label = result[0]['label'].lower()
                score = result[0]['score']
                if 'pos' in label:
                    return 'positive', score
                elif 'neg' in label:
                    return 'negative', score
                else:
                    return 'neutral', score
        except Exception:
            pass
        return 'neutral', 0.0
    
    def _add_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add duration-aware features for alerts."""
        logger.info("  Adding duration-aware features...")
        
        text_col = None
        for col in ['description_text', 'alert_text', 'header_text', 'text']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col:
            df['alert_text_length'] = df[text_col].fillna('').str.len()
            df['alert_word_count'] = df[text_col].fillna('').str.split().str.len()
            
            df['alert_has_url'] = df[text_col].fillna('').str.contains(
                r'http[s]?://', regex=True, na=False
            ).astype(int)
            df['alert_has_phone'] = df[text_col].fillna('').str.contains(
                r'\d{2,4}[-\s]?\d{2,4}[-\s]?\d{2,4}', regex=True, na=False
            ).astype(int)
            df['alert_has_email'] = df[text_col].fillna('').str.contains(
                r'[\w.-]+@[\w.-]+', regex=True, na=False
            ).astype(int)
        
        start_col = None
        end_col = None
        for sc in ['active_period_start', 'alert_start', 'start_time']:
            if sc in df.columns:
                start_col = sc
                break
        for ec in ['active_period_end', 'alert_end', 'end_time']:
            if ec in df.columns:
                end_col = ec
                break
        
        ts_col = None
        for tc in ['timestamp', 'feed_timestamp', 'event_time']:
            if tc in df.columns:
                ts_col = tc
                break
        
        if start_col and end_col:
            start = pd.to_datetime(df[start_col], errors='coerce')
            end = pd.to_datetime(df[end_col], errors='coerce')
            df['alert_duration_seconds'] = (end - start).dt.total_seconds()
            df['alert_duration_minutes'] = df['alert_duration_seconds'] / 60
            df['alert_duration_hours'] = df['alert_duration_seconds'] / 3600
            df['alert_duration_days'] = df['alert_duration_seconds'] / 86400
        
        if ts_col and end_col:
            current = pd.to_datetime(df[ts_col], errors='coerce')
            end = pd.to_datetime(df[end_col], errors='coerce')
            df['alert_remaining_seconds'] = (end - current).dt.total_seconds()
            df['alert_remaining_hours'] = df['alert_remaining_seconds'] / 3600
            df['alert_is_expired'] = (df['alert_remaining_seconds'] <= 0).astype(int)
            df['alert_is_imminent'] = (
                (df['alert_remaining_seconds'] > 0) & 
                (df['alert_remaining_seconds'] <= 3600)
            ).astype(int)
        
        if start_col and ts_col:
            start = pd.to_datetime(df[start_col], errors='coerce')
            current = pd.to_datetime(df[ts_col], errors='coerce')
            df['alert_age_seconds'] = (current - start).dt.total_seconds()
            df['alert_age_minutes'] = df['alert_age_seconds'] / 60
            df['alert_age_hours'] = df['alert_age_seconds'] / 3600
            df['alert_is_new'] = (df['alert_age_minutes'] <= 15).astype(int)
        
        if 'cause' in df.columns:
            df['alert_cause_id'] = df['cause'].map(CAUSE_MAP).fillna(-1).astype(int)
        
        if 'effect' in df.columns:
            df['alert_effect_id'] = df['effect'].map(EFFECT_MAP).fillna(-1).astype(int)
        
        if 'cause' in df.columns and 'effect' in df.columns:
            df['alert_severity_composite'] = (
                df['alert_cause_id'].abs() + df['alert_effect_id'].abs()
            )
        
        return df
    
    def _add_language_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add language detection features."""
        logger.info("  Adding language detection features...")
        
        text_col = None
        for col in ['description_text', 'alert_text', 'header_text', 'text']:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col or not self.enable_language_detection:
            df['alert_language'] = 'unknown'
            df['alert_language_confidence'] = 0.0
            return df
        
        self._load_models()
        if not self._lang_detector:
            df['alert_language'] = 'unknown'
            df['alert_language_confidence'] = 0.0
            return df
        
        texts = df[text_col].fillna('').tolist()
        languages = []
        confidences = []
        
        for text in texts:
            lang, conf = self._detect_language(text)
            languages.append(lang)
            confidences.append(conf)
        
        df['alert_language'] = languages
        df['alert_language_confidence'] = confidences
        
        df['alert_is_dutch'] = (df['alert_language'] == 'nl').astype(int)
        df['alert_is_english'] = (df['alert_language'] == 'en').astype(int)
        df['alert_is_german'] = (df['alert_language'] == 'de').astype(int)
        df['alert_is_french'] = (df['alert_language'] == 'fr').astype(int)
        
        return df
    
    def _add_ner_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add named entity recognition features."""
        logger.info("  Adding NER features...")
        
        text_col = None
        for col in ['description_text', 'alert_text', 'header_text', 'text']:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col or not self.enable_ner:
            df['alert_entities'] = '[]'
            df['alert_entity_count'] = 0
            return df
        
        self._load_models()
        if not self._ner_pipeline:
            df['alert_entities'] = '[]'
            df['alert_entity_count'] = 0
            return df
        
        import json
        all_entities = []
        entity_counts = []
        
        for text in df[text_col].fillna('').tolist():
            entities = self._extract_entities(text)
            all_entities.append(json.dumps(entities))
            entity_counts.append(len(entities))
        
        df['alert_entities'] = all_entities
        df['alert_entity_count'] = entity_counts
        
        entity_types = ['LOC', 'ORG', 'PER', 'TIME', 'MISC']
        for et in entity_types:
            df[f'alert_has_{et.lower()}'] = df['alert_entities'].str.contains(
                f'"{et}"', regex=False, na=False
            ).astype(int)
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment analysis features."""
        logger.info("  Adding sentiment features...")
        
        text_col = None
        for col in ['description_text', 'alert_text', 'header_text', 'text']:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col or not self.enable_sentiment:
            df['alert_sentiment'] = 'neutral'
            df['alert_sentiment_confidence'] = 0.0
            return df
        
        self._load_models()
        if not self._sentiment_pipeline:
            df['alert_sentiment'] = 'neutral'
            df['alert_sentiment_confidence'] = 0.0
            return df
        
        sentiments = []
        confidences = []
        
        for text in df[text_col].fillna('').tolist():
            sent, conf = self._analyze_sentiment(text)
            sentiments.append(sent)
            confidences.append(conf)
        
        df['alert_sentiment'] = sentiments
        df['alert_sentiment_confidence'] = confidences
        
        df['alert_is_negative'] = (df['alert_sentiment'] == 'negative').astype(int)
        df['alert_is_positive'] = (df['alert_sentiment'] == 'positive').astype(int)
        df['alert_is_neutral'] = (df['alert_sentiment'] == 'neutral').astype(int)
        
        df['alert_urgency_score'] = (
            (df['alert_is_negative'].astype(float) * df['alert_sentiment_confidence']) +
            (df['effect_id'].abs() if 'effect_id' in df.columns else 0) * 0.5
        )
        
        return df
    
    def _add_topic_features(self, df: pd.DataFrame, n_topics: int = 10) -> pd.DataFrame:
        """Add topic modeling features."""
        logger.info("  Adding topic modeling features...")
        
        text_col = None
        for col in ['description_text', 'alert_text', 'header_text', 'text']:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col or not self.enable_topic:
            df['alert_topic_id'] = -1
            df['alert_topic_probability'] = 0.0
            return df
        
        BERTopic = _lazy_import_bertopic()
        SentenceTransformer = _lazy_import_sentence_transformers()
        
        if not BERTopic or not SentenceTransformer:
            df['alert_topic_id'] = -1
            df['alert_topic_probability'] = 0.0
            return df
        
        try:
            embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            topic_model = BERTopic(
                embedding_model=embedding_model,
                nr_topics=n_topics
            )
            
            texts = df[text_col].fillna('').tolist()
            topics, probs = topic_model.fit_transform(texts)
            
            df['alert_topic_id'] = topics
            df['alert_topic_probability'] = probs
        except Exception as e:
            logger.warning(f"  Topic modeling failed: {e}")
            df['alert_topic_id'] = -1
            df['alert_topic_probability'] = 0.0
        
        return df
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all NLP enrichment to alert DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with alert text columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with NLP features added
        """
        logger.info("=" * 60)
        logger.info("ALERT NLP ENRICHMENT")
        logger.info("=" * 60)
        
        if df.empty:
            logger.warning("  Empty DataFrame - skipping NLP enrichment")
            return df
        
        self._load_models()
        
        out = df.copy()
        
        out = self._add_duration_features(out)
        
        out = self._add_language_features(out)
        
        out = self._add_ner_features(out)
        
        out = self._add_sentiment_features(out)
        
        if self.enable_topic:
            out = self._add_topic_features(out)
        
        new_cols = [c for c in out.columns if c not in df.columns]
        logger.info(f"  Added {len(new_cols)} NLP features")
        
        return out


def add_alert_nlp_features(
    df: pd.DataFrame,
    enable_language_detection: bool = True,
    enable_ner: bool = True,
    enable_sentiment: bool = True,
    enable_topic: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to add NLP features to alerts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with alert text columns
    enable_language_detection : bool
        Enable language detection (default True)
    enable_ner : bool
        Enable named entity recognition (default True)
    enable_sentiment : bool
        Enable sentiment analysis (default True)
    enable_topic : bool
        Enable topic modeling (default False)
    **kwargs
        Additional arguments for AlertNLPEnricher
        
    Returns
    -------
    pd.DataFrame
        DataFrame with NLP features added
    """
    enricher = AlertNLPEnricher(
        enable_language_detection=enable_language_detection,
        enable_ner=enable_ner,
        enable_sentiment=enable_sentiment,
        enable_topic=enable_topic,
        **kwargs
    )
    return enricher.enrich(df)


class AlertDurationAnalyzer:
    """
    Duration-aware analysis for service alerts.
    
    Methods
    -------
    compute_active_duration() - Calculate how long alert has been active
    compute_remaining_duration() - Calculate remaining active time
    is_active() - Check if alert is currently active
    is_expiring_soon() - Check if alert expires within threshold
    aggregate_by_duration() - Aggregate metrics by duration bucket
    """
    
    def __init__(self, time_column: str = 'timestamp'):
        self.time_column = time_column
    
    def compute_active_duration(
        self,
        df: pd.DataFrame,
        start_column: str = 'active_period_start'
    ) -> pd.DataFrame:
        """Compute alert active duration."""
        if start_column not in df.columns or self.time_column not in df.columns:
            return df
        
        start = pd.to_datetime(df[start_column], errors='coerce')
        current = pd.to_datetime(df[self.time_column], errors='coerce')
        
        df['alert_active_duration_seconds'] = (current - start).dt.total_seconds()
        df['alert_active_duration_minutes'] = df['alert_active_duration_seconds'] / 60
        df['alert_active_duration_hours'] = df['alert_active_duration_seconds'] / 3600
        
        return df
    
    def compute_remaining_duration(
        self,
        df: pd.DataFrame,
        end_column: str = 'active_period_end'
    ) -> pd.DataFrame:
        """Compute remaining alert duration."""
        if end_column not in df.columns or self.time_column not in df.columns:
            return df
        
        end = pd.to_datetime(df[end_column], errors='coerce')
        current = pd.to_datetime(df[self.time_column], errors='coerce')
        
        df['alert_remaining_seconds'] = (end - current).dt.total_seconds()
        df['alert_remaining_hours'] = df['alert_remaining_seconds'] / 3600
        
        return df
    
    def is_active(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check if alert is currently active."""
        if 'alert_remaining_seconds' not in df.columns:
            if 'active_period_end' in df.columns and self.time_column in df.columns:
                df = self.compute_remaining_duration(df)
        
        if 'alert_remaining_seconds' in df.columns:
            df['alert_is_active'] = (df['alert_remaining_seconds'] > 0).astype(int)
        else:
            df['alert_is_active'] = 1
        
        return df
    
    def is_expiring_soon(
        self,
        df: pd.DataFrame,
        threshold_minutes: int = 60
    ) -> pd.DataFrame:
        """Check if alert expires within threshold."""
        if 'alert_remaining_seconds' not in df.columns:
            if 'active_period_end' in df.columns and self.time_column in df.columns:
                df = self.compute_remaining_duration(df)
        
        if 'alert_remaining_seconds' in df.columns:
            df['alert_is_expiring_soon'] = (
                (df['alert_remaining_seconds'] > 0) &
                (df['alert_remaining_seconds'] <= threshold_minutes * 60)
            ).astype(int)
        else:
            df['alert_is_expiring_soon'] = 0
        
        return df
    
    def aggregate_by_duration(
        self,
        df: pd.DataFrame,
        duration_column: str = 'alert_duration_hours'
    ) -> pd.DataFrame:
        """Aggregate metrics by duration bucket."""
        if duration_column not in df.columns:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy['duration_bucket'] = pd.cut(
            df_copy[duration_column],
            bins=[0, 1, 6, 24, 48, np.inf],
            labels=['<1h', '1-6h', '6-24h', '24-48h', '>48h']
        )
        
        agg = df_copy.groupby('duration_bucket').agg({
            'alert_is_active': ['sum', 'mean'],
            'alert_is_expiring_soon': ['sum', 'mean'] if 'alert_is_expiring_soon' in df_copy.columns else 'count'
        }).reset_index()
        
        return agg


def _lazy_import_geopy():
    """Lazy import for geopy library."""
    try:
        from geopy.geocoders import Nominatim
        return Nominatim
    except ImportError:
        logger.warning("geopy not installed - geocoding disabled")
        return None


class AlertGeocoder:
    """
    Geocoding for alert location entities using Nominatim.
    
    Methods
    -------
    geocode_locations() - Geocode location entities from NER
    add_risk_map() - Add risk level for map visualization
    build_folium_map() - Create interactive Folium map
    """
    
    def __init__(
        self,
        user_agent: str = "gtfs_disruption_pipeline",
        timeout: int = 10,
        cache_dir: Optional[str] = None
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.cache_dir = cache_dir or "visualizations"
        self._geolocator = None
        self._cache = {}
    
    def _get_geolocator(self):
        """Get or create Nominatim geolocator."""
        if self._geolocator is None:
            Nominatim = _lazy_import_geopy()
            if Nominatim:
                self._geolocator = Nominatim(user_agent=self.user_agent, timeout=self.timeout)
        return self._geolocator
    
    def _is_valid_netherlands_coords(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Netherlands bounds."""
        return (
            NETHERLANDS_LAT_MIN <= lat <= NETHERLANDS_LAT_MAX and
            NETHERLANDS_LON_MIN <= lon <= NETHERLANDS_LON_MAX and
            not (abs(lat) < 1 and abs(lon) < 1)
        )
    
    def geocode_location(
        self,
        location_text: str,
        country_code: str = "nl"
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Geocode a single location string.
        
        Returns
        -------
        Tuple[lat, lon, full_address]
        """
        if not location_text or pd.isna(location_text):
            return None, None, None
        
        location_text = str(location_text).strip()
        if not location_text:
            return None, None, None
        
        if location_text in self._cache:
            return self._cache[location_text]
        
        geolocator = self._get_geolocator()
        if not geolocator:
            return None, None, None
        
        try:
            query = f"{location_text}, {country_code}"
            result = geolocator.geocode(query)
            if result:
                lat, lon = result.latitude, result.longitude
                if self._is_valid_netherlands_coords(lat, lon):
                    self._cache[location_text] = (lat, lon, result.address)
                    return lat, lon, result.address
        except Exception as e:
            logger.debug(f"Geocoding failed for '{location_text}': {e}")
        
        self._cache[location_text] = (None, None, None)
        return None, None, None
    
    def geocode_ner_locations(
        self,
        df: pd.DataFrame,
        entity_column: str = 'alert_entities',
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract location entities from NER and geocode them.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with NER entities
        entity_column : str
            Column containing NER entity JSON
        text_column : str, optional
            Fallback text column if no location entities found
            
        Returns
        -------
        pd.DataFrame
            DataFrame with geocoded coordinates added
        """
        logger.info("  Geocoding NER location entities...")
        
        df = df.copy()
        df['geo_lat'] = np.nan
        df['geo_lon'] = np.nan
        df['geo_address'] = None
        
        for idx, row in df.iterrows():
            lat, lon, address = None, None, None
            
            if entity_column in df.columns and pd.notna(row.get(entity_column)):
                try:
                    entities = json.loads(row[entity_column])
                    loc_entities = [e for e in entities if e.get('entity_type') == 'LOC']
                    for ent in loc_entities:
                        loc_text = ent.get('entity_text', '')
                        lat, lon, address = self.geocode_location(loc_text)
                        if lat is not None:
                            break
                except (json.JSONDecodeError, TypeError):
                    pass
            
            if lat is None and text_column and text_column in df.columns:
                text = str(row.get(text_column, ''))
                if text:
                    import re
                    dutch_cities = [
                        'amsterdam', 'rotterdam', 'the hague', 'den haag', 'utrecht',
                        'eindhoven', 'tilburg', 'groningen', 'almere', 'breda',
                        'nijmegen', 'enschede', 'haarlem', 'arnhem', 'amersfoort',
                        'haarlemmermeer', 'zuidhorn', 'maastricht', 'leiden',
                        'apeldoorn', 'amersfoort', 'zoetermeer', 'purmerend'
                    ]
                    text_lower = text.lower()
                    for city in dutch_cities:
                        if city in text_lower:
                            lat, lon, address = self.geocode_location(city)
                            if lat is not None:
                                break
            
            if pd.notna(row.get('first_lat')) and pd.notna(row.get('first_lon')):
                lat = row['first_lat']
                lon = row['first_lon']
                address = row.get('first_loc_text')
            
            df.at[idx, 'geo_lat'] = lat
            df.at[idx, 'geo_lon'] = lon
            df.at[idx, 'geo_address'] = address
        
        logger.info(f"  Geocoded {df['geo_lat'].notna().sum()} locations")
        return df
    
    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean coordinates to Netherlands bounds."""
        logger.info("  Cleaning coordinates to Netherlands bounds...")
        
        df = df.copy()
        
        lat_col = 'geo_lat' if 'geo_lat' in df.columns else 'first_lat'
        lon_col = 'geo_lon' if 'geo_lon' in df.columns else 'first_lon'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            return df
        
        bad_mask = (
            df[lat_col].isna() |
            df[lon_col].isna() |
            (df[lat_col] < NETHERLANDS_LAT_MIN) |
            (df[lat_col] > NETHERLANDS_LAT_MAX) |
            (df[lon_col] < NETHERLANDS_LON_MIN) |
            (df[lon_col] > NETHERLANDS_LON_MAX) |
            (
                (df[lat_col].abs() < 1) &
                (df[lon_col].abs() < 1)
            )
        )
        
        n_bad = bad_mask.sum()
        if n_bad > 0:
            logger.info(f"  Removing {n_bad} out-of-bounds coordinates")
            df.loc[bad_mask, [lat_col, lon_col]] = np.nan
        
        return df
    
    def add_risk_levels(
        self,
        df: pd.DataFrame,
        target_column: str = 'disruption_target'
    ) -> pd.DataFrame:
        """Add risk level column for map visualization."""
        logger.info("  Adding risk levels...")
        
        df = df.copy()
        
        if 'risk_level' not in df.columns:
            if target_column in df.columns:
                df['risk_level'] = df[target_column].map({
                    1: "high", 0: "low"
                }).fillna("low")
            else:
                df['risk_level'] = "low"
        
        if 'disruption_class' not in df.columns:
            df['disruption_class'] = "unknown"
        
        return df
    
    def build_folium_map(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Build interactive Folium map of disruptions.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with geocoded locations and risk levels
        output_path : str, optional
            Path to save HTML map
            
        Returns
        -------
        folium.Map or None
        """
        try:
            import folium
            from folium.plugins import MarkerCluster
        except ImportError:
            logger.warning("folium not installed - map generation disabled")
            return None
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        lat_col = 'geo_lat' if 'geo_lat' in df.columns else 'first_lat'
        lon_col = 'geo_lon' if 'geo_lon' in df.columns else 'first_lon'
        
        map_df = df.dropna(subset=[lat_col, lon_col]).copy()
        map_df = map_df[
            (map_df[lat_col] >= NETHERLANDS_LAT_MIN) &
            (map_df[lat_col] <= NETHERLANDS_LAT_MAX) &
            (map_df[lon_col] >= NETHERLANDS_LON_MIN) &
            (map_df[lon_col] <= NETHERLANDS_LON_MAX)
        ]
        
        if map_df.empty:
            logger.warning("No valid coordinates for map")
            return None
        
        RISK_COLOURS = {
            "critical": "#ff4757",
            "high": "#ff6b35",
            "moderate": "#ffa502",
            "low": "#2ed573",
            "unknown": "#808080",
        }
        
        center_lat = float(map_df[lat_col].mean())
        center_lon = float(map_df[lon_col].mean())
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="CartoDB positron"
        )
        
        risk_levels = map_df["risk_level"].fillna("unknown").unique().tolist()
        risk_clusters = {}
        for level in risk_levels:
            risk_clusters[level] = MarkerCluster(name=f"Risk: {level}").add_to(m)
        
        for _, row in map_df.iterrows():
            level = str(row.get("risk_level", "unknown")).lower()
            colour = RISK_COLOURS.get(level, "#808080")
            
            popup_lines = [f"<b>Risk level:</b> {level}"]
            for col in ["disruption_class", "alert_sentiment", "disruption_target",
                        "geo_address", "route_id", "cause", "effect"]:
                if col in row.index and pd.notna(row.get(col)):
                    popup_lines.append(f"<b>{col}:</b> {row[col]}")
            
            popup_html = "<br>".join(popup_lines)
            
            folium.CircleMarker(
                location=[float(row[lat_col]), float(row[lon_col])],
                radius=7,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=level,
            ).add_to(risk_clusters.get(level, m))
        
        folium.LayerControl(collapsed=False).add_to(m)
        
        legend_html = """
        <div style="position:fixed; bottom:50px; left:50px; z-index:9999;
                    background:white; border:2px solid #ccc; border-radius:6px;
                    padding:12px; font-size:12px; font-family:sans-serif;">
          <b>Risk Level</b><br>
        """
        for level, colour in RISK_COLOURS.items():
            legend_html += (
                f'<i style="background:{colour};width:12px;height:12px;'
                f'display:inline-block;margin-right:6px;border-radius:50%;"></i>'
                f'{level}<br>'
            )
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))
        
        output_path = output_path or os.path.join(self.cache_dir, "disruption_risk_map.html")
        m.save(output_path)
        logger.info(f"  Map saved to {output_path}")
        
        return m


def add_geocoding_features(
    df: pd.DataFrame,
    user_agent: str = "gtfs_disruption_pipeline"
) -> pd.DataFrame:
    """
    Convenience function to add geocoding features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with alert data
    user_agent : str
        User agent for Nominatim
        
    Returns
    -------
    pd.DataFrame
        DataFrame with geocoded coordinates
    """
    geocoder = AlertGeocoder(user_agent=user_agent)
    
    df = geocoder.geocode_ner_locations(df)
    df = geocoder.clean_coordinates(df)
    df = geocoder.add_risk_levels(df)
    
    return df