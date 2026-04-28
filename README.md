# GTFS Disruption Detection - Modular Package

A modular, production-ready pipeline for detecting and classifying transit disruptions using GTFS (General Transit Feed Specification) data.

## Project Structure

```
gtfs_disruption/
├── __init__.py              # Package initialization
├── config.yaml              # Configuration file
├── pipeline.py              # Main pipeline orchestrator
├── features/                # Feature engineering module
│   ├── __init__.py          # DisruptionFeatureBuilder
│   ├── classifier.py        # DisruptionClassifier
│   └── analyzer.py          # DisruptionAnalyzer
├── modeling/                # Modeling module
│   ├── __init__.py          # Temporal splitting, class balancing
│   └── leakage.py           # Leakage detection utilities
├── evaluation/              # Evaluation module
│   └── __init__.py          # Metrics and visualization
├── utils/                   # Utility functions
│   └── __init__.py          # Config loading, logging, memory monitoring
└── tests/                   # Unit tests
    ├── test_features.py     # Feature engineering tests
    └── test_modeling.py     # Modeling tests
```

## Key Improvements Over Original Notebook

### 1. **Modular Structure**
- Separated concerns into distinct modules
- Each module has a single responsibility
- Easy to import and use individual components

### 2. **Configuration Management**
- All parameters in `config.yaml`
- No hardcoded values
- Easy to modify thresholds and settings

### 3. **Logging**
- Replaced print statements with proper logging
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Logs to both file and console

### 4. **Unit Tests**
- Comprehensive test coverage
- pytest-based testing framework
- Tests for all major components

### 5. **Leakage Prevention**
- Chronological temporal splitting
- Temporal-aware class balancing
- Leakage detection utilities
- Backward-looking feature computation

### 6. **Code Quality**
- Type hints throughout
- Comprehensive docstrings
- Error handling with specific exceptions
- No bare `except:` clauses

## Usage

### Basic Usage

```python
from gtfs_disruption import DisruptionPipeline

# Initialize pipeline
pipeline = DisruptionPipeline("config.yaml")

# Run full pipeline
results = pipeline.run_full_pipeline(merged_df, gtfs_data)

# Access results
feature_df = results['feature_df']
classified_df = results['classified_df']
model = results['model']
metrics = results['metrics']
```

### Individual Components

```python
from gtfs_disruption.features import DisruptionFeatureBuilder
from gtfs_disruption.features.classifier import DisruptionClassifier
from gtfs_disruption.modeling import chronological_split

# Feature engineering
builder = DisruptionFeatureBuilder(merged_df, gtfs_data)
feature_df = builder.build()

# Classification
classifier = DisruptionClassifier()
classified_df = classifier.classify(feature_df)

# Temporal split
train_df, val_df, test_df = chronological_split(
    classified_df, 
    'feed_timestamp',
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
)
```

## Configuration

Edit `config.yaml` to customize:

- **Disruption thresholds**: Delay, speed, severity scoring
- **Temporal split ratios**: Train/val/test proportions
- **Class balancing strategy**: class_weight, oversample, none
- **Leakage detection**: Correlation thresholds, lookback windows
- **Logging**: Level, format, output file

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest gtfs_disruption/tests/ -v

# Run specific test file
pytest gtfs_disruption/tests/test_features.py -v

# Run with coverage
pytest gtfs_disruption/tests/ --cov=gtfs_disruption --cov-report=html
```

## Dependencies

- pandas
- numpy
- geopandas
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn
- shap
- matplotlib
- seaborn
- pyyaml
- psutil

Install with:
```bash
pip install -r requirements.txt
```

## Key Classes

### DisruptionFeatureBuilder
Fuses trip updates, vehicle positions, and service alerts into a feature DataFrame.

### DisruptionClassifier
Labels each stop event with disruption type and severity score.

### DisruptionAnalyzer
Provides higher-level analytics (hot spots, time profiles, alert breakdowns).

### TemporalAwareBalancer
Handles class imbalance without breaking temporal ordering.

### WalkForwardCV
Walk-forward cross-validation for temporal data.

## Leakage Prevention

The package includes several mechanisms to prevent data leakage:

1. **Chronological Splitting**: Always split by time order, never randomly
2. **Temporal-Aware Balancing**: Use class weights instead of SMOTE
3. **Leakage Detection**: Flag features with suspicious correlation differences
4. **Backward-Looking Features**: Compute network load using only historical data
5. **Train-Only Computation**: Calculate stop load using only training data

## License

This project is part of an academic thesis on GTFS disruption detection.
