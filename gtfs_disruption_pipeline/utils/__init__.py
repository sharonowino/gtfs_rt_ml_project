"""
GTFS Disruption Detection - Utility Functions
"""
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'logs/pipeline.log')
    
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor and report memory usage."""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def get_available_memory():
        """Get available system memory in GB."""
        import psutil
        return psutil.virtual_memory().available / 1024 / 1024 / 1024
    
    @staticmethod
    def estimate_dataframe_memory(df):
        """Estimate DataFrame memory usage in MB."""
        return df.memory_usage(deep=True).sum() / 1024 / 1024
    
    @staticmethod
    def print_memory_status(label=""):
        """Print current memory status."""
        used = MemoryMonitor.get_memory_usage()
        available = MemoryMonitor.get_available_memory()
        print(f" Memory {label}: {used:.1f} MB used | {available:.1f} GB available")


# Import experiment tracking module
from .experiment_tracking import (
    ExperimentTracker,
    ExperimentManager,
    create_experiment_tracker,
    create_experiment_manager
)

# Import monitoring module
from .monitoring import (
    DriftDetector,
    PerformanceTracker,
    ModelMonitor,
    create_monitor
)

# Import test data module
from .test_data import (
    load_test_data,
    load_sample_for_dashboard,
    load_vehicle_positions,
    load_alerts,
    load_trip_updates,
    add_test_features
)

__all__ = [
    'load_config',
    'setup_logging',
    'MemoryMonitor',
    'ExperimentTracker',
    'ExperimentManager',
    'create_experiment_tracker',
    'create_experiment_manager',
    'DriftDetector',
    'PerformanceTracker',
    'ModelMonitor',
    'create_monitor',
    'load_test_data',
    'load_sample_for_dashboard',
    'load_vehicle_positions',
    'load_alerts',
    'load_trip_updates',
    'add_test_features',
]
