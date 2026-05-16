"""
Pytest configuration and shared fixtures
"""
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    data_dir = Path("data/temp")
    data_dir.mkdir(parents=True, exist_ok=True)
    yield data_dir
    # Cleanup after tests
    # In production, you might want to clean up test data


@pytest.fixture(scope="session")
def test_models_dir():
    """Create temporary directory for test models"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    yield models_dir


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "model_registry_uri": "http://localhost:5000"
        },
        "feature_store": {
            "project_name": "test_features",
            "online_store": "redis",
            "offline_store": "parquet"
        },
        "model_monitoring": {
            "drift_threshold": 0.15,
            "alert_on_drift": False
        }
    }
