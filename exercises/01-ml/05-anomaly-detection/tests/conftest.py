"""Pytest configuration and shared fixtures for FraudShield tests"""

import pytest
import pandas as pd
import numpy as np

from src.data import load_and_split
from src.features import FeatureEngineer


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )


@pytest.fixture(scope="session")
def sample_data():
    """Generate small sample dataset for testing."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
        n_samples=1000,
        n_features=10,
        contamination=0.1,
        random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


@pytest.fixture
def feature_engineer():
    """Create feature engineer instance."""
    return FeatureEngineer(scale_features=True)


@pytest.fixture
def sample_features():
    """Generate sample feature array."""
    return pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f"feature_{i+1}" for i in range(10)]
    )
