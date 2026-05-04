"""Pytest configuration and shared fixtures for EnsembleAI tests"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


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


@pytest.fixture
def sample_data():
    """Generate sample classification data for testing.
    
    Returns:
        Tuple of (X, y) with 100 samples, 10 features
    """
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    feature_names = [f"feature_{i:02d}" for i in range(10)]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    
    return X, y


@pytest.fixture
def sample_splits(sample_data):
    """Generate train/test splits from sample data.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def temp_config(tmp_path):
    """Create temporary config file for testing.
    
    Args:
        tmp_path: pytest temporary directory fixture
    
    Returns:
        Path to temporary config file
    """
    import yaml
    
    config = {
        'data': {
            'dataset': 'make_classification',
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        },
        'features': {
            'scale_features': True,
            'feature_selection': True,
            'top_k_features': 8
        },
        'base_models': {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        },
        'ensemble': {
            'voting': {
                'strategy': 'soft'
            }
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path
