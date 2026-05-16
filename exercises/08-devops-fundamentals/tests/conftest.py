"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from src.model import MLModel, create_dummy_data
from src.api import app as flask_app


@pytest.fixture
def model():
    """Create a trained model for testing."""
    model = MLModel()
    X, y = create_dummy_data(n_samples=100)
    model.train(X, y)
    return model


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    return create_dummy_data(n_samples=50)


@pytest.fixture
def app():
    """Create Flask app for testing."""
    flask_app.config['TESTING'] = True
    return flask_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_features():
    """Sample features for prediction testing."""
    return [[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.5, 2.5, 3.5, 4.5]]
