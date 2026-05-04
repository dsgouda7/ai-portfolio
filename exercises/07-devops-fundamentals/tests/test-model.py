"""Tests for ML model."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from src.model import MLModel, create_dummy_data


def test_model_creation():
    """Test model initialization."""
    model = MLModel()
    assert model.model is not None


def test_model_training(model, dummy_data):
    """Test model training."""
    X, y = dummy_data
    model.train(X, y)
    # Model should be able to predict after training
    predictions = model.predict(X[:5])
    assert len(predictions) == 5
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)


def test_model_prediction(model):
    """Test model predictions."""
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    predictions = model.predict(X)
    assert predictions.shape == (1,)
    assert isinstance(predictions[0], (int, float, np.number))


def test_model_save_load(model, dummy_data):
    """Test model persistence."""
    X, y = dummy_data
    model.train(X, y)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        model.save(str(model_path))
        assert model_path.exists()
        
        # Load model
        new_model = MLModel()
        new_model.load(str(model_path))
        
        # Predictions should be identical
        predictions1 = model.predict(X[:5])
        predictions2 = new_model.predict(X[:5])
        np.testing.assert_array_almost_equal(predictions1, predictions2)


def test_create_dummy_data():
    """Test dummy data generation."""
    X, y = create_dummy_data(n_samples=100)
    assert X.shape == (100, 5)
    assert y.shape == (100,)


def test_model_predict_raises_without_training():
    """Test that prediction raises error without training."""
    model = MLModel()
    model.model = None  # Force None
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    
    with pytest.raises(ValueError, match="Model not trained or loaded"):
        model.predict(X)
