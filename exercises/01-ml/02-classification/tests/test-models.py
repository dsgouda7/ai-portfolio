"""Tests for model training and registry"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.models import ModelRegistry
from src.data import load_and_split


@pytest.fixture
def sample_data():
    """Fixture providing sample train/test data."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(random_state=42)
    
    # Use smaller subset for faster tests
    X_train_small = X_train[:50]
    X_val_small = X_val[:20]
    y_train_small = y_train[:50]
    y_val_small = y_val[:20]
    
    return X_train_small, X_val_small, y_train_small, y_val_small


@pytest.fixture
def registry():
    """Fixture providing empty ModelRegistry."""
    return ModelRegistry()


class TestModelRegistry:
    """Test suite for ModelRegistry class."""
    
    def test_init(self, registry):
        """Test initialization of registry."""
        assert len(registry.models) == 0
        assert len(registry.cv_scores) == 0
        assert registry.best_model_name is None
    
    def test_train_logistic_regression(self, registry, sample_data):
        """Test training LogisticRegression model."""
        X_train, _, y_train, _ = sample_data
        
        metrics = registry.train_logistic_regression(X_train, y_train, C=1.0, cv_folds=3)
        
        # Check model stored
        assert "logistic_regression" in registry.models
        assert "logistic_regression" in registry.cv_scores
        
        # Check metrics
        assert "train_accuracy" in metrics
        assert "cv_accuracy" in metrics
        
        # Metrics should be reasonable
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        assert 0.0 <= metrics["cv_accuracy"] <= 1.0
    
    def test_predict_without_training(self, registry, sample_data):
        """Test that predict before training raises error."""
        X_train, _, _, _ = sample_data
        
        with pytest.raises(RuntimeError, match="No models trained"):
            registry.predict(X_train)
    
    def test_predict_with_model_name(self, registry, sample_data):
        """Test prediction with specific model name."""
        X_train, X_val, y_train, _ = sample_data
        
        # Train model
        registry.train_logistic_regression(X_train, y_train, cv_folds=3)
        
        # Predict
        predictions = registry.predict(X_val, model_name="logistic_regression")
        
        assert len(predictions) == len(X_val)
        assert predictions.dtype in [np.int32, np.int64]
    
    def test_save_and_load_model(self, registry, sample_data):
        """Test saving and loading model."""
        X_train, X_val, y_train, _ = sample_data
        
        # Train model
        registry.train_logistic_regression(X_train, y_train, cv_folds=3)
        predictions_before = registry.predict(X_val, model_name="logistic_regression")
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lr_test.pkl"
            registry.save_model("logistic_regression", str(model_path))
            
            # Create new registry and load
            new_registry = ModelRegistry()
            new_registry.load_model("logistic_regression", str(model_path))
            
            # Predictions should match
            predictions_after = new_registry.predict(X_val, model_name="logistic_regression")
            np.testing.assert_array_equal(predictions_before, predictions_after)
