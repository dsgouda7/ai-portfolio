"""Tests for model training"""

import pytest
import numpy as np

from src.models import ModelRegistry


class TestModelRegistry:
    """Test suite for ModelRegistry class."""
    
    def test_model_registry_init(self):
        """Test ModelRegistry initialization."""
        registry = ModelRegistry()
        
        assert registry.models == {}
        assert registry.best_model_name is None
        assert registry.histories == {}
    
    @pytest.mark.slow
    def test_train_dense_nn(self):
        """Test training dense neural network."""
        # Generate small synthetic dataset
        X_train = np.random.randn(200, 10)
        y_train = np.random.randint(0, 3, 200)
        X_val = np.random.randn(50, 10)
        y_val = np.random.randint(0, 3, 50)
        
        registry = ModelRegistry()
        metrics = registry.train_dense_nn(
            X_train, y_train, X_val, y_val,
            architecture=[32, 16],
            epochs=5,
            early_stopping_patience=2
        )
        
        # Check model is stored
        assert "dense_nn" in registry.models
        assert "dense_nn" in registry.histories
        
        # Check metrics
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        assert 0.0 <= metrics["val_accuracy"] <= 1.0
    
    @pytest.mark.slow
    def test_train_cnn_1d(self):
        """Test training 1D CNN."""
        X_train = np.random.randn(200, 10)
        y_train = np.random.randint(0, 3, 200)
        X_val = np.random.randn(50, 10)
        y_val = np.random.randint(0, 3, 50)
        
        registry = ModelRegistry()
        metrics = registry.train_cnn_1d(
            X_train, y_train, X_val, y_val,
            filters=[16, 8],
            epochs=5,
            early_stopping_patience=2
        )
        
        # Check model is stored
        assert "cnn_1d" in registry.models
        
        # Check metrics
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
    
    @pytest.mark.slow
    def test_predict(self):
        """Test prediction with trained model."""
        X_train = np.random.randn(200, 10)
        y_train = np.random.randint(0, 3, 200)
        X_val = np.random.randn(50, 10)
        y_val = np.random.randint(0, 3, 50)
        X_test = np.random.randn(30, 10)
        
        registry = ModelRegistry()
        registry.train_dense_nn(
            X_train, y_train, X_val, y_val,
            architecture=[32, 16],
            epochs=3
        )
        
        predictions = registry.predict(X_test, "dense_nn")
        
        # Check predictions
        assert predictions.shape == (30,)
        assert np.all((predictions >= 0) & (predictions < 3))
    
    def test_predict_model_not_found(self):
        """Test prediction with non-existent model raises ValueError."""
        registry = ModelRegistry()
        X_test = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="not found"):
            registry.predict(X_test, "nonexistent_model")
    
    @pytest.mark.slow
    def test_predict_proba(self):
        """Test probability prediction."""
        X_train = np.random.randn(200, 10)
        y_train = np.random.randint(0, 3, 200)
        X_val = np.random.randn(50, 10)
        y_val = np.random.randint(0, 3, 50)
        X_test = np.random.randn(30, 10)
        
        registry = ModelRegistry()
        registry.train_dense_nn(
            X_train, y_train, X_val, y_val,
            architecture=[32, 16],
            epochs=3
        )
        
        probabilities = registry.predict_proba(X_test, "dense_nn")
        
        # Check probabilities
        assert probabilities.shape == (30, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
