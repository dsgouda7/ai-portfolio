"""Tests for model training and registry"""

import pytest
import pandas as pd
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
    X_train_small = X_train.head(200)
    X_val_small = X_val.head(50)
    y_train_small = y_train.head(200)
    y_val_small = y_val.head(50)
    
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
    
    def test_train_ridge(self, registry, sample_data):
        """Test training Ridge model."""
        X_train, _, y_train, _ = sample_data
        
        metrics = registry.train_ridge(X_train, y_train, alpha=1.0, cv_folds=3)
        
        # Check model stored
        assert "ridge" in registry.models
        assert "ridge" in registry.cv_scores
        
        # Check metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "cv_mae" in metrics
        
        # Metrics should be reasonable
        assert metrics["mae"] > 0
        assert metrics["r2"] <= 1.0
    
    def test_train_lasso(self, registry, sample_data):
        """Test training Lasso model."""
        X_train, _, y_train, _ = sample_data
        
        metrics = registry.train_lasso(X_train, y_train, alpha=0.1, cv_folds=3)
        
        # Check model stored
        assert "lasso" in registry.models
        assert "lasso" in registry.cv_scores
        
        # Check Lasso-specific metrics
        assert "n_nonzero_coef" in metrics
        assert metrics["n_nonzero_coef"] > 0
    
    def test_train_xgboost(self, registry, sample_data):
        """Test training XGBoost model."""
        X_train, _, y_train, _ = sample_data
        
        metrics = registry.train_xgboost(
            X_train, y_train,
            n_estimators=10,  # Small for speed
            max_depth=3,
            cv_folds=3
        )
        
        # Check model stored
        assert "xgboost" in registry.models
        assert "xgboost" in registry.cv_scores
        
        # Check metrics
        assert metrics["mae"] > 0
        assert metrics["cv_mae"] > 0
    
    def test_train_ridge_invalid_alpha(self, registry, sample_data):
        """Test that negative alpha raises error."""
        X_train, _, y_train, _ = sample_data
        
        with pytest.raises(ValueError):
            registry.train_ridge(X_train, y_train, alpha=-1.0)
    
    def test_predict_without_training(self, registry, sample_data):
        """Test that predict before training raises error."""
        X_train, _, _, _ = sample_data
        
        with pytest.raises(RuntimeError, match="No models trained"):
            registry.predict(X_train)
    
    def test_predict_with_model_name(self, registry, sample_data):
        """Test prediction with specific model name."""
        X_train, X_val, y_train, _ = sample_data
        
        # Train model
        registry.train_ridge(X_train, y_train, cv_folds=3)
        
        # Predict
        predictions = registry.predict(X_val, model_name="ridge")
        
        assert len(predictions) == len(X_val)
        assert predictions.dtype in [np.float64, np.float32]
    
    def test_predict_best_model(self, registry, sample_data):
        """Test prediction with best model (no model_name specified)."""
        X_train, X_val, y_train, _ = sample_data
        
        # Train multiple models
        registry.train_ridge(X_train, y_train, alpha=1.0, cv_folds=3)
        registry.train_lasso(X_train, y_train, alpha=0.1, cv_folds=3)
        
        # Predict with best model
        predictions = registry.predict(X_val)
        
        assert len(predictions) == len(X_val)
    
    def test_predict_invalid_model_name(self, registry, sample_data):
        """Test that invalid model name raises error."""
        X_train, X_val, y_train, _ = sample_data
        
        registry.train_ridge(X_train, y_train, cv_folds=3)
        
        with pytest.raises(ValueError, match="not found"):
            registry.predict(X_val, model_name="nonexistent")
    
    def test_get_best_model_name(self, registry, sample_data):
        """Test getting best model name."""
        X_train, _, y_train, _ = sample_data
        
        # Train multiple models
        registry.train_ridge(X_train, y_train, alpha=1.0, cv_folds=3)
        registry.train_lasso(X_train, y_train, alpha=0.1, cv_folds=3)
        
        best_name = registry.get_best_model_name()
        
        assert best_name in ["ridge", "lasso"]
        assert best_name == registry.best_model_name
    
    def test_get_best_model_name_no_models(self, registry):
        """Test that getting best model without training raises error."""
        with pytest.raises(RuntimeError, match="No models trained"):
            registry.get_best_model_name()
    
    def test_save_and_load_model(self, registry, sample_data):
        """Test saving and loading model."""
        X_train, X_val, y_train, _ = sample_data
        
        # Train model
        registry.train_ridge(X_train, y_train, cv_folds=3)
        predictions_before = registry.predict(X_val, model_name="ridge")
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "ridge_test.pkl"
            registry.save_model("ridge", str(model_path))
            
            # Create new registry and load
            new_registry = ModelRegistry()
            new_registry.load_model("ridge", str(model_path))
            
            # Predictions should match
            predictions_after = new_registry.predict(X_val, model_name="ridge")
            np.testing.assert_array_almost_equal(predictions_before, predictions_after)
    
    def test_save_nonexistent_model(self, registry):
        """Test that saving nonexistent model raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            
            with pytest.raises(ValueError, match="not found"):
                registry.save_model("nonexistent", str(model_path))
    
    def test_load_nonexistent_file(self, registry):
        """Test that loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            registry.load_model("model", "/nonexistent/path/model.pkl")
    
    def test_get_model_summary(self, registry, sample_data):
        """Test getting model summary."""
        X_train, _, y_train, _ = sample_data
        
        # Train multiple models
        registry.train_ridge(X_train, y_train, cv_folds=3)
        registry.train_lasso(X_train, y_train, cv_folds=3)
        
        summary = registry.get_model_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "model" in summary.columns
        assert "cv_mae" in summary.columns
    
    def test_get_model_summary_empty(self, registry):
        """Test getting summary with no models."""
        summary = registry.get_model_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
    
    def test_multiple_models_comparison(self, registry, sample_data):
        """Test training multiple models and comparing performance."""
        X_train, _, y_train, _ = sample_data
        
        # Train all model types
        ridge_metrics = registry.train_ridge(X_train, y_train, alpha=1.0, cv_folds=3)
        lasso_metrics = registry.train_lasso(X_train, y_train, alpha=0.1, cv_folds=3)
        xgb_metrics = registry.train_xgboost(
            X_train, y_train,
            n_estimators=10,
            max_depth=3,
            cv_folds=3
        )
        
        # All should have positive MAE
        assert ridge_metrics["cv_mae"] > 0
        assert lasso_metrics["cv_mae"] > 0
        assert xgb_metrics["cv_mae"] > 0
        
        # Best model should be one of them
        best = registry.get_best_model_name()
        assert best in ["ridge", "lasso", "xgboost"]
