"""Tests for model training and registry for FraudShield"""

import pytest
import pandas as pd
import numpy as np

from src.models import ModelRegistry


@pytest.fixture
def sample_anomaly_data():
    """Generate sample anomaly detection data."""
    np.random.seed(42)
    
    # Normal data
    X_normal = np.random.randn(800, 10)
    y_normal = np.zeros(800)
    
    # Anomaly data (shifted distribution)
    X_anomaly = np.random.randn(200, 10) * 2 + 3
    y_anomaly = np.ones(200)
    
    # Combine
    X = pd.DataFrame(np.vstack([X_normal, X_anomaly]))
    y = pd.Series(np.concatenate([y_normal, y_anomaly]))
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)
    
    return X, y


class TestModelRegistry:
    """Test suite for ModelRegistry class."""
    
    def test_initialization(self):
        """Test initialization of registry."""
        registry = ModelRegistry()
        
        assert registry.models == {}
        assert registry.best_model_name is None
        assert registry.validation_scores == {}
    
    def test_train_isolation_forest(self, sample_anomaly_data):
        """Test training Isolation Forest."""
        X, y = sample_anomaly_data
        registry = ModelRegistry()
        
        metrics = registry.train_isolation_forest(X, y, contamination=0.2)
        
        # Check metrics returned
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'accuracy' in metrics
        
        # Check model stored
        assert 'isolation_forest' in registry.models
        assert 'isolation_forest' in registry.validation_scores
        
        # Metrics should be reasonable
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
    
    def test_train_one_class_svm(self, sample_anomaly_data):
        """Test training One-Class SVM."""
        X, y = sample_anomaly_data
        registry = ModelRegistry()
        
        metrics = registry.train_one_class_svm(X, y, nu=0.2)
        
        # Check metrics returned
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check model stored
        assert 'one_class_svm' in registry.models
    
    @pytest.mark.slow
    def test_train_autoencoder(self, sample_anomaly_data):
        """Test training Autoencoder."""
        pytest.importorskip("tensorflow")
        
        X, y = sample_anomaly_data
        registry = ModelRegistry()
        
        metrics = registry.train_autoencoder(
            X, y, encoding_dim=5, epochs=5, batch_size=32
        )
        
        # Check metrics returned
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # Check model stored
        assert 'autoencoder' in registry.models
        assert 'model' in registry.models['autoencoder']
        assert 'threshold' in registry.models['autoencoder']
    
    def test_train_isolation_forest_invalid_contamination(self, sample_anomaly_data):
        """Test training with invalid contamination raises error."""
        X, y = sample_anomaly_data
        registry = ModelRegistry()
        
        with pytest.raises(ValueError):
            registry.train_isolation_forest(X, y, contamination=-0.1)
    
    def test_predict_before_training_raises(self):
        """Test predict before training raises error."""
        X = pd.DataFrame(np.random.randn(10, 10))
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="not found"):
            registry.predict(X, "isolation_forest")
    
    def test_predict_isolation_forest(self, sample_anomaly_data):
        """Test prediction with Isolation Forest."""
        X, y = sample_anomaly_data
        registry = ModelRegistry()
        
        registry.train_isolation_forest(X, y, contamination=0.2)
        predictions = registry.predict(X, "isolation_forest")
        
        # Check output
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_scores_isolation_forest(self, sample_anomaly_data):
        """Test getting anomaly scores."""
        X, y = sample_anomaly_data
        registry = ModelRegistry()
        
        registry.train_isolation_forest(X, y, contamination=0.2)
        scores = registry.predict_scores(X, "isolation_forest")
        
        # Check output
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(X)
        assert all(np.isfinite(scores))
    
    def test_select_best_model_no_models_raises(self):
        """Test selecting best model with no trained models raises error."""
        registry = ModelRegistry()
        
        with pytest.raises(RuntimeError, match="No models trained"):
            registry.select_best_model()
    
    def test_select_best_model(self, sample_anomaly_data):
        """Test selecting best model."""
        X, y = sample_anomaly_data
        registry = ModelRegistry()
        
        registry.train_isolation_forest(X, y)
        registry.train_one_class_svm(X, y)
        
        best = registry.select_best_model()
        
        assert best in ['isolation_forest', 'one_class_svm']
        assert registry.best_model_name == best
    
    def test_save_and_load_model(self, sample_anomaly_data, tmp_path):
        """Test saving and loading model."""
        X, y = sample_anomaly_data
        registry = ModelRegistry()
        
        registry.train_isolation_forest(X, y)
        
        # Save
        model_path = tmp_path / "test_model.pkl"
        registry.save_model("isolation_forest", str(model_path))
        
        assert model_path.exists()
        
        # Load
        registry2 = ModelRegistry()
        registry2.load_model("isolation_forest", str(model_path))
        
        assert "isolation_forest" in registry2.models
        
        # Predictions should match
        pred1 = registry.predict(X.head(10), "isolation_forest")
        pred2 = registry2.predict(X.head(10), "isolation_forest")
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_load_model_not_found_raises(self, tmp_path):
        """Test loading non-existent model raises error."""
        registry = ModelRegistry()
        
        with pytest.raises(FileNotFoundError):
            registry.load_model("test", str(tmp_path / "nonexistent.pkl"))
