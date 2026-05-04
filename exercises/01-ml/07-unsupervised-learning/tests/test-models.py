"""Tests for model training module"""

import pytest
import pandas as pd
import numpy as np
from src.models import ModelRegistry
from src.data import load_and_prepare
from src.features import FeatureEngineer


class TestModelRegistry:
    """Test model training and registry."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data."""
        X, _ = load_and_prepare(dataset="iris")
        engineer = FeatureEngineer(scale_features=True, n_components_pca=2)
        X_transformed = engineer.fit_transform(X)
        return X_transformed
    
    def test_initialization(self):
        """Test registry initialization."""
        registry = ModelRegistry()
        
        assert isinstance(registry.models, dict)
        assert len(registry.models) == 0
        assert registry.best_model_name is None
    
    def test_train_kmeans(self, sample_data):
        """Test KMeans training."""
        registry = ModelRegistry()
        metrics = registry.train_kmeans(sample_data, n_clusters=3)
        
        assert "kmeans" in registry.models
        assert "silhouette_score" in metrics
        assert "n_clusters" in metrics
        assert "inertia" in metrics
        assert metrics["n_clusters"] == 3
        assert -1 <= metrics["silhouette_score"] <= 1
    
    def test_train_dbscan(self, sample_data):
        """Test DBSCAN training."""
        registry = ModelRegistry()
        metrics = registry.train_dbscan(sample_data, eps=0.5, min_samples=5)
        
        assert "dbscan" in registry.models
        assert "n_clusters" in metrics
        assert "n_noise" in metrics
        assert "noise_ratio" in metrics
        assert metrics["n_clusters"] >= 0
    
    def test_train_hierarchical(self, sample_data):
        """Test Hierarchical clustering training."""
        registry = ModelRegistry()
        metrics = registry.train_hierarchical(sample_data, n_clusters=3, linkage='ward')
        
        assert "hierarchical" in registry.models
        assert "silhouette_score" in metrics
        assert "n_clusters" in metrics
        assert "linkage" in metrics
        assert metrics["n_clusters"] == 3
        assert metrics["linkage"] == "ward"
    
    def test_train_gmm(self, sample_data):
        """Test GMM training."""
        registry = ModelRegistry()
        metrics = registry.train_gmm(sample_data, n_components=3)
        
        assert "gmm" in registry.models
        assert "silhouette_score" in metrics
        assert "n_components" in metrics
        assert "bic" in metrics
        assert "aic" in metrics
        assert metrics["n_components"] == 3
    
    def test_predict_kmeans(self, sample_data):
        """Test prediction with KMeans."""
        registry = ModelRegistry()
        registry.train_kmeans(sample_data, n_clusters=3)
        
        labels = registry.predict(sample_data, "kmeans")
        
        assert len(labels) == len(sample_data)
        assert set(labels).issubset({0, 1, 2})
    
    def test_predict_nonexistent_model(self, sample_data):
        """Test error when predicting with nonexistent model."""
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="Model .* not found"):
            registry.predict(sample_data, "nonexistent")
    
    def test_get_best_model(self, sample_data):
        """Test getting best model by silhouette score."""
        registry = ModelRegistry()
        registry.train_kmeans(sample_data, n_clusters=3)
        registry.train_hierarchical(sample_data, n_clusters=3)
        
        best = registry.get_best_model()
        
        assert best in ["kmeans", "hierarchical"]
        assert registry.best_model_name == best
    
    def test_get_best_model_no_training(self):
        """Test error when getting best model without training."""
        registry = ModelRegistry()
        
        with pytest.raises(RuntimeError, match="No models trained"):
            registry.get_best_model()
    
    def test_invalid_n_clusters(self, sample_data):
        """Test error for invalid n_clusters."""
        registry = ModelRegistry()
        
        with pytest.raises(ValueError):
            registry.train_kmeans(sample_data, n_clusters=0)
        
        with pytest.raises(ValueError):
            registry.train_kmeans(sample_data, n_clusters=len(sample_data) + 1)
    
    def test_invalid_dbscan_params(self, sample_data):
        """Test error for invalid DBSCAN parameters."""
        registry = ModelRegistry()
        
        with pytest.raises(ValueError):
            registry.train_dbscan(sample_data, eps=-1)
        
        with pytest.raises(ValueError):
            registry.train_dbscan(sample_data, min_samples=0)
    
    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading models."""
        registry1 = ModelRegistry()
        registry1.train_kmeans(sample_data, n_clusters=3)
        
        # Save model
        model_path = tmp_path / "kmeans.pkl"
        registry1.save("kmeans", str(model_path))
        
        assert model_path.exists()
        
        # Load model
        registry2 = ModelRegistry()
        registry2.load("kmeans", str(model_path))
        
        assert "kmeans" in registry2.models
        
        # Predictions should match
        labels1 = registry1.predict(sample_data, "kmeans")
        labels2 = registry2.predict(sample_data, "kmeans")
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_cluster_labels_storage(self, sample_data):
        """Test that cluster labels are stored."""
        registry = ModelRegistry()
        registry.train_kmeans(sample_data, n_clusters=3)
        
        assert "kmeans" in registry.cluster_labels
        assert len(registry.cluster_labels["kmeans"]) == len(sample_data)
    
    @pytest.mark.slow
    def test_multiple_models_comparison(self, sample_data):
        """Test training and comparing multiple models."""
        registry = ModelRegistry()
        
        # Train all models
        registry.train_kmeans(sample_data, n_clusters=3)
        registry.train_dbscan(sample_data, eps=0.5, min_samples=5)
        registry.train_hierarchical(sample_data, n_clusters=3)
        registry.train_gmm(sample_data, n_components=3)
        
        # Check all models trained
        assert len(registry.models) == 4
        assert len(registry.scores) == 4
        
        # Get best model
        best = registry.get_best_model()
        assert best in registry.models
        assert best in registry.scores
