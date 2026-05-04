"""Tests for Flask API"""

import pytest
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.api import app, load_model_and_config
from src.models import ModelRegistry
from src.features import FeatureEngineer
from src.data import load_and_prepare


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create test client with mocked model."""
    # Change to tmp directory
    monkeypatch.chdir(tmp_path)
    
    # Create directories
    (tmp_path / "models").mkdir()
    (tmp_path / "logs").mkdir()
    
    # Create minimal config
    config = {
        "api": {"host": "0.0.0.0", "port": 5000, "debug": False},
        "monitoring": {"enabled": True}
    }
    with open(tmp_path / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f)
    
    # Train and save a simple model
    X, _ = load_and_prepare(dataset="iris")
    engineer = FeatureEngineer(scale_features=True, n_components_pca=2)
    X_transformed = engineer.fit_transform(X)
    
    registry = ModelRegistry()
    registry.train_kmeans(X_transformed, n_clusters=3)
    
    # Save model and feature engineer
    joblib.dump(registry.models["kmeans"], tmp_path / "models" / "best_model.pkl")
    joblib.dump(engineer, tmp_path / "models" / "feature_engineer.pkl")
    
    # Load model
    load_model_and_config()
    
    # Configure app for testing
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check with loaded model."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestClusterEndpoint:
    """Test clustering endpoint."""
    
    def test_cluster_valid_request(self, client):
        """Test clustering with valid request."""
        request_data = {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        
        response = client.post(
            '/cluster',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "cluster_id" in data
        assert "model" in data
        assert isinstance(data["cluster_id"], int)
        assert 0 <= data["cluster_id"] <= 2
    
    def test_cluster_invalid_features(self, client):
        """Test clustering with invalid features."""
        request_data = {
            "features": [5.1, 3.5, float('nan'), 0.2]
        }
        
        response = client.post(
            '/cluster',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_cluster_missing_features(self, client):
        """Test clustering without features field."""
        request_data = {}
        
        response = client.post(
            '/cluster',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_cluster_empty_features(self, client):
        """Test clustering with empty features."""
        request_data = {
            "features": []
        }
        
        response = client.post(
            '/cluster',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400


class TestBatchClusterEndpoint:
    """Test batch clustering endpoint."""
    
    def test_batch_cluster_valid(self, client):
        """Test batch clustering with valid requests."""
        request_data = {
            "samples": [
                {"features": [5.1, 3.5, 1.4, 0.2]},
                {"features": [4.9, 3.0, 1.4, 0.2]},
                {"features": [6.5, 3.0, 5.2, 2.0]}
            ]
        }
        
        response = client.post(
            '/cluster/batch',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "predictions" in data
        assert "cluster_distribution" in data
        assert "n_samples" in data
        assert len(data["predictions"]) == 3
        assert data["n_samples"] == 3
    
    def test_batch_cluster_empty(self, client):
        """Test batch clustering with empty samples."""
        request_data = {
            "samples": []
        }
        
        response = client.post(
            '/cluster/batch',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert b'clustering_latency_seconds' in response.data


class TestInfoEndpoint:
    """Test model info endpoint."""
    
    def test_info_endpoint(self, client):
        """Test info endpoint returns model metadata."""
        response = client.get('/info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "model_type" in data
        assert "model_loaded" in data
        assert data["model_loaded"] is True
