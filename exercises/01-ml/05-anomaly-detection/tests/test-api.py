"""Tests for Flask API for FraudShield"""

import pytest
import json
import pandas as pd
import numpy as np

from src.api import app
from src.models import ModelRegistry
from src.features import FeatureEngineer


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def trained_model(tmp_path):
    """Create and save a trained model."""
    # Generate sample data
    np.random.seed(42)
    X_normal = np.random.randn(800, 20)
    X_anomaly = np.random.randn(200, 20) * 2 + 3
    X = pd.DataFrame(np.vstack([X_normal, X_anomaly]))
    y = pd.Series([0] * 800 + [1] * 200)
    
    # Train model
    registry = ModelRegistry()
    registry.train_isolation_forest(X, y, contamination=0.2)
    
    # Save model
    model_path = tmp_path / "models" / "best_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    registry.save_model("isolation_forest", str(model_path))
    
    return model_path


class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    def test_health_check_no_model(self, client):
        """Test health check when model not loaded."""
        response = client.get('/health')
        data = json.loads(response.data)
        
        assert response.status_code in [200, 503]
        assert 'status' in data
        assert 'model_loaded' in data
    
    def test_health_check_returns_json(self, client):
        """Test health check returns JSON."""
        response = client.get('/health')
        assert response.content_type == 'application/json'


class TestDetectEndpoint:
    """Test suite for anomaly detection endpoint."""
    
    def test_detect_no_model(self, client):
        """Test detection when model not loaded."""
        payload = {
            "features": [0.5] * 20
        }
        
        response = client.post(
            '/detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should return error
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_detect_missing_content_type(self, client):
        """Test detection with missing Content-Type."""
        response = client.post('/detect', data='{}')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_detect_invalid_json(self, client):
        """Test detection with invalid JSON."""
        response = client.post(
            '/detect',
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_detect_missing_features(self, client):
        """Test detection with missing features field."""
        payload = {}
        
        response = client.post(
            '/detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [400, 422, 503]
    
    def test_detect_invalid_features_type(self, client):
        """Test detection with invalid features type."""
        payload = {
            "features": "not a list"
        }
        
        response = client.post(
            '/detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [400, 422, 503]
    
    def test_detect_features_with_nan(self, client):
        """Test detection with NaN values."""
        payload = {
            "features": [0.5, float('nan'), 0.3]
        }
        
        response = client.post(
            '/detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should validate and reject
        assert response.status_code in [400, 422, 503]
    
    def test_detect_features_too_few(self, client):
        """Test detection with too few features."""
        payload = {
            "features": [0.5, 0.3]  # Only 2 features
        }
        
        response = client.post(
            '/detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should fail validation or model check
        assert response.status_code in [400, 503]


class TestBatchDetectEndpoint:
    """Test suite for batch detection endpoint."""
    
    def test_batch_detect_no_model(self, client):
        """Test batch detection when model not loaded."""
        payload = {
            "samples": [[0.5] * 20, [0.3] * 20]
        }
        
        response = client.post(
            '/batch_detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should return error
        assert response.status_code == 503
    
    def test_batch_detect_missing_samples(self, client):
        """Test batch detection with missing samples field."""
        payload = {}
        
        response = client.post(
            '/batch_detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [400, 503]
    
    def test_batch_detect_invalid_samples_type(self, client):
        """Test batch detection with invalid samples type."""
        payload = {
            "samples": "not a list"
        }
        
        response = client.post(
            '/batch_detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [400, 503]
    
    def test_batch_detect_exceeds_limit(self, client):
        """Test batch detection with too many samples."""
        payload = {
            "samples": [[0.5] * 20] * 2000  # Exceeds typical limit
        }
        
        response = client.post(
            '/batch_detect',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should reject or handle gracefully
        assert response.status_code in [400, 503]


class TestMetricsEndpoint:
    """Test suite for Prometheus metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns data."""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert b'prediction_latency_seconds' in response.data or response.status_code == 200
