"""
Tests for model serving API
"""
import pytest
from fastapi.testclient import TestClient
import numpy as np
from pathlib import Path
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.serving import app, model_server


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def trained_model():
    """Create and save a trained model"""
    # Train simple model
    X = np.random.randn(100, 8)
    y = np.random.randn(100)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "production_model_latest.pkl"
    joblib.dump(model, model_path)
    
    # Load into model server
    model_server.load_model(str(model_path))
    
    yield model
    
    # Cleanup
    if model_path.exists():
        model_path.unlink()


def test_health_endpoint(client, trained_model):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data['status'] == 'healthy'
    assert 'model_version' in data
    assert 'timestamp' in data


def test_ready_endpoint(client, trained_model):
    """Test readiness check endpoint"""
    response = client.get("/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert data['status'] == 'ready'
    assert data['model_loaded'] == True


def test_predict_single(client, trained_model):
    """Test single prediction"""
    payload = {
        "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
        "model_version": "latest"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert 'predictions' in data
    assert len(data['predictions']) == 1
    assert 'model_version' in data
    assert 'latency_ms' in data
    assert data['latency_ms'] > 0


def test_predict_batch(client, trained_model):
    """Test batch prediction"""
    batch_size = 10
    payload = {
        "features": [[float(i+j) for j in range(8)] for i in range(batch_size)],
        "model_version": "latest"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data['predictions']) == batch_size


def test_predict_invalid_features(client, trained_model):
    """Test prediction with invalid features"""
    payload = {
        "features": [[1.0, 2.0]]  # Wrong number of features
    }
    
    response = client.post("/predict", json=payload)
    # Should return error
    assert response.status_code in [400, 500]


def test_model_info_endpoint(client, trained_model):
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert 'model_version' in data
    assert 'model_type' in data
    assert data['model_type'] == 'RandomForestRegressor'


def test_metrics_endpoint(client, trained_model):
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # Metrics should be in Prometheus format
    content = response.text
    assert 'model_predictions_total' in content or 'python_info' in content


def test_health_without_model(client):
    """Test health check fails when model not loaded"""
    # Reset model server
    original_model = model_server.model
    model_server.model = None
    
    response = client.get("/health")
    assert response.status_code == 503
    
    # Restore
    model_server.model = original_model
