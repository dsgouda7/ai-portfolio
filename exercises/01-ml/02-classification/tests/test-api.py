"""Tests for Flask API"""

import pytest
import json
import numpy as np

from src.api import app


@pytest.fixture
def client():
    """Fixture providing Flask test client."""
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Fixture providing sample image data."""
    # Random 64x64 grayscale image (flattened)
    return np.random.rand(4096).tolist()


class TestHealthEndpoint:
    """Test suite for /health endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        
        assert response.status_code in [200, 503]
        
        data = json.loads(response.data)
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Test suite for /predict endpoint."""
    
    def test_predict_invalid_method(self, client):
        """Test that GET to /predict is not allowed."""
        response = client.get('/predict')
        assert response.status_code == 405
    
    def test_predict_no_json(self, client):
        """Test prediction with non-JSON content type."""
        response = client.post(
            '/predict',
            data='not json',
            content_type='text/plain'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_predict_empty_json(self, client):
        """Test prediction with empty JSON."""
        response = client.post(
            '/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_predict_wrong_image_size(self, client):
        """Test prediction with wrong image size."""
        invalid_request = {"image": [0.5] * 100}  # Too small
        
        response = client.post(
            '/predict',
            data=json.dumps(invalid_request),
            content_type='application/json'
        )
        
        assert response.status_code == 400


class TestMetricsEndpoint:
    """Test suite for /metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get('/metrics')
        
        assert response.status_code == 200


class TestInfoEndpoint:
    """Test suite for /info endpoint."""
    
    def test_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get('/info')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert "version" in data
        assert "expected_input" in data
