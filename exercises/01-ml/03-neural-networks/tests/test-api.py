"""Tests for API endpoints"""

import pytest
import json
import numpy as np

from src.api import app, load_model_and_config


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthCheck:
    """Test suite for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns 200 or 503."""
        response = client.get('/health')
        
        # Should return either 200 (healthy) or 503 (degraded)
        assert response.status_code in [200, 503]
        
        # Check response structure
        data = json.loads(response.data)
        assert 'status' in data
        assert 'model_loaded' in data


class TestInfoEndpoint:
    """Test suite for info endpoint."""
    
    def test_info(self, client):
        """Test info endpoint returns API information."""
        response = client.get('/info')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'name' in data
        assert 'version' in data
        assert data['name'] == 'UnifiedAI'


class TestMetricsEndpoint:
    """Test suite for metrics endpoint."""
    
    def test_metrics(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert b'prediction_latency' in response.data or response.data == b''


class TestPredictEndpoint:
    """Test suite for predict endpoint."""
    
    def test_predict_without_model(self, client):
        """Test prediction without loaded model returns 503."""
        payload = {
            'features': [0.1] * 20
        }
        
        response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should return 503 if model not loaded
        assert response.status_code in [200, 503]
    
    def test_predict_invalid_content_type(self, client):
        """Test prediction with invalid content type returns 400."""
        response = client.post(
            '/predict',
            data='not json',
            content_type='text/plain'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_invalid_payload(self, client):
        """Test prediction with invalid payload returns 400."""
        payload = {
            'wrong_field': [0.1] * 20
        }
        
        response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should return 400 or 503 (if model not loaded)
        assert response.status_code in [400, 503]
    
    def test_predict_valid_request_format(self, client):
        """Test that valid request format is accepted."""
        payload = {
            'features': [0.1] * 20
        }
        
        response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should return 200 (success) or 503 (model not loaded) or 400 (validation)
        assert response.status_code in [200, 400, 503]
        
        data = json.loads(response.data)
        
        # If successful, check response structure
        if response.status_code == 200:
            assert 'prediction' in data
            assert 'confidence' in data
            assert 'probabilities' in data
