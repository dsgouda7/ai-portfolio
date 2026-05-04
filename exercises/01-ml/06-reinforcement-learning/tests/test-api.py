"""Tests for Flask API endpoints"""

import pytest
import json

from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test suite for /health endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns status."""
        response = client.get('/health')
        data = json.loads(response.data)
        
        assert response.status_code in [200, 503]
        assert 'status' in data
        assert 'policy_loaded' in data
        assert 'config_loaded' in data


class TestActionEndpoint:
    """Test suite for /act endpoint."""
    
    def test_action_endpoint_validation(self, client):
        """Test that invalid request returns 400."""
        # Missing state field
        response = client.post(
            '/act',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code in [400, 503]
    
    def test_action_endpoint_valid_request(self, client):
        """Test action endpoint with valid request."""
        # Valid CartPole state
        request_data = {
            "state": [0.1, 0.2, -0.3, 0.4],
            "epsilon": 0.05
        }
        
        response = client.post(
            '/act',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        # May return 503 if policy not loaded (expected in test)
        assert response.status_code in [200, 503]
        
        data = json.loads(response.data)
        
        if response.status_code == 200:
            assert 'action' in data
            assert 'policy' in data
            assert 'epsilon' in data
            assert data['action'] in [0, 1]


class TestTrainEndpoint:
    """Test suite for /train endpoint."""
    
    def test_train_endpoint_not_implemented(self, client):
        """Test that train endpoint returns 501 (not implemented)."""
        request_data = {
            "episodes": 100,
            "save_policy": True
        }
        
        response = client.post(
            '/train',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        # Should return 501 (not implemented) or validation error
        assert response.status_code in [400, 501]


class TestMetricsEndpoint:
    """Test suite for /metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        # Prometheus metrics are plain text
        assert response.content_type.startswith('text/plain')


class TestInfoEndpoint:
    """Test suite for /info endpoint."""
    
    def test_info_endpoint(self, client):
        """Test info endpoint returns API information."""
        response = client.get('/info')
        
        # May return 503 if not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'api_version' in data


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_404_handler(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
