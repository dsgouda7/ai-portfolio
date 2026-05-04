"""Tests for API endpoints"""

import pytest
import json
from unittest.mock import Mock, patch

# Import after mocking to avoid model loading
@pytest.fixture
def client():
    """Create test client."""
    with patch('src.api.load_model_and_config'):
        from src.api import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client


class TestHealthEndpoint:
    """Test suite for /health endpoint."""
    
    def test_health_check_no_model(self, client):
        """Test health check when model is not loaded."""
        response = client.get('/health')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert data['status'] == 'degraded'
        assert not data['model_loaded']
    
    def test_health_check_with_model(self, client):
        """Test health check when model is loaded."""
        with patch('src.api.model', Mock()):
            response = client.get('/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'healthy'
            assert data['model_loaded']


class TestRecommendEndpoint:
    """Test suite for /recommend endpoint."""
    
    def test_recommend_no_model(self, client):
        """Test recommendation when model is not loaded."""
        response = client.post(
            '/recommend',
            data=json.dumps({'user_id': 123, 'k': 10}),
            content_type='application/json'
        )
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_recommend_valid_request(self, client):
        """Test recommendation with valid request."""
        # Mock model
        mock_model = Mock()
        mock_model.recommend_items.return_value = [
            (456, 4.5),
            (789, 4.2)
        ]
        
        with patch('src.api.model', mock_model):
            response = client.post(
                '/recommend',
                data=json.dumps({'user_id': 123, 'k': 2}),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['user_id'] == 123
            assert len(data['recommendations']) == 2
            assert data['recommendations'][0]['item_id'] == 456
            assert data['recommendations'][0]['score'] == 4.5
    
    def test_recommend_invalid_request(self, client):
        """Test recommendation with invalid request."""
        with patch('src.api.model', Mock()):
            response = client.post(
                '/recommend',
                data=json.dumps({'k': 10}),  # Missing user_id
                content_type='application/json'
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
    
    def test_recommend_invalid_k(self, client):
        """Test recommendation with invalid K value."""
        with patch('src.api.model', Mock()):
            response = client.post(
                '/recommend',
                data=json.dumps({'user_id': 123, 'k': 0}),  # Invalid K
                content_type='application/json'
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
    
    def test_recommend_user_not_found(self, client):
        """Test recommendation for non-existent user."""
        mock_model = Mock()
        mock_model.recommend_items.side_effect = KeyError("User not found")
        
        with patch('src.api.model', mock_model):
            response = client.post(
                '/recommend',
                data=json.dumps({'user_id': 999, 'k': 10}),
                content_type='application/json'
            )
            
            assert response.status_code == 404
            data = json.loads(response.data)
            assert 'error' in data


class TestMetricsEndpoint:
    """Test suite for /metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert 'text/plain' in response.content_type


class TestInfoEndpoint:
    """Test suite for /info endpoint."""
    
    def test_info_endpoint(self, client):
        """Test service info endpoint."""
        response = client.get('/info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['service'] == 'FlixAI'
        assert 'version' in data
        assert 'endpoints' in data
