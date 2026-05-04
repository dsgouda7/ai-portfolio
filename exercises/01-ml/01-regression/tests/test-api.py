"""Tests for Flask API"""

import pytest
import json
from pathlib import Path
import tempfile

from src.api import app, load_model_and_config


@pytest.fixture
def client():
    """Fixture providing Flask test client."""
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_prediction_request():
    """Fixture providing valid prediction request."""
    return {
        "MedInc": 3.5,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -120.0
    }


class TestHealthEndpoint:
    """Test suite for /health endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        
        assert response.status_code in [200, 503]  # May be degraded if model not loaded
        
        data = json.loads(response.data)
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_health_check_method_not_allowed(self, client):
        """Test that POST to /health is not allowed."""
        response = client.post('/health')
        assert response.status_code == 405


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
        assert "json" in data["message"].lower()
    
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
    
    def test_predict_missing_fields(self, client, sample_prediction_request):
        """Test prediction with missing required fields."""
        # Remove one required field
        incomplete_request = sample_prediction_request.copy()
        del incomplete_request["MedInc"]
        
        response = client.post(
            '/predict',
            data=json.dumps(incomplete_request),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_predict_invalid_field_types(self, client, sample_prediction_request):
        """Test prediction with invalid field types."""
        invalid_request = sample_prediction_request.copy()
        invalid_request["MedInc"] = "not a number"
        
        response = client.post(
            '/predict',
            data=json.dumps(invalid_request),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_predict_out_of_range_values(self, client, sample_prediction_request):
        """Test prediction with out-of-range values."""
        invalid_request = sample_prediction_request.copy()
        invalid_request["MedInc"] = 100.0  # Way above max of 15
        
        response = client.post(
            '/predict',
            data=json.dumps(invalid_request),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_predict_negative_values(self, client, sample_prediction_request):
        """Test prediction with negative values where not allowed."""
        invalid_request = sample_prediction_request.copy()
        invalid_request["Population"] = -100.0
        
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
        assert b'prediction_latency_seconds' in response.data or b'# HELP' in response.data
    
    def test_metrics_method_not_allowed(self, client):
        """Test that POST to /metrics is not allowed."""
        response = client.post('/metrics')
        assert response.status_code == 405


class TestInfoEndpoint:
    """Test suite for /info endpoint."""
    
    def test_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get('/info')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert "version" in data
        assert "features_required" in data
        
        # Check all required features listed
        required_features = data["features_required"]
        assert "MedInc" in required_features
        assert "HouseAge" in required_features
        assert len(required_features) == 8


class TestErrorHandlers:
    """Test suite for error handlers."""
    
    def test_404_handler(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent-endpoint')
        
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert "error" in data
        assert data["error"] == "Not found"
    
    def test_404_helpful_message(self, client):
        """Test that 404 message lists available endpoints."""
        response = client.get('/wrong-path')
        
        data = json.loads(response.data)
        assert "health" in data["message"] or "predict" in data["message"]


class TestAPIIntegration:
    """Integration tests for API."""
    
    def test_health_then_predict_flow(self, client, sample_prediction_request):
        """Test typical API usage flow."""
        # Check health
        health_response = client.get('/health')
        assert health_response.status_code in [200, 503]
        
        # Get info
        info_response = client.get('/info')
        assert info_response.status_code == 200
        
        # Make prediction (may fail if model not loaded, which is expected)
        predict_response = client.post(
            '/predict',
            data=json.dumps(sample_prediction_request),
            content_type='application/json'
        )
        
        # Either succeeds (200) or model not loaded (503)
        assert predict_response.status_code in [200, 503]
    
    def test_multiple_predictions(self, client, sample_prediction_request):
        """Test making multiple predictions."""
        for i in range(3):
            modified_request = sample_prediction_request.copy()
            modified_request["MedInc"] = 2.0 + i
            
            response = client.post(
                '/predict',
                data=json.dumps(modified_request),
                content_type='application/json'
            )
            
            # Either succeeds or model not loaded
            assert response.status_code in [200, 503]
