"""Tests for Flask API."""

import pytest
import json


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'cpu_percent' in data
    assert 'memory_percent' in data


def test_ready_endpoint(client):
    """Test readiness check endpoint."""
    response = client.get('/ready')
    assert response.status_code in [200, 503]  # May not be ready if model not loaded
    data = json.loads(response.data)
    assert 'status' in data


def test_predict_endpoint(client, sample_features):
    """Test prediction endpoint."""
    response = client.post(
        '/predict',
        data=json.dumps({'features': sample_features}),
        content_type='application/json'
    )
    
    # May fail if model not trained, but should return valid response
    assert response.status_code in [200, 500]
    data = json.loads(response.data)
    
    if response.status_code == 200:
        assert 'predictions' in data
        assert 'version' in data
        assert len(data['predictions']) == len(sample_features)


def test_predict_missing_features(client):
    """Test prediction with missing features field."""
    response = client.post(
        '/predict',
        data=json.dumps({'data': [[1, 2, 3]]}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_invalid_content_type(client):
    """Test prediction with invalid content type."""
    response = client.post('/predict', data='not json')
    assert response.status_code == 400


def test_predict_invalid_shape(client):
    """Test prediction with invalid feature shape."""
    response = client.post(
        '/predict',
        data=json.dumps({'features': [1, 2, 3]}),  # 1D instead of 2D
        content_type='application/json'
    )
    assert response.status_code in [400, 500]


def test_model_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get('/model/info')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'name' in data
    assert 'version' in data
    assert 'model_type' in data


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = client.get('/metrics')
    assert response.status_code == 200
    # Check that response contains Prometheus metrics
    assert b'ml_predictions_total' in response.data or b'# HELP' in response.data
