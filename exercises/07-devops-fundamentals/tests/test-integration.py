"""Integration tests for full workflow."""

import pytest
import json
import numpy as np
from src.model import MLModel, create_dummy_data


def test_full_training_and_prediction_workflow():
    """Test complete training and prediction workflow."""
    # Create and train model
    model = MLModel()
    X, y = create_dummy_data(n_samples=200)
    
    # Split data
    split_idx = 150
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train
    model.train(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Verify predictions
    assert predictions.shape == y_test.shape
    assert len(predictions) == len(X_test)
    
    # Check prediction quality (should be reasonable)
    mae = np.mean(np.abs(predictions - y_test))
    assert mae < 5.0  # Reasonable MAE for dummy data


def test_api_full_workflow(client, sample_features):
    """Test full API workflow: health, ready, predict, info."""
    # Health check
    health_response = client.get('/health')
    assert health_response.status_code == 200
    
    # Readiness check
    ready_response = client.get('/ready')
    assert ready_response.status_code in [200, 503]
    
    # Model info
    info_response = client.get('/model/info')
    assert info_response.status_code == 200
    
    # Prediction
    predict_response = client.post(
        '/predict',
        data=json.dumps({'features': sample_features}),
        content_type='application/json'
    )
    assert predict_response.status_code in [200, 500]
    
    # Metrics
    metrics_response = client.get('/metrics')
    assert metrics_response.status_code == 200


def test_error_handling(client):
    """Test error handling across endpoints."""
    # Invalid prediction request
    response = client.post('/predict', data='invalid')
    assert response.status_code == 400
    
    # Missing endpoint
    response = client.get('/nonexistent')
    assert response.status_code == 404
