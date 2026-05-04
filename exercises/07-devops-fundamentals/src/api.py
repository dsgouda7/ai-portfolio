"""Flask API for ML predictions."""

import os
import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app
import logging

from src.model import MLModel
from src.utils import setup_logging, load_config
from src.health import HealthChecker

logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS = Counter('ml_predictions_total', 'Total predictions')
PREDICTION_TIME = Histogram('ml_prediction_duration_seconds', 'Prediction duration')
ERRORS = Counter('ml_errors_total', 'Total errors', ['error_type'])

# Initialize Flask app
app = Flask(__name__)

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# Load configuration
config = load_config()
setup_logging(config['logging']['level'])

# Initialize model and health checker
model = MLModel(model_path=os.getenv('MODEL_PATH', 'models/model.pkl'))
health_checker = HealthChecker(model)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return health_checker.health_check()


@app.route('/ready', methods=['GET'])
def ready():
    """Readiness check endpoint."""
    return health_checker.readiness_check()


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions.
    
    Expected JSON:
    {
        "features": [[1.0, 2.0, 3.0, 4.0, 5.0], ...]
    }
    
    Returns:
    {
        "predictions": [10.5, ...],
        "version": "1.0.0"
    }
    """
    try:
        with PREDICTION_TIME.time():
            # Validate request
            if not request.is_json:
                ERRORS.labels(error_type='invalid_content_type').inc()
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            
            if 'features' not in data:
                ERRORS.labels(error_type='missing_features').inc()
                return jsonify({'error': 'Missing "features" field'}), 400
            
            # Convert to numpy array
            features = np.array(data['features'])
            
            if features.ndim != 2:
                ERRORS.labels(error_type='invalid_shape').inc()
                return jsonify({'error': 'Features must be 2D array'}), 400
            
            # Make predictions
            predictions = model.predict(features)
            PREDICTIONS.inc()
            
            return jsonify({
                'predictions': predictions.tolist(),
                'version': config['application']['version'],
                'environment': os.getenv('ENVIRONMENT', 'dev')
            })
    
    except Exception as e:
        ERRORS.labels(error_type='prediction_error').inc()
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify({
        'name': config['application']['name'],
        'version': config['application']['version'],
        'model_type': config['model']['type'],
        'environment': os.getenv('ENVIRONMENT', 'dev')
    })


if __name__ == '__main__':
    port = config['application']['port']
    logger.info(f"Starting ML API on port {port}")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('ENVIRONMENT') == 'dev'
    )
