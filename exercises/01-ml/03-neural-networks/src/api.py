"""Flask REST API for UnifiedAI

Provides: Production-ready neural network classification API with validation and monitoring
"""

import logging
from pathlib import Path

import joblib
import yaml
import numpy as np
from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, Field, ValidationError

from src.monitoring import (
    track_prediction_latency,
    track_prediction_count,
    track_error,
    track_model_status,
    get_metrics,
    CONTENT_TYPE_LATEST
)
from src.utils import setup_logging


# Initialize Flask app
app = Flask(__name__)

# Setup logging
logger = setup_logging("INFO", "logs/api.log")

# Global model placeholder
model = None
feature_engineer = None
config = None
model_name = "dense_nn"  # Default model


class PredictionRequest(BaseModel):
    """Schema for prediction request.
    
    Attributes:
        features: List of feature values (length must match n_features)
    """
    features: list = Field(..., min_items=1, description="Feature vector for classification")


def load_model_and_config() -> None:
    """Load trained model, feature engineer, and configuration."""
    global model, feature_engineer, config, model_name
    
    try:
        # Load configuration
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("config.yaml not found")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded")
        
        # Load model
        model_path = Path("models/best_model.h5")
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Using fallback.")
            return
        
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        track_model_status("production", True)
        
        # Load feature engineer
        fe_path = Path("models/feature_engineer.pkl")
        if fe_path.exists():
            feature_engineer = joblib.load(fe_path)
            logger.info("Feature engineer loaded")
        else:
            logger.warning("Feature engineer not found. Using raw features.")
        
    except Exception as e:
        logger.error(f"Failed to load model/config: {e}")
        track_error("model_loading")
        raise RuntimeError(f"Model loading failed: {e}") from e


@app.route("/health", methods=["GET"])
def health_check() -> tuple:
    """Health check endpoint."""
    model_loaded = model is not None
    
    status = {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "config_loaded": config is not None,
    }
    
    status_code = 200 if model_loaded else 503
    
    return jsonify(status), status_code


@app.route("/predict", methods=["POST"])
@track_prediction_latency
def predict() -> tuple:
    """Prediction endpoint."""
    try:
        # Check model loaded
        if model is None:
            track_error("model_not_loaded")
            return jsonify({
                "error": "Model not loaded",
                "message": "Service not ready. Model must be trained and loaded first."
            }), 503
        
        # Validate request
        if not request.is_json:
            track_error("validation")
            return jsonify({
                "error": "Invalid request",
                "message": "Content-Type must be application/json"
            }), 400
        
        # Parse and validate input
        try:
            input_data = PredictionRequest(**request.json)
        except ValidationError as e:
            track_error("validation")
            return jsonify({
                "error": "Validation failed",
                "details": e.errors()
            }), 400
        
        # Convert to numpy array
        X = np.array([input_data.features])
        
        # Validate feature count
        expected_features = config['data']['n_features']
        if X.shape[1] != expected_features:
            track_error("validation")
            return jsonify({
                "error": "Feature count mismatch",
                "message": f"Expected {expected_features} features, got {X.shape[1]}"
            }), 400
        
        # Apply feature engineering if available
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Reshape for CNN if needed
        if model_name == "cnn_1d":
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Make prediction
        predictions_proba = model.predict(X, verbose=0)
        prediction = int(np.argmax(predictions_proba[0]))
        confidence = float(predictions_proba[0][prediction])
        
        # Track metrics
        track_prediction_count(model_name, str(prediction), "success")
        
        response = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": predictions_proba[0].tolist(),
            "model": model_name
        }
        
        logger.info(f"Prediction: class={prediction}, confidence={confidence:.3f}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        track_error("prediction")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(get_metrics(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/info", methods=["GET"])
def info() -> tuple:
    """API information endpoint."""
    info = {
        "name": "UnifiedAI",
        "version": "1.0.0",
        "model": model_name,
        "features_required": config['data']['n_features'] if config else None,
        "n_classes": config['data']['n_classes'] if config else None,
    }
    
    return jsonify(info), 200


if __name__ == "__main__":
    # Load model on startup
    load_model_and_config()
    
    # Run server
    host = config['api']['host'] if config else '0.0.0.0'
    port = config['api']['port'] if config else 5000
    debug = config['api']['debug'] if config else False
    
    logger.info(f"Starting UnifiedAI API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
