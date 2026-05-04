"""Flask REST API for FaceAI

Provides: Production-ready face classification API with validation and monitoring
"""

import logging
from pathlib import Path
import base64

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


class PredictionRequest(BaseModel):
    """Schema for prediction request.
    
    Attributes:
        image: Base64-encoded image or flattened pixel array (4096 values for 64x64)
    """
    image: list = Field(..., min_items=4096, max_items=4096, description="Flattened 64x64 grayscale image")


def load_model_and_config() -> None:
    """Load trained model, feature engineer, and configuration."""
    global model, feature_engineer, config
    
    try:
        # Load configuration
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("config.yaml not found")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded")
        
        # Load model
        model_path = Path("models/best_model.pkl")
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Using fallback.")
            return
        
        model = joblib.load(model_path)
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
        X = np.array([input_data.image])
        
        # Apply feature engineering if available
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = float(proba.max())
        else:
            confidence = None
        
        # Track metrics
        track_prediction_count("production", int(prediction), "success")
        
        response = {
            "predicted_class": int(prediction),
            "confidence": confidence,
            "model": getattr(model, "__class__", type(model)).__name__,
        }
        
        logger.info(f"Prediction made: class={prediction}, confidence={confidence}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        track_error("server")
        track_prediction_count("production", -1, "failure")
        
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(get_metrics(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/info", methods=["GET"])
def info() -> tuple:
    """API information endpoint."""
    from src import __version__
    
    info_data = {
        "version": __version__,
        "model_type": getattr(model, "__class__", type(model)).__name__ if model else None,
        "expected_input": "Flattened 64x64 grayscale image (4096 values)",
    }
    
    return jsonify(info_data), 200


@app.errorhandler(404)
def not_found(error) -> tuple:
    """Handle 404 errors."""
    return jsonify({
        "error": "Not found",
        "message": "Endpoint not found. Try /health, /predict, /metrics, or /info"
    }), 404


@app.errorhandler(500)
def internal_error(error) -> tuple:
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    track_error("server")
    
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later."
    }), 500


def create_app() -> Flask:
    """Application factory for testing."""
    load_model_and_config()
    return app


if __name__ == "__main__":
    # Load model and config
    load_model_and_config()
    
    # Get config or use defaults
    api_config = config.get("api", {}) if config else {}
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 5000)
    debug = api_config.get("debug", False)
    
    logger.info(f"Starting FaceAI API on {host}:{port}")
    
    # Run Flask app
    app.run(host=host, port=port, debug=debug)
