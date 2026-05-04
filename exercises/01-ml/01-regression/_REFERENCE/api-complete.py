"""Flask REST API for SmartVal AI

Provides: Production-ready prediction API with validation and monitoring
"""

import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import yaml
from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, Field, ValidationError

from src.monitoring import (
    track_prediction_latency,
    track_prediction_count,
    track_error,
    track_model_status,
    track_prediction_value,
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
        MedInc: Median income in block group
        HouseAge: Median house age in block group
        AveRooms: Average number of rooms per household
        AveBedrms: Average number of bedrooms per household
        Population: Block group population
        AveOccup: Average number of household members
        Latitude: Block group latitude
        Longitude: Block group longitude
    """
    MedInc: float = Field(..., ge=0, le=15, description="Median income (0-15)")
    HouseAge: float = Field(..., ge=1, le=52, description="House age (1-52 years)")
    AveRooms: float = Field(..., ge=0, le=50, description="Average rooms (0-50)")
    AveBedrms: float = Field(..., ge=0, le=10, description="Average bedrooms (0-10)")
    Population: float = Field(..., ge=0, le=50000, description="Population (0-50k)")
    AveOccup: float = Field(..., ge=0, le=50, description="Average occupancy (0-50)")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude (32-42)")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude (-125 to -114)")


def load_model_and_config() -> None:
    """Load trained model, feature engineer, and configuration.
    
    Raises:
        FileNotFoundError: If model or config files not found
        RuntimeError: If model loading fails
    """
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
            # In production, this would raise an error
            # For now, we'll handle gracefully in predict endpoint
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
    """Health check endpoint.
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl http://localhost:5000/health
        {"status": "healthy", "model_loaded": true}
    """
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
    """Prediction endpoint.
    
    Request body:
        JSON with house features (see PredictionRequest schema)
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl -X POST http://localhost:5000/predict \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"MedInc": 3.5, "HouseAge": 20, ...}'
        {"prediction": 2.5, "model": "ridge", "units": "$100k"}
    """
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
        
        # Convert to DataFrame for prediction
        import pandas as pd
        X = pd.DataFrame([input_data.dict()])
        
        # Apply feature engineering if available
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Track metrics
        track_prediction_count("production", "success")
        track_prediction_value(float(prediction))
        
        response = {
            "prediction": float(prediction),
            "units": "$100k",
            "model": getattr(model, "__class__", type(model)).__name__,
        }
        
        logger.info(f"Prediction made: {prediction:.2f}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        track_error("server")
        track_prediction_count("production", "failure")
        
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint.
    
    Returns:
        Prometheus-formatted metrics
    
    Example:
        >>> curl http://localhost:5000/metrics
        # HELP prediction_latency_seconds Time spent processing prediction
        # TYPE prediction_latency_seconds histogram
        ...
    """
    return Response(get_metrics(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/info", methods=["GET"])
def info() -> tuple:
    """API information endpoint.
    
    Returns:
        API version and model information
    
    Example:
        >>> curl http://localhost:5000/info
        {"version": "1.0.0", "model_type": "Ridge"}
    """
    from src import __version__
    
    info_data = {
        "version": __version__,
        "model_type": getattr(model, "__class__", type(model)).__name__ if model else None,
        "features_required": list(PredictionRequest.__fields__.keys()),
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
    """Application factory for testing.
    
    Returns:
        Configured Flask app
    """
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
    
    logger.info(f"Starting SmartVal API on {host}:{port}")
    
    # Run Flask app
    app.run(host=host, port=port, debug=debug)
