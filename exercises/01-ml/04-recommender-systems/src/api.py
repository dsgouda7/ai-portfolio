"""Flask REST API for FlixAI

Provides: Production-ready recommendation API with validation and monitoring
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import joblib
import yaml
from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, Field, ValidationError

from src.monitoring import (
    track_prediction_latency,
    track_prediction_count,
    track_error,
    track_model_status,
    track_recommendation_k,
    get_metrics
)
from src.utils import setup_logging


# Initialize Flask app
app = Flask(__name__)

# Setup logging
logger = setup_logging("INFO", "logs/api.log")

# Global model placeholder
model = None
config = None


class RecommendationRequest(BaseModel):
    """Schema for recommendation request.
    
    Attributes:
        user_id: User ID for recommendations
        k: Number of recommendations (default: 10)
        exclude_seen: Whether to exclude already-seen items (default: True)
        model_name: Model to use for recommendations (default: "matrix_factorization")
    """
    user_id: int = Field(..., ge=0, description="User ID")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations (1-100)")
    exclude_seen: bool = Field(True, description="Exclude already-seen items")
    model_name: str = Field("matrix_factorization", description="Model name")


def load_model_and_config() -> None:
    """Load trained model and configuration.
    
    Raises:
        FileNotFoundError: If model or config files not found
        RuntimeError: If model loading fails
    """
    global model, config
    
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
            return
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        track_model_status("production", True)
        
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


@app.route("/recommend", methods=["POST"])
@track_prediction_latency
def recommend() -> tuple:
    """Recommendation endpoint.
    
    Request body:
        JSON with user_id, k (optional), exclude_seen (optional)
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl -X POST http://localhost:5000/recommend \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"user_id": 123, "k": 10}'
        {
          "user_id": 123,
          "recommendations": [
            {"item_id": 456, "score": 4.2},
            {"item_id": 789, "score": 3.9}
          ],
          "model": "matrix_factorization",
          "latency_ms": 45
        }
    """
    start_time = time.time()
    
    try:
        # Check model loaded
        if model is None:
            track_error("model_not_loaded")
            return jsonify({
                "error": "Model not loaded",
                "message": "Service is starting up. Please try again shortly."
            }), 503
        
        # Validate request
        try:
            request_data = RecommendationRequest(**request.json)
        except ValidationError as e:
            track_error("validation_error")
            return jsonify({
                "error": "Validation error",
                "details": e.errors()
            }), 400
        
        # Track K value
        track_recommendation_k(request_data.k)
        
        # Get recommendations
        try:
            recommendations = model.recommend_items(
                user_id=request_data.user_id,
                k=request_data.k,
                exclude_seen=request_data.exclude_seen,
                model_name=request_data.model_name
            )
            
            # Format response
            rec_list = [
                {"item_id": int(item_id), "score": float(score)}
                for item_id, score in recommendations
            ]
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            response = {
                "user_id": request_data.user_id,
                "recommendations": rec_list,
                "model": request_data.model_name,
                "k": len(rec_list),
                "latency_ms": latency_ms
            }
            
            track_prediction_count(request_data.model_name, "success")
            
            return jsonify(response), 200
            
        except KeyError as e:
            track_error("user_not_found")
            return jsonify({
                "error": "User not found",
                "message": f"User {request_data.user_id} not found in training data"
            }), 404
        
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            track_error("recommendation_error")
            track_prediction_count(request_data.model_name, "failure")
            return jsonify({
                "error": "Recommendation failed",
                "message": str(e)
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        track_error("unexpected_error")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint.
    
    Returns:
        Response with Prometheus metrics
    
    Example:
        >>> curl http://localhost:5000/metrics
        # HELP prediction_latency_seconds Time spent processing recommendation request
        # TYPE prediction_latency_seconds histogram
        ...
    """
    metrics_text, content_type = get_metrics()
    return Response(metrics_text, mimetype=content_type)


@app.route("/info", methods=["GET"])
def info() -> tuple:
    """Service information endpoint.
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl http://localhost:5000/info
        {"service": "FlixAI", "version": "1.0.0", ...}
    """
    info_data = {
        "service": "FlixAI",
        "version": "1.0.0",
        "description": "Production movie recommendation API",
        "endpoints": {
            "/health": "Health check",
            "/recommend": "Get recommendations (POST)",
            "/metrics": "Prometheus metrics",
            "/info": "Service information"
        }
    }
    
    if config:
        info_data["config"] = {
            "default_k": config.get("api", {}).get("default_k", 10),
            "models_available": list(model.models.keys()) if model else []
        }
    
    return jsonify(info_data), 200


if __name__ == "__main__":
    # Load model on startup
    try:
        load_model_and_config()
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        logger.warning("API starting without model. Endpoints will return 503.")
    
    # Start Flask app
    host = config.get("api", {}).get("host", "0.0.0.0") if config else "0.0.0.0"
    port = config.get("api", {}).get("port", 5000) if config else 5000
    debug = config.get("api", {}).get("debug", False) if config else False
    
    logger.info(f"Starting FlixAI API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
