"""Flask REST API for FraudShield

Provides: Production-ready anomaly detection API with validation, monitoring, and explanations
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import joblib
import yaml
import numpy as np
from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.monitoring import (
    track_prediction_latency,
    track_prediction,
    track_error,
    track_model_status,
    track_anomaly_score,
    track_and_update_anomaly_rate,
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
model_name = None


class DetectionRequest(BaseModel):
    """Schema for anomaly detection request.
    
    Attributes:
        features: List of feature values (must match expected number of features)
    """
    features: List[float] = Field(..., min_length=1, max_length=100)
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Validate features are finite numbers."""
        if not all(np.isfinite(x) for x in v):
            raise ValueError("All features must be finite numbers (no NaN or inf)")
        return v


def load_model_and_config() -> None:
    """Load trained model, feature engineer, and configuration.
    
    Raises:
        FileNotFoundError: If model or config files not found
        RuntimeError: If model loading fails
    """
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
        model_path = Path("models/best_model.pkl")
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Using fallback.")
            # In production, this would raise an error
            # For now, we'll handle gracefully in detect endpoint
            return
        
        model = joblib.load(model_path)
        model_name = "production"
        logger.info(f"Model loaded from {model_path}")
        track_model_status(model_name, True)
        
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


@app.route("/detect", methods=["POST"])
@track_prediction_latency
def detect() -> tuple:
    """Anomaly detection endpoint.
    
    Request body:
        JSON with features list
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl -X POST http://localhost:5000/detect \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"features": [0.5, 1.2, -0.3, ...]}'
        {
            "is_anomaly": true,
            "anomaly_score": 0.85,
            "confidence": "high",
            "model": "isolation_forest"
        }
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
            input_data = DetectionRequest(**request.json)
        except ValidationError as e:
            track_error("validation")
            return jsonify({
                "error": "Validation failed",
                "details": e.errors()
            }), 400
        
        # Convert to DataFrame for prediction
        import pandas as pd
        X = pd.DataFrame([input_data.features])
        
        # Check feature count
        expected_features = config.get('data', {}).get('n_features', 20)
        if X.shape[1] != expected_features:
            track_error("validation")
            return jsonify({
                "error": "Feature count mismatch",
                "message": f"Expected {expected_features} features, got {X.shape[1]}"
            }), 400
        
        # Apply feature engineering if available
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Make prediction
        # Handle both ModelRegistry.predict and direct sklearn model
        if hasattr(model, 'predict'):
            # Direct sklearn model
            prediction_raw = model.predict(X)
            if prediction_raw[0] == -1:
                is_anomaly = True
                prediction = 1
            else:
                is_anomaly = False
                prediction = 0
            
            # Get anomaly score
            if hasattr(model, 'decision_function'):
                score_raw = model.decision_function(X)[0]
                anomaly_score = float(-score_raw)  # More negative = more anomalous
            elif hasattr(model, 'score_samples'):
                score_raw = model.score_samples(X)[0]
                anomaly_score = float(-score_raw)
            else:
                anomaly_score = 1.0 if is_anomaly else 0.0
        else:
            # Shouldn't happen in production, but handle gracefully
            is_anomaly = False
            anomaly_score = 0.0
            prediction = 0
        
        # Normalize score to [0, 1] range for consistency
        # (approximation, since score ranges vary by model)
        if anomaly_score > 1.0:
            anomaly_score_normalized = min(anomaly_score / 10.0, 1.0)
        else:
            anomaly_score_normalized = anomaly_score
        
        # Determine confidence level
        if anomaly_score_normalized > 0.7:
            confidence = "high"
        elif anomaly_score_normalized > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Track metrics
        track_prediction(model_name or "production", is_anomaly)
        track_anomaly_score(anomaly_score_normalized)
        current_anomaly_rate = track_and_update_anomaly_rate(is_anomaly)
        
        response = {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score_normalized),
            "confidence": confidence,
            "model": model_name or "production",
            "current_anomaly_rate": float(current_anomaly_rate),
        }
        
        # Add explanation if enabled
        if config.get('api', {}).get('explanation_enabled', False):
            response["explanation"] = _generate_explanation(is_anomaly, anomaly_score_normalized)
        
        logger.info(
            f"Detection made: is_anomaly={is_anomaly}, score={anomaly_score_normalized:.3f}"
        )
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        track_error("server")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route("/batch_detect", methods=["POST"])
@track_prediction_latency
def batch_detect() -> tuple:
    """Batch anomaly detection endpoint.
    
    Request body:
        JSON with list of feature lists
    
    Returns:
        Tuple of (response_dict, status_code) with results for each sample
    
    Example:
        >>> curl -X POST http://localhost:5000/batch_detect \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"samples": [[0.5, 1.2, ...], [0.3, -0.5, ...]]}'
    """
    try:
        # Check model loaded
        if model is None:
            track_error("model_not_loaded")
            return jsonify({
                "error": "Model not loaded"
            }), 503
        
        # Validate request
        if not request.is_json:
            track_error("validation")
            return jsonify({
                "error": "Invalid request"
            }), 400
        
        samples = request.json.get('samples', [])
        if not samples or not isinstance(samples, list):
            track_error("validation")
            return jsonify({
                "error": "Missing or invalid 'samples' field"
            }), 400
        
        # Limit batch size
        max_batch_size = 1000
        if len(samples) > max_batch_size:
            track_error("validation")
            return jsonify({
                "error": f"Batch size exceeds limit of {max_batch_size}"
            }), 400
        
        # Process batch
        import pandas as pd
        X = pd.DataFrame(samples)
        
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Make predictions
        predictions_raw = model.predict(X)
        predictions = [(p == -1) for p in predictions_raw]
        
        # Get scores
        if hasattr(model, 'decision_function'):
            scores = -model.decision_function(X)
        else:
            scores = [1.0 if p else 0.0 for p in predictions]
        
        # Build response
        results = [
            {
                "is_anomaly": bool(pred),
                "anomaly_score": float(score)
            }
            for pred, score in zip(predictions, scores)
        ]
        
        # Track metrics
        for is_anomaly in predictions:
            track_prediction(model_name or "production", is_anomaly)
        
        response = {
            "count": len(results),
            "results": results,
            "anomaly_count": sum(predictions),
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        track_error("server")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint.
    
    Returns:
        Prometheus metrics in exposition format
    
    Example:
        >>> curl http://localhost:5000/metrics
    """
    return Response(get_metrics(), mimetype=CONTENT_TYPE_LATEST)


def _generate_explanation(is_anomaly: bool, score: float) -> str:
    """Generate human-readable explanation for prediction.
    
    Args:
        is_anomaly: Whether prediction was anomaly
        score: Anomaly score
    
    Returns:
        Explanation string
    """
    if is_anomaly:
        if score > 0.8:
            return "Strong anomaly signal detected. This transaction deviates significantly from normal patterns."
        elif score > 0.6:
            return "Moderate anomaly detected. This transaction shows unusual characteristics."
        else:
            return "Weak anomaly signal. This transaction is slightly suspicious."
    else:
        return "Normal transaction. This matches expected patterns."


if __name__ == "__main__":
    # Load model on startup
    try:
        load_model_and_config()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    
    # Get config
    api_config = config.get('api', {}) if config else {}
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting FraudShield API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
