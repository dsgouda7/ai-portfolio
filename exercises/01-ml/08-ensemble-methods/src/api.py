"""Flask REST API for EnsembleAI

Provides: Production-ready ensemble prediction API with validation and monitoring
"""

import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import yaml
from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, Field, ValidationError

from src.monitoring import (
    track_prediction_latency,
    track_prediction_count,
    track_error,
    track_model_status,
    track_prediction_value,
    track_ensemble_agreement,
    track_confidence_score,
    get_metrics,
    CONTENT_TYPE_LATEST
)
from src.utils import setup_logging


# Initialize Flask app
app = Flask(__name__)

# Setup logging
logger = setup_logging("INFO", "logs/api.log")

# Global model placeholder
ensemble_model = None
base_models = {}
feature_engineer = None
config = None


class PredictionRequest(BaseModel):
    """Schema for prediction request.
    
    Attributes:
        features: Dictionary of feature name -> value pairs
    
    Example:
        {
            "features": {
                "feature_00": 0.5,
                "feature_01": -1.2,
                ...
            }
        }
    """
    features: Dict[str, float] = Field(..., description="Feature values")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "feature_00": 0.5,
                    "feature_01": -1.2,
                    "feature_02": 0.8,
                    "feature_03": -0.3
                }
            }
        }


def load_model_and_config() -> None:
    """Load trained ensemble model, base models, feature engineer, and configuration.
    
    Raises:
        FileNotFoundError: If model or config files not found
        RuntimeError: If model loading fails
    """
    global ensemble_model, base_models, feature_engineer, config
    
    try:
        # Load configuration
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("config.yaml not found")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded")
        
        # Load ensemble model
        ensemble_path = Path("models/voting_ensemble.pkl")  # or stacking_ensemble.pkl
        if not ensemble_path.exists():
            ensemble_path = Path("models/stacking_ensemble.pkl")
        
        if ensemble_path.exists():
            ensemble_model = joblib.load(ensemble_path)
            logger.info(f"Ensemble model loaded from {ensemble_path}")
            track_model_status("ensemble", True)
        else:
            logger.warning(f"Ensemble model not found. Using base models only.")
        
        # Load base models (for individual predictions)
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pkl"):
                if "ensemble" not in model_file.name:
                    model_name = model_file.stem
                    base_models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded base model: {model_name}")
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
        {"status": "healthy", "ensemble_loaded": true, "base_models_loaded": 5}
    """
    ensemble_loaded = ensemble_model is not None
    base_models_loaded = len(base_models)
    
    status = {
        "status": "healthy" if (ensemble_loaded or base_models_loaded > 0) else "degraded",
        "ensemble_loaded": ensemble_loaded,
        "base_models_loaded": base_models_loaded,
        "base_model_names": list(base_models.keys()),
        "config_loaded": config is not None,
    }
    
    status_code = 200 if (ensemble_loaded or base_models_loaded > 0) else 503
    
    return jsonify(status), status_code


@app.route("/predict", methods=["POST"])
@track_prediction_latency
def predict() -> tuple:
    """Prediction endpoint with ensemble and individual model predictions.
    
    Request body:
        JSON with features dict (see PredictionRequest schema)
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl -X POST http://localhost:5000/predict \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"features": {"feature_00": 0.5, ...}}'
        {
            "ensemble_prediction": 1,
            "ensemble_confidence": 0.92,
            "individual_predictions": {
                "xgboost": 1,
                "lightgbm": 1,
                "catboost": 0
            },
            "confidence_scores": {
                "xgboost": 0.95,
                "lightgbm": 0.88,
                "catboost": 0.45
            },
            "agreement_rate": 0.67,
            "model_used": "voting"
        }
    """
    try:
        # Check models loaded
        if ensemble_model is None and len(base_models) == 0:
            track_error("model_not_loaded")
            return jsonify({
                "error": "No models loaded",
                "details": "Server not ready for predictions"
            }), 503
        
        # Parse and validate request
        try:
            data = request.get_json()
            req = PredictionRequest(**data)
        except ValidationError as e:
            track_error("validation")
            return jsonify({
                "error": "Validation failed",
                "details": str(e)
            }), 400
        
        # Convert features to DataFrame
        import pandas as pd
        X = pd.DataFrame([req.features])
        
        # Apply feature engineering
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Get ensemble prediction
        ensemble_pred = None
        ensemble_confidence = None
        model_used = None
        
        if ensemble_model is not None:
            ensemble_pred = int(ensemble_model.predict(X)[0])
            ensemble_proba = ensemble_model.predict_proba(X)[0]
            ensemble_confidence = float(np.max(ensemble_proba))
            model_used = "ensemble"
            
            track_prediction_value(ensemble_pred)
            track_confidence_score(ensemble_confidence)
        
        # Get individual base model predictions
        individual_predictions = {}
        confidence_scores = {}
        
        return_individual = config.get('api', {}).get('return_individual_predictions', True)
        
        if return_individual and base_models:
            for model_name, model in base_models.items():
                pred = int(model.predict(X)[0])
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
                
                individual_predictions[model_name] = pred
                confidence_scores[model_name] = confidence
        
        # Compute agreement rate
        agreement_rate = None
        if len(individual_predictions) > 1:
            predictions_array = list(individual_predictions.values())
            majority_vote = max(set(predictions_array), key=predictions_array.count)
            agreement_rate = float(predictions_array.count(majority_vote) / len(predictions_array))
            
            track_ensemble_agreement("base_models", agreement_rate)
        
        # Build response
        response = {}
        
        if ensemble_pred is not None:
            response["ensemble_prediction"] = ensemble_pred
            response["ensemble_confidence"] = round(ensemble_confidence, 4)
        
        if individual_predictions:
            response["individual_predictions"] = individual_predictions
        
        return_confidence = config.get('api', {}).get('return_confidence_scores', True)
        if return_confidence and confidence_scores:
            response["confidence_scores"] = {k: round(v, 4) for k, v in confidence_scores.items()}
        
        if agreement_rate is not None:
            response["agreement_rate"] = round(agreement_rate, 4)
        
        response["model_used"] = model_used or "base_models"
        response["status"] = "success"
        
        # Track success
        track_prediction_count(response["model_used"], "success")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        track_error("server")
        track_prediction_count("unknown", "failure")
        
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint.
    
    Returns:
        Response with Prometheus metrics
    
    Example:
        >>> curl http://localhost:5000/metrics
    """
    return Response(get_metrics(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/models", methods=["GET"])
def list_models() -> tuple:
    """List available models endpoint.
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl http://localhost:5000/models
        {
            "ensemble_model": "voting",
            "base_models": ["xgboost", "lightgbm", "catboost", "random_forest", "logistic_regression"]
        }
    """
    response = {
        "ensemble_model": "loaded" if ensemble_model else "not_loaded",
        "base_models": list(base_models.keys()),
        "feature_engineer": "loaded" if feature_engineer else "not_loaded"
    }
    
    return jsonify(response), 200


if __name__ == "__main__":
    # Load models on startup
    try:
        load_model_and_config()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    
    # Get API config
    api_config = config.get('api', {}) if config else {}
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting EnsembleAI API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
