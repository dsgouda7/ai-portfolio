"""Flask REST API for SegmentAI

Provides: Production-ready clustering API with validation and monitoring
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import joblib
import yaml
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.monitoring import (
    track_prediction_latency,
    track_prediction_count,
    track_error,
    track_model_status,
    track_cluster_distribution,
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


class ClusteringRequest(BaseModel):
    """Schema for clustering request.
    
    Attributes:
        features: List of feature values (must match training feature count)
    
    Example:
        {
            "features": [5.1, 3.5, 1.4, 0.2]  # Iris features
        }
    """
    features: List[float] = Field(..., min_length=1, max_length=100)
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Validate feature values."""
        if any(np.isnan(val) or np.isinf(val) for val in v):
            raise ValueError("Features contain NaN or infinite values")
        return v


class BatchClusteringRequest(BaseModel):
    """Schema for batch clustering request.
    
    Attributes:
        samples: List of samples, each with features
    
    Example:
        {
            "samples": [
                {"features": [5.1, 3.5, 1.4, 0.2]},
                {"features": [4.9, 3.0, 1.4, 0.2]}
            ]
        }
    """
    samples: List[ClusteringRequest] = Field(..., min_length=1, max_length=1000)


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
            # For now, we'll handle gracefully in cluster endpoint
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


@app.route("/cluster", methods=["POST"])
@track_prediction_latency
def cluster() -> tuple:
    """Clustering endpoint.
    
    Request body:
        JSON with features (see ClusteringRequest schema)
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl -X POST http://localhost:5000/cluster \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
        {
            "cluster_id": 0,
            "centroid_distance": 0.45,
            "cluster_size": 50,
            "model": "kmeans"
        }
    """
    try:
        # Check model loaded
        if model is None:
            track_error("model_not_loaded")
            return jsonify({
                "error": "Model not loaded",
                "message": "Train and save a model first"
            }), 503
        
        # Validate request
        try:
            data = ClusteringRequest(**request.json)
        except ValidationError as e:
            track_error("validation")
            return jsonify({
                "error": "Validation failed",
                "details": e.errors()
            }), 400
        
        # Prepare features
        features = np.array(data.features).reshape(1, -1)
        
        # Check feature count matches training
        if feature_engineer is not None:
            expected_features = len(feature_engineer.feature_names) if hasattr(feature_engineer, 'feature_names') else None
            if expected_features and len(data.features) != expected_features:
                track_error("validation")
                return jsonify({
                    "error": "Feature count mismatch",
                    "expected": expected_features,
                    "received": len(data.features)
                }), 400
        
        # Create DataFrame for consistency
        feature_names = [f"feature_{i}" for i in range(len(data.features))]
        X = pd.DataFrame(features, columns=feature_names)
        
        # Apply feature engineering if available
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Predict cluster
        cluster_id = int(model.predict(X)[0])
        
        # Compute additional metrics
        response = {
            "cluster_id": cluster_id,
            "model": "production"
        }
        
        # Add centroid distance for KMeans
        if hasattr(model, 'cluster_centers_'):
            centroid = model.cluster_centers_[cluster_id]
            distance = float(np.linalg.norm(X.values[0] - centroid))
            response["centroid_distance"] = round(distance, 4)
        
        # Add cluster size if available
        if hasattr(model, 'labels_'):
            cluster_size = int(np.sum(model.labels_ == cluster_id))
            response["cluster_size"] = cluster_size
        
        # Track metrics
        track_prediction_count("production", "success")
        track_cluster_distribution([cluster_id])
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        track_error("server")
        track_prediction_count("production", "failure")
        
        return jsonify({
            "error": "Clustering failed",
            "message": str(e)
        }), 500


@app.route("/cluster/batch", methods=["POST"])
@track_prediction_latency
def cluster_batch() -> tuple:
    """Batch clustering endpoint.
    
    Request body:
        JSON with multiple samples (see BatchClusteringRequest schema)
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl -X POST http://localhost:5000/cluster/batch \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"samples": [{"features": [5.1, 3.5, 1.4, 0.2]}, ...]}'
        {
            "predictions": [
                {"cluster_id": 0, "centroid_distance": 0.45},
                {"cluster_id": 1, "centroid_distance": 0.32}
            ],
            "cluster_distribution": {0: 1, 1: 1}
        }
    """
    try:
        # Check model loaded
        if model is None:
            track_error("model_not_loaded")
            return jsonify({
                "error": "Model not loaded"
            }), 503
        
        # Validate request
        try:
            data = BatchClusteringRequest(**request.json)
        except ValidationError as e:
            track_error("validation")
            return jsonify({
                "error": "Validation failed",
                "details": e.errors()
            }), 400
        
        # Prepare features
        features_list = [sample.features for sample in data.samples]
        features = np.array(features_list)
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(len(features_list[0]))]
        X = pd.DataFrame(features, columns=feature_names)
        
        # Apply feature engineering if available
        if feature_engineer is not None:
            X = feature_engineer.transform(X)
        
        # Predict clusters
        cluster_ids = model.predict(X).tolist()
        
        # Build responses
        predictions = []
        for i, cluster_id in enumerate(cluster_ids):
            pred = {"cluster_id": cluster_id}
            
            # Add centroid distance for KMeans
            if hasattr(model, 'cluster_centers_'):
                centroid = model.cluster_centers_[cluster_id]
                distance = float(np.linalg.norm(X.values[i] - centroid))
                pred["centroid_distance"] = round(distance, 4)
            
            predictions.append(pred)
        
        # Compute cluster distribution
        from collections import Counter
        cluster_dist = dict(Counter(cluster_ids))
        
        # Track metrics
        track_prediction_count("production", "success")
        track_cluster_distribution(cluster_ids)
        
        return jsonify({
            "predictions": predictions,
            "cluster_distribution": cluster_dist,
            "n_samples": len(cluster_ids)
        }), 200
        
    except Exception as e:
        logger.error(f"Batch clustering failed: {e}")
        track_error("server")
        track_prediction_count("production", "failure")
        
        return jsonify({
            "error": "Batch clustering failed",
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


@app.route("/info", methods=["GET"])
def info() -> tuple:
    """Model information endpoint.
    
    Returns:
        Model metadata and configuration
    
    Example:
        >>> curl http://localhost:5000/info
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    info_dict = {
        "model_type": type(model).__name__,
        "model_loaded": True
    }
    
    # Add model-specific info
    if hasattr(model, 'n_clusters'):
        info_dict["n_clusters"] = int(model.n_clusters)
    
    if hasattr(model, 'cluster_centers_'):
        info_dict["n_features"] = int(model.cluster_centers_.shape[1])
    
    if config:
        info_dict["config"] = {
            "api": config.get("api", {}),
            "monitoring": config.get("monitoring", {})
        }
    
    return jsonify(info_dict), 200


if __name__ == "__main__":
    # Load model on startup
    try:
        load_model_and_config()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
    
    # Run server
    api_config = config.get("api", {}) if config else {}
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 5000)
    debug = api_config.get("debug", False)
    
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
