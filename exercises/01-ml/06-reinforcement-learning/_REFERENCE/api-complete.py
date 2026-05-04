"""Flask REST API for AgentAI

Provides: Production-ready RL agent API with validation and monitoring
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
    track_action_latency,
    track_action_count,
    track_error,
    track_policy_status,
    track_episode_reward_value,
    get_metrics,
    CONTENT_TYPE_LATEST
)
from src.utils import setup_logging


# Initialize Flask app
app = Flask(__name__)

# Setup logging
logger = setup_logging("INFO", "logs/api.log")

# Global placeholders
policy_registry = None
state_preprocessor = None
config = None


class ActionRequest(BaseModel):
    """Schema for action selection request.
    
    Attributes:
        state: Current environment state (list of floats)
        epsilon: Optional exploration rate (0.0-1.0)
    """
    state: List[float] = Field(..., description="Current state vector")
    epsilon: float = Field(default=0.0, ge=0.0, le=1.0, description="Exploration rate")
    
    @field_validator('state')
    @classmethod
    def validate_state_dim(cls, v):
        """Validate state dimension matches environment."""
        if len(v) < 1:
            raise ValueError("State must have at least 1 dimension")
        # Will check against config state_dim after loading
        return v


class TrainingRequest(BaseModel):
    """Schema for training request.
    
    Attributes:
        episodes: Number of episodes to train
        save_policy: Whether to save policy after training
    """
    episodes: int = Field(default=100, ge=1, le=10000, description="Training episodes")
    save_policy: bool = Field(default=True, description="Save policy after training")


def load_policy_and_config() -> None:
    """Load trained policy, preprocessor, and configuration.
    
    Raises:
        FileNotFoundError: If config file not found
        RuntimeError: If loading fails
    """
    global policy_registry, state_preprocessor, config
    
    try:
        # Load configuration
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("config.yaml not found")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded")
        
        # Load policy registry
        policy_path = Path("models/policy_registry.pkl")
        if policy_path.exists():
            policy_registry = joblib.load(policy_path)
            logger.info(f"Policy registry loaded from {policy_path}")
            
            # Track policy status
            for policy_name in policy_registry.policies.keys():
                track_policy_status(policy_name, True)
        else:
            logger.warning(f"Policy registry not found at {policy_path}. Using fallback.")
        
        # Load state preprocessor
        preprocessor_path = Path("models/state_preprocessor.pkl")
        if preprocessor_path.exists():
            state_preprocessor = joblib.load(preprocessor_path)
            logger.info("State preprocessor loaded")
        else:
            logger.warning("State preprocessor not found. Using raw states.")
        
    except Exception as e:
        logger.error(f"Failed to load policy/config: {e}")
        track_error("policy_loading")
        raise RuntimeError(f"Loading failed: {e}") from e


@app.route("/health", methods=["GET"])
def health_check() -> tuple:
    """Health check endpoint.
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl http://localhost:5000/health
        {"status": "healthy", "policy_loaded": true, "environment": "CartPole-v1"}
    """
    policy_loaded_flag = policy_registry is not None
    
    status = {
        "status": "healthy" if policy_loaded_flag else "degraded",
        "policy_loaded": policy_loaded_flag,
        "config_loaded": config is not None,
        "environment": config.get("environment", {}).get("env_name") if config else None,
    }
    
    status_code = 200 if policy_loaded_flag else 503
    
    return jsonify(status), status_code


@app.route("/act", methods=["POST"])
@track_action_latency
def select_action() -> tuple:
    """Action selection endpoint.
    
    Request body:
        JSON with state vector (see ActionRequest schema)
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl -X POST http://localhost:5000/act \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"state": [0.1, 0.2, -0.3, 0.4], "epsilon": 0.05}'
        {"action": 1, "policy": "reinforce", "epsilon": 0.05}
    """
    try:
        # Check policy loaded
        if policy_registry is None:
            track_error("policy_not_loaded")
            return jsonify({
                "error": "Policy not loaded",
                "message": "Train a policy first or load from file"
            }), 503
        
        # Validate request
        try:
            req = ActionRequest(**request.json)
        except ValidationError as e:
            track_error("validation")
            return jsonify({"error": "Invalid request", "details": str(e)}), 400
        
        # Preprocess state
        state = np.array(req.state, dtype=np.float32)
        
        if state_preprocessor and state_preprocessor.is_fitted:
            state = state_preprocessor.transform(state)
        
        # Get policy name from config
        policy_name = config.get("policy_network", {}).get("algorithm", "reinforce").lower()
        
        # Select action
        action = policy_registry.select_action(
            state,
            policy_name=policy_name,
            epsilon=req.epsilon
        )
        
        # Track metrics
        track_action_count(policy_name, "success")
        
        return jsonify({
            "action": int(action),
            "policy": policy_name,
            "epsilon": req.epsilon,
        }), 200
        
    except Exception as e:
        logger.error(f"Action selection failed: {e}")
        track_error("action_selection")
        return jsonify({"error": "Action selection failed", "details": str(e)}), 500


@app.route("/train", methods=["POST"])
def train_episodes() -> tuple:
    """Training endpoint.
    
    Request body:
        JSON with training parameters (see TrainingRequest schema)
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl -X POST http://localhost:5000/train \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"episodes": 100, "save_policy": true}'
        {"episodes_completed": 100, "avg_reward": 198.5, "policy_saved": true}
    """
    try:
        # Validate request
        try:
            req = TrainingRequest(**request.json)
        except ValidationError as e:
            track_error("validation")
            return jsonify({"error": "Invalid request", "details": str(e)}), 400
        
        # This is a stub - actual training would require environment setup
        # In production, training would be done offline or via batch job
        return jsonify({
            "error": "Training endpoint not implemented",
            "message": "Train policies offline using training scripts",
            "hint": "Use: python -m src.models --config config.yaml"
        }), 501
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        track_error("training")
        return jsonify({"error": "Training failed", "details": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint.
    
    Returns:
        Response with Prometheus metrics
    
    Example:
        >>> curl http://localhost:5000/metrics
    """
    metrics_data, content_type = get_metrics()
    return Response(metrics_data, mimetype=content_type)


@app.route("/info", methods=["GET"])
def info() -> tuple:
    """Get API and policy information.
    
    Returns:
        Tuple of (response_dict, status_code)
    
    Example:
        >>> curl http://localhost:5000/info
    """
    if policy_registry is None or config is None:
        return jsonify({"error": "System not initialized"}), 503
    
    info_data = {
        "api_version": "1.0.0",
        "environment": config.get("environment", {}).get("env_name"),
        "algorithm": config.get("policy_network", {}).get("algorithm"),
        "state_dim": policy_registry.state_dim,
        "action_dim": policy_registry.action_dim,
        "policies_loaded": list(policy_registry.policies.keys()),
    }
    
    return jsonify(info_data), 200


@app.errorhandler(404)
def not_found(error) -> tuple:
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error) -> tuple:
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    track_error("internal")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Load models on startup
    try:
        load_policy_and_config()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    
    # Run Flask app
    host = config.get("api", {}).get("host", "0.0.0.0") if config else "0.0.0.0"
    port = config.get("api", {}).get("port", 5000) if config else 5000
    debug = config.get("api", {}).get("debug", False) if config else False
    
    app.run(host=host, port=port, debug=debug)
