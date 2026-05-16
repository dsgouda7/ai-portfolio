"""
Model serving with FastAPI and ONNX Runtime
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import joblib
from pathlib import Path
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
from .utils import setup_logging, load_config

logger = setup_logging()

# Prometheus metrics
prediction_counter = Counter('model_predictions_total', 'Total number of predictions')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency in seconds')
error_counter = Counter('model_errors_total', 'Total number of prediction errors')

app = FastAPI(
    title="ML Model Serving API",
    description="Production model serving with health checks and monitoring",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    """Request schema for predictions"""
    features: List[List[float]] = Field(..., description="List of feature vectors")
    model_version: Optional[str] = Field("latest", description="Model version to use")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [[1.5, 2.3, 4.1, 0.8]],
                "model_version": "latest"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predictions: List[float] = Field(..., description="Model predictions")
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class ModelServer:
    """
    Production model serving with caching and monitoring
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model server"""
        self.config = load_config(config_path)
        self.serving_config = self.config["serving"]
        self.model = None
        self.model_version = None
        
        logger.info("Model server initialized")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load model from disk
        
        Args:
            model_path: Path to model file (uses config default if None)
        """
        if model_path is None:
            model_name = self.serving_config["model_name"]
            model_version = self.serving_config["model_version"]
            model_path = f"models/{model_name}_{model_version}.pkl"
        
        try:
            self.model = joblib.load(model_path)
            self.model_version = Path(model_path).stem.split('_')[-1]
            logger.info(f"Loaded model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            features: Input feature array
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            predictions = self.model.predict(features)
            prediction_counter.inc(len(features))
            return predictions
        except Exception as e:
            error_counter.inc()
            logger.error(f"Prediction error: {e}")
            raise
        finally:
            latency = time.time() - start_time
            prediction_latency.observe(latency)


# Global model server instance
model_server = ModelServer()


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model_server.load_model()
        logger.info("Model server ready")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    if model_server.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "status": "healthy",
        "model_version": model_server.model_version,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """Readiness check endpoint"""
    return {
        "status": "ready",
        "model_loaded": model_server.model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate predictions for input features
    
    Args:
        request: Prediction request with features
    
    Returns:
        Predictions and metadata
    """
    start_time = time.time()
    
    try:
        # Convert to numpy array
        features = np.array(request.features)
        
        # Generate predictions
        predictions = model_server.predict(features)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_version=model_server.model_version,
            latency_ms=round(latency_ms, 2)
        )
    
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info")
async def model_info() -> Dict[str, Any]:
    """Get model metadata"""
    if model_server.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_version": model_server.model_version,
        "model_type": type(model_server.model).__name__,
        "features_count": getattr(model_server.model, 'n_features_in_', 'unknown')
    }
