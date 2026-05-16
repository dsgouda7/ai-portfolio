"""
Monitoring and metrics collection for PixelSmith
"""

import time
from functools import wraps
from typing import Callable, Dict, Any
import mlflow
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from .utils import setup_logger


logger = setup_logger(__name__)


# Prometheus metrics
REQUEST_COUNT = Counter(
    'pixelsmith_requests_total',
    'Total number of requests',
    ['endpoint', 'modality', 'status']
)

REQUEST_DURATION = Histogram(
    'pixelsmith_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'modality']
)

ACTIVE_REQUESTS = Gauge(
    'pixelsmith_active_requests',
    'Number of active requests',
    ['endpoint']
)

MODEL_LOAD_TIME = Gauge(
    'pixelsmith_model_load_seconds',
    'Time taken to load model',
    ['model_name']
)

GENERATION_TIME = Histogram(
    'pixelsmith_generation_duration_seconds',
    'Image generation duration in seconds'
)

CLIP_SCORE = Histogram(
    'pixelsmith_clip_score',
    'CLIP similarity scores'
)


class MetricsCollector:
    """
    Collect and track metrics for PixelSmith.
    """
    
    def __init__(self, mlflow_uri: str = "file:./mlruns", experiment_name: str = "pixelsmith"):
        """
        Initialize metrics collector.
        
        Args:
            mlflow_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
        """
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        
        # Set up MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Initialized metrics collector with MLflow URI: {mlflow_uri}")
    
    def log_request(
        self,
        endpoint: str,
        modality: str,
        duration: float,
        status: str = "success"
    ):
        """
        Log API request metrics.
        
        Args:
            endpoint: API endpoint name
            modality: Modality being processed
            duration: Request duration in seconds
            status: Request status (success/error)
        """
        REQUEST_COUNT.labels(endpoint=endpoint, modality=modality, status=status).inc()
        REQUEST_DURATION.labels(endpoint=endpoint, modality=modality).observe(duration)
    
    def log_model_load(self, model_name: str, load_time: float):
        """
        Log model loading time.
        
        Args:
            model_name: Name of the model
            load_time: Time taken to load in seconds
        """
        MODEL_LOAD_TIME.labels(model_name=model_name).set(load_time)
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
    
    def log_generation(self, duration: float, clip_score: float = None):
        """
        Log image generation metrics.
        
        Args:
            duration: Generation duration in seconds
            clip_score: CLIP similarity score
        """
        GENERATION_TIME.observe(duration)
        
        if clip_score is not None:
            CLIP_SCORE.observe(clip_score)
    
    def start_mlflow_run(self, run_name: str = None) -> Any:
        """
        Start MLflow run.
        
        Args:
            run_name: Name for the run
        
        Returns:
            MLflow run context
        """
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
        """
        mlflow.log_metrics(metrics)
    
    def log_artifact(self, file_path: str):
        """
        Log artifact to MLflow.
        
        Args:
            file_path: Path to artifact file
        """
        mlflow.log_artifact(file_path)


def monitor_performance(endpoint: str, modality: str = "unknown"):
    """
    Decorator to monitor endpoint performance.
    
    Args:
        endpoint: Endpoint name
        modality: Modality being processed
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Track active requests
            ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
            
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise e
            finally:
                duration = time.time() - start_time
                
                # Log metrics
                REQUEST_COUNT.labels(
                    endpoint=endpoint,
                    modality=modality,
                    status=status
                ).inc()
                
                REQUEST_DURATION.labels(
                    endpoint=endpoint,
                    modality=modality
                ).observe(duration)
                
                ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()
                
                logger.info(f"{endpoint} completed in {duration:.3f}s with status: {status}")
        
        return wrapper
    return decorator


def start_metrics_server(port: int = 8000):
    """
    Start Prometheus metrics server.
    
    Args:
        port: Port to serve metrics on
    """
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")


class PerformanceTracker:
    """
    Track performance of model operations.
    """
    
    def __init__(self):
        self.timings = {}
    
    def start(self, operation: str):
        """Start timing an operation."""
        self.timings[operation] = {"start": time.time()}
    
    def end(self, operation: str) -> float:
        """
        End timing an operation.
        
        Returns:
            Duration in seconds
        """
        if operation not in self.timings:
            logger.warning(f"Operation {operation} not started")
            return 0.0
        
        duration = time.time() - self.timings[operation]["start"]
        self.timings[operation]["duration"] = duration
        
        return duration
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all timings.
        
        Returns:
            Dictionary of operation -> duration
        """
        return {
            op: timing.get("duration", 0.0)
            for op, timing in self.timings.items()
        }
