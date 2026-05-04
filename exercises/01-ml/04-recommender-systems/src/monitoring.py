"""Production monitoring for FlixAI

Provides: Prometheus metrics for API monitoring
"""

import logging
import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


logger = logging.getLogger("flixai")


# Prometheus metrics
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time spent processing recommendation request',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

predictions_total = Counter(
    'predictions_total',
    'Total number of recommendations made',
    ['model_name', 'status']
)

errors_total = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type']
)

model_loaded = Gauge(
    'model_loaded',
    'Whether model is loaded (1=loaded, 0=not loaded)',
    ['model_name']
)

recommendation_count = Histogram(
    'recommendation_count',
    'Distribution of recommendation counts (K)',
    buckets=(1, 5, 10, 20, 50, 100)
)


def track_prediction_latency(func: Callable) -> Callable:
    """Decorator to track recommendation latency.
    
    Args:
        func: Function to track
    
    Returns:
        Wrapped function with latency tracking
    
    Example:
        >>> @track_prediction_latency
        ... def recommend(user_id, k):
        ...     return model.recommend_items(user_id, k)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            prediction_latency.observe(elapsed)
            logger.debug(f"Recommendation latency: {elapsed:.3f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            prediction_latency.observe(elapsed)
            raise
    
    return wrapper


def track_prediction_count(model_name: str, status: str = "success") -> None:
    """Track recommendation count by model and status.
    
    Args:
        model_name: Name of model used
        status: Status of recommendation (success/failure)
    
    Example:
        >>> track_prediction_count("matrix_factorization", "success")
    """
    predictions_total.labels(model_name=model_name, status=status).inc()


def track_error(error_type: str) -> None:
    """Track error occurrence by type.
    
    Args:
        error_type: Type of error (e.g., 'validation', 'model_error')
    
    Example:
        >>> track_error("validation_error")
    """
    errors_total.labels(error_type=error_type).inc()
    logger.warning(f"Error tracked: {error_type}")


def track_model_status(model_name: str, loaded: bool) -> None:
    """Track model loading status.
    
    Args:
        model_name: Name of model
        loaded: Whether model is loaded
    
    Example:
        >>> track_model_status("matrix_factorization", True)
    """
    model_loaded.labels(model_name=model_name).set(1 if loaded else 0)
    logger.info(f"Model '{model_name}' status: {'loaded' if loaded else 'unloaded'}")


def track_recommendation_k(k: int) -> None:
    """Track recommendation count (K value).
    
    Args:
        k: Number of recommendations
    
    Example:
        >>> track_recommendation_k(10)
    """
    recommendation_count.observe(k)


def get_metrics() -> tuple:
    """Get Prometheus metrics in exposition format.
    
    Returns:
        Tuple of (metrics_text, content_type)
    
    Example:
        >>> metrics, content_type = get_metrics()
        >>> # Return from Flask endpoint
    """
    return generate_latest(), CONTENT_TYPE_LATEST
