"""Production monitoring for FaceAI

Provides: Prometheus metrics for API monitoring
"""

import logging
import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


logger = logging.getLogger("faceai")


# Prometheus metrics
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction request',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_name', 'status', 'predicted_class']
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


def track_prediction_latency(func: Callable) -> Callable:
    """Decorator to track prediction latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            prediction_latency.observe(elapsed)
            logger.debug(f"Prediction latency: {elapsed:.3f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            prediction_latency.observe(elapsed)
            raise
    
    return wrapper


def track_prediction_count(model_name: str, predicted_class: int, status: str = "success") -> None:
    """Track prediction count by model, class, and status."""
    predictions_total.labels(
        model_name=model_name,
        status=status,
        predicted_class=str(predicted_class)
    ).inc()


def track_error(error_type: str) -> None:
    """Track error occurrence by type."""
    errors_total.labels(error_type=error_type).inc()
    logger.warning(f"Error tracked: {error_type}")


def track_model_status(model_name: str, loaded: bool) -> None:
    """Track whether model is loaded."""
    model_loaded.labels(model_name=model_name).set(1 if loaded else 0)


def get_metrics() -> bytes:
    """Get Prometheus metrics in exposition format."""
    return generate_latest()
