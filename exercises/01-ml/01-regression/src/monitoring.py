"""Production monitoring for SmartVal AI

Provides: Prometheus metrics for API monitoring
"""

import logging
import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


logger = logging.getLogger("smartval")


# Prometheus metrics
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction request',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
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

prediction_value = Histogram(
    'prediction_value',
    'Distribution of prediction values',
    buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
)


def track_prediction_latency(func: Callable) -> Callable:
    """Decorator to track prediction latency.
    
    Args:
        func: Function to track
    
    Returns:
        Wrapped function with latency tracking
    
    Example:
        >>> @track_prediction_latency
        ... def predict(data):
        ...     return model.predict(data)
    """
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


def track_prediction_count(model_name: str, status: str = "success") -> None:
    """Track prediction count by model and status.
    
    Args:
        model_name: Name of model used
        status: Status of prediction (success/failure)
    
    Example:
        >>> track_prediction_count("ridge", "success")
    """
    predictions_total.labels(model_name=model_name, status=status).inc()


def track_error(error_type: str) -> None:
    """Track error occurrence by type.
    
    Args:
        error_type: Type of error (validation/model/server)
    
    Example:
        >>> track_error("validation")
    """
    errors_total.labels(error_type=error_type).inc()
    logger.warning(f"Error tracked: {error_type}")


def track_model_status(model_name: str, loaded: bool) -> None:
    """Track whether model is loaded.
    
    Args:
        model_name: Name of model
        loaded: Whether model is loaded
    
    Example:
        >>> track_model_status("ridge", True)
    """
    model_loaded.labels(model_name=model_name).set(1 if loaded else 0)


def track_prediction_value(value: float) -> None:
    """Track distribution of prediction values.
    
    Args:
        value: Predicted value
    
    Example:
        >>> track_prediction_value(3.5)
    """
    prediction_value.observe(value)


def get_metrics() -> bytes:
    """Get Prometheus metrics in exposition format.
    
    Returns:
        Metrics in Prometheus format
    
    Example:
        >>> metrics = get_metrics()
        >>> return Response(metrics, mimetype=CONTENT_TYPE_LATEST)
    """
    return generate_latest()


class MetricsCollector:
    """Collect and aggregate custom metrics.
    
    Useful for tracking business metrics beyond standard Prometheus counters.
    
    Attributes:
        metrics: Dictionary of collected metrics
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.record("mae", 25.3)
        >>> collector.get_summary()
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {}
        logger.info("Initialized MetricsCollector")
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value.
        
        Args:
            metric_name: Name of metric
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
    
    def get_summary(self) -> dict:
        """Get summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, min, max, count for each metric
        """
        import numpy as np
        
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.metrics.clear()
        logger.info("Metrics collector reset")


# Global collector instance
collector = MetricsCollector()
