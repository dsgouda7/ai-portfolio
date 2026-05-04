"""Production monitoring for EnsembleAI

Provides: Prometheus metrics for API monitoring and ensemble agreement tracking
"""

import logging
import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


logger = logging.getLogger("ensembleai")


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
    'Distribution of prediction values (class labels)',
    buckets=(0, 1)
)

ensemble_agreement_rate = Gauge(
    'ensemble_agreement_rate',
    'Rate of agreement among base models in ensemble',
    ['ensemble_type']
)

confidence_score = Histogram(
    'confidence_score',
    'Distribution of prediction confidence scores',
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0)
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
        >>> track_prediction_count("voting", "success")
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
        >>> track_model_status("voting", True)
    """
    model_loaded.labels(model_name=model_name).set(1 if loaded else 0)


def track_prediction_value(value: int) -> None:
    """Track distribution of prediction values (class labels).
    
    Args:
        value: Predicted class label
    
    Example:
        >>> track_prediction_value(1)
    """
    prediction_value.observe(value)


def track_ensemble_agreement(ensemble_type: str, agreement_rate: float) -> None:
    """Track agreement rate among ensemble base models.
    
    Args:
        ensemble_type: Type of ensemble (voting/stacking)
        agreement_rate: Rate of agreement (0.0 to 1.0)
    
    Example:
        >>> track_ensemble_agreement("voting", 0.85)
    """
    ensemble_agreement_rate.labels(ensemble_type=ensemble_type).set(agreement_rate)
    logger.debug(f"{ensemble_type} ensemble agreement: {agreement_rate:.2%}")


def track_confidence_score(confidence: float) -> None:
    """Track distribution of prediction confidence scores.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
    
    Example:
        >>> track_confidence_score(0.92)
    """
    confidence_score.observe(confidence)


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
    
    Useful for tracking application-specific metrics that don't fit
    standard Prometheus patterns.
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_metric("custom_metric", 42.0)
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {}
        logger.info("Initialized MetricsCollector")
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
    
    def get_summary(self) -> dict:
        """Get summary statistics for all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        import numpy as np
        
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary
