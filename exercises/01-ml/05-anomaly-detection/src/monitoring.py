"""Production monitoring for FraudShield

Provides: Prometheus metrics for API monitoring and anomaly rate tracking
"""

import logging
import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


logger = logging.getLogger("fraudshield")


# Prometheus metrics
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction request',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_name', 'prediction']
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

anomaly_score_distribution = Histogram(
    'anomaly_score',
    'Distribution of anomaly scores',
    buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0)
)

anomaly_rate = Gauge(
    'anomaly_rate',
    'Current anomaly detection rate (rolling window)'
)


def track_prediction_latency(func: Callable) -> Callable:
    """Decorator to track prediction latency.
    
    Args:
        func: Function to track
    
    Returns:
        Wrapped function with latency tracking
    
    Example:
        >>> @track_prediction_latency
        ... def detect(data):
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


def track_prediction(model_name: str, is_anomaly: bool) -> None:
    """Track prediction count by model and prediction type.
    
    Args:
        model_name: Name of model used
        is_anomaly: Whether prediction was anomaly (True) or normal (False)
    
    Example:
        >>> track_prediction("isolation_forest", True)
    """
    prediction_label = "anomaly" if is_anomaly else "normal"
    predictions_total.labels(model_name=model_name, prediction=prediction_label).inc()


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
        >>> track_model_status("isolation_forest", True)
    """
    model_loaded.labels(model_name=model_name).set(1 if loaded else 0)


def track_anomaly_score(score: float) -> None:
    """Track distribution of anomaly scores.
    
    Args:
        score: Anomaly score
    
    Example:
        >>> track_anomaly_score(0.85)
    """
    anomaly_score_distribution.observe(score)


def update_anomaly_rate(rate: float) -> None:
    """Update current anomaly rate gauge.
    
    Args:
        rate: Anomaly rate (0.0 to 1.0)
    
    Example:
        >>> update_anomaly_rate(0.12)  # 12% anomaly rate
    """
    anomaly_rate.set(rate)
    
    # Log alert if rate exceeds threshold
    if rate > 0.15:
        logger.warning(f"High anomaly rate detected: {rate:.1%}")


def get_metrics() -> bytes:
    """Get Prometheus metrics in exposition format.
    
    Returns:
        Metrics in Prometheus format
    
    Example:
        >>> metrics = get_metrics()
        >>> return Response(metrics, mimetype=CONTENT_TYPE_LATEST)
    """
    return generate_latest()


class AnomalyRateTracker:
    """Track anomaly rate over a rolling window.
    
    Maintains a sliding window of predictions to compute
    real-time anomaly rate for monitoring.
    
    Attributes:
        window_size: Size of rolling window
        predictions: List of recent predictions (0/1)
    
    Example:
        >>> tracker = AnomalyRateTracker(window_size=100)
        >>> tracker.add_prediction(is_anomaly=True)
        >>> rate = tracker.get_rate()
    """
    
    def __init__(self, window_size: int = 1000):
        """Initialize tracker.
        
        Args:
            window_size: Number of predictions to track
        """
        self.window_size = window_size
        self.predictions = []
    
    def add_prediction(self, is_anomaly: bool) -> None:
        """Add a prediction to the window.
        
        Args:
            is_anomaly: Whether prediction was anomaly
        """
        self.predictions.append(1 if is_anomaly else 0)
        
        # Keep only last window_size predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
    
    def get_rate(self) -> float:
        """Get current anomaly rate.
        
        Returns:
            Anomaly rate (0.0 to 1.0)
        """
        if not self.predictions:
            return 0.0
        
        return sum(self.predictions) / len(self.predictions)
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.predictions = []


# Global anomaly rate tracker
_anomaly_rate_tracker = AnomalyRateTracker(window_size=1000)


def track_and_update_anomaly_rate(is_anomaly: bool) -> float:
    """Track prediction and update anomaly rate gauge.
    
    Args:
        is_anomaly: Whether prediction was anomaly
    
    Returns:
        Current anomaly rate
    
    Example:
        >>> rate = track_and_update_anomaly_rate(True)
    """
    _anomaly_rate_tracker.add_prediction(is_anomaly)
    rate = _anomaly_rate_tracker.get_rate()
    update_anomaly_rate(rate)
    return rate
