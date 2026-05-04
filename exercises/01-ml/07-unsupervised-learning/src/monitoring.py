"""Production monitoring for SegmentAI

Provides: Prometheus metrics for API monitoring
"""

import logging
import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


logger = logging.getLogger("segmentai")


# Prometheus metrics
prediction_latency = Histogram(
    'clustering_latency_seconds',
    'Time spent processing clustering request',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

predictions_total = Counter(
    'clustering_requests_total',
    'Total number of clustering requests',
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

cluster_distribution = Gauge(
    'cluster_distribution',
    'Number of samples in each cluster',
    ['cluster_id']
)

cluster_count = Gauge(
    'cluster_count',
    'Total number of clusters identified'
)


def track_prediction_latency(func: Callable) -> Callable:
    """Decorator to track clustering latency.
    
    Args:
        func: Function to track
    
    Returns:
        Wrapped function with latency tracking
    
    Example:
        >>> @track_prediction_latency
        ... def cluster(data):
        ...     return model.predict(data)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            prediction_latency.observe(elapsed)
            logger.debug(f"Clustering latency: {elapsed:.3f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            prediction_latency.observe(elapsed)
            raise
    
    return wrapper


def track_prediction_count(model_name: str, status: str = "success") -> None:
    """Track clustering request count by model and status.
    
    Args:
        model_name: Name of model used
        status: Status of request (success/failure)
    
    Example:
        >>> track_prediction_count("kmeans", "success")
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
        >>> track_model_status("kmeans", True)
    """
    model_loaded.labels(model_name=model_name).set(1 if loaded else 0)


def track_cluster_distribution(cluster_labels: list) -> None:
    """Track distribution of samples across clusters.
    
    Args:
        cluster_labels: List of cluster assignments
    
    Example:
        >>> track_cluster_distribution([0, 0, 1, 1, 2, 2])
    """
    # Count samples per cluster
    from collections import Counter
    cluster_counts = Counter(cluster_labels)
    
    # Update gauges
    for cluster_id, count in cluster_counts.items():
        if cluster_id != -1:  # Exclude noise
            cluster_distribution.labels(cluster_id=str(cluster_id)).set(count)
    
    # Track total cluster count (excluding noise)
    n_clusters = len([c for c in cluster_counts.keys() if c != -1])
    cluster_count.set(n_clusters)


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
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_latency(0.123)
        >>> print(collector.get_summary())
    """
    
    def __init__(self):
        """Initialize collector."""
        self.latencies = []
        self.cluster_sizes = []
    
    def record_latency(self, latency: float) -> None:
        """Record request latency."""
        self.latencies.append(latency)
    
    def record_cluster_size(self, size: int) -> None:
        """Record cluster size."""
        self.cluster_sizes.append(size)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        import numpy as np
        
        summary = {}
        
        if self.latencies:
            summary['latency_p50'] = np.percentile(self.latencies, 50)
            summary['latency_p95'] = np.percentile(self.latencies, 95)
            summary['latency_p99'] = np.percentile(self.latencies, 99)
        
        if self.cluster_sizes:
            summary['avg_cluster_size'] = np.mean(self.cluster_sizes)
            summary['min_cluster_size'] = np.min(self.cluster_sizes)
            summary['max_cluster_size'] = np.max(self.cluster_sizes)
        
        return summary
    
    def reset(self) -> None:
        """Reset collected metrics."""
        self.latencies = []
        self.cluster_sizes = []
