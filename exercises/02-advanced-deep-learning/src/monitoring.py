"""Monitoring and Metrics Collection

Implements Prometheus metrics for production monitoring.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from flask import Response
import time
from functools import wraps
from src.utils import setup_logger


logger = setup_logger()


# Define Prometheus metrics
PREDICTION_COUNTER = Counter(
    'productioncv_predictions_total',
    'Total number of predictions made',
    ['model_type', 'status']
)

PREDICTION_LATENCY = Histogram(
    'productioncv_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_type'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

DETECTION_COUNT = Histogram(
    'productioncv_detections_per_image',
    'Number of detections per image',
    buckets=[0, 1, 2, 5, 10, 20, 50, 100]
)

MODEL_SIZE_MB = Gauge(
    'productioncv_model_size_mb',
    'Model size in megabytes',
    ['model_name']
)

INFERENCE_FPS = Gauge(
    'productioncv_inference_fps',
    'Inference frames per second',
    ['model_name']
)

API_REQUEST_COUNT = Counter(
    'productioncv_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

API_REQUEST_LATENCY = Histogram(
    'productioncv_api_request_latency_seconds',
    'API request latency in seconds',
    ['endpoint', 'method'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ERROR_COUNTER = Counter(
    'productioncv_errors_total',
    'Total errors',
    ['error_type']
)


class MetricsCollector:
    """
    Centralized metrics collection for ProductionCV.
    """
    
    def __init__(self):
        logger.info("Initialized MetricsCollector")
    
    @staticmethod
    def track_prediction(model_type: str, status: str = 'success'):
        """
        Track a prediction event.
        
        Args:
            model_type: Type of model used
            status: Prediction status ('success' or 'error')
        """
        PREDICTION_COUNTER.labels(model_type=model_type, status=status).inc()
    
    @staticmethod
    def track_latency(model_type: str, latency_seconds: float):
        """
        Track prediction latency.
        
        Args:
            model_type: Type of model used
            latency_seconds: Latency in seconds
        """
        PREDICTION_LATENCY.labels(model_type=model_type).observe(latency_seconds)
    
    @staticmethod
    def track_detections(num_detections: int):
        """
        Track number of detections.
        
        Args:
            num_detections: Number of objects detected
        """
        DETECTION_COUNT.observe(num_detections)
    
    @staticmethod
    def set_model_size(model_name: str, size_mb: float):
        """
        Set model size metric.
        
        Args:
            model_name: Name of the model
            size_mb: Size in megabytes
        """
        MODEL_SIZE_MB.labels(model_name=model_name).set(size_mb)
    
    @staticmethod
    def set_inference_fps(model_name: str, fps: float):
        """
        Set inference FPS metric.
        
        Args:
            model_name: Name of the model
            fps: Frames per second
        """
        INFERENCE_FPS.labels(model_name=model_name).set(fps)
    
    @staticmethod
    def track_api_request(endpoint: str, method: str, status: int):
        """
        Track API request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status: HTTP status code
        """
        API_REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()
    
    @staticmethod
    def track_api_latency(endpoint: str, method: str, latency_seconds: float):
        """
        Track API request latency.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            latency_seconds: Latency in seconds
        """
        API_REQUEST_LATENCY.labels(
            endpoint=endpoint,
            method=method
        ).observe(latency_seconds)
    
    @staticmethod
    def track_error(error_type: str):
        """
        Track error occurrence.
        
        Args:
            error_type: Type of error
        """
        ERROR_COUNTER.labels(error_type=error_type).inc()
    
    @staticmethod
    def get_metrics() -> Response:
        """
        Get Prometheus metrics in exposition format.
        
        Returns:
            Flask Response with metrics
        """
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


def monitor_prediction(model_type: str = 'faster_rcnn'):
    """
    Decorator to monitor prediction functions.
    
    Args:
        model_type: Type of model being used
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                
                # Track number of detections if result contains boxes
                if isinstance(result, dict) and 'boxes' in result:
                    num_detections = len(result['boxes'])
                    MetricsCollector.track_detections(num_detections)
                
                return result
                
            except Exception as e:
                status = 'error'
                MetricsCollector.track_error(type(e).__name__)
                raise
                
            finally:
                # Track prediction and latency
                latency = time.time() - start_time
                MetricsCollector.track_prediction(model_type, status)
                MetricsCollector.track_latency(model_type, latency)
        
        return wrapper
    return decorator


def monitor_api_request(func):
    """
    Decorator to monitor API endpoint requests.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from flask import request
        
        start_time = time.time()
        endpoint = request.endpoint or 'unknown'
        method = request.method
        status = 200
        
        try:
            result = func(*args, **kwargs)
            
            # Extract status code if result is a tuple
            if isinstance(result, tuple) and len(result) > 1:
                status = result[1]
            
            return result
            
        except Exception as e:
            status = 500
            MetricsCollector.track_error(type(e).__name__)
            raise
            
        finally:
            # Track API metrics
            latency = time.time() - start_time
            MetricsCollector.track_api_request(endpoint, method, status)
            MetricsCollector.track_api_latency(endpoint, method, latency)
    
    return wrapper


# Initialize global metrics collector
metrics_collector = MetricsCollector()
