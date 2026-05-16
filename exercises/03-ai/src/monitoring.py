"""
Prometheus monitoring for PizzaBot
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable, Any
from .utils import setup_logger

logger = setup_logger(__name__)

# Metrics

# Conversation metrics
conversations_total = Counter(
    'pizzabot_conversations_total',
    'Total number of conversations',
    ['session_status']
)

messages_total = Counter(
    'pizzabot_messages_total',
    'Total number of messages',
    ['role', 'intent']
)

# Intent detection metrics
intent_predictions = Counter(
    'pizzabot_intent_predictions_total',
    'Total intent predictions',
    ['intent', 'confidence_level']
)

intent_accuracy = Gauge(
    'pizzabot_intent_accuracy',
    'Current intent detection accuracy'
)

# RAG metrics
retrieval_operations = Counter(
    'pizzabot_retrieval_operations_total',
    'Total retrieval operations',
    ['status']
)

retrieval_latency = Histogram(
    'pizzabot_retrieval_latency_seconds',
    'Retrieval operation latency',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

documents_retrieved = Histogram(
    'pizzabot_documents_retrieved',
    'Number of documents retrieved per query',
    buckets=[0, 1, 2, 3, 5, 10, 20]
)

retrieval_accuracy = Gauge(
    'pizzabot_retrieval_accuracy',
    'Average similarity score of retrieved documents'
)

# Generation metrics
generation_operations = Counter(
    'pizzabot_generation_operations_total',
    'Total generation operations',
    ['status', 'model']
)

generation_latency = Histogram(
    'pizzabot_generation_latency_seconds',
    'Generation operation latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

tokens_used = Counter(
    'pizzabot_tokens_used_total',
    'Total tokens used',
    ['model', 'operation_type']
)

# Order metrics
orders_total = Counter(
    'pizzabot_orders_total',
    'Total pizza orders',
    ['status']
)

order_value = Histogram(
    'pizzabot_order_value_dollars',
    'Order value distribution',
    buckets=[10, 20, 30, 50, 75, 100, 150]
)

# API metrics
api_requests = Counter(
    'pizzabot_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

api_latency = Histogram(
    'pizzabot_api_latency_seconds',
    'API request latency',
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Error metrics
errors_total = Counter(
    'pizzabot_errors_total',
    'Total errors',
    ['error_type', 'component']
)

# System info
system_info = Info(
    'pizzabot_system',
    'System information'
)


def track_retrieval(func: Callable) -> Callable:
    """Decorator to track retrieval operations."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        status = 'success'
        
        try:
            result = func(*args, **kwargs)
            
            # Track metrics
            if isinstance(result, list):
                documents_retrieved.observe(len(result))
                if result:
                    avg_similarity = sum(
                        doc.get('similarity', 0) for doc in result
                    ) / len(result)
                    retrieval_accuracy.set(avg_similarity)
            
            return result
            
        except Exception as e:
            status = 'error'
            errors_total.labels(
                error_type=type(e).__name__,
                component='retrieval'
            ).inc()
            raise
        
        finally:
            latency = time.time() - start_time
            retrieval_latency.observe(latency)
            retrieval_operations.labels(status=status).inc()
    
    return wrapper


def track_generation(func: Callable) -> Callable:
    """Decorator to track generation operations."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        status = 'success'
        model = kwargs.get('model', 'unknown')
        
        try:
            result = func(*args, **kwargs)
            
            # Track tokens if available
            if isinstance(result, dict) and 'tokens_used' in result:
                tokens_used.labels(
                    model=model,
                    operation_type='generation'
                ).inc(result['tokens_used'])
            
            return result
            
        except Exception as e:
            status = 'error'
            errors_total.labels(
                error_type=type(e).__name__,
                component='generation'
            ).inc()
            raise
        
        finally:
            latency = time.time() - start_time
            generation_latency.observe(latency)
            generation_operations.labels(status=status, model=model).inc()
    
    return wrapper


def track_api_request(func: Callable) -> Callable:
    """Decorator to track API requests."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        endpoint = func.__name__
        method = 'POST'  # Most endpoints are POST
        status = '200'
        
        try:
            result = func(*args, **kwargs)
            return result
            
        except Exception as e:
            status = '500'
            errors_total.labels(
                error_type=type(e).__name__,
                component='api'
            ).inc()
            raise
        
        finally:
            latency = time.time() - start_time
            api_latency.observe(latency)
            api_requests.labels(
                endpoint=endpoint,
                method=method,
                status=status
            ).inc()
    
    return wrapper


def update_conversation_metrics(
    session_status: str,
    intent: str = None,
    confidence: float = None
) -> None:
    """
    Update conversation-related metrics.
    
    Args:
        session_status: Status of session (started, ended)
        intent: Detected intent
        confidence: Intent confidence score
    """
    conversations_total.labels(session_status=session_status).inc()
    
    if intent:
        confidence_level = 'high' if confidence and confidence > 0.8 else 'low'
        intent_predictions.labels(
            intent=intent,
            confidence_level=confidence_level
        ).inc()


def update_order_metrics(status: str, value: float = None) -> None:
    """
    Update order-related metrics.
    
    Args:
        status: Order status (placed, cancelled, completed)
        value: Order value in dollars
    """
    orders_total.labels(status=status).inc()
    
    if value:
        order_value.observe(value)


def log_system_info(config: dict) -> None:
    """
    Log system information.
    
    Args:
        config: Configuration dictionary
    """
    system_info.info({
        'version': '1.0.0',
        'embedding_model': config['models']['embedding_model'],
        'llm_model': config['models']['llm_model'],
        'vector_db': config['vector_db']['type']
    })
    
    logger.info("System info logged to Prometheus")
