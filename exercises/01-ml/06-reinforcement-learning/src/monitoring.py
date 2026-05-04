"""Production monitoring for AgentAI

Provides: Prometheus metrics for RL API monitoring
"""

import logging
import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


logger = logging.getLogger("agentai")


# Prometheus metrics
action_latency = Histogram(
    'action_selection_latency_seconds',
    'Time spent selecting action',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0)
)

actions_total = Counter(
    'actions_total',
    'Total number of actions selected',
    ['policy_name', 'status']
)

episode_reward = Gauge(
    'episode_reward',
    'Most recent episode reward',
    ['policy_name']
)

training_episodes_total = Counter(
    'training_episodes_total',
    'Total number of training episodes completed',
    ['policy_name']
)

errors_total = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type']
)

policy_loaded = Gauge(
    'policy_loaded',
    'Whether policy is loaded (1=loaded, 0=not loaded)',
    ['policy_name']
)


def track_action_latency(func: Callable) -> Callable:
    """Decorator to track action selection latency.
    
    Args:
        func: Function to track
    
    Returns:
        Wrapped function with latency tracking
    
    Example:
        >>> @track_action_latency
        ... def select_action(state):
        ...     return policy.predict(state)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            action_latency.observe(elapsed)
            logger.debug(f"Action selection latency: {elapsed:.3f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            action_latency.observe(elapsed)
            raise
    
    return wrapper


def track_action_count(policy_name: str, status: str = "success") -> None:
    """Track action selection count by policy and status.
    
    Args:
        policy_name: Name of policy used
        status: Status of action selection (success/failure)
    
    Example:
        >>> track_action_count("reinforce", "success")
    """
    actions_total.labels(policy_name=policy_name, status=status).inc()


def track_episode_reward_value(policy_name: str, reward: float) -> None:
    """Track episode reward.
    
    Args:
        policy_name: Name of policy
        reward: Episode reward
    
    Example:
        >>> track_episode_reward_value("reinforce", 195.0)
    """
    episode_reward.labels(policy_name=policy_name).set(reward)


def track_training_episode(policy_name: str) -> None:
    """Track training episode completion.
    
    Args:
        policy_name: Name of policy being trained
    
    Example:
        >>> track_training_episode("reinforce")
    """
    training_episodes_total.labels(policy_name=policy_name).inc()


def track_error(error_type: str) -> None:
    """Track error occurrence by type.
    
    Args:
        error_type: Type of error (e.g., 'environment', 'policy', 'api')
    
    Example:
        >>> track_error("environment")
    """
    errors_total.labels(error_type=error_type).inc()
    logger.warning(f"Error tracked: {error_type}")


def track_policy_status(policy_name: str, loaded: bool) -> None:
    """Track policy load status.
    
    Args:
        policy_name: Name of policy
        loaded: Whether policy is loaded
    
    Example:
        >>> track_policy_status("reinforce", True)
    """
    policy_loaded.labels(policy_name=policy_name).set(1 if loaded else 0)


def get_metrics() -> tuple:
    """Get current Prometheus metrics.
    
    Returns:
        Tuple of (metrics_text, content_type)
    
    Example:
        >>> metrics, content_type = get_metrics()
    """
    return generate_latest(), CONTENT_TYPE_LATEST
