"""Utility functions for AgentAI

Provides: logging setup, reproducibility helpers, timing decorators
"""

import logging
import random
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Configure structured logging for production.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs to console only.
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logging("INFO", "logs/agentai.log")
        >>> logger.info("Policy training started")
    """
    logger = logging.getLogger("agentai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    
    Example:
        >>> set_seed(42)  # Ensures reproducible episode generation
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function with timing
    
    Example:
        >>> @timer
        ... def train_policy():
        ...     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger("agentai")
        start = time.time()
        
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        
        elapsed = time.time() - start
        logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
        
        return result
    
    return wrapper


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name of parameter (for error message)
    
    Raises:
        ValueError: If value is not positive
    
    Example:
        >>> validate_positive(0.001, "learning_rate")
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_in_range(value: float, name: str, min_val: float, max_val: float) -> None:
    """Validate that a value is within a range.
    
    Args:
        value: Value to validate
        name: Name of parameter (for error message)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Raises:
        ValueError: If value is out of range
    
    Example:
        >>> validate_in_range(0.99, "gamma", 0.0, 1.0)
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")
