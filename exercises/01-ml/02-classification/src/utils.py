"""Utility functions for FaceAI

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
        >>> logger = setup_logging("INFO", "logs/faceai.log")
        >>> logger.info("Model training started")
    """
    logger = logging.getLogger("faceai")
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
        >>> set_seed(42)  # Ensures reproducible train/test splits
    """
    random.seed(seed)
    np.random.seed(seed)


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function with timing
    
    Example:
        >>> @timer
        ... def train_model():
        ...     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger("faceai")
        start = time.time()
        
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        
        elapsed = time.time() - start
        logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
        
        return result
    
    return wrapper


def validate_positive(value: float, name: str) -> None:
    """Validate that a numeric value is positive.
    
    Args:
        value: Value to validate
        name: Parameter name for error message
    
    Raises:
        ValueError: If value is not positive
    
    Example:
        >>> validate_positive(0.1, "learning_rate")  # OK
        >>> validate_positive(-1, "C")  # Raises ValueError
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_in_range(value: float, name: str, min_val: float, max_val: float) -> None:
    """Validate that a value is within specified range.
    
    Args:
        value: Value to validate
        name: Parameter name for error message
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Raises:
        ValueError: If value is outside range
    
    Example:
        >>> validate_in_range(0.5, "test_size", 0.0, 1.0)  # OK
        >>> validate_in_range(1.5, "test_size", 0.0, 1.0)  # Raises ValueError
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")


class LatencyTracker:
    """Track API request latencies for monitoring.
    
    Attributes:
        latencies: List of recorded latencies in seconds
    
    Example:
        >>> tracker = LatencyTracker()
        >>> with tracker:
        ...     # Do work
        ...     pass
        >>> tracker.get_percentile(95)  # p95 latency
    """
    
    def __init__(self):
        self.latencies = []
        self._start_time = None
    
    def __enter__(self):
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time:
            elapsed = time.time() - self._start_time
            self.latencies.append(elapsed)
    
    def get_percentile(self, percentile: float) -> float:
        """Get specified percentile of latencies.
        
        Args:
            percentile: Percentile to compute (0-100)
        
        Returns:
            Latency at specified percentile in seconds
        """
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, percentile))
    
    def get_stats(self) -> dict:
        """Get summary statistics of latencies.
        
        Returns:
            Dictionary with mean, median, p50, p95, p99 latencies
        """
        if not self.latencies:
            return {
                "mean": 0.0, "median": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0
            }
        
        return {
            "mean": float(np.mean(self.latencies)),
            "median": float(np.median(self.latencies)),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
        }
