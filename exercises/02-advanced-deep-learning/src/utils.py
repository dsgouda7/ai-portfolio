"""Utility Functions for ProductionCV

Provides logging, timing, file I/O, and ONNX helper utilities.
"""

import os
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
import onnx
import onnxruntime as ort


def setup_logger(name: str = "productioncv", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("productioncv")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def get_model_size_mb(model_path: str) -> float:
    """
    Get model file size in megabytes.
    
    Args:
        model_path: Path to model file
        
    Returns:
        File size in MB
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def validate_onnx_model(onnx_path: str) -> bool:
    """
    Validate ONNX model structure.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        True if valid, False otherwise
    """
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return True
    except Exception as e:
        logger = logging.getLogger("productioncv")
        logger.error(f"ONNX validation failed: {str(e)}")
        return False


def create_onnx_session(onnx_path: str, providers: Optional[list] = None) -> ort.InferenceSession:
    """
    Create ONNX Runtime inference session.
    
    Args:
        onnx_path: Path to ONNX model
        providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
    Returns:
        ONNX Runtime inference session
    """
    if providers is None:
        providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.logger = logging.getLogger("productioncv")
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        self.logger.info(f"{self.name} took {self.elapsed:.2f}s")
