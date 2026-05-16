"""Utility functions for logging and configuration."""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with JSON formatting for production.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S"
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    config['application']['port'] = int(os.getenv('APP_PORT', config['application']['port']))
    config['logging']['level'] = os.getenv('LOG_LEVEL', config['logging']['level'])
    
    return config


def get_env() -> str:
    """Get current deployment environment.
    
    Returns:
        Environment name (dev, staging, prod)
    """
    return os.getenv('ENVIRONMENT', 'dev')
