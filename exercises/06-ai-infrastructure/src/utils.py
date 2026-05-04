"""
Utility functions for AI Infrastructure
"""
import logging
import yaml
from pathlib import Path
from typing import Dict, Any


def setup_logging(level: str = "INFO", logger_name: str = "ai-infra") -> logging.Logger:
    """
    Configure logging for infrastructure components
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        logger_name: Name for the logger
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler with formatting
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(handler)
    
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_path(model_name: str, version: str = "latest") -> Path:
    """
    Get standardized model artifact path
    
    Args:
        model_name: Name of the model
        version: Model version
    
    Returns:
        Path to model artifact
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    if version == "latest":
        return models_dir / f"{model_name}_latest.pkl"
    return models_dir / f"{model_name}_v{version}.pkl"


def validate_environment() -> Dict[str, bool]:
    """
    Validate that required infrastructure components are available
    
    Returns:
        Dictionary of component availability
    """
    checks = {
        "config_file": Path("config.yaml").exists(),
        "models_dir": Path("models").exists(),
        "data_dir": Path("data").exists(),
        "mlruns_dir": Path("mlruns").exists(),
    }
    
    return checks
