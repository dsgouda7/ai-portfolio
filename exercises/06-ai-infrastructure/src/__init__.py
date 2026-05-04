"""
AI Infrastructure Exercise - Production ML Deployment Components
"""

__version__ = "1.0.0"

# Export key infrastructure components
from .utils import setup_logging, load_config
from .api import app as infrastructure_api

__all__ = [
    "setup_logging",
    "load_config",
    "infrastructure_api",
]
