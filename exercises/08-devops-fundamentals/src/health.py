"""Health check and readiness probe implementations."""

import psutil
from flask import jsonify
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health and readiness checks for Kubernetes."""
    
    def __init__(self, model):
        """Initialize health checker.
        
        Args:
            model: ML model instance
        """
        self.model = model
    
    def health_check(self) -> tuple:
        """Liveness probe - is the service alive?
        
        Returns:
            JSON response and status code
        """
        try:
            # Check if process is responding
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return jsonify({
                'status': 'healthy',
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent
            }), 200
        
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 503
    
    def readiness_check(self) -> tuple:
        """Readiness probe - is the service ready to accept traffic?
        
        Returns:
            JSON response and status code
        """
        try:
            # Check if model is loaded
            if self.model.model is None:
                return jsonify({
                    'status': 'not_ready',
                    'reason': 'model_not_loaded'
                }), 503
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return jsonify({
                    'status': 'not_ready',
                    'reason': 'high_memory_usage',
                    'memory_percent': memory.percent
                }), 503
            
            return jsonify({
                'status': 'ready',
                'model_loaded': True,
                'memory_percent': memory.percent
            }), 200
        
        except Exception as e:
            logger.error(f"Readiness check failed: {str(e)}")
            return jsonify({
                'status': 'not_ready',
                'error': str(e)
            }), 503
