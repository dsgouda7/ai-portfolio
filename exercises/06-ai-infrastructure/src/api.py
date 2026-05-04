"""
Infrastructure API - Health checks, metrics, and system status
"""
from flask import Flask, jsonify, request
from prometheus_client import Counter, Gauge, generate_latest
from typing import Dict, Any
import psutil
import time
from .utils import setup_logging, load_config, validate_environment

logger = setup_logging()

# Create Flask app
app = Flask(__name__)

# Prometheus metrics
system_cpu_usage = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
system_memory_usage = Gauge('system_memory_usage_percent', 'System memory usage percentage')
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])


@app.before_request
def before_request():
    """Track request start time"""
    request.start_time = time.time()


@app.after_request
def after_request(response):
    """Record request metrics"""
    endpoint = request.endpoint or 'unknown'
    status = response.status_code
    api_requests.labels(endpoint=endpoint, status=status).inc()
    
    return response


@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint
    
    Returns:
        System health status
    """
    try:
        # Check environment
        env_checks = validate_environment()
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Update metrics
        system_cpu_usage.set(cpu_percent)
        system_memory_usage.set(memory_percent)
        
        health_status = {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": env_checks,
            "resources": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory_percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route('/ready', methods=['GET'])
def readiness_check() -> Dict[str, str]:
    """
    Readiness check endpoint for Kubernetes
    
    Returns:
        Readiness status
    """
    try:
        # Check critical dependencies
        env_checks = validate_environment()
        
        if all(env_checks.values()):
            return jsonify({"status": "ready"}), 200
        else:
            return jsonify({
                "status": "not_ready",
                "checks": env_checks
            }), 503
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            "status": "not_ready",
            "error": str(e)
        }), 503


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Prometheus metrics endpoint
    
    Returns:
        Metrics in Prometheus format
    """
    return generate_latest()


@app.route('/info', methods=['GET'])
def system_info() -> Dict[str, Any]:
    """
    Get system and infrastructure information
    
    Returns:
        System information
    """
    try:
        config = load_config()
        
        info = {
            "version": "1.0.0",
            "environment": config.get("cloud", {}).get("environment", "unknown"),
            "components": {
                "mlflow_uri": config.get("mlflow", {}).get("tracking_uri"),
                "feature_store": config.get("feature_store", {}).get("project_name"),
                "kubernetes_namespace": config.get("kubernetes", {}).get("namespace")
            },
            "system": {
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "python_version": "3.11+"
            }
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/config', methods=['GET'])
def get_config() -> Dict[str, Any]:
    """
    Get current infrastructure configuration
    
    Returns:
        Configuration dictionary (sensitive values masked)
    """
    try:
        config = load_config()
        
        # Mask sensitive values
        safe_config = _mask_sensitive_values(config)
        
        return jsonify(safe_config), 200
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        return jsonify({"error": str(e)}), 500


def _mask_sensitive_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mask sensitive configuration values
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Config with masked sensitive values
    """
    sensitive_keys = ['password', 'secret', 'token', 'key', 'credential']
    
    def mask_dict(d: Dict) -> Dict:
        masked = {}
        for key, value in d.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = mask_dict(value)
            else:
                masked[key] = value
        return masked
    
    return mask_dict(config)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
