"""
Load testing scenarios with Locust
"""
from locust import HttpUser, task, between
from typing import Dict, Any
import random
import json
from .utils import setup_logging, load_config

logger = setup_logging()


class ModelServingUser(HttpUser):
    """
    Simulate user requests to model serving API
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.config = load_config()
        self.load_config = self.config.get("load_testing", {})
        logger.info("Load test user started")
    
    @task(3)
    def predict_single(self):
        """
        Task: Single prediction request (weight=3, most common)
        """
        # Generate random features
        features = [[random.uniform(0, 10) for _ in range(8)]]
        
        payload = {
            "features": features,
            "model_version": "latest"
        }
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "predictions" in result:
                    response.success()
                else:
                    response.failure("No predictions in response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def predict_batch(self):
        """
        Task: Batch prediction request (weight=2)
        """
        # Generate batch of random features
        batch_size = random.randint(5, 20)
        features = [[random.uniform(0, 10) for _ in range(8)] for _ in range(batch_size)]
        
        payload = {
            "features": features,
            "model_version": "latest"
        }
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if len(result.get("predictions", [])) == batch_size:
                    response.success()
                else:
                    response.failure("Batch size mismatch")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Task: Health check request (weight=1)
        """
        with self.client.get(
            "/health",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def model_info(self):
        """
        Task: Model info request (weight=1)
        """
        with self.client.get(
            "/model/info",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "model_version" in result:
                    response.success()
                else:
                    response.failure("Missing model info")
            else:
                response.failure(f"Got status code {response.status_code}")


class StressTestUser(HttpUser):
    """
    High-load stress testing scenario
    """
    wait_time = between(0.1, 0.5)  # Aggressive load
    
    @task
    def rapid_predictions(self):
        """
        Rapid-fire prediction requests to test system limits
        """
        features = [[random.uniform(0, 10) for _ in range(8)]]
        payload = {"features": features}
        
        self.client.post("/predict", json=payload)


def generate_load_test_report(stats: Dict[str, Any]) -> None:
    """
    Generate load test report summary
    
    Args:
        stats: Load test statistics from Locust
    """
    report = {
        "total_requests": stats.get("num_requests", 0),
        "failure_rate": stats.get("num_failures", 0) / max(stats.get("num_requests", 1), 1),
        "average_response_time": stats.get("avg_response_time", 0),
        "p95_response_time": stats.get("response_time_percentile_95", 0),
        "p99_response_time": stats.get("response_time_percentile_99", 0),
        "requests_per_second": stats.get("requests_per_second", 0)
    }
    
    logger.info(f"Load test report: {json.dumps(report, indent=2)}")
    
    # Check SLA compliance
    if report["p99_response_time"] > 100:  # 100ms threshold
        logger.warning(f"⚠️ P99 latency ({report['p99_response_time']}ms) exceeds SLA (100ms)")
    
    if report["failure_rate"] > 0.01:  # 1% threshold
        logger.warning(f"⚠️ Failure rate ({report['failure_rate']:.2%}) exceeds SLA (1%)")
    
    return report
