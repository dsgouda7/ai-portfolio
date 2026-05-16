"""MLOps Infrastructure Components with Experiment Framework

This module provides:
- Abstract InfrastructureComponent interface for plug-and-play architecture
- Concrete implementations: ModelServer, MetricsCollector, ExperimentTracker (with TODOs)
- ExperimentRunner for infrastructure orchestration
- Immediate feedback with rich console output

Learning objectives:
1. Implement production model serving with FastAPI
2. Set up Prometheus metrics collection (latency, throughput, errors)
3. Configure MLflow experiment tracking for ML workflows
4. Orchestrate complete infrastructure stack
5. Measure and optimize API performance (p95/p99 latency)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("ml_infra")
console = Console()


@dataclass
class InfraConfig:
    """Configuration for infrastructure components."""
    model_path: str = "models/model.pkl"
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "ml_infrastructure"
    metrics_port: int = 8000
    api_port: int = 5001
    verbose: bool = True


class InfrastructureComponent(ABC):
    """Abstract base class for all infrastructure components.
    
    Provides common interface for plug-and-play MLOps experimentation.
    Subclasses implement setup() and execute() methods with immediate feedback.
    """
    
    def __init__(self, name: str):
        """Initialize infrastructure component with name for display."""
        self.name = name
        self.status = {}
        self.is_ready = False
    
    @abstractmethod
    def setup(self, config: InfraConfig) -> Dict[str, Any]:
        """Set up infrastructure component and return status with immediate console feedback.
        
        Args:
            config: Infrastructure configuration
        
        Returns:
            Dictionary with status: {"ready": bool, "latency_ms": float, "details": Any}
        """
        pass
    
    @abstractmethod
    def execute(self, data: Any) -> Dict[str, Any]:
        """Execute primary component function (serve, collect, track) and return results.
        
        Args:
            data: Input data (varies by component)
        
        Returns:
            Dictionary with execution results
        """
        pass
    
    def health_check(self) -> bool:
        """Check if component is healthy and ready."""
        return self.is_ready


# ============================================
# Pydantic Models for API Validation
# ============================================

class PredictionRequest(BaseModel):
    """Request schema for model predictions."""
    features: List[List[float]] = Field(..., description="List of feature vectors")
    model_version: Optional[str] = Field("latest", description="Model version to use")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [[1.5, 2.3, 4.1, 0.8]],
                "model_version": "latest"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: List[float] = Field(..., description="Model predictions")
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")


# ============================================
# MODEL SERVING COMPONENT
# ============================================

class ModelServer(InfrastructureComponent):
    """Production model serving with FastAPI and health checks.
    
    Provides REST API endpoints for:
    - /predict: Generate predictions with latency tracking
    - /health: Health check endpoint
    - /metrics: Prometheus metrics endpoint
    """
    
    def __init__(self):
        """Initialize model server."""
        super().__init__("Model Server")
        self.model = None
        self.model_version = None
        self.app = FastAPI(title="ML Model Serving API", version="1.0.0")
        
        # Prometheus metrics (defined but not yet implemented)
        self.prediction_counter = Counter('model_predictions_total', 'Total predictions')
        self.prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency')
        self.error_counter = Counter('model_errors_total', 'Total prediction errors')
    
    def setup(self, config: InfraConfig) -> Dict[str, Any]:
        """TODO: Implement model server setup (load model, configure FastAPI routes, test prediction)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement model server setup")
    
    def execute(self, data: PredictionRequest) -> PredictionResponse:
        """TODO: Implement model prediction endpoint (validate ready, predict, record metrics)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement prediction execution")


# ============================================
# METRICS COLLECTION COMPONENT
# ============================================

class MetricsCollector(InfrastructureComponent):
    """Prometheus metrics collection for production monitoring.
    
    Tracks critical production metrics:
    - API latency (p50, p95, p99 percentiles)
    - Request throughput (requests/second)
    - Error rate (errors/total requests)
    - System resources (CPU, memory)
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        super().__init__("Metrics Collector")
        
        # Prometheus metrics (defined but not yet implemented)
        self.request_latency = Histogram(
            'api_request_latency_seconds',
            'API request latency in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        self.request_throughput = Counter('api_requests_total', 'Total API requests')
        self.error_rate = Counter('api_errors_total', 'Total API errors')
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        
        self.latencies = []  # Store recent latencies for stats
    
    def setup(self, config: InfraConfig) -> Dict[str, Any]:
        """TODO: Implement metrics collector setup (initialize storage, configure Prometheus endpoint)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement metrics collector setup")
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: Implement metrics collection and aggregation (record latencies, calculate percentiles)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement metrics collection")


# ============================================
# EXPERIMENT TRACKING COMPONENT
# ============================================

class ExperimentTracker(InfrastructureComponent):
    """MLflow experiment tracking for ML workflows.
    
    Tracks:
    - Model parameters (hyperparameters, configuration)
    - Training metrics (accuracy, loss, validation scores)
    - Artifacts (models, plots, data files)
    - Model versioning and lifecycle
    """
    
    def __init__(self):
        """Initialize experiment tracker."""
        super().__init__("Experiment Tracker")
        self.client = None
        self.experiment_id = None
        self.run_id = None
    
    def setup(self, config: InfraConfig) -> Dict[str, Any]:
        """TODO: Implement MLflow experiment tracker setup (configure URI, create/get experiment)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment tracker setup")
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: Implement experiment logging (log params, metrics, artifacts to MLflow)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment logging")


# ============================================
# EXPERIMENT ORCHESTRATION
# ============================================

class ExperimentRunner:
    """Orchestrate infrastructure components and run experiments.
    
    Coordinates:
    1. Component setup (server, metrics, tracking)
    2. Load testing and performance measurement
    3. Result aggregation and reporting
    """
    
    def __init__(self, config: InfraConfig):
        """Initialize experiment runner."""
        self.config = config
        self.components: List[InfrastructureComponent] = []
        self.results = []
    
    def add_component(self, component: InfrastructureComponent) -> None:
        """Add infrastructure component to experiment."""
        self.components.append(component)
        logger.info(f"Added component: {component.name}")
    
    def setup_all(self) -> pd.DataFrame:
        """TODO: Set up all infrastructure components (orchestrate setup, create summary)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement infrastructure setup")
    
    def run_load_test(self, num_requests: int = 100) -> pd.DataFrame:
        """TODO: Run load test against model server (generate requests, measure latency, calculate stats)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement load testing")
    
    def display_leaderboard(self) -> None:
        """Display infrastructure performance leaderboard."""
        if not self.results:
            console.print("[yellow]No results to display yet[/yellow]")
            return
        
        console.print("\n[bold cyan]📊 INFRASTRUCTURE LEADERBOARD[/bold cyan]")
        
        table = Table(title="Component Performance", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Component", style="yellow")
        table.add_column("Setup Time", justify="right")
        table.add_column("Latency (Avg)", justify="right")
        table.add_column("Status", style="green")
        
        # Sort by setup time
        sorted_results = sorted(self.results, key=lambda x: x.get("latency_ms", float('inf')))
        
        for i, result in enumerate(sorted_results, 1):
            table.add_row(
                f"#{i}",
                result.get("name", "Unknown"),
                f"{result.get('latency_ms', 0):.1f}ms",
                f"{result.get('avg_latency_ms', 0):.1f}ms",
                "✓ Ready" if result.get("ready", False) else "✗ Failed"
            )
        
        console.print(table)


def create_sample_model(output_path: str = "models/model.pkl") -> None:
    """Create a simple trained model for testing infrastructure.
    
    Args:
        output_path: Path to save the model
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    
    console.print("\n[bold cyan]🔨 Creating sample model for infrastructure testing[/bold cyan]")
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
    
    # Train simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    
    console.print(f"  ✓ Sample model saved to: {output_path}")
