"""API request validation and metric aggregation for ML infrastructure

This module provides:
- Request validation with Pydantic schemas
- Input data preprocessing and validation
- Metric aggregation and statistical analysis
- Load testing utilities

Learning objectives:
1. Validate API requests with type checking and constraints
2. Aggregate performance metrics (latency, throughput, errors)
3. Generate load testing data for stress testing
4. Analyze system performance under load
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from rich.console import Console

logger = logging.getLogger("ml_infra")
console = Console()


# ============================================
# PYDANTIC VALIDATION SCHEMAS
# ============================================

class FeatureVector(BaseModel):
    """Individual feature vector with validation."""
    values: List[float] = Field(..., min_items=1, max_items=1000)
    feature_names: Optional[List[str]] = None
    
    @validator('values')
    def validate_values(cls, v):
        """TODO: Implement feature vector validation (check NaN/Inf, validate value range)"""
        # TODO: Your implementation here
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request with validation."""
    features: List[FeatureVector] = Field(..., min_items=1, max_items=1000)
    model_version: str = Field(default="latest", regex="^(latest|v\\d+)$")
    timeout_ms: int = Field(default=5000, ge=100, le=60000)
    
    @validator('features')
    def validate_batch_size(cls, v):
        """TODO: Implement batch size validation (check non-empty, uniform length, size limits)"""
        # TODO: Your implementation here
        return v


class MetricSnapshot(BaseModel):
    """Point-in-time metric snapshot."""
    timestamp: float = Field(..., description="Unix timestamp")
    latency_ms: float = Field(..., ge=0, description="Request latency in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    error: bool = Field(default=False, description="Whether request failed")
    error_message: Optional[str] = None


# ============================================
# REQUEST VALIDATION
# ============================================

@dataclass
class ValidationConfig:
    """Configuration for request validation."""
    max_batch_size: int = 1000
    max_feature_dim: int = 1000
    max_value_magnitude: float = 1e6
    timeout_ms: int = 5000


class RequestValidator:
    """Validate API requests with detailed error messages."""
    
    def __init__(self, config: ValidationConfig):
        """Initialize request validator."""
        self.config = config
        self.validation_count = 0
        self.rejection_count = 0
    
    def validate_prediction_request(
        self,
        features: List[List[float]],
        model_version: str = "latest"
    ) -> Tuple[bool, Optional[str]]:
        """TODO: Implement prediction request validation (batch size, dimensions, values, version format)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement request validation")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        rejection_rate = (
            self.rejection_count / self.validation_count * 100
            if self.validation_count > 0
            else 0
        )
        
        return {
            "total_validations": self.validation_count,
            "rejections": self.rejection_count,
            "rejection_rate_percent": rejection_rate,
            "acceptance_rate_percent": 100 - rejection_rate
        }


# ============================================
# METRIC AGGREGATION
# ============================================

class MetricAggregator:
    """Aggregate and analyze performance metrics."""
    
    def __init__(self):
        """Initialize metric aggregator."""
        self.snapshots: List[MetricSnapshot] = []
        self.window_size = 100  # Keep last 100 metrics
    
    def record(self, snapshot: MetricSnapshot) -> None:
        """Record a metric snapshot."""
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots
        if len(self.snapshots) > self.window_size:
            self.snapshots = self.snapshots[-self.window_size:]
    
    def aggregate(self, time_window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """TODO: Implement metric aggregation (latencies, percentiles, error rate, throughput)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement metric aggregation")
    
    def print_summary(self) -> None:
        """Print metric summary to console."""
        stats = self.aggregate()
        
        if stats["count"] == 0:
            console.print("[yellow]No metrics recorded yet[/yellow]")
            return
        
        console.print("\n[bold cyan]📊 METRIC SUMMARY[/bold cyan]")
        console.print(f"  Total requests: {stats['count']}")
        console.print(f"  Avg latency: {stats['avg_latency_ms']:.2f}ms")
        console.print(f"  Median latency: {stats['median_latency_ms']:.2f}ms")
        console.print(f"  p95 latency: {stats['p95_latency_ms']:.2f}ms")
        console.print(f"  p99 latency: {stats['p99_latency_ms']:.2f}ms")
        console.print(f"  Error rate: {stats['error_rate_percent']:.2f}%")
        console.print(f"  Throughput: {stats['throughput_rps']:.1f} req/s")


# ============================================
# LOAD TESTING UTILITIES
# ============================================

class LoadTestGenerator:
    """Generate synthetic load for testing."""
    
    def __init__(self, feature_dim: int = 10, random_state: int = 42):
        """Initialize load test generator."""
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(random_state)
    
    def generate_batch(
        self,
        batch_size: int,
        distribution: str = "normal"
    ) -> List[List[float]]:
        """TODO: Implement load test data generation (support normal, uniform, adversarial distributions)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement load test generation")
    
    def generate_traffic_pattern(
        self,
        duration_seconds: int,
        requests_per_second: float,
        spike_probability: float = 0.1
    ) -> List[float]:
        """TODO: Implement realistic traffic pattern generation (with jitter and random spikes)"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement traffic pattern generation")


# ============================================
# PERFORMANCE ANALYSIS
# ============================================

def analyze_latency_distribution(
    latencies: List[float],
    target_p99_ms: float = 100.0
) -> Dict[str, Any]:
    """TODO: Implement latency distribution analysis (percentiles, SLO compliance, outlier detection)"""
    # TODO: Your implementation here
    raise NotImplementedError("Implement latency analysis")


def compare_infrastructure_configs(
    results_df: pd.DataFrame,
    group_by: str = "component"
) -> pd.DataFrame:
    """Compare infrastructure configurations by performance.
    
    Args:
        results_df: DataFrame with infrastructure performance results
        group_by: Column to group by (component, version, etc.)
    
    Returns:
        Comparison DataFrame with aggregated metrics
    """
    comparison = results_df.groupby(group_by).agg({
        "latency_ms": ["mean", "std", "min", "max"],
        "throughput_rps": ["mean", "std"],
        "error_rate_percent": ["mean", "max"]
    }).round(2)
    
    console.print(f"\n[bold cyan]📊 INFRASTRUCTURE COMPARISON (grouped by {group_by})[/bold cyan]")
    console.print(comparison.to_string())
    
    return comparison
