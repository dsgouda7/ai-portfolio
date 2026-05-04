"""MLOps Infrastructure - Interactive Infrastructure Stack Demonstration

This script demonstrates:
1. Infrastructure component setup with immediate feedback
2. Model serving with FastAPI and health checks
3. Prometheus metrics collection (latency, throughput, errors)
4. MLflow experiment tracking for ML workflows
5. Load testing and performance analysis

Usage:
    python main.py

Expected runtime: 3-5 minutes (after TODOs completed)
Expected output: Console shows infrastructure setup, load test results, and performance metrics
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.models import (
    ModelServer,
    MetricsCollector,
    ExperimentTracker,
    ExperimentRunner,
    InfraConfig,
    create_sample_model,
)
from src.features import (
    RequestValidator,
    MetricAggregator,
    LoadTestGenerator,
    ValidationConfig,
    analyze_latency_distribution,
    MetricSnapshot,
)

console = Console()


def main():
    """Run complete ML infrastructure demonstration with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]MLOps Infrastructure Platform[/bold cyan]\n"
        "Production ML Deployment Stack",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Configuration
    # ============================================
    console.print("\n[bold cyan]⚙️  CONFIGURATION[/bold cyan]")
    
    config = InfraConfig(
        model_path="models/sample_model.pkl",
        mlflow_uri="http://localhost:5000",  # Local MLflow server
        experiment_name="ml_infrastructure_demo",
        metrics_port=8000,
        api_port=5001,
        verbose=True
    )
    
    console.print(f"  Model path: {config.model_path}")
    console.print(f"  MLflow URI: {config.mlflow_uri}")
    console.print(f"  Experiment: {config.experiment_name}")
    console.print(f"  Metrics port: {config.metrics_port}")
    console.print(f"  API port: {config.api_port}")
    
    # ============================================
    # STEP 2: Create Sample Model (if needed)
    # ============================================
    if not Path(config.model_path).exists():
        create_sample_model(config.model_path)
    else:
        console.print(f"\n  ✓ Using existing model: {config.model_path}")
    
    # ============================================
    # STEP 3: Initialize Infrastructure Components
    # ============================================
    console.print("\n[bold cyan]🏗️  INITIALIZING INFRASTRUCTURE[/bold cyan]")
    
    runner = ExperimentRunner(config)
    
    # Add infrastructure components
    model_server = ModelServer()
    metrics_collector = MetricsCollector()
    experiment_tracker = ExperimentTracker()
    
    runner.add_component(model_server)
    runner.add_component(metrics_collector)
    runner.add_component(experiment_tracker)
    
    console.print(f"  ✓ Registered {len(runner.components)} components")
    
    # ============================================
    # STEP 4: Set Up All Components
    # ============================================
    try:
        setup_df = runner.setup_all()
        console.print(f"\n  ✓ All components ready!")
    except NotImplementedError as e:
        console.print(f"\n[yellow]⚠️  Setup incomplete: {e}[/yellow]")
        console.print("\n[bold yellow]💡 NEXT STEPS:[/bold yellow]")
        console.print("  1. Implement ModelServer.setup() in src/models.py")
        console.print("  2. Implement MetricsCollector.setup() in src/models.py")
        console.print("  3. Implement ExperimentTracker.setup() in src/models.py")
        console.print("  4. Run this script again to continue")
        return
    
    # ============================================
    # STEP 5: Initialize Request Validation
    # ============================================
    console.print("\n[bold cyan]🔒 REQUEST VALIDATION[/bold cyan]")
    
    validation_config = ValidationConfig(
        max_batch_size=1000,
        max_feature_dim=1000,
        max_value_magnitude=1e6,
        timeout_ms=5000
    )
    
    validator = RequestValidator(validation_config)
    console.print(f"  ✓ Request validator initialized")
    console.print(f"    Max batch size: {validation_config.max_batch_size}")
    console.print(f"    Max feature dim: {validation_config.max_feature_dim}")
    
    # ============================================
    # STEP 6: Generate Load Test Data
    # ============================================
    console.print("\n[bold cyan]📦 GENERATING LOAD TEST DATA[/bold cyan]")
    
    load_gen = LoadTestGenerator(feature_dim=4, random_state=42)
    
    try:
        # Generate test batches with different distributions
        normal_batch = load_gen.generate_batch(100, distribution="normal")
        uniform_batch = load_gen.generate_batch(100, distribution="uniform")
        
        console.print(f"  ✓ Generated normal distribution batch: {len(normal_batch)} samples")
        console.print(f"  ✓ Generated uniform distribution batch: {len(uniform_batch)} samples")
        
        # Validate test data
        is_valid, error = validator.validate_prediction_request(normal_batch)
        if is_valid:
            console.print(f"  ✓ Test data validation: PASSED")
        else:
            console.print(f"  ✗ Test data validation: FAILED - {error}")
    
    except NotImplementedError:
        console.print("[yellow]  ⚠️  Load test generation not yet implemented[/yellow]")
        console.print("     Implement LoadTestGenerator.generate_batch() in src/features.py")
    
    # ============================================
    # STEP 7: Run Load Test
    # ============================================
    try:
        console.print("\n[bold cyan]🚀 LOAD TESTING[/bold cyan]")
        
        num_requests = 100
        load_test_df = runner.run_load_test(num_requests=num_requests)
        
        # Analyze latency distribution
        if "latency_ms" in load_test_df.columns:
            latencies = load_test_df["latency_ms"].tolist()
            analysis = analyze_latency_distribution(latencies, target_p99_ms=100.0)
            
            console.print(f"\n  ✓ Load test analysis complete")
    
    except NotImplementedError as e:
        console.print(f"[yellow]  ⚠️  Load testing not yet implemented: {e}[/yellow]")
        console.print("\n[bold yellow]💡 NEXT STEPS:[/bold yellow]")
        console.print("  1. Implement ModelServer.execute() in src/models.py")
        console.print("  2. Implement MetricsCollector.execute() in src/models.py")
        console.print("  3. Implement ExperimentRunner.run_load_test() in src/models.py")
    
    # ============================================
    # STEP 8: Log Experiment to MLflow
    # ============================================
    console.print("\n[bold cyan]📝 EXPERIMENT TRACKING[/bold cyan]")
    
    try:
        # Prepare experiment data
        experiment_data = {
            "params": {
                "model_type": "RandomForestRegressor",
                "n_estimators": 10,
                "random_state": 42,
                "infrastructure": "FastAPI + Prometheus + MLflow"
            },
            "metrics": {
                "avg_latency_ms": load_test_df["latency_ms"].mean() if "latency_ms" in load_test_df else 0,
                "p95_latency_ms": load_test_df["latency_ms"].quantile(0.95) if "latency_ms" in load_test_df else 0,
                "p99_latency_ms": load_test_df["latency_ms"].quantile(0.99) if "latency_ms" in load_test_df else 0,
                "error_rate": load_test_df["error"].sum() / len(load_test_df) if "error" in load_test_df else 0,
                "throughput_rps": num_requests / (load_test_df["latency_ms"].sum() / 1000) if "latency_ms" in load_test_df else 0
            },
            "artifacts": []
        }
        
        # Log to MLflow
        result = experiment_tracker.execute(experiment_data)
        console.print(f"  ✓ Experiment logged to MLflow")
        console.print(f"    Run ID: {result.get('run_id', 'N/A')[:8]}...")
    
    except NotImplementedError:
        console.print("[yellow]  ⚠️  MLflow tracking not yet implemented[/yellow]")
        console.print("     Implement ExperimentTracker.execute() in src/models.py")
    except Exception as e:
        console.print(f"[yellow]  ⚠️  MLflow logging failed: {e}[/yellow]")
        console.print("     (This is expected if MLflow server is not running)")
    
    # ============================================
    # STEP 9: Display Infrastructure Leaderboard
    # ============================================
    runner.display_leaderboard()
    
    # ============================================
    # STEP 10: Validation Statistics
    # ============================================
    console.print("\n[bold cyan]🔍 VALIDATION STATISTICS[/bold cyan]")
    
    try:
        val_stats = validator.get_validation_stats()
        console.print(f"  Total validations: {val_stats['total_validations']}")
        console.print(f"  Rejections: {val_stats['rejections']}")
        console.print(f"  Rejection rate: {val_stats['rejection_rate_percent']:.2f}%")
        console.print(f"  Acceptance rate: {val_stats['acceptance_rate_percent']:.2f}%")
    except Exception:
        console.print("[yellow]  ⚠️  Validation stats not available yet[/yellow]")
    
    # ============================================
    # COMPLETION SUMMARY
    # ============================================
    console.print("\n[bold green]✅ INFRASTRUCTURE DEMO COMPLETE[/bold green]")
    console.print("\n[bold cyan]📚 What You Built:[/bold cyan]")
    console.print("  ✓ Model Serving: FastAPI with health checks and validation")
    console.print("  ✓ Metrics Collection: Prometheus for latency/throughput tracking")
    console.print("  ✓ Experiment Tracking: MLflow for ML workflow management")
    console.print("  ✓ Load Testing: Performance analysis under production load")
    console.print("  ✓ Request Validation: Input sanitization and error handling")
    
    console.print("\n[bold cyan]🎯 Production MLOps Concepts Learned:[/bold cyan]")
    console.print("  • Model serving patterns (REST API, versioning)")
    console.print("  • Observability (metrics, logging, health checks)")
    console.print("  • Experiment tracking (params, metrics, artifacts)")
    console.print("  • Performance analysis (latency percentiles, SLOs)")
    console.print("  • Request validation (input sanitization, rate limiting)")
    
    console.print("\n[bold cyan]🚀 Next Steps:[/bold cyan]")
    console.print("  1. Deploy to Kubernetes (see k8s/ directory)")
    console.print("  2. Set up CI/CD pipelines (see ci/ directory)")
    console.print("  3. Configure Grafana dashboards for monitoring")
    console.print("  4. Implement data drift detection with Evidently")
    console.print("  5. Add feature store integration with Feast")
    
    console.print("\n[bold magenta]════════════════════════════════════════════════[/bold magenta]")


if __name__ == "__main__":
    main()
