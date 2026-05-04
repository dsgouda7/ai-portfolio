"""SmartVal AI - Interactive Model Training and Experimentation

This script demonstrates:
1. Feature engineering with immediate feedback
2. Plug-and-play model comparison (Ridge, Lasso, XGBoost)
3. Hyperparameter tuning with grid search
4. Beautiful console output with leaderboards

Usage:
    python main.py

Expected runtime: 2-3 minutes (9 models)
Expected output: Console shows progress, leaderboard, and saves best model
"""

from rich.console import Console
from rich.panel import Panel

from src.data import load_and_split
from src.features import FeatureEngineer
from src.models import (
    RidgeRegressor,
    LassoRegressor,
    XGBoostRegressor,
    ExperimentRunner,
    ModelConfig,
)

console = Console()


def main():
    """Run complete ML pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]SmartVal AI[/bold cyan]\n"
        "Interactive Regression Model Training",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load Data
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    X_train, X_test, y_train, y_test = load_and_split()
    console.print(f"  ✓ Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    console.print(f"  ✓ Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    console.print(f"  ✓ Target range: ${y_train.min():.0f} - ${y_train.max():.0f}")
    
    # ============================================
    # STEP 2: Feature Engineering
    # ============================================
    console.print("\n[bold cyan]🔧 FEATURE ENGINEERING[/bold cyan]")
    console.print("→ Applying polynomial features (degree=2) + VIF filtering...")
    
    fe = FeatureEngineer(
        polynomial_degree=2,
        scale_features=True,
        vif_threshold=10.0
    )
    
    X_train_transformed = fe.fit_transform(X_train)
    X_test_transformed = fe.transform(X_test)
    
    console.print(f"  ✓ Features: {X_train.shape[1]} → {X_train_transformed.shape[1]} (after poly + VIF)")
    
    # ============================================
    # STEP 3: Model Training (Plug-and-Play Comparison)
    # ============================================
    console.print("\n[bold cyan]🤖 MODEL TRAINING[/bold cyan]")
    console.print("Comparing 9 models with different hyperparameters...")
    
    runner = ExperimentRunner()
    
    # TODO: Register different Ridge alphas (try α=0.01, 0.1, 1.0, 10.0)
    runner.register("Ridge (α=0.01)", RidgeRegressor(alpha=0.01))
    runner.register("Ridge (α=0.1)", RidgeRegressor(alpha=0.1))
    runner.register("Ridge (α=1.0)", RidgeRegressor(alpha=1.0))
    runner.register("Ridge (α=10.0)", RidgeRegressor(alpha=10.0))
    
    # TODO: Try Lasso with different alphas
    runner.register("Lasso (α=0.1)", LassoRegressor(alpha=0.1))
    runner.register("Lasso (α=1.0)", LassoRegressor(alpha=1.0))
    
    # TODO: Try XGBoost with different depths and estimators
    runner.register("XGBoost (d=3, n=100)", XGBoostRegressor(max_depth=3, n_estimators=100))
    runner.register("XGBoost (d=6, n=100)", XGBoostRegressor(max_depth=6, n_estimators=100))
    runner.register("XGBoost (d=6, n=200)", XGBoostRegressor(max_depth=6, n_estimators=200))
    
    # Run all experiments (prints after each model)
    runner.run_experiment(X_train_transformed, y_train, ModelConfig(cv_folds=5))
    
    # Print leaderboard
    runner.print_leaderboard()
    
    # ============================================
    # STEP 4: Final Evaluation on Test Set
    # ============================================
    console.print("\n[bold cyan]✅ FINAL EVALUATION[/bold cyan]")
    
    best_model = runner.get_best_model()
    console.print(f"→ Evaluating {best_model.name} on test set...")
    
    # TODO: Evaluate best model on test set and print metrics
    
    # ============================================
    # STEP 5: Save Best Model
    # ============================================
    console.print("\n[bold cyan]💾 SAVING MODEL[/bold cyan]")
    
    # TODO: Save best model and feature engineer to disk
    
    console.print("\n[bold green]✨ Training complete![/bold green]\n")
    
    # ============================================
    # NEXT STEPS (Optional)
    # ============================================
    console.print("[dim]Next steps:")
    console.print("  1. Try different feature engineering (polynomial degree 3?)")
    console.print("  2. Grid search over more hyperparameters")
    console.print("  3. Deploy API: make docker-build && make docker-up")
    console.print("  4. View metrics: http://localhost:9090 (Prometheus)[/dim]")


if __name__ == "__main__":
    main()
