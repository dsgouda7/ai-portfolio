"""FaceAI - Interactive Model Training and Experimentation

This script demonstrates:
1. HOG feature extraction with PCA dimensionality reduction
2. Plug-and-play model comparison (Logistic, SVM, RandomForest)
3. Hyperparameter tuning with cross-validation
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
    LogisticRegressor,
    SVMClassifier,
    RandomForestClassifier,
    ExperimentRunner,
    ModelConfig,
)
from src.evaluate import evaluate_model

console = Console()


def main():
    """Run complete ML pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]FaceAI[/bold cyan]\n"
        "Interactive Face Classification Training",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load Data
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    X_train, X_test, y_train, y_test = load_and_split()
    console.print(f"  ✓ Train: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} pixels")
    console.print(f"  ✓ Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]:,} pixels")
    console.print(f"  ✓ Classes: {len(set(y_train))} unique faces")
    
    # ============================================
    # STEP 2: Feature Engineering
    # ============================================
    console.print("\n[bold cyan]🔧 FEATURE ENGINEERING[/bold cyan]")
    console.print("→ Applying HOG feature extraction + PCA...")
    
    fe = FeatureEngineer(
        hog_orientations=9,
        hog_pixels_per_cell=(8, 8),
        hog_cells_per_block=(2, 2),
        pca_components=100,
        scale_features=True
    )
    
    X_train_transformed = fe.fit_transform(X_train)
    X_test_transformed = fe.transform(X_test)
    
    console.print(f"  ✓ Features: {X_train.shape[1]:,} → {X_train_transformed.shape[1]} (after HOG + PCA)")
    
    # ============================================
    # STEP 3: Model Training (Plug-and-Play Comparison)
    # ============================================
    console.print("\n[bold cyan]🤖 MODEL TRAINING[/bold cyan]")
    console.print("Comparing 9 models with different hyperparameters...")
    
    runner = ExperimentRunner()
    
    # Logistic Regression with different regularization
    runner.register("Logistic (C=0.01)", LogisticRegressor(C=0.01))
    runner.register("Logistic (C=0.1)", LogisticRegressor(C=0.1))
    runner.register("Logistic (C=1.0)", LogisticRegressor(C=1.0))
    runner.register("Logistic (C=10.0)", LogisticRegressor(C=10.0))
    
    # SVM with different kernels
    runner.register("SVM (linear)", SVMClassifier(C=1.0, kernel='linear'))
    runner.register("SVM (rbf)", SVMClassifier(C=1.0, kernel='rbf'))
    
    # Random Forest with different configurations
    runner.register("RF (n=50, d=5)", RandomForestClassifier(n_estimators=50, max_depth=5))
    runner.register("RF (n=100, d=10)", RandomForestClassifier(n_estimators=100, max_depth=10))
    runner.register("RF (n=200, d=15)", RandomForestClassifier(n_estimators=200, max_depth=15))
    
    # Run all experiments (prints after each model)
    runner.run_experiment(X_train_transformed, y_train, ModelConfig(cv_folds=5))
    
    # Print leaderboard
    runner.print_leaderboard()
    
    # ============================================
    # STEP 4: Final Evaluation on Test Set
    # ============================================
    console.print("\n[bold cyan]🎯 TEST SET EVALUATION[/bold cyan]")
    
    # TODO: Get best model from runner and evaluate on test set
    console.print("\n[yellow]TODO: Implement test set evaluation (see main.py)[/yellow]")
    
    # ============================================
    # STEP 5: Save Best Model
    # ============================================
    console.print("\n[bold cyan]💾 SAVING MODEL[/bold cyan]")
    
    # TODO: Save best model to disk using joblib
    console.print("\n[yellow]TODO: Implement model saving (see main.py)[/yellow]")
    
    # ============================================
    # Summary
    # ============================================
    console.print("\n[bold green]✅ TRAINING COMPLETE![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Try different hyperparameters in the model registration")
    console.print("  2. Experiment with PCA components (50, 100, 150)")
    console.print("  3. Compare HOG orientations (6, 9, 12)")
    console.print("  4. Deploy with Docker: make docker-build && make docker-up")


if __name__ == "__main__":
    main()
