"""EnsembleAI - Interactive Ensemble Method Training and Comparison

This script demonstrates:
1. Feature engineering with immediate feedback
2. Plug-and-play ensemble comparison (Bagging, Boosting, Stacking)
3. Feature importance analysis across methods
4. Beautiful console output with leaderboards

Usage:
    python main.py

Expected runtime: 3-5 minutes (depends on ensemble configuration)
Expected output: Console shows progress, leaderboard, feature importance comparison
"""

from rich.console import Console
from rich.panel import Panel

from src.data import load_and_split
from src.features import FeatureEngineer, FeatureImportanceAnalyzer
from src.models import (
    BaggingEnsemble,
    BoostingEnsemble,
    StackingEnsemble,
    ExperimentRunner,
    ModelConfig,
)
from src.evaluate import evaluate_classification

console = Console()


def main():
    """Run complete ensemble ML pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]EnsembleAI[/bold cyan]\n"
        "Interactive Ensemble Method Training & Comparison",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load Data
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    X_train, X_test, y_train, y_test = load_and_split()
    console.print(f"  ✓ Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    console.print(f"  ✓ Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    console.print(f"  ✓ Classes: {y_train.nunique()} ({y_train.value_counts().to_dict()})")
    
    # ============================================
    # STEP 2: Feature Engineering
    # ============================================
    console.print("\n[bold cyan]🔧 FEATURE ENGINEERING[/bold cyan]")
    console.print("→ Applying feature selection (mutual information) + scaling...")
    
    fe = FeatureEngineer(
        scale_features=True,
        feature_selection=True,
        top_k_features=15  # Select top 15 most informative features
    )
    
    X_train_transformed = fe.fit_transform(X_train, y_train)
    X_test_transformed = fe.transform(X_test)
    
    # ============================================
    # STEP 3: Ensemble Training (Plug-and-Play Comparison)
    # ============================================
    console.print("\n[bold cyan]🎯 ENSEMBLE TRAINING[/bold cyan]")
    console.print("Comparing 6 ensemble configurations across 3 methods...")
    
    runner = ExperimentRunner()
    
    # TODO: Register different Bagging configurations
    # Experiment: More estimators = better? What about max_samples?
    runner.register("Bagging-30", BaggingEnsemble(n_estimators=30, max_samples=0.8))
    runner.register("Bagging-50", BaggingEnsemble(n_estimators=50, max_samples=1.0))
    
    # TODO: Register different Boosting configurations
    # Experiment: Learning rate vs n_estimators tradeoff
    runner.register("GradBoost-50-0.1", BoostingEnsemble(n_estimators=50, learning_rate=0.1, max_depth=3))
    runner.register("GradBoost-100-0.05", BoostingEnsemble(n_estimators=100, learning_rate=0.05, max_depth=3))
    
    # TODO: Register different Stacking configurations
    # Experiment: Which meta-learner works best?
    runner.register("Stacking-Logistic", StackingEnsemble(meta_learner_type="logistic"))
    runner.register("Stacking-GBM", StackingEnsemble(meta_learner_type="gbm"))
    
    # Run all experiments (prints after each ensemble trains)
    runner.run_experiment(X_train_transformed, y_train, ModelConfig(cv_folds=5))
    
    # Print leaderboard
    runner.print_leaderboard()
    
    # ============================================
    # STEP 4: Feature Importance Analysis
    # ============================================
    console.print("\n[bold cyan]📈 FEATURE IMPORTANCE ANALYSIS[/bold cyan]")
    console.print("Comparing feature rankings across ensemble methods...")
    
    analyzer = FeatureImportanceAnalyzer()
    
    # TODO: Add models to analyzer (skip if you want to implement this later)
    # Extract importance from trained ensembles
    for name, ensemble in runner.ensembles.items():
        analyzer.add_model(name, ensemble, X_train_transformed.columns.tolist())
    
    # Print comparison table
    analyzer.print_comparison(top_k=10)
    
    # Plot importance (optional - saves to file)
    # analyzer.plot_importance(top_k=10, save_path="models/feature_importance.png")
    
    # ============================================
    # STEP 5: Final Evaluation on Test Set
    # ============================================
    console.print("\n[bold cyan]✅ FINAL TEST EVALUATION[/bold cyan]")
    console.print("Evaluating best ensemble on held-out test set...")
    
    best_ensemble = runner.get_best_model()
    y_pred = best_ensemble.predict(X_test_transformed)
    
    test_metrics = evaluate_classification(y_test, y_pred)
    
    console.print(f"\n🏆 Best Ensemble: [bold green]{best_ensemble.name}[/bold green]")
    console.print(f"  • Test Accuracy: {test_metrics['accuracy']:.4f}")
    console.print(f"  • Precision: {test_metrics['precision']:.4f}")
    console.print(f"  • Recall: {test_metrics['recall']:.4f}")
    console.print(f"  • F1 Score: {test_metrics['f1']:.4f}")
    
    # Save best model
    console.print("\n[bold cyan]💾 SAVING BEST MODEL[/bold cyan]")
    best_ensemble.save("models/best_ensemble.pkl")
    console.print(f"  ✓ Saved to models/best_ensemble.pkl")
    
    # ============================================
    # STEP 6: Insights & Recommendations
    # ============================================
    console.print("\n[bold cyan]💡 ENSEMBLE INSIGHTS[/bold cyan]")
    console.print("Key observations from your experiments:")
    
    # Check if Bagging has OOB score
    bagging_results = [r for r in runner.results if "Bagging" in r["model"]]
    if bagging_results:
        avg_oob = np.mean([r.get("oob_score", 0) for r in bagging_results])
        console.print(f"  • Bagging OOB score ≈ {avg_oob:.4f} (free validation!)")
    
    # Check boosting feature importance
    boosting_results = [r for r in runner.results if "GradBoost" in r["model"]]
    if boosting_results:
        console.print(f"  • Boosting leverages {len(X_train_transformed.columns)} features sequentially")
    
    # Check stacking diversity
    stacking_results = [r for r in runner.results if "Stacking" in r["model"]]
    if stacking_results:
        console.print(f"  • Stacking combines diverse base learners for robustness")
    
    # Performance comparison
    cv_scores = [r.get("cv_accuracy", 0) for r in runner.results]
    best_cv = max(cv_scores)
    worst_cv = min(cv_scores)
    improvement = ((best_cv - worst_cv) / worst_cv) * 100
    console.print(f"  • Best vs Worst: {improvement:.1f}% improvement")
    
    console.print("\n" + "="*60)
    console.print("[bold green]✅ ENSEMBLE TRAINING COMPLETE![/bold green]")
    console.print("="*60)
    console.print("\n[bold yellow]Next Steps:[/bold yellow]")
    console.print("  1. Experiment with different ensemble configurations")
    console.print("  2. Try different base learners (e.g., SVM, KNN)")
    console.print("  3. Tune hyperparameters via grid search")
    console.print("  4. Deploy best model via Docker (make docker-up)")


if __name__ == "__main__":
    import numpy as np
    main()
