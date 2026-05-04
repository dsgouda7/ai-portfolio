"""FraudShield - Interactive Anomaly Detection Training and Experimentation

This script demonstrates:
1. Feature engineering with immediate feedback (standardization + PCA)
2. Plug-and-play detector comparison (IsolationForest, LOF, Autoencoder)
3. Anomaly visualization and scoring
4. Beautiful console output with leaderboards

Usage:
    python main.py

Expected runtime: 3-5 minutes (3 detectors)
Expected output: Console shows progress, leaderboard, and visualizes detected anomalies
"""

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_and_split
from src.features import FeatureEngineer
from src.models import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    AutoencoderDetector,
    ExperimentRunner,
    ModelConfig,
)

console = Console()


def main():
    """Run complete anomaly detection pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]FraudShield AI[/bold cyan]\n"
        "Interactive Anomaly Detection Training",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load Data
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    X_train, X_test, y_train, y_test = load_and_split()
    
    # Count anomalies
    n_train_anomalies = y_train.sum()
    n_test_anomalies = y_test.sum()
    train_contamination = n_train_anomalies / len(y_train)
    test_contamination = n_test_anomalies / len(y_test)
    
    console.print(f"  ✓ Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    console.print(f"    - Normal: {len(y_train) - n_train_anomalies:,} | Anomalies: {n_train_anomalies:,} ({train_contamination:.1%})")
    console.print(f"  ✓ Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    console.print(f"    - Normal: {len(y_test) - n_test_anomalies:,} | Anomalies: {n_test_anomalies:,} ({test_contamination:.1%})")
    
    # ============================================
    # STEP 2: Feature Engineering
    # ============================================
    console.print("\n[bold cyan]🔧 FEATURE ENGINEERING[/bold cyan]")
    console.print("Applying standardization (critical for distance-based methods)...")
    
    # TODO: Create feature engineer with scaling and optional PCA
    # fe = FeatureEngineer(
    #     scale_features=True,
    #     n_components_pca=None  # Try 10 or 15 for dimensionality reduction
    # )
    # 
    # X_train_transformed = fe.fit_transform(X_train)
    # X_test_transformed = fe.transform(X_test)
    #
    # Time estimate: 5 minutes
    
    # For now, use placeholder (remove when implementing TODO)
    fe = FeatureEngineer(scale_features=True, n_components_pca=None)
    X_train_transformed = fe.fit_transform(X_train)
    X_test_transformed = fe.transform(X_test)
    
    # ============================================
    # STEP 3: Model Training (Plug-and-Play Comparison)
    # ============================================
    console.print("\n[bold cyan]🔍 ANOMALY DETECTOR TRAINING[/bold cyan]")
    console.print("Comparing 5 detectors with different hyperparameters...")
    
    # TODO: Register different detectors
    # Hint: Try different n_estimators, n_neighbors, encoding_dim
    #
    # runner = ExperimentRunner()
    # 
    # # IsolationForest with different tree counts
    # runner.register("IF-50", IsolationForestDetector(n_estimators=50))
    # runner.register("IF-100", IsolationForestDetector(n_estimators=100))
    # 
    # # LOF with different neighbor counts
    # runner.register("LOF-10", LocalOutlierFactorDetector(n_neighbors=10))
    # runner.register("LOF-20", LocalOutlierFactorDetector(n_neighbors=20))
    # 
    # # Autoencoder (requires TensorFlow)
    # # runner.register("AE-10", AutoencoderDetector(encoding_dim=10, epochs=50))
    # 
    # # Run all experiments (prints after each detector)
    # runner.run_experiment(
    #     X_train_transformed, 
    #     y_train, 
    #     ModelConfig(contamination=train_contamination)
    # )
    # 
    # # Print leaderboard
    # runner.print_leaderboard()
    #
    # Time estimate: 5 minutes
    
    runner = ExperimentRunner()
    runner.register("IF-100", IsolationForestDetector(n_estimators=100))
    runner.register("LOF-20", LocalOutlierFactorDetector(n_neighbors=20))
    
    runner.run_experiment(
        X_train_transformed,
        y_train,
        ModelConfig(contamination=train_contamination)
    )
    
    runner.print_leaderboard()
    
    # ============================================
    # STEP 4: Final Evaluation on Test Set
    # ============================================
    console.print("\n[bold cyan]✅ FINAL EVALUATION[/bold cyan]")
    
    best_detector = runner.get_best_detector()
    console.print(f"→ Evaluating {best_detector.name} on test set...")
    
    # TODO: Predict on test set and compute metrics
    # Steps:
    # 1. Get predictions:
    #    y_pred_test = best_detector.predict(X_test_transformed)
    # 2. Get anomaly scores:
    #    scores_test = best_detector.predict_scores(X_test_transformed)
    # 3. Compute metrics:
    #    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    #    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    #    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    #    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    #    test_roc_auc = roc_auc_score(y_test, scores_test)
    # 4. Print results:
    #    console.print(f"  Test Precision: {test_precision:.3f}", style="green")
    #    console.print(f"  Test Recall:    {test_recall:.3f}", style="green")
    #    console.print(f"  Test F1:        {test_f1:.3f}", style="green")
    #    console.print(f"  Test ROC-AUC:   {test_roc_auc:.3f}", style="green")
    #
    # Time estimate: 10 minutes
    
    # ============================================
    # STEP 5: Visualize Anomaly Detection
    # ============================================
    console.print("\n[bold cyan]📈 VISUALIZING ANOMALIES[/bold cyan]")
    
    # TODO: Create visualization of detected anomalies
    # Steps:
    # 1. Get predictions and scores:
    #    y_pred_test = best_detector.predict(X_test_transformed)
    #    scores_test = best_detector.predict_scores(X_test_transformed)
    # 
    # 2. Create 2x2 subplot figure:
    #    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    #    fig.suptitle(f'Anomaly Detection Results - {best_detector.name}', 
    #                 fontsize=16, fontweight='bold')
    # 
    # 3. Plot 1 - Anomaly Score Distribution:
    #    axes[0, 0].hist(scores_test[y_test == 0], bins=50, alpha=0.6, 
    #                    label='Normal', color='blue')
    #    axes[0, 0].hist(scores_test[y_test == 1], bins=50, alpha=0.6, 
    #                    label='Anomaly', color='red')
    #    axes[0, 0].set_xlabel('Anomaly Score')
    #    axes[0, 0].set_ylabel('Count')
    #    axes[0, 0].set_title('Anomaly Score Distribution')
    #    axes[0, 0].legend()
    #    axes[0, 0].grid(True, alpha=0.3)
    # 
    # 4. Plot 2 - Top Features Scatter (if not using PCA):
    #    if fe.n_components_pca is None and X_test.shape[1] >= 2:
    #        # Use first two features for visualization
    #        feat1_idx, feat2_idx = 0, 1
    #        axes[0, 1].scatter(
    #            X_test.iloc[y_test == 0, feat1_idx],
    #            X_test.iloc[y_test == 0, feat2_idx],
    #            c='blue', alpha=0.3, s=30, label='Normal (True)'
    #        )
    #        axes[0, 1].scatter(
    #            X_test.iloc[y_test == 1, feat1_idx],
    #            X_test.iloc[y_test == 1, feat2_idx],
    #            c='red', alpha=0.6, s=50, label='Anomaly (True)', marker='x'
    #        )
    #        # Overlay detected anomalies
    #        detected_anomalies = (y_pred_test == 1)
    #        axes[0, 1].scatter(
    #            X_test.iloc[detected_anomalies, feat1_idx],
    #            X_test.iloc[detected_anomalies, feat2_idx],
    #            facecolors='none', edgecolors='orange', s=100, 
    #            linewidths=2, label='Detected', marker='o'
    #        )
    #        axes[0, 1].set_xlabel(X_test.columns[feat1_idx])
    #        axes[0, 1].set_ylabel(X_test.columns[feat2_idx])
    #        axes[0, 1].set_title('Anomaly Detection in Feature Space')
    #        axes[0, 1].legend()
    #        axes[0, 1].grid(True, alpha=0.3)
    #    else:
    #        axes[0, 1].text(0.5, 0.5, 'Feature visualization\nrequires >= 2 raw features',
    #                       ha='center', va='center', fontsize=12)
    #        axes[0, 1].axis('off')
    # 
    # 5. Plot 3 - Confusion Matrix:
    #    from sklearn.metrics import confusion_matrix
    #    cm = confusion_matrix(y_test, y_pred_test)
    #    im = axes[1, 0].imshow(cm, cmap='Blues')
    #    axes[1, 0].set_xticks([0, 1])
    #    axes[1, 0].set_yticks([0, 1])
    #    axes[1, 0].set_xticklabels(['Normal', 'Anomaly'])
    #    axes[1, 0].set_yticklabels(['Normal', 'Anomaly'])
    #    axes[1, 0].set_xlabel('Predicted')
    #    axes[1, 0].set_ylabel('True')
    #    axes[1, 0].set_title('Confusion Matrix')
    #    # Add text annotations
    #    for i in range(2):
    #        for j in range(2):
    #            axes[1, 0].text(j, i, str(cm[i, j]),
    #                           ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black',
    #                           fontsize=16, fontweight='bold')
    # 
    # 6. Plot 4 - ROC Curve:
    #    from sklearn.metrics import roc_curve
    #    fpr, tpr, thresholds = roc_curve(y_test, scores_test)
    #    axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={test_roc_auc:.3f})')
    #    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    #    axes[1, 1].set_xlabel('False Positive Rate')
    #    axes[1, 1].set_ylabel('True Positive Rate')
    #    axes[1, 1].set_title('ROC Curve')
    #    axes[1, 1].legend()
    #    axes[1, 1].grid(True, alpha=0.3)
    # 
    # 7. Show plot:
    #    plt.tight_layout()
    #    plt.savefig('anomaly_detection_results.png', dpi=150, bbox_inches='tight')
    #    console.print("  ✓ Visualization saved to anomaly_detection_results.png", style="green")
    #    plt.show()
    #
    # Time estimate: 20-30 minutes
    
    console.print("  ⚠ Implement visualization TODOs in main.py", style="yellow")
    
    # ============================================
    # STEP 6: Save Best Model
    # ============================================
    console.print("\n[bold cyan]💾 SAVING MODEL[/bold cyan]")
    
    # TODO: Save best detector and feature engineer
    # import joblib
    # best_detector.save("models/best_detector.pkl")
    # joblib.dump(fe, "models/feature_engineer.pkl")
    # console.print("  ✓ Detector saved to models/best_detector.pkl", style="green")
    # console.print("  ✓ Feature engineer saved to models/feature_engineer.pkl", style="green")
    
    console.print("\n[bold green]✨ Training complete![/bold green]\n")
    
    # ============================================
    # NEXT STEPS (Optional)
    # ============================================
    console.print("[dim]Next steps:")
    console.print("  1. Try different contamination rates (0.05, 0.15, 0.2)")
    console.print("  2. Experiment with PCA dimensionality reduction")
    console.print("  3. Tune detector hyperparameters for better F1 score")
    console.print("  4. Deploy API: make docker-build && make docker-up")
    console.print("  5. View metrics: http://localhost:9090 (Prometheus)[/dim]")


if __name__ == "__main__":
    main()
