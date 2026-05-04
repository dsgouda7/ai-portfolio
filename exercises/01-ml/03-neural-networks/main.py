"""UnifiedAI - Interactive Neural Network Training and Experimentation

This script demonstrates:
1. Feature engineering with standardization/PCA
2. Plug-and-play neural network comparison (MLP, CNN)
3. Live epoch-by-epoch training feedback
4. Beautiful console output with leaderboards

Usage:
    python main.py

Expected runtime: 5-10 minutes (6 models with 50 epochs each)
Expected output: Console shows live epoch feedback, leaderboard, and saves best model
"""

from rich.console import Console
from rich.panel import Panel

from src.data import load_and_split
from src.features import FeatureEngineer
from src.models import (
    MLPClassifier,
    CNNClassifier,
    ExperimentRunner,
    ModelConfig,
)

console = Console()


def main():
    """Run complete neural network pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]UnifiedAI[/bold cyan]\n"
        "Interactive Neural Network Classification",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load Data
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
    console.print(f"  ✓ Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    console.print(f"  ✓ Val:   {X_val.shape[0]:,} samples × {X_val.shape[1]} features")
    console.print(f"  ✓ Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    console.print(f"  ✓ Classes: {len(set(y_train))}")
    
    # ============================================
    # STEP 2: Feature Engineering
    # ============================================
    console.print("\n[bold cyan]🔧 FEATURE ENGINEERING[/bold cyan]")
    console.print("→ Applying standardization (zero mean, unit variance)...")
    
    fe = FeatureEngineer(
        scale_features=True,
        pca_components=None  # Try setting to 20 to see dimensionality reduction
    )
    
    X_train_transformed = fe.fit_transform(X_train)
    X_val_transformed = fe.transform(X_val)
    X_test_transformed = fe.transform(X_test)
    
    console.print(f"  ✓ Features: {X_train.shape[1]} → {X_train_transformed.shape[1]}")
    console.print(f"  ✓ Feature scaling complete (mean≈0, std≈1)")
    
    # ============================================
    # STEP 3: Neural Network Training (Plug-and-Play Comparison)
    # ============================================
    console.print("\n[bold cyan]🧠 NEURAL NETWORK TRAINING[/bold cyan]")
    console.print("Comparing 6 neural architectures with live epoch feedback...")
    
    runner = ExperimentRunner()
    
    # TODO: Register different MLP architectures (try varying depth/width)
    runner.register("Shallow MLP (64, 32)", MLPClassifier(
        architecture=[64, 32],
        dropout=0.3,
        learning_rate=0.001
    ))
    
    runner.register("Medium MLP (128, 64, 32)", MLPClassifier(
        architecture=[128, 64, 32],
        dropout=0.3,
        learning_rate=0.001
    ))
    
    runner.register("Deep MLP (256, 128, 64, 32)", MLPClassifier(
        architecture=[256, 128, 64, 32],
        dropout=0.4,  # More dropout for deeper network
        learning_rate=0.001
    ))
    
    # TODO: Try different dropout rates (experiment with 0.1 vs 0.5)
    runner.register("Low Dropout MLP (128, 64)", MLPClassifier(
        architecture=[128, 64],
        dropout=0.1,
        learning_rate=0.001
    ))
    
    # TODO: Try CNNs with different filter configurations
    runner.register("Shallow CNN (32, 16)", CNNClassifier(
        filters=[32, 16],
        kernel_size=3,
        dropout=0.3,
        learning_rate=0.001
    ))
    
    runner.register("Deep CNN (64, 32, 16)", CNNClassifier(
        filters=[64, 32, 16],
        kernel_size=3,
        dropout=0.3,
        learning_rate=0.001
    ))
    
    # Run all experiments (prints live epoch-by-epoch feedback)
    config = ModelConfig(
        batch_size=32,
        epochs=50,
        early_stopping_patience=5
    )
    
    runner.run_experiment(X_train_transformed, y_train, X_val_transformed, y_val, config)
    
    # Print leaderboard
    runner.print_leaderboard()
    
    # ============================================
    # STEP 4: Final Evaluation on Test Set
    # ============================================
    console.print("\n[bold cyan]🎯 TEST SET EVALUATION[/bold cyan]")
    console.print("→ Evaluating best model on held-out test set...")
    
    # TODO: Get best model, evaluate on test set, print classification report
    
    # TODO: Save best model to models/best_neural_network.keras
    
    # ============================================
    # STEP 5: Visualization (Optional)
    # ============================================
    console.print("\n[bold cyan]📈 TRAINING VISUALIZATION[/bold cyan]")
    console.print("→ To plot training curves, uncomment code below:")
    
    # TODO: Plot training history (loss and accuracy curves)
    
    # ============================================
    # Final Summary
    # ============================================
    console.print("\n" + "="*60)
    console.print("[bold green]✓ EXPERIMENT COMPLETE![/bold green]")
    console.print("="*60)
    console.print("\n[bold cyan]Key Observations to Note:[/bold cyan]")
    console.print("  1. Which architecture achieved highest validation accuracy?")
    console.print("  2. Did deeper networks overfit (train acc >> val acc)?")
    console.print("  3. How many epochs did models need to converge?")
    console.print("  4. Compare MLP vs CNN performance on this dataset")
    console.print("  5. Effect of dropout rate on generalization")
    console.print("\n[dim]Next steps: Uncomment test evaluation and visualization code![/dim]")


if __name__ == "__main__":
    main()
