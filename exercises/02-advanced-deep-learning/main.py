"""Advanced Deep Learning - Interactive Model Training and Comparison

This script demonstrates:
1. Data augmentation with immediate visual feedback
2. Transfer learning with frozen/unfrozen layers
3. Plug-and-play architecture comparison (ResNet, ViT, EfficientNet)
4. Mixed precision training and gradient accumulation
5. Beautiful console output with training curves

Usage:
    python main.py

Expected runtime: 10-15 minutes (3 models × 5 epochs each on CPU)
Expected output: Training progress, leaderboard, saved best model

Advanced concepts demonstrated:
- ResNet: Residual connections for deep networks
- Vision Transformer: Self-attention across image patches
- Transfer learning: Freeze early layers, fine-tune last layers
- Mixed precision: FP16 training for 2-3x speedup
- Data augmentation: Rotation, flip, color jitter, random erasing
"""

from rich.console import Console
from rich.panel import Panel

from src.features import DatasetBuilder, ImageAugmenter
from src.models import (
    ResNetModel,
    TransformerModel,
    EfficientNetModel,
    ExperimentRunner,
    TrainingConfig,
    count_parameters,
)

console = Console()


def main():
    """Run complete deep learning pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]Advanced Deep Learning[/bold cyan]\n"
        "Transfer Learning & Model Comparison",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load Data with Augmentation
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    
    dataset_builder = DatasetBuilder(
        dataset_name="cifar10",  # 10 classes: airplane, car, bird, etc.
        data_dir="./data",
        val_split=0.1,
        num_workers=2
    )
    
    console.print("→ Building datasets with augmentation...")
    train_dataset, val_dataset, test_dataset = dataset_builder.build_datasets()
    
    console.print("→ Creating DataLoaders...")
    train_loader, val_loader, test_loader = dataset_builder.create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=32
    )
    
    # ============================================
    # STEP 2: Configure Training
    # ============================================
    console.print("\n[bold cyan]⚙️  TRAINING CONFIGURATION[/bold cyan]")
    
    config = TrainingConfig(
        epochs=5,  # Quick demo (increase to 20-50 for real training)
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        mixed_precision=True,  # FP16 for 2-3x speedup
        gradient_accumulation_steps=1,
        early_stopping_patience=5
    )
    
    console.print(f"  • Epochs: {config.epochs}")
    console.print(f"  • Batch size: {config.batch_size}")
    console.print(f"  • Learning rate: {config.learning_rate}")
    console.print(f"  • Mixed precision: {config.mixed_precision}")
    console.print(f"  • Device: {config.device}")
    
    # ============================================
    # STEP 3: Model Training (Plug-and-Play Comparison)
    # ============================================
    console.print("\n[bold cyan]🤖 MODEL TRAINING[/bold cyan]")
    console.print("Comparing 3 architectures with transfer learning...")
    
    runner = ExperimentRunner()
    
    # TODO: Register ResNet with different freeze levels
    # Experiment: Which freeze level works best?
    # - freeze_layers=0: Train all (slower, more flexible)
    # - freeze_layers=3: Freeze early layers (faster, good for similar domains)
    # - freeze_layers=4: Freeze all except FC (fastest, limited adaptation)
    console.print("\n[dim]Registering models...[/dim]")
    runner.register("ResNet-50 (freeze=3)", ResNetModel(num_classes=10, freeze_layers=3))
    runner.register("ResNet-50 (freeze=0)", ResNetModel(num_classes=10, freeze_layers=0))
    
    # TODO: Register Vision Transformer
    # ViT needs more data → freeze encoder for small datasets
    runner.register("ViT-B/16 (freeze=True)", TransformerModel(num_classes=10, freeze_encoder=True))
    
    # TODO: Register EfficientNet (efficient baseline)
    runner.register("EfficientNet-B0", EfficientNetModel(num_classes=10))
    
    # Print model sizes
    console.print("\n[bold cyan]📐 MODEL SIZES[/bold cyan]")
    for name, model in runner.models.items():
        model.build_model()
        params = count_parameters(model.model)
        console.print(f"  • {name}: {params:,} parameters")
    
    # Run all experiments (prints after each model)
    console.print("\n[bold cyan]🏃 RUNNING EXPERIMENTS[/bold cyan]")
    runner.run_experiment(train_loader, val_loader, config)
    
    # Print leaderboard
    runner.print_leaderboard()
    
    # ============================================
    # STEP 4: Final Evaluation on Test Set
    # ============================================
    console.print("\n[bold cyan]✅ FINAL EVALUATION[/bold cyan]")
    
    best_model = runner.get_best_model()
    console.print(f"→ Evaluating {best_model.name} on test set...")
    
    # TODO: Compute test accuracy
    # Hint: Loop through test_loader, accumulate correct predictions
    # test_correct = 0
    # test_total = 0
    # best_model.model.eval()
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images, labels = images.to(config.device), labels.to(config.device)
    #         outputs = best_model.predict(images)
    #         _, predicted = torch.max(outputs, 1)
    #         test_total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()
    # test_acc = 100.0 * test_correct / test_total
    # console.print(f"  Test Accuracy: {test_acc:.2f}%", style="green")
    
    # ============================================
    # STEP 5: Save Best Model
    # ============================================
    console.print("\n[bold cyan]💾 SAVING MODEL[/bold cyan]")
    
    # TODO: Save best model
    # best_model.save("models/best_model.pth")
    # console.print("  ✓ Model saved to models/best_model.pth", style="green")
    
    console.print("\n[bold green]✨ Training complete![/bold green]\n")
    
    # ============================================
    # NEXT STEPS (Optional)
    # ============================================
    console.print("[dim]Next steps:")
    console.print("  1. Increase epochs (20-50) for better accuracy")
    console.print("  2. Try different augmentation strengths")
    console.print("  3. Experiment with learning rate schedules")
    console.print("  4. Implement Mixup/CutMix for +2-3% accuracy")
    console.print("  5. Fine-tune with unfrozen layers (two-stage training)[/dim]")
    
    # ============================================
    # TRANSFER LEARNING INSIGHTS
    # ============================================
    console.print("\n[bold yellow]💡 TRANSFER LEARNING INSIGHTS[/bold yellow]")
    console.print("[dim]")
    console.print("Why transfer learning works:")
    console.print("  • Early layers learn generic features (edges, textures)")
    console.print("  • Later layers learn task-specific features (objects, scenes)")
    console.print("  • Freezing early layers → faster training, less overfitting")
    console.print("  • ImageNet pretraining → strong initialization")
    console.print("")
    console.print("When to freeze layers:")
    console.print("  • Small dataset (<10k images): Freeze more (3-4 layer groups)")
    console.print("  • Large dataset (>100k): Freeze less (0-2 groups)")
    console.print("  • Similar domain (e.g., ImageNet → CIFAR): Freeze more")
    console.print("  • Different domain (e.g., natural → medical): Freeze less")
    console.print("[/dim]")


if __name__ == "__main__":
    main()
