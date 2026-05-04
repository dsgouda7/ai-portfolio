"""Model evaluation and diagnostics for UnifiedAI

Provides: AutoEvaluator for comprehensive neural network assessment
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


logger = logging.getLogger("unifiedai")


class AutoEvaluator:
    """Automated neural network evaluation with visualizations.
    
    Provides:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix
    - Learning curves (loss and accuracy over epochs)
    - Per-class performance analysis
    
    Attributes:
        metrics: Dictionary of computed metrics
        predictions: Model predictions
    
    Example:
        >>> evaluator = AutoEvaluator()
        >>> metrics = evaluator.evaluate(model, X_test, y_test)
        >>> evaluator.plot_learning_curves(history)
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        self.predictions = None
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)
        
        logger.info("Initialized AutoEvaluator")
    
    def evaluate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "model",
        set_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate neural network model on dataset.
        
        Args:
            model: Trained Keras model
            X: Features
            y: True labels
            model_name: Name of model (for CNN reshaping)
            set_name: Name of dataset (for logging)
        
        Returns:
            Dictionary with accuracy, precision, recall, f1
        
        Example:
            >>> metrics = evaluator.evaluate(model, X_test, y_test, "dense_nn", "test")
        """
        logger.info(f"Evaluating {model_name} on {set_name} set ({len(X)} samples)")
        
        # Reshape for CNN if needed
        if model_name == "cnn_1d":
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Make predictions
        predictions_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(predictions_proba, axis=1)
        self.predictions = y_pred
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        
        # Store results
        self.metrics[set_name] = metrics
        
        logger.info(
            f"{set_name.capitalize()} metrics - "
            f"Acc: {accuracy:.3f}, P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}"
        )
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save figure
        
        Example:
            >>> evaluator.plot_confusion_matrix(y_test, predictions)
        """
        logger.info("Generating confusion matrix")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(
        self,
        history: Dict,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot training and validation learning curves.
        
        Args:
            history: Training history dictionary from model.fit()
            save_path: Optional path to save figure
        
        Example:
            >>> evaluator.plot_learning_curves(history)
        """
        logger.info("Generating learning curves")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Example:
            >>> evaluator.print_classification_report(y_test, predictions)
        """
        logger.info("Generating classification report")
        
        report = classification_report(y_true, y_pred, zero_division=0)
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(report)
        print("="*60 + "\n")
    
    def get_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """Get per-class performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary mapping class labels to metrics
        
        Example:
            >>> per_class = evaluator.get_per_class_metrics(y_test, predictions)
        """
        classes = np.unique(y_true)
        per_class_metrics = {}
        
        for cls in classes:
            # Binary mask for current class
            mask = (y_true == cls)
            cls_pred = (y_pred == cls)
            
            # Compute metrics
            tp = np.sum(mask & cls_pred)
            fp = np.sum(~mask & cls_pred)
            fn = np.sum(mask & ~cls_pred)
            tn = np.sum(~mask & ~cls_pred)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[int(cls)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(np.sum(mask))
            }
        
        return per_class_metrics
