"""Model evaluation and diagnostics for FaceAI

Provides: AutoEvaluator for comprehensive classification assessment
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


logger = logging.getLogger("faceai")


class AutoEvaluator:
    """Automated classification model evaluation with visualizations.
    
    Provides:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix
    - Classification report
    - Per-class performance analysis
    
    Attributes:
        metrics: Dictionary of computed metrics
        predictions: Model predictions
    
    Example:
        >>> evaluator = AutoEvaluator()
        >>> metrics = evaluator.evaluate(model, X_test, y_test)
        >>> evaluator.plot_confusion_matrix(y_test, predictions)
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
        set_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate classification model on dataset.
        
        Args:
            model: Trained model with predict() method
            X: Features
            y: True labels
            set_name: Name of dataset (for logging)
        
        Returns:
            Dictionary with accuracy, precision, recall, f1
        
        Example:
            >>> metrics = evaluator.evaluate(model, X_test, y_test, "test")
            >>> print(f"Test Accuracy: {metrics['accuracy']:.3f}")
        """
        logger.info(f"Evaluating model on {set_name} set ({len(X)} samples)")
        
        # Make predictions
        y_pred = model.predict(X)
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
        normalize: bool = False,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize confusion matrix
            save_path: Path to save plot (if None, displays only)
        
        Example:
            >>> evaluator.plot_confusion_matrix(y_test, predictions, normalize=True)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Classification report string
        
        Example:
            >>> report = evaluator.get_classification_report(y_test, predictions)
            >>> print(report)
        """
        return classification_report(y_true, y_pred)
    
    def get_metrics_summary(self) -> dict:
        """Get summary of all computed metrics.
        
        Returns:
            Dictionary with metrics for each evaluated set
        
        Example:
            >>> summary = evaluator.get_metrics_summary()
            >>> print(summary)
        """
        return self.metrics
