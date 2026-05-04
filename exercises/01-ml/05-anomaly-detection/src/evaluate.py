"""Model evaluation and diagnostics for FraudShield

Provides: AnomalyEvaluator for comprehensive anomaly detection assessment
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)


logger = logging.getLogger("fraudshield")


class AnomalyEvaluator:
    """Automated anomaly detection evaluation with visualizations and diagnostics.
    
    Provides:
    - Binary classification metrics (precision, recall, F1, ROC-AUC)
    - Precision@K for top-K predictions
    - Confusion matrix
    - ROC curve
    - Score distribution plots
    
    Attributes:
        metrics: Dictionary of computed metrics
        confusion_mat: Confusion matrix
    
    Example:
        >>> evaluator = AnomalyEvaluator()
        >>> metrics = evaluator.evaluate(y_test, predictions, scores)
        >>> evaluator.plot_confusion_matrix()
        >>> evaluator.plot_roc_curve()
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        self.confusion_mat = None
        self.y_true = None
        self.y_pred = None
        self.scores = None
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        logger.info("Initialized AnomalyEvaluator")
    
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        scores: Optional[np.ndarray] = None,
        set_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate anomaly detection model.
        
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            y_pred: Predicted labels
            scores: Anomaly scores (optional, for ROC-AUC and precision@K)
            set_name: Name of dataset (for logging)
        
        Returns:
            Dictionary with precision, recall, f1, accuracy, roc_auc
        
        Example:
            >>> metrics = evaluator.evaluate(y_test, predictions, scores, "test")
        """
        logger.info(f"Evaluating model on {set_name} set ({len(y_true)} samples)")
        
        # Store for plotting
        self.y_true = y_true
        self.y_pred = y_pred
        self.scores = scores
        
        # Compute basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((y_true == 1).sum()),
        }
        
        # ROC-AUC (requires scores)
        if scores is not None:
            try:
                roc_auc = roc_auc_score(y_true, scores)
                metrics["roc_auc"] = float(roc_auc)
            except ValueError as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = None
        
        # Confusion matrix
        self.confusion_mat = confusion_matrix(y_true, y_pred)
        
        # Store results
        self.metrics[set_name] = metrics
        
        logger.info(
            f"{set_name.capitalize()} metrics - "
            f"Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"F1: {f1:.3f}" + 
            (f", ROC-AUC: {metrics['roc_auc']:.3f}" if metrics.get('roc_auc') else "")
        )
        
        return metrics
    
    def compute_precision_at_k(
        self,
        y_true: pd.Series,
        scores: np.ndarray,
        k_values: List[int] = [10, 20, 50]
    ) -> Dict[int, float]:
        """Compute precision@K for top-K highest scoring predictions.
        
        Useful metric for anomaly detection: what fraction of our top K
        predictions are true anomalies?
        
        Args:
            y_true: True labels
            scores: Anomaly scores (higher = more anomalous)
            k_values: List of K values to evaluate
        
        Returns:
            Dictionary mapping K to precision@K
        
        Example:
            >>> precision_at_k = evaluator.compute_precision_at_k(y_test, scores)
            >>> print(f"Precision@10: {precision_at_k[10]:.1%}")
        """
        results = {}
        
        # Sort indices by score (descending)
        sorted_indices = np.argsort(-scores)
        
        for k in k_values:
            if k > len(y_true):
                logger.warning(f"K={k} exceeds dataset size ({len(y_true)}), skipping")
                continue
            
            # Top K predictions
            top_k_indices = sorted_indices[:k]
            top_k_labels = y_true.iloc[top_k_indices]
            
            # Precision = fraction of top K that are true anomalies
            precision = (top_k_labels == 1).sum() / k
            results[k] = float(precision)
            
            logger.info(f"Precision@{k}: {precision:.1%}")
        
        return results
    
    def plot_confusion_matrix(
        self,
        normalize: bool = False,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            normalize: Whether to normalize by row (show rates instead of counts)
            save_path: Path to save plot (if None, displays only)
        
        Example:
            >>> evaluator.plot_confusion_matrix(save_path="plots/confusion.png")
        """
        if self.confusion_mat is None:
            raise RuntimeError("No confusion matrix available. Call evaluate() first.")
        
        cm = self.confusion_mat
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2%' if normalize else 'd',
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """Plot ROC curve.
        
        Args:
            save_path: Path to save plot (if None, displays only)
        
        Example:
            >>> evaluator.plot_roc_curve(save_path="plots/roc.png")
        """
        if self.scores is None:
            raise RuntimeError("No scores available. Call evaluate() with scores.")
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.scores)
        roc_auc = roc_auc_score(self.y_true, self.scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_score_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot distribution of anomaly scores by class.
        
        Args:
            save_path: Path to save plot (if None, displays only)
        
        Example:
            >>> evaluator.plot_score_distribution(save_path="plots/scores.png")
        """
        if self.scores is None:
            raise RuntimeError("No scores available. Call evaluate() with scores.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate scores by class
        normal_scores = self.scores[self.y_true == 0]
        anomaly_scores = self.scores[self.y_true == 1]
        
        # Plot histograms
        ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', edgecolor='black')
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', edgecolor='black')
        
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Anomaly Scores by Class', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Score distribution saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_classification_report(self) -> None:
        """Print detailed classification report.
        
        Example:
            >>> evaluator.print_classification_report()
        """
        if self.y_pred is None:
            raise RuntimeError("No predictions available. Call evaluate() first.")
        
        report = classification_report(
            self.y_true,
            self.y_pred,
            target_names=['Normal', 'Anomaly'],
            digits=3
        )
        
        print("\nClassification Report:")
        print("=" * 60)
        print(report)
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of all evaluation results.
        
        Returns:
            Dictionary with metrics and diagnostic information
        
        Example:
            >>> summary = evaluator.get_summary()
            >>> print(summary['test']['f1'])
        """
        summary = {
            "metrics": self.metrics,
            "confusion_matrix": self.confusion_mat.tolist() if self.confusion_mat is not None else None,
        }
        
        return summary
