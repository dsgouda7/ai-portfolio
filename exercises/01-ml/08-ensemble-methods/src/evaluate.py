"""Evaluation and ensemble analysis for EnsembleAI

Provides: Comprehensive classification metrics, ensemble diversity analysis
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

from src.utils import timer


logger = logging.getLogger("ensembleai")


class AutoEvaluator:
    """Automatic evaluation and ensemble analysis.
    
    Provides:
    - Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
    - Confusion matrix visualization
    - ROC curve comparison
    - Ensemble diversity metrics
    - Individual vs ensemble performance comparison
    
    Example:
        >>> evaluator = AutoEvaluator()
        >>> metrics = evaluator.evaluate_classification(y_test, y_pred)
        >>> evaluator.compare_models(y_test, predictions_dict)
    """
    
    def __init__(self):
        """Initialize evaluator."""
        logger.info("Initialized AutoEvaluator")
    
    @timer
    def evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)
            model_name: Name of model for logging
        
        Returns:
            Dictionary of metrics
        
        Example:
            >>> metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)
        """
        metrics = {}
        
        # Core metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # ROC-AUC (if probabilities provided)
        if y_proba is not None:
            if len(y_proba.shape) == 2:
                # Multi-class or binary with proba for both classes
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
            except ValueError as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics['roc_auc'] = np.nan
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str = "model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            save_path: Path to save plot (None = show only)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_models(
        self,
        y_true: pd.Series,
        predictions: Dict[str, np.ndarray],
        probabilities: Optional[Dict[str, np.ndarray]] = None
    ) -> pd.DataFrame:
        """Compare multiple models side-by-side.
        
        Args:
            y_true: True labels
            predictions: Dictionary mapping model_name -> predictions
            probabilities: Optional dict mapping model_name -> probabilities
        
        Returns:
            DataFrame with comparison metrics
        
        Example:
            >>> predictions = {
            ...     'xgboost': xgb_pred,
            ...     'lightgbm': lgbm_pred,
            ...     'voting': voting_pred
            ... }
            >>> comparison = evaluator.compare_models(y_test, predictions, probabilities)
        """
        results = []
        
        for model_name, y_pred in predictions.items():
            y_proba = probabilities.get(model_name) if probabilities else None
            metrics = self.evaluate_classification(y_true, y_pred, y_proba, model_name)
            metrics['model'] = model_name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.set_index('model')
        
        # Sort by F1 score
        df = df.sort_values('f1', ascending=False)
        
        logger.info("\n" + "="*60)
        logger.info("Model Comparison:")
        logger.info("\n" + df.to_string())
        logger.info("="*60)
        
        return df
    
    def plot_roc_curves(
        self,
        y_true: pd.Series,
        probabilities: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curves for multiple models.
        
        Args:
            y_true: True labels
            probabilities: Dictionary mapping model_name -> probabilities
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, y_proba in probabilities.items():
            # Extract positive class probabilities
            if len(y_proba.shape) == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
            auc = roc_auc_score(y_true, y_proba_pos)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
        
        # Diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_ensemble_diversity(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Analyze diversity of ensemble models.
        
        Computes pairwise diversity metrics:
        - Q-statistic: Agreement between pairs of classifiers
        - Disagreement measure: Fraction of cases where models disagree
        - Double-fault measure: Fraction where both models are wrong
        
        Args:
            predictions: Dictionary mapping model_name -> predictions
            y_true: True labels (optional, required for double-fault)
        
        Returns:
            Dictionary with diversity metrics
        
        Example:
            >>> diversity = evaluator.analyze_ensemble_diversity(predictions, y_test)
        """
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        # Pairwise metrics
        q_statistics = []
        disagreements = []
        double_faults = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_i = model_names[i]
                model_j = model_names[j]
                
                pred_i = predictions[model_i]
                pred_j = predictions[model_j]
                
                # Q-statistic
                n11 = np.sum((pred_i == 1) & (pred_j == 1))
                n10 = np.sum((pred_i == 1) & (pred_j == 0))
                n01 = np.sum((pred_i == 0) & (pred_j == 1))
                n00 = np.sum((pred_i == 0) & (pred_j == 0))
                
                if (n11 * n00 + n01 * n10) != 0:
                    q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                    q_statistics.append(q)
                
                # Disagreement measure
                disagreement = np.mean(pred_i != pred_j)
                disagreements.append(disagreement)
                
                # Double-fault measure (if true labels provided)
                if y_true is not None:
                    double_fault = np.mean((pred_i != y_true) & (pred_j != y_true))
                    double_faults.append(double_fault)
        
        results = {
            'q_statistic_mean': np.mean(q_statistics) if q_statistics else np.nan,
            'q_statistic_std': np.std(q_statistics) if q_statistics else np.nan,
            'disagreement_mean': np.mean(disagreements) if disagreements else np.nan,
            'disagreement_std': np.std(disagreements) if disagreements else np.nan,
        }
        
        if double_faults:
            results['double_fault_mean'] = np.mean(double_faults)
            results['double_fault_std'] = np.std(double_faults)
        
        logger.info("Ensemble Diversity Analysis:")
        logger.info(f"  Q-statistic (mean): {results['q_statistic_mean']:.4f} "
                   f"(lower = more diverse)")
        logger.info(f"  Disagreement (mean): {results['disagreement_mean']:.4f} "
                   f"(higher = more diverse)")
        
        return results
    
    def compare_individual_vs_ensemble(
        self,
        y_true: pd.Series,
        base_predictions: Dict[str, np.ndarray],
        ensemble_predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Compare individual base models against ensemble methods.
        
        Args:
            y_true: True labels
            base_predictions: Dict of base model predictions
            ensemble_predictions: Dict of ensemble predictions
        
        Returns:
            DataFrame with comparison showing improvement
        """
        # Evaluate base models
        base_metrics = []
        for name, pred in base_predictions.items():
            acc = accuracy_score(y_true, pred)
            f1 = f1_score(y_true, pred, average='binary')
            base_metrics.append({
                'model': name,
                'type': 'base',
                'accuracy': acc,
                'f1': f1
            })
        
        # Evaluate ensemble models
        ensemble_metrics = []
        for name, pred in ensemble_predictions.items():
            acc = accuracy_score(y_true, pred)
            f1 = f1_score(y_true, pred, average='binary')
            ensemble_metrics.append({
                'model': name,
                'type': 'ensemble',
                'accuracy': acc,
                'f1': f1
            })
        
        # Combine and compute improvement
        df = pd.DataFrame(base_metrics + ensemble_metrics)
        
        # Get best base model performance
        best_base_acc = df[df['type'] == 'base']['accuracy'].max()
        best_base_f1 = df[df['type'] == 'base']['f1'].max()
        
        # Compute improvement for ensemble models
        df['acc_improvement'] = df.apply(
            lambda row: (row['accuracy'] - best_base_acc) * 100 if row['type'] == 'ensemble' else np.nan,
            axis=1
        )
        df['f1_improvement'] = df.apply(
            lambda row: (row['f1'] - best_base_f1) * 100 if row['type'] == 'ensemble' else np.nan,
            axis=1
        )
        
        logger.info("\n" + "="*60)
        logger.info("Individual vs Ensemble Performance:")
        logger.info(f"Best Base Model - Accuracy: {best_base_acc:.4f}, F1: {best_base_f1:.4f}")
        logger.info("\n" + df.to_string(index=False))
        logger.info("="*60)
        
        return df
