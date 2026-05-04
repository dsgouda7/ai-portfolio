"""Model evaluation and diagnostics for SmartVal AI

Provides: AutoEvaluator for comprehensive model assessment
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve


logger = logging.getLogger("smartval")


class AutoEvaluator:
    """Automated model evaluation with visualizations and diagnostics.
    
    Provides:
    - Regression metrics (MAE, RMSE, R², MAPE)
    - Residual analysis
    - Learning curves
    - Feature importance (for tree models)
    - Prediction vs actual plots
    
    Attributes:
        metrics: Dictionary of computed metrics
        residuals: Prediction residuals
    
    Example:
        >>> evaluator = AutoEvaluator()
        >>> metrics = evaluator.evaluate(model, X_test, y_test)
        >>> evaluator.plot_residuals()
        >>> evaluator.plot_predictions_vs_actual(y_test, predictions)
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        self.residuals = None
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        logger.info("Initialized AutoEvaluator")
    
    def evaluate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            model: Trained model with predict() method
            X: Features
            y: True labels
            set_name: Name of dataset (for logging)
        
        Returns:
            Dictionary with mae, rmse, r2, mape
        
        Example:
            >>> metrics = evaluator.evaluate(model, X_test, y_test, "test")
            >>> print(f"Test MAE: {metrics['mae']:.2f}")
        """
        logger.info(f"Evaluating model on {set_name} set ({len(X)} samples)")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Compute metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mape = self._compute_mape(y, y_pred)
        
        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
        }
        
        # Store results
        self.metrics[set_name] = metrics
        self.residuals = y - y_pred
        
        logger.info(
            f"{set_name.capitalize()} metrics - "
            f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}, MAPE: {mape:.2f}%"
        )
        
        return metrics
    
    def plot_residuals(
        self,
        y_pred: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot residual analysis (residuals vs predictions, histogram, Q-Q plot).
        
        Args:
            y_pred: Predictions (if None, uses last evaluate() results)
            save_path: Path to save plot (if None, displays only)
        
        Example:
            >>> evaluator.plot_residuals(save_path="plots/residuals.png")
        """
        if self.residuals is None:
            raise RuntimeError("No residuals available. Call evaluate() first.")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Residuals vs Predictions
        axes[0].scatter(y_pred if y_pred is not None else range(len(self.residuals)),
                       self.residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predictions')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(self.residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(self.residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved residual plot to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot predictions vs actual values.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
        
        Example:
            >>> evaluator.plot_predictions_vs_actual(y_test, predictions)
        """
        plt.figure(figsize=(8, 8))
        
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        plt.xlabel('Actual Values ($100k)')
        plt.ylabel('Predicted Values ($100k)')
        plt.title('Predictions vs Actual Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        save_path: Optional[str] = None
    ) -> None:
        """Plot learning curves (training and validation scores vs training size).
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            save_path: Path to save plot
        
        Example:
            >>> evaluator.plot_learning_curves(model, X_train, y_train)
        """
        logger.info("Computing learning curves (this may take a while)...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        
        # Convert to positive MAE
        train_mae = -train_scores.mean(axis=1)
        val_mae = -val_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_std = val_scores.std(axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_mae, 'o-', label='Training MAE', linewidth=2)
        plt.plot(train_sizes, val_mae, 'o-', label='Validation MAE', linewidth=2)
        
        plt.fill_between(train_sizes, train_mae - train_std, train_mae + train_std, alpha=0.1)
        plt.fill_between(train_sizes, val_mae - val_std, val_mae + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Mean Absolute Error')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curves to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        model,
        feature_names: list,
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance (for tree-based models).
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save plot
        
        Example:
            >>> evaluator.plot_feature_importance(xgb_model, X_train.columns)
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of all computed metrics.
        
        Returns:
            DataFrame with metrics for each evaluated set
        
        Example:
            >>> summary = evaluator.get_metrics_summary()
            >>> print(summary)
        """
        if not self.metrics:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics).T
    
    def _compute_mape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            MAPE as percentage
        """
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return 0.0
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape)
