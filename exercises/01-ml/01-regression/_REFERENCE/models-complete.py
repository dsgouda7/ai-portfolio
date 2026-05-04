"""Model training and registry for SmartVal AI

Provides: ModelRegistry for training Ridge, Lasso, XGBoost with persistence
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from src.utils import timer, validate_positive


logger = logging.getLogger("smartval")


class ModelRegistry:
    """Registry for training and managing multiple regression models.
    
    Supported models:
    - Ridge regression (L2 regularization)
    - Lasso regression (L1 regularization)
    - XGBoost (gradient boosting)
    
    Attributes:
        models: Dictionary of trained models
        best_model_name: Name of best performing model
        cv_scores: Cross-validation scores for each model
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.train_ridge(X_train, y_train, alpha=1.0)
        >>> registry.train_xgboost(X_train, y_train, n_estimators=100)
        >>> predictions = registry.predict(X_test, "ridge")
    """
    
    def __init__(self):
        """Initialize empty model registry."""
        self.models = {}
        self.best_model_name = None
        self.cv_scores = {}
        
        logger.info("Initialized ModelRegistry")
    
    @timer
    def train_ridge(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float = 1.0,
        max_iter: int = 1000,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Train Ridge regression model.
        
        Args:
            X: Training features
            y: Training labels
            alpha: Regularization strength (higher = more regularization)
            max_iter: Maximum iterations for solver
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with training metrics (mae, rmse, r2)
        
        Raises:
            ValueError: If alpha is not positive
            RuntimeError: If training fails
        
        Example:
            >>> metrics = registry.train_ridge(X_train, y_train, alpha=1.0)
            >>> print(f"CV MAE: {metrics['cv_mae']:.2f}")
        """
        validate_positive(alpha, "alpha")
        
        logger.info(f"Training Ridge (alpha={alpha}, max_iter={max_iter})")
        
        try:
            model = Ridge(alpha=alpha, max_iter=max_iter, random_state=42)
            model.fit(X, y)
            
            # Cross-validation
            cv_mae = self._cross_validate(model, X, y, cv_folds)
            
            # Store model
            self.models["ridge"] = model
            self.cv_scores["ridge"] = cv_mae
            
            # Training metrics
            y_pred = model.predict(X)
            metrics = self._compute_metrics(y, y_pred)
            metrics["cv_mae"] = cv_mae
            
            logger.info(f"Ridge trained - CV MAE: {cv_mae:.2f}, Train MAE: {metrics['mae']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ridge training failed: {e}")
            raise RuntimeError(f"Ridge training failed: {e}") from e
    
    @timer
    def train_lasso(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float = 0.1,
        max_iter: int = 1000,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Train Lasso regression model.
        
        Args:
            X: Training features
            y: Training labels
            alpha: Regularization strength
            max_iter: Maximum iterations
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_lasso(X_train, y_train, alpha=0.1)
        """
        validate_positive(alpha, "alpha")
        
        logger.info(f"Training Lasso (alpha={alpha}, max_iter={max_iter})")
        
        try:
            model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
            model.fit(X, y)
            
            # Cross-validation
            cv_mae = self._cross_validate(model, X, y, cv_folds)
            
            # Store model
            self.models["lasso"] = model
            self.cv_scores["lasso"] = cv_mae
            
            # Training metrics
            y_pred = model.predict(X)
            metrics = self._compute_metrics(y, y_pred)
            metrics["cv_mae"] = cv_mae
            metrics["n_nonzero_coef"] = np.sum(model.coef_ != 0)
            
            logger.info(
                f"Lasso trained - CV MAE: {cv_mae:.2f}, "
                f"Non-zero coef: {metrics['n_nonzero_coef']}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Lasso training failed: {e}")
            raise RuntimeError(f"Lasso training failed: {e}") from e
    
    @timer
    def train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        cv_folds: int = 5,
        **kwargs
    ) -> Dict[str, float]:
        """Train XGBoost regression model.
        
        Args:
            X: Training features
            y: Training labels
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            cv_folds: Number of cross-validation folds
            **kwargs: Additional XGBoost parameters
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_xgboost(
            ...     X_train, y_train,
            ...     n_estimators=100,
            ...     max_depth=6
            ... )
        """
        logger.info(
            f"Training XGBoost (n_estimators={n_estimators}, "
            f"max_depth={max_depth}, lr={learning_rate})"
        )
        
        try:
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                **kwargs
            )
            model.fit(X, y)
            
            # Cross-validation
            cv_mae = self._cross_validate(model, X, y, cv_folds)
            
            # Store model
            self.models["xgboost"] = model
            self.cv_scores["xgboost"] = cv_mae
            
            # Training metrics
            y_pred = model.predict(X)
            metrics = self._compute_metrics(y, y_pred)
            metrics["cv_mae"] = cv_mae
            
            logger.info(f"XGBoost trained - CV MAE: {cv_mae:.2f}, Train MAE: {metrics['mae']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            raise RuntimeError(f"XGBoost training failed: {e}") from e
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions using specified model.
        
        Args:
            X: Features to predict on
            model_name: Name of model to use (None = best model)
        
        Returns:
            Array of predictions
        
        Raises:
            ValueError: If model_name not found
            RuntimeError: If no models trained
        
        Example:
            >>> predictions = registry.predict(X_test, "ridge")
        """
        if not self.models:
            raise RuntimeError("No models trained. Train a model first.")
        
        if model_name is None:
            model_name = self.get_best_model_name()
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        return model.predict(X)
    
    def get_best_model_name(self) -> str:
        """Get name of model with lowest cross-validation MAE.
        
        Returns:
            Name of best model
        
        Raises:
            RuntimeError: If no models trained
        """
        if not self.cv_scores:
            raise RuntimeError("No models trained with cross-validation")
        
        best_name = min(self.cv_scores, key=self.cv_scores.get)
        self.best_model_name = best_name
        
        return best_name
    
    def save_model(self, model_name: str, path: str) -> None:
        """Save model to disk using joblib.
        
        Args:
            model_name: Name of model to save
            path: File path for saved model
        
        Raises:
            ValueError: If model not found
        
        Example:
            >>> registry.save_model("ridge", "models/ridge_v1.pkl")
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.models[model_name], model_path)
        logger.info(f"Saved {model_name} to {path}")
    
    def load_model(self, model_name: str, path: str) -> None:
        """Load model from disk.
        
        Args:
            model_name: Name to assign loaded model
            path: File path of saved model
        
        Raises:
            FileNotFoundError: If model file not found
        
        Example:
            >>> registry.load_model("ridge", "models/ridge_v1.pkl")
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.models[model_name] = joblib.load(model_path)
        logger.info(f"Loaded {model_name} from {path}")
    
    def _cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series, cv_folds: int) -> float:
        """Perform cross-validation and return mean MAE.
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Labels
            cv_folds: Number of folds
        
        Returns:
            Mean absolute error across folds
        """
        scores = cross_val_score(
            model, X, y,
            cv=cv_folds,
            scoring='neg_mean_absolute_error'
        )
        mae = -scores.mean()  # Convert back to positive MAE
        
        return float(mae)
    
    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with mae, rmse, r2
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models.
        
        Returns:
            DataFrame with model names and CV scores
        
        Example:
            >>> summary = registry.get_model_summary()
            >>> print(summary)
        """
        if not self.cv_scores:
            return pd.DataFrame()
        
        return pd.DataFrame({
            "model": list(self.cv_scores.keys()),
            "cv_mae": list(self.cv_scores.values()),
        }).sort_values("cv_mae")
