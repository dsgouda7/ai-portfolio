"""Model training with experiment framework for SmartVal AI

This module provides:
- Abstract Regressor interface for plug-and-play models
- Concrete implementations: Ridge, Lasso, XGBoost (with TODOs)
- ExperimentRunner for comparing multiple models
- Immediate feedback with rich console output

Learning objectives:
1. Implement Ridge/Lasso/XGBoost training with cross-validation
2. Compare models using plug-and-play registry pattern
3. See results immediately after each model trains
4. Experiment with hyperparameters and observe impact
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

logger = logging.getLogger("smartval")
console = Console()


@dataclass
class ModelConfig:
    """Configuration for model training."""
    cv_folds: int = 5
    random_state: int = 42
    verbose: bool = True


class Regressor(ABC):
    """Abstract base class for all regressors.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement train() and predict() methods.
    """
    
    def __init__(self, name: str):
        """Initialize regressor with name for display."""
        self.name = name
        self.model = None
        self.metrics = {}
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """Train model and return metrics with immediate console feedback.
        
        Args:
            X: Training features
            y: Training labels
            config: Training configuration
        
        Returns:
            Dictionary with metrics: {"mae": float, "rmse": float, "r2": float, "cv_mae": float}
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Regressor":
        """Load trained model from disk."""
        model = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = model
        return instance


class RidgeRegressor(Regressor):
    """Ridge regression with L2 regularization.
    
    Ridge adds penalty term: loss = MSE + alpha * sum(coef^2)
    Higher alpha = more regularization = simpler model
    """
    
    def __init__(self, alpha: float = 1.0):
        """Initialize Ridge regressor.
        
        Args:
            alpha: Regularization strength (higher = more regularization)
        """
        super().__init__(f"Ridge (α={alpha})")
        self.alpha = alpha
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Train Ridge regression with cross-validation and return metrics.
        
        📖 **See:** notes/01-ml/01_regression/ch05_regularization/README.md § 2.2 for Ridge (L2) penalty math and cross-validation implementation
        """
        raise NotImplementedError("TODO: Implement Ridge training with cross-validation")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained Ridge model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class LassoRegressor(Regressor):
    """Lasso regression with L1 regularization.
    
    Lasso adds penalty term: loss = MSE + alpha * sum(|coef|)
    L1 penalty drives some coefficients to exactly zero (feature selection)
    """
    
    def __init__(self, alpha: float = 0.1):
        """Initialize Lasso regressor.
        
        Args:
            alpha: Regularization strength
        """
        super().__init__(f"Lasso (α={alpha})")
        self.alpha = alpha
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Train Lasso regression with cross-validation and count non-zero features.
        
        📖 **See:** notes/01-ml/01_regression/ch05_regularization/README.md § 2.3 for Lasso (L1) penalty and automatic feature selection
        """
        raise NotImplementedError("TODO: Implement Lasso training with L1 regularization")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained Lasso model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class XGBoostRegressor(Regressor):
    """XGBoost gradient boosting regressor.
    
    XGBoost builds ensemble of decision trees sequentially,
    where each tree corrects errors of previous trees.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        """Initialize XGBoost regressor.
        
        Args:
            n_estimators: Number of boosting rounds (trees)
            max_depth: Maximum tree depth
            learning_rate: Shrinkage factor for each tree
        """
        super().__init__(f"XGBoost (d={max_depth}, n={n_estimators})")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Train XGBoost gradient boosting regressor with cross-validation.
        
        📖 **See:** notes/01-ml/01_regression/ch07_hyperparameter_tuning/README.md § 1-2 for XGBoost gradient boosting fundamentals and cross-validation
        """
        raise NotImplementedError("TODO: Implement XGBoost training with gradient boosting")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained XGBoost model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class ExperimentRunner:
    """Run experiments with multiple regressors and compare results.
    
    Provides plug-and-play framework for trying different models:
    1. Register regressors to try
    2. Run all experiments with immediate feedback
    3. Print leaderboard sorted by performance
    
    Example:
        >>> runner = ExperimentRunner()
        >>> runner.register("Ridge (α=0.1)", RidgeRegressor(alpha=0.1))
        >>> runner.register("Ridge (α=1.0)", RidgeRegressor(alpha=1.0))
        >>> runner.register("XGBoost", XGBoostRegressor(n_estimators=100))
        >>> runner.run_experiment(X_train, y_train, ModelConfig())
        >>> runner.print_leaderboard()
    """
    
    def __init__(self):
        """Initialize empty experiment runner."""
        self.regressors: Dict[str, Regressor] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, regressor: Regressor):
        """Register a regressor to try in experiments.
        
        Args:
            name: Display name for results
            regressor: Regressor instance to train
        """
        self.regressors[name] = regressor
        console.print(f"Registered: {name}", style="dim")
    
    def run_experiment(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig):
        """TODO: Train all registered models and store results for comparison.
        
        📖 **See:** notes/01-ml/01_regression/ch06_metrics/README.md § 6 for cross-validation strategy and multiple model comparison framework
        """
        raise NotImplementedError("TODO: Implement experiment runner loop")
    
    def print_leaderboard(self):
        """TODO: Display sorted model comparison table with metrics.
        
        📖 **See:** notes/01-ml/01_regression/ch06_metrics/README.md § 2.1 for metrics journey table format and model comparison best practices
        """
        raise NotImplementedError("TODO: Implement leaderboard display with Rich table")
    
    def get_best_model(self) -> Regressor:
        """Return regressor with lowest CV MAE."""
        if not self.results:
            raise ValueError("No experiments run yet")
        best_result = min(self.results, key=lambda x: x["cv_mae"])
        return self.regressors[best_result["model"]]
