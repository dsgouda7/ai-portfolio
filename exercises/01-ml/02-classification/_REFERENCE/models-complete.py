"""Model training and registry for FaceAI

Provides: ModelRegistry for training LogisticRegression, SVM, RandomForest with persistence
"""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.utils import timer, validate_positive


logger = logging.getLogger("faceai")


class ModelRegistry:
    """Registry for training and managing multiple classification models.
    
    Supported models:
    - Logistic Regression (multi-class)
    - Support Vector Machine (SVM)
    - Random Forest
    
    Attributes:
        models: Dictionary of trained models
        best_model_name: Name of best performing model
        cv_scores: Cross-validation scores for each model
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.train_logistic_regression(X_train, y_train, C=1.0)
        >>> registry.train_svm(X_train, y_train, C=1.0, kernel='rbf')
        >>> predictions = registry.predict(X_test, "logistic_regression")
    """
    
    def __init__(self):
        """Initialize empty model registry."""
        self.models = {}
        self.best_model_name = None
        self.cv_scores = {}
        
        logger.info("Initialized ModelRegistry")
    
    @timer
    def train_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
        max_iter: int = 1000,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Train Logistic Regression model.
        
        Args:
            X: Training features
            y: Training labels
            C: Inverse of regularization strength (smaller = more regularization)
            max_iter: Maximum iterations for solver
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with training metrics (accuracy, cv_accuracy)
        
        Raises:
            ValueError: If C is not positive
            RuntimeError: If training fails
        
        Example:
            >>> metrics = registry.train_logistic_regression(X_train, y_train, C=1.0)
            >>> print(f"CV Accuracy: {metrics['cv_accuracy']:.3f}")
        """
        validate_positive(C, "C")
        
        logger.info(f"Training LogisticRegression (C={C}, max_iter={max_iter})")
        
        try:
            model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                multi_class='multinomial',
                solver='lbfgs',
                random_state=42
            )
            model.fit(X, y)
            
            # Cross-validation
            cv_accuracy = self._cross_validate(model, X, y, cv_folds)
            
            # Store model
            self.models["logistic_regression"] = model
            self.cv_scores["logistic_regression"] = cv_accuracy
            
            # Training metrics
            train_accuracy = model.score(X, y)
            
            metrics = {
                "train_accuracy": float(train_accuracy),
                "cv_accuracy": float(cv_accuracy),
            }
            
            logger.info(
                f"LogisticRegression trained - "
                f"CV Acc: {cv_accuracy:.3f}, Train Acc: {train_accuracy:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"LogisticRegression training failed: {e}")
            raise RuntimeError(f"LogisticRegression training failed: {e}") from e
    
    @timer
    def train_svm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Train Support Vector Machine model.
        
        Args:
            X: Training features
            y: Training labels
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale' or 'auto' or float)
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_svm(X_train, y_train, C=1.0, kernel='rbf')
        """
        validate_positive(C, "C")
        
        logger.info(f"Training SVM (C={C}, kernel={kernel}, gamma={gamma})")
        
        try:
            model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                random_state=42
            )
            model.fit(X, y)
            
            # Cross-validation
            cv_accuracy = self._cross_validate(model, X, y, cv_folds)
            
            # Store model
            self.models["svm"] = model
            self.cv_scores["svm"] = cv_accuracy
            
            # Training metrics
            train_accuracy = model.score(X, y)
            
            metrics = {
                "train_accuracy": float(train_accuracy),
                "cv_accuracy": float(cv_accuracy),
                "n_support_vectors": int(model.n_support_.sum()),
            }
            
            logger.info(
                f"SVM trained - CV Acc: {cv_accuracy:.3f}, "
                f"Support vectors: {metrics['n_support_vectors']}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"SVM training failed: {e}")
            raise RuntimeError(f"SVM training failed: {e}") from e
    
    @timer
    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 10,
        cv_folds: int = 5,
        **kwargs
    ) -> Dict[str, float]:
        """Train Random Forest model.
        
        Args:
            X: Training features
            y: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            cv_folds: Number of cross-validation folds
            **kwargs: Additional RandomForest parameters
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_random_forest(
            ...     X_train, y_train,
            ...     n_estimators=100,
            ...     max_depth=10
            ... )
        """
        logger.info(
            f"Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})"
        )
        
        try:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                **kwargs
            )
            model.fit(X, y)
            
            # Cross-validation
            cv_accuracy = self._cross_validate(model, X, y, cv_folds)
            
            # Store model
            self.models["random_forest"] = model
            self.cv_scores["random_forest"] = cv_accuracy
            
            # Training metrics
            train_accuracy = model.score(X, y)
            
            metrics = {
                "train_accuracy": float(train_accuracy),
                "cv_accuracy": float(cv_accuracy),
            }
            
            logger.info(
                f"RandomForest trained - CV Acc: {cv_accuracy:.3f}, "
                f"Train Acc: {train_accuracy:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"RandomForest training failed: {e}")
            raise RuntimeError(f"RandomForest training failed: {e}") from e
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
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
            >>> predictions = registry.predict(X_test, "logistic_regression")
        """
        if not self.models:
            raise RuntimeError("No models trained. Train a model first.")
        
        if model_name is None:
            model_name = self.get_best_model_name()
        
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(self.models.keys())}"
            )
        
        model = self.models[model_name]
        return model.predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Features to predict on
            model_name: Name of model to use (None = best model)
        
        Returns:
            Array of class probabilities
        
        Raises:
            ValueError: If model doesn't support predict_proba
        """
        if not self.models:
            raise RuntimeError("No models trained. Train a model first.")
        
        if model_name is None:
            model_name = self.get_best_model_name()
        
        model = self.models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model '{model_name}' does not support predict_proba")
        
        return model.predict_proba(X)
    
    def get_best_model_name(self) -> str:
        """Get name of model with highest cross-validation accuracy.
        
        Returns:
            Name of best model
        
        Raises:
            RuntimeError: If no models trained
        """
        if not self.cv_scores:
            raise RuntimeError("No models trained with cross-validation")
        
        best_name = max(self.cv_scores, key=self.cv_scores.get)
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
            >>> registry.save_model("logistic_regression", "models/lr_v1.pkl")
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
            >>> registry.load_model("logistic_regression", "models/lr_v1.pkl")
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.models[model_name] = joblib.load(model_path)
        logger.info(f"Loaded {model_name} from {path}")
    
    def _cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int
    ) -> float:
        """Perform cross-validation and return mean accuracy.
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Labels
            cv_folds: Number of folds
        
        Returns:
            Mean accuracy across folds
        """
        scores = cross_val_score(
            model, X, y,
            cv=cv_folds,
            scoring='accuracy'
        )
        accuracy = scores.mean()
        
        return float(accuracy)
    
    def get_model_summary(self) -> dict:
        """Get summary of all trained models.
        
        Returns:
            Dictionary with model names and CV scores
        
        Example:
            >>> summary = registry.get_model_summary()
            >>> print(summary)
        """
        if not self.cv_scores:
            return {}
        
        return {
            "models": list(self.cv_scores.keys()),
            "cv_scores": self.cv_scores,
            "best_model": self.get_best_model_name() if self.cv_scores else None,
        }
