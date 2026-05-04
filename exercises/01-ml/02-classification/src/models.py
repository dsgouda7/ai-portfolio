"""Model training with experiment framework for FaceAI

This module provides:
- Abstract Classifier interface for plug-and-play models
- Concrete implementations: LogisticRegression, SVM, RandomForest (with TODOs)
- ExperimentRunner for comparing multiple models
- Immediate feedback with rich console output

Learning objectives:
1. Implement Logistic Regression/SVM/RandomForest with cross-validation
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
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger("faceai")
console = Console()


@dataclass
class ModelConfig:
    """Configuration for model training."""
    cv_folds: int = 5
    random_state: int = 42
    verbose: bool = True



class Classifier(ABC):
    """Abstract base class for all classifiers.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement train() and predict() methods.
    """
    
    def __init__(self, name: str):
        """Initialize classifier with name for display."""
        self.name = name
        self.model = None
        self.metrics = {}
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, config: ModelConfig) -> Dict[str, float]:
        """Train model and return metrics with immediate console feedback.
        
        Args:
            X: Training features
            y: Training labels
            config: Training configuration
        
        Returns:
            Dictionary with metrics: {"accuracy": float, "cv_accuracy": float, ...}
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (if supported)."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.name} does not support predict_proba")
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Classifier":
        """Load trained model from disk."""
        model = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = model
        return instance



class LogisticRegressor(Classifier):
    """Logistic Regression for multi-class classification.
    
    Logistic Regression uses sigmoid function: P(y=1|x) = 1/(1 + e^(-wx))
    For multi-class: uses softmax + multinomial loss
    Regularization: C parameter (smaller C = more regularization)
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """Initialize Logistic Regression classifier.
        
        Args:
            C: Inverse regularization strength (smaller = more regularization)
            max_iter: Maximum iterations for solver
        """
        super().__init__(f"Logistic (C={C})")
        self.C = C
        self.max_iter = max_iter
    
    def train(self, X: np.ndarray, y: np.ndarray, config: ModelConfig) -> Dict[str, float]:
        """TODO: Implement Logistic Regression training with cross-validation and metrics."""
        raise NotImplementedError("Implement Logistic Regression training")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained Logistic Regression model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class SVMClassifier(Classifier):
    """Support Vector Machine classifier with kernel trick.
    
    SVM finds optimal hyperplane that maximizes margin between classes.
    Kernel trick: Projects data to higher dimensions for non-linear boundaries
    - 'linear': Works well for linearly separable data
    - 'rbf': Radial basis function for complex boundaries
    - 'poly': Polynomial kernel for polynomial relationships
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale'):
        """Initialize SVM classifier.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale' or 'auto' or float)
        """
        super().__init__(f"SVM (C={C}, {kernel})")
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
    
    def train(self, X: np.ndarray, y: np.ndarray, config: ModelConfig) -> Dict[str, float]:
        """TODO: Implement SVM training with cross-validation and support vector metrics."""
        raise NotImplementedError("Implement SVM training")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained SVM model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class RandomForestClassifier(Classifier):
    """Random Forest ensemble classifier.
    
    Random Forest builds multiple decision trees on random subsets:
    1. Bootstrap sampling: Each tree trained on random sample (with replacement)
    2. Random features: Each split considers random subset of features
    3. Voting: Final prediction is majority vote across all trees
    
    Key advantages:
    - Handles non-linear relationships naturally
    - Resistant to overfitting (with enough trees)
    - Provides feature importance scores
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        """Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in forest
            max_depth: Maximum depth of each tree
        """
        super().__init__(f"RandomForest (n={n_estimators}, d={max_depth})")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def train(self, X: np.ndarray, y: np.ndarray, config: ModelConfig) -> Dict[str, float]:
        """TODO: Implement Random Forest training with cross-validation."""
        raise NotImplementedError("Implement Random Forest training")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained Random Forest model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class ExperimentRunner:
    """Run experiments with multiple classifiers and compare results.
    
    Provides plug-and-play framework for trying different models:
    1. Register classifiers to try
    2. Run all experiments with immediate feedback
    3. Print leaderboard sorted by performance
    
    Example:
        >>> runner = ExperimentRunner()
        >>> runner.register("Logistic (C=0.1)", LogisticRegressor(C=0.1))
        >>> runner.register("Logistic (C=1.0)", LogisticRegressor(C=1.0))
        >>> runner.register("SVM (RBF)", SVMClassifier(C=1.0, kernel='rbf'))
        >>> runner.run_experiment(X_train, y_train, ModelConfig())
        >>> runner.print_leaderboard()
    """
    
    def __init__(self):
        """Initialize empty experiment runner."""
        self.classifiers: Dict[str, Classifier] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, classifier: Classifier):
        """Register a classifier to try in experiments.
        
        Args:
            name: Display name for results
            classifier: Classifier instance to train
        """
        self.classifiers[name] = classifier
        console.print(f"Registered: {name}", style="dim")
    
    def run_experiment(self, X: np.ndarray, y: np.ndarray, config: ModelConfig):
        """TODO: Train all registered classifiers and store results."""
        raise NotImplementedError("Implement experiment runner")
    
    def print_leaderboard(self):
        """TODO: Display sorted leaderboard table comparing all trained models."""
        raise NotImplementedError("Implement leaderboard")
    
    def get_best_model(self) -> Classifier:
        """Return classifier with highest CV accuracy."""
        if not self.results:
            raise ValueError("No experiments run yet")
        best_result = max(self.results, key=lambda x: x["cv_accuracy"])
        return self.classifiers[best_result["model"]]

