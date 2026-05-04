"""Model training with ensemble methods for EnsembleAI

This module provides:
- Abstract EnsembleModel interface for plug-and-play ensembles
- Concrete implementations: Bagging, Boosting, Stacking (with TODOs)
- ExperimentRunner for comparing multiple ensemble methods
- Immediate feedback with rich console output

Learning objectives:
1. Implement Bagging with bootstrap sampling and OOB error
2. Implement Boosting with sequential weighting (AdaBoost/Gradient Boosting)
3. Implement Stacking with meta-learner training
4. Compare ensemble diversity and performance
5. Understand bias-variance tradeoff through experimentation
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
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger("ensembleai")
console = Console()


@dataclass
class ModelConfig:
    """Configuration for ensemble training."""
    cv_folds: int = 5
    random_state: int = 42
    verbose: bool = True


class EnsembleModel(ABC):
    """Abstract base class for all ensemble models.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement train() and predict() methods.
    
    Ensemble methods combine multiple weak learners to create
    a strong learner. Key concepts:
    - Bagging: Reduces variance via bootstrap + aggregation
    - Boosting: Reduces bias via sequential error correction
    - Stacking: Combines diverse models via meta-learner
    """
    
    def __init__(self, name: str):
        """Initialize ensemble model with name for display."""
        self.name = name
        self.model = None
        self.metrics = {}
        self.base_learners = []
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """Train ensemble model and return metrics with immediate console feedback.
        
        Args:
            X: Training features
            y: Training labels (classification)
            config: Training configuration
        
        Returns:
            Dictionary with metrics: {"accuracy": float, "cv_accuracy": float, ...}
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.name} does not support probability prediction")
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        """Load trained model from disk."""
        model = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = model
        return instance


class BaggingEnsemble(EnsembleModel):
    """Bagging (Bootstrap Aggregating) ensemble classifier.
    
    Bagging reduces variance by:
    1. Creating bootstrap samples (random sampling with replacement)
    2. Training independent models on each sample
    3. Aggregating predictions (voting for classification)
    
    Key features:
    - Parallel training (base learners are independent)
    - Out-of-bag (OOB) error estimation (free validation)
    - Works best with high-variance models (e.g., deep decision trees)
    """
    
    def __init__(self, n_estimators: int = 50, max_samples: float = 1.0, base_estimator=None):
        """Initialize Bagging ensemble.
        
        Args:
            n_estimators: Number of base learners
            max_samples: Fraction of samples for each bootstrap (0.0-1.0)
            base_estimator: Base learner (default: DecisionTreeClassifier)
        """
        super().__init__(f"Bagging (n={n_estimators}, samples={max_samples:.1f})")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=10)
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Create BaggingClassifier with oob_score=True, fit, compute metrics, return dict"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement Bagging training - see TODO above")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained Bagging ensemble."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class BoostingEnsemble(EnsembleModel):
    """Gradient Boosting ensemble classifier.
    
    Boosting reduces bias by:
    1. Training models sequentially
    2. Each model focuses on correcting previous errors
    3. Combining via weighted sum
    
    Key features:
    - Sequential training (models depend on previous predictions)
    - Learning rate controls contribution of each tree
    - Works best with shallow trees (weak learners)
    - Powerful but can overfit without regularization
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        """Initialize Gradient Boosting ensemble.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Shrinkage factor (lower = more robust but slower)
            max_depth: Maximum tree depth (3-5 typical for boosting)
        """
        super().__init__(f"GradBoost (n={n_estimators}, lr={learning_rate}, d={max_depth})")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Create GradientBoostingClassifier, fit, compute CV/metrics, extract feature_importances_"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement Boosting training - see TODO above")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained Boosting ensemble."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class StackingEnsemble(EnsembleModel):
    """Stacking (Stacked Generalization) ensemble classifier.
    
    Stacking combines diverse models by:
    1. Training multiple diverse base models
    2. Using base model predictions as features
    3. Training meta-learner on these features
    
    Key features:
    - Leverages diversity of different model types
    - Meta-learner learns optimal combination
    - Two-level architecture (base + meta)
    - Usually outperforms single models and voting
    """
    
    def __init__(self, meta_learner_type: str = "logistic"):
        """Initialize Stacking ensemble.
        
        Args:
            meta_learner_type: Type of meta-learner ("logistic", "rf", "gbm")
        """
        super().__init__(f"Stacking (meta={meta_learner_type})")
        self.meta_learner_type = meta_learner_type
        
        # Define diverse base learners
        self.base_learners = [
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
            ("gbm", GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
            ("dt", DecisionTreeClassifier(max_depth=10, random_state=42))
        ]
    
    def train(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> Dict[str, float]:
        """TODO: Create StackingClassifier with meta-learner, fit with CV, compute metrics"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement Stacking training - see TODO above")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained Stacking ensemble."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class ExperimentRunner:
    """Run experiments with multiple ensemble methods and compare results.
    
    Provides plug-and-play framework for trying different ensembles:
    1. Register ensemble models to try
    2. Run all experiments with immediate feedback
    3. Print leaderboard sorted by performance
    4. Analyze base learner diversity
    
    Example:
        >>> runner = ExperimentRunner()
        >>> runner.register("Bagging-50", BaggingEnsemble(n_estimators=50))
        >>> runner.register("Boosting-100", BoostingEnsemble(n_estimators=100))
        >>> runner.register("Stacking", StackingEnsemble(meta_learner_type="logistic"))
        >>> runner.run_experiment(X_train, y_train, ModelConfig())
        >>> runner.print_leaderboard()
    """
    
    def __init__(self):
        """Initialize empty experiment runner."""
        self.ensembles: Dict[str, EnsembleModel] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, ensemble: EnsembleModel):
        """Register an ensemble model to try in experiments.
        
        Args:
            name: Display name for results
            ensemble: EnsembleModel instance to train
        """
        self.ensembles[name] = ensemble
        console.print(f"Registered: {name}", style="dim")
    
    def run_experiment(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig):
        """TODO: Loop through registered ensembles, train each, store results with metrics"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner - see TODO above")
    
    def print_leaderboard(self):
        """TODO: Create Rich table with results sorted by CV accuracy, show winner and insights"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement leaderboard - see TODO above")
    
    def get_best_model(self) -> EnsembleModel:
        """Return ensemble with highest CV accuracy."""
        if not self.results:
            raise ValueError("No experiments run yet")
        best_result = max(self.results, key=lambda x: x.get("cv_accuracy", 0))
        return self.ensembles[best_result["model"]]
    
    def analyze_diversity(self, X: pd.DataFrame, y: pd.Series):
        """
        TODO (BONUS): Analyze base learner diversity and agreement
        
        Steps:
        1. Print header: console.print("\n[bold cyan]🔍 DIVERSITY ANALYSIS[/bold cyan]")
        2. For each ensemble model:
           - Get predictions from each base learner
           - Calculate pairwise agreement (what % of predictions match)
           - Print diversity metric (lower agreement = higher diversity)
        3. Create visualization showing:
           - Agreement matrix (heatmap of pairwise agreements)
           - Diversity score (1 - mean agreement)
        
        Time estimate: 30-40 minutes (optional advanced feature)
        
        Theory:
        - High diversity → ensembles learn complementary patterns
        - Low diversity → limited benefit from ensembling
        - Bagging: moderate diversity (same algorithm, different data)
        - Boosting: high diversity (sequential error correction)
        - Stacking: highest diversity (different algorithms)
        """
        # TODO (BONUS): Your implementation here
        pass
