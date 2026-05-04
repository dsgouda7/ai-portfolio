"""Model training with experiment framework for FlixAI

This module provides:
- Abstract Recommender interface for plug-and-play models
- Concrete implementations: Collaborative Filtering, Matrix Factorization (with TODOs)
- ExperimentRunner for comparing multiple recommendation algorithms
- Immediate feedback with rich console output

Learning objectives:
1. Implement collaborative filtering with user-item similarity
2. Implement matrix factorization using SVD decomposition
3. Compare models using plug-and-play registry pattern
4. See results immediately after each model trains
5. Experiment with different hyperparameters and observe impact
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("flixai")
console = Console()


@dataclass
class ModelConfig:
    """Configuration for model training."""
    random_state: int = 42
    verbose: bool = True


class Recommender(ABC):
    """Abstract base class for all recommenders.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement train() and recommend() methods.
    """
    
    def __init__(self, name: str):
        """Initialize recommender with name for display."""
        self.name = name
        self.model = None
        self.metrics = {}
        self.user_item_matrix = None
    
    @abstractmethod
    def train(
        self,
        ratings: pd.DataFrame,
        user_item_matrix: pd.DataFrame,
        config: ModelConfig
    ) -> Dict[str, float]:
        """Train recommender and return metrics with immediate console feedback.
        
        Args:
            ratings: DataFrame with user_id, item_id, rating columns
            user_item_matrix: User-item matrix (users × items)
            config: Training configuration
        
        Returns:
            Dictionary with metrics: {"rmse": float, "mae": float, "coverage": float}
        """
        pass
    
    @abstractmethod
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Generate top-k recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            k: Number of recommendations
            exclude_seen: Whether to exclude items user has already rated
        
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        pass
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'user_item_matrix': self.user_item_matrix,
            'metrics': self.metrics
        }, path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Recommender":
        """Load trained model from disk."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = data['model']
        instance.user_item_matrix = data['user_item_matrix']
        instance.metrics = data.get('metrics', {})
        return instance


class CollaborativeFilteringRecommender(Recommender):
    """User-based collaborative filtering recommender.
    
    Finds similar users and recommends items they liked.
    Uses cosine similarity to measure user-user similarity.
    """
    
    def __init__(self, n_neighbors: int = 20):
        """Initialize collaborative filtering recommender.
        
        Args:
            n_neighbors: Number of similar users to consider
        """
        super().__init__(f"Collaborative Filtering (k={n_neighbors})")
        self.n_neighbors = n_neighbors
        self.user_similarity = None
    
    def train(
        self,
        ratings: pd.DataFrame,
        user_item_matrix: pd.DataFrame,
        config: ModelConfig
    ) -> Dict[str, float]:
        """
        TODO: Implement collaborative filtering training
        
        Args:
            ratings: DataFrame with user_id, item_id, rating columns
            user_item_matrix: User-item matrix (users × items)
            config: Training configuration
        
        Returns:
            Dictionary with metrics: {"rmse": float, "mae": float, "coverage": float}
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement collaborative filtering training")
    
    def _predict_rating(self, user_id: int, item_id: int) -> float:
        """
        TODO: Predict rating for user-item pair using k-nearest neighbors
        
        Args:
            user_id: User ID
            item_id: Item ID
        
        Returns:
            Predicted rating (1-5)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement rating prediction")
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Generate top-k recommendations using collaborative filtering.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude already-rated items
        
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if self.user_similarity is None:
            raise ValueError("Model not trained yet")
        
        # Get all item predictions
        n_items = self.user_item_matrix.shape[1]
        predictions = []
        
        for item_id in range(n_items):
            # Skip if user already rated this item
            if exclude_seen and self.user_item_matrix.iloc[user_id, item_id] > 0:
                continue
            
            pred = self._predict_rating(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Sort by predicted rating and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]


class MatrixFactorizationRecommender(Recommender):
    """Matrix factorization recommender using SVD.
    
    Decomposes user-item matrix into user and item latent factors:
    R ≈ U @ Σ @ V^T
    
    Lower-rank approximation captures latent preferences.
    """
    
    def __init__(self, n_factors: int = 50):
        """Initialize matrix factorization recommender.
        
        Args:
            n_factors: Number of latent factors
        """
        super().__init__(f"Matrix Factorization (k={n_factors})")
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.sigma = None
    
    def train(
        self,
        ratings: pd.DataFrame,
        user_item_matrix: pd.DataFrame,
        config: ModelConfig
    ) -> Dict[str, float]:
        """
        TODO: Implement matrix factorization training using SVD
        
        Args:
            ratings: DataFrame with ratings
            user_item_matrix: User-item matrix
            config: Training configuration
        
        Returns:
            Dictionary with metrics
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement matrix factorization training")
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        TODO: Generate top-k recommendations using matrix factorization
        
        Args:
            user_id: User ID to generate recommendations for
            k: Number of recommendations
            exclude_seen: Whether to exclude items user has already rated
        
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement recommendation generation")


class ExperimentRunner:
    """Run experiments with multiple recommenders and compare results.
    
    Provides plug-and-play framework for trying different models:
    1. Register recommenders to try
    2. Run all experiments with immediate feedback
    3. Print leaderboard sorted by performance
    
    Example:
        >>> runner = ExperimentRunner()
        >>> runner.register("CF (k=20)", CollaborativeFilteringRecommender(n_neighbors=20))
        >>> runner.register("MF (k=50)", MatrixFactorizationRecommender(n_factors=50))
        >>> runner.run_experiment(ratings_train, user_item_train, ModelConfig())
        >>> runner.print_leaderboard()
    """
    
    def __init__(self):
        """Initialize empty experiment runner."""
        self.recommenders: Dict[str, Recommender] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, recommender: Recommender):
        """Register a recommender to try in experiments.
        
        Args:
            name: Display name for results
            recommender: Recommender instance to train
        """
        self.recommenders[name] = recommender
        console.print(f"Registered: {name}", style="dim")
    
    def run_experiment(
        self,
        ratings: pd.DataFrame,
        user_item_matrix: pd.DataFrame,
        config: ModelConfig
    ):
        """
        TODO: Run all registered recommenders and compare
        
        Args:
            ratings: DataFrame with ratings
            user_item_matrix: User-item matrix
            config: Training configuration
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner")
    
    def print_leaderboard(self):
        """
        TODO: Print sorted leaderboard of all models
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement leaderboard")
    
    def get_best_model(self) -> Recommender:
        """Return recommender with lowest RMSE."""
        if not self.results:
            raise ValueError("No experiments run yet")
        best_result = min(self.results, key=lambda x: x["rmse"])
        return self.recommenders[best_result["model"]]
    
    def evaluate_recommendations(
        self,
        test_ratings: pd.DataFrame,
        k: int = 10
    ) -> Dict[str, float]:
        """
        TODO: Evaluate recommendation quality on test set
        
        Args:
            test_ratings: Test ratings DataFrame
            k: Number of recommendations
        
        Returns:
            Dictionary with precision and recall metrics
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement recommendation evaluation")
