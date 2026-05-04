"""Model training with experiment framework for SegmentAI

This module provides:
- Abstract UnsupervisedModel interface for plug-and-play clustering
- Concrete implementations: KMeans, DBSCAN, PCA (with TODOs)
- ExperimentRunner for comparing multiple clustering algorithms
- Immediate feedback with rich console output

Learning objectives:
1. Implement KMeans/DBSCAN/PCA with silhouette score evaluation
2. Compare clustering algorithms using plug-and-play registry pattern
3. See clustering metrics immediately after each model trains
4. Experiment with hyperparameters and observe cluster quality
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

logger = logging.getLogger("segmentai")
console = Console()


@dataclass
class ClusterConfig:
    """Configuration for clustering."""
    random_state: int = 42
    verbose: bool = True



class UnsupervisedModel(ABC):
    """Abstract base class for all unsupervised learning models.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement fit() and transform() methods.
    """
    
    def __init__(self, name: str):
        """Initialize model with name for display."""
        self.name = name
        self.model = None
        self.labels = None
        self.metrics = {}
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, config: ClusterConfig) -> Dict[str, Any]:
        """Fit model and return metrics with immediate console feedback.
        
        Args:
            X: Training features (scaled)
            config: Clustering configuration
        
        Returns:
            Dictionary with metrics (silhouette_score, n_clusters, etc.)
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted model (predict clusters or reduce dimensions)."""
        pass
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved {self.name} to {path}")
    
    @classmethod
    def load(cls, path: str) -> "UnsupervisedModel":
        """Load trained model from disk."""
        model = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = model
        return instance


class KMeansClusterer(UnsupervisedModel):
    """KMeans clustering with centroid-based partitioning.
    
    KMeans partitions data into K clusters by:
    1. Initialize K random centroids
    2. Assign each point to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence
    
    Metrics:
    - Silhouette score: [-1, 1] - measures cluster cohesion/separation
    - Inertia: Sum of squared distances to centroids (lower is better)
    """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, n_init: int = 10):
        """Initialize KMeans clusterer.
        
        Args:
            n_clusters: Number of clusters K
            max_iter: Maximum iterations for convergence
            n_init: Number of random initializations (best kept)
        """
        super().__init__(f"KMeans (K={n_clusters})")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
    
    def fit(self, X: pd.DataFrame, config: ClusterConfig) -> Dict[str, Any]:
        """
        TODO: Implement KMeans clustering with evaluation metrics
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement KMeans clustering - see TODO above")
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


class DBSCANClusterer(UnsupervisedModel):
    """DBSCAN density-based clustering.
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
    - Finds clusters of arbitrary shape
    - Identifies outliers as noise (label = -1)
    - No need to specify number of clusters
    
    Key parameters:
    - eps: Maximum distance for points to be neighbors
    - min_samples: Minimum points to form dense region
    
    Metrics:
    - Silhouette score (excluding noise points)
    - Number of clusters found
    - Noise ratio (percentage of outliers)
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """Initialize DBSCAN clusterer.
        
        Args:
            eps: Maximum distance between two samples to be neighbors
            min_samples: Minimum samples in neighborhood for core point
        """
        super().__init__(f"DBSCAN (ε={eps}, min={min_samples})")
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(self, X: pd.DataFrame, config: ClusterConfig) -> Dict[str, Any]:
        """
        TODO: Implement DBSCAN clustering with noise handling
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement DBSCAN clustering - see TODO above")
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """DBSCAN doesn't support predict() for new data - must refit."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        logger.warning("DBSCAN doesn't support transform() - returning training labels")
        return self.labels


class PCAReducer(UnsupervisedModel):
    """PCA dimensionality reduction.
    
    PCA (Principal Component Analysis):
    - Linear dimensionality reduction via variance maximization
    - Projects data onto principal components (orthogonal directions)
    - Preserves most variance with fewer dimensions
    
    Use cases:
    - Visualization (reduce to 2D/3D)
    - Noise reduction
    - Speed up clustering
    
    Metrics:
    - Explained variance ratio per component
    - Cumulative explained variance
    """
    
    def __init__(self, n_components: int = 2):
        """Initialize PCA reducer.
        
        Args:
            n_components: Number of principal components to keep
        """
        super().__init__(f"PCA (n={n_components})")
        self.n_components = n_components
    
    def fit(self, X: pd.DataFrame, config: ClusterConfig) -> Dict[str, Any]:
        """
        TODO: Implement PCA dimensionality reduction
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement PCA dimensionality reduction - see TODO above")
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data to principal component space."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.transform(X)



class ExperimentRunner:
    """Run experiments with multiple clustering algorithms and compare results.
    
    Provides plug-and-play framework for comparing clustering approaches:
    1. Register models to try (KMeans with different K, DBSCAN, PCA)
    2. Run all experiments with immediate feedback
    3. Print leaderboard sorted by silhouette score
    
    Example:
        >>> runner = ExperimentRunner()
        >>> runner.register("KMeans (K=3)", KMeansClusterer(n_clusters=3))
        >>> runner.register("KMeans (K=5)", KMeansClusterer(n_clusters=5))
        >>> runner.register("DBSCAN", DBSCANClusterer(eps=0.5, min_samples=5))
        >>> runner.run_experiment(X_scaled, ClusterConfig())
        >>> runner.print_leaderboard()
    """
    
    def __init__(self):
        """Initialize empty experiment runner."""
        self.models: Dict[str, UnsupervisedModel] = {}
        self.results: List[Dict[str, Any]] = []
    
    def register(self, name: str, model: UnsupervisedModel):
        """Register a model to try in experiments.
        
        Args:
            name: Display name for results
            model: UnsupervisedModel instance to train
        """
        self.models[name] = model
        console.print(f"Registered: {name}", style="dim")
    
    def run_experiment(self, X: pd.DataFrame, config: ClusterConfig):
        """
        TODO: Run all registered models and compare
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner - see TODO above")
    
    def print_leaderboard(self):
        """
        TODO: Print leaderboard sorted by silhouette score
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement leaderboard - see TODO above")
    
    def get_best_model(self) -> str:
        """Get name of best performing model by silhouette score.
        
        Returns:
            Name of best model
        
        Raises:
            RuntimeError: If no models trained
        """
        if not self.results:
            raise RuntimeError("No models trained yet")
        
        best = max(self.results, key=lambda x: x.get("silhouette_score", -999))
        return best["model"]
    
    def find_optimal_k(self, X: pd.DataFrame, k_range: range, config: ClusterConfig):
        """
        TODO: Elbow method - find optimal K by plotting inertia vs K
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement elbow method - see TODO above")

