"""Model training and registry for SegmentAI

Provides: ModelRegistry for training KMeans, DBSCAN, Hierarchical, GMM
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from src.utils import timer, validate_positive


logger = logging.getLogger("segmentai")


class ModelRegistry:
    """Registry for training and managing multiple clustering models.
    
    Supported models:
    - KMeans (centroid-based clustering)
    - DBSCAN (density-based clustering)
    - Hierarchical (agglomerative clustering)
    - GMM (Gaussian Mixture Model - probabilistic clustering)
    
    Attributes:
        models: Dictionary of trained models
        best_model_name: Name of best performing model (by silhouette score)
        scores: Silhouette scores for each model
        cluster_labels: Cluster assignments for each model
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.train_kmeans(X, n_clusters=3)
        >>> registry.train_dbscan(X, eps=0.5, min_samples=5)
        >>> labels = registry.predict(X, "kmeans")
    """
    
    def __init__(self):
        """Initialize empty model registry."""
        self.models = {}
        self.best_model_name = None
        self.scores = {}
        self.cluster_labels = {}
        
        logger.info("Initialized ModelRegistry")
    
    @timer
    def train_kmeans(
        self,
        X: pd.DataFrame,
        n_clusters: int = 3,
        max_iter: int = 300,
        n_init: int = 10
    ) -> Dict[str, Any]:
        """Train KMeans clustering model.
        
        Args:
            X: Training features
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            n_init: Number of initializations
        
        Returns:
            Dictionary with clustering metrics
        
        Raises:
            ValueError: If n_clusters is invalid
            RuntimeError: If training fails
        
        Example:
            >>> metrics = registry.train_kmeans(X, n_clusters=3)
            >>> print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
        """
        validate_positive(n_clusters, "n_clusters")
        
        if n_clusters >= len(X):
            raise ValueError(f"n_clusters ({n_clusters}) must be < n_samples ({len(X)})")
        
        logger.info(f"Training KMeans (n_clusters={n_clusters}, n_init={n_init})")
        
        try:
            model = KMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                n_init=n_init,
                random_state=42
            )
            labels = model.fit_predict(X)
            
            # Store model and labels
            self.models["kmeans"] = model
            self.cluster_labels["kmeans"] = labels
            
            # Compute metrics
            metrics = self._compute_metrics(X, labels)
            metrics["n_clusters"] = n_clusters
            metrics["inertia"] = model.inertia_
            
            self.scores["kmeans"] = metrics["silhouette_score"]
            
            logger.info(
                f"KMeans trained - Silhouette: {metrics['silhouette_score']:.3f}, "
                f"Inertia: {metrics['inertia']:.2f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"KMeans training failed: {e}")
            raise RuntimeError(f"KMeans training failed: {e}") from e
    
    @timer
    def train_dbscan(
        self,
        X: pd.DataFrame,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ) -> Dict[str, Any]:
        """Train DBSCAN clustering model.
        
        DBSCAN finds clusters of arbitrary shape and identifies outliers as noise (-1).
        
        Args:
            X: Training features
            eps: Maximum distance between two samples to be in same neighborhood
            min_samples: Minimum samples in neighborhood for core point
            metric: Distance metric
        
        Returns:
            Dictionary with clustering metrics
        
        Example:
            >>> metrics = registry.train_dbscan(X, eps=0.5, min_samples=5)
            >>> print(f"Found {metrics['n_clusters']} clusters, {metrics['n_noise']} noise points")
        """
        validate_positive(eps, "eps")
        validate_positive(min_samples, "min_samples")
        
        logger.info(f"Training DBSCAN (eps={eps}, min_samples={min_samples})")
        
        try:
            model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = model.fit_predict(X)
            
            # Store model and labels
            self.models["dbscan"] = model
            self.cluster_labels["dbscan"] = labels
            
            # Count clusters and noise
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Compute metrics (exclude noise points if present)
            if n_clusters > 1 and n_noise < len(X):
                # Only compute silhouette on non-noise points
                mask = labels != -1
                X_filtered = X.values[mask]
                labels_filtered = labels[mask]
                
                metrics = self._compute_metrics(
                    pd.DataFrame(X_filtered, columns=X.columns),
                    labels_filtered
                )
            else:
                # Can't compute silhouette with <2 clusters or all noise
                metrics = {"silhouette_score": -1.0}
            
            metrics["n_clusters"] = n_clusters
            metrics["n_noise"] = n_noise
            metrics["noise_ratio"] = n_noise / len(X)
            
            self.scores["dbscan"] = metrics["silhouette_score"]
            
            logger.info(
                f"DBSCAN trained - Clusters: {n_clusters}, Noise: {n_noise}, "
                f"Silhouette: {metrics['silhouette_score']:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"DBSCAN training failed: {e}")
            raise RuntimeError(f"DBSCAN training failed: {e}") from e
    
    @timer
    def train_hierarchical(
        self,
        X: pd.DataFrame,
        n_clusters: int = 3,
        linkage: str = 'ward',
        metric: str = 'euclidean'
    ) -> Dict[str, Any]:
        """Train Hierarchical (Agglomerative) clustering model.
        
        Args:
            X: Training features
            n_clusters: Number of clusters
            linkage: Linkage criterion (ward, complete, average, single)
            metric: Distance metric
        
        Returns:
            Dictionary with clustering metrics
        
        Example:
            >>> metrics = registry.train_hierarchical(X, n_clusters=3, linkage='ward')
        """
        validate_positive(n_clusters, "n_clusters")
        
        if n_clusters >= len(X):
            raise ValueError(f"n_clusters ({n_clusters}) must be < n_samples ({len(X)})")
        
        logger.info(f"Training Hierarchical (n_clusters={n_clusters}, linkage={linkage})")
        
        try:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=metric if linkage != 'ward' else None  # ward requires euclidean
            )
            labels = model.fit_predict(X)
            
            # Store model and labels
            self.models["hierarchical"] = model
            self.cluster_labels["hierarchical"] = labels
            
            # Compute metrics
            metrics = self._compute_metrics(X, labels)
            metrics["n_clusters"] = n_clusters
            metrics["linkage"] = linkage
            
            self.scores["hierarchical"] = metrics["silhouette_score"]
            
            logger.info(
                f"Hierarchical trained - Silhouette: {metrics['silhouette_score']:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Hierarchical training failed: {e}")
            raise RuntimeError(f"Hierarchical training failed: {e}") from e
    
    @timer
    def train_gmm(
        self,
        X: pd.DataFrame,
        n_components: int = 3,
        covariance_type: str = 'full',
        max_iter: int = 100
    ) -> Dict[str, Any]:
        """Train Gaussian Mixture Model (probabilistic clustering).
        
        Args:
            X: Training features
            n_components: Number of Gaussian components (clusters)
            covariance_type: Type of covariance (full, tied, diag, spherical)
            max_iter: Maximum iterations
        
        Returns:
            Dictionary with clustering metrics
        
        Example:
            >>> metrics = registry.train_gmm(X, n_components=3)
        """
        validate_positive(n_components, "n_components")
        
        if n_components >= len(X):
            raise ValueError(f"n_components ({n_components}) must be < n_samples ({len(X)})")
        
        logger.info(f"Training GMM (n_components={n_components}, covariance={covariance_type})")
        
        try:
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                max_iter=max_iter,
                random_state=42
            )
            model.fit(X)
            labels = model.predict(X)
            
            # Store model and labels
            self.models["gmm"] = model
            self.cluster_labels["gmm"] = labels
            
            # Compute metrics
            metrics = self._compute_metrics(X, labels)
            metrics["n_components"] = n_components
            metrics["bic"] = model.bic(X)
            metrics["aic"] = model.aic(X)
            
            self.scores["gmm"] = metrics["silhouette_score"]
            
            logger.info(
                f"GMM trained - Silhouette: {metrics['silhouette_score']:.3f}, "
                f"BIC: {metrics['bic']:.2f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"GMM training failed: {e}")
            raise RuntimeError(f"GMM training failed: {e}") from e
    
    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            X: Features to cluster
            model_name: Name of trained model to use
        
        Returns:
            Cluster labels
        
        Raises:
            ValueError: If model not found
        
        Example:
            >>> labels = registry.predict(X_new, "kmeans")
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Handle different prediction APIs
        if hasattr(model, 'predict'):
            return model.predict(X)
        elif hasattr(model, 'fit_predict'):
            # DBSCAN doesn't have predict(), need to use fit_predict on new data
            logger.warning(f"{model_name} doesn't support predict(), using fit_predict()")
            return model.fit_predict(X)
        else:
            raise RuntimeError(f"Model {model_name} doesn't support prediction")
    
    def get_best_model(self) -> str:
        """Get name of best performing model by silhouette score.
        
        Returns:
            Name of best model
        
        Raises:
            RuntimeError: If no models trained
        """
        if not self.scores:
            raise RuntimeError("No models trained yet")
        
        self.best_model_name = max(self.scores, key=self.scores.get)
        
        logger.info(
            f"Best model: {self.best_model_name} "
            f"(silhouette={self.scores[self.best_model_name]:.3f})"
        )
        
        return self.best_model_name
    
    def save(self, model_name: str, filepath: str) -> None:
        """Save trained model to disk.
        
        Args:
            model_name: Name of model to save
            filepath: Output file path
        
        Raises:
            ValueError: If model not found
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], filepath)
        
        logger.info(f"Saved {model_name} to {filepath}")
    
    def load(self, model_name: str, filepath: str) -> None:
        """Load trained model from disk.
        
        Args:
            model_name: Name to assign to loaded model
            filepath: Model file path
        
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.models[model_name] = joblib.load(filepath)
        
        logger.info(f"Loaded {model_name} from {filepath}")
    
    def _compute_metrics(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering metrics.
        
        Args:
            X: Features
            labels: Cluster labels
        
        Returns:
            Dictionary with metrics
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters < 2:
            return {"silhouette_score": -1.0}
        
        try:
            silhouette = silhouette_score(X, labels)
        except:
            silhouette = -1.0
        
        return {"silhouette_score": silhouette}
