"""Feature engineering for SegmentAI

Provides: Scaling, PCA, and UMAP for dimensionality reduction
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap


logger = logging.getLogger("segmentai")


class FeatureEngineer:
    """Feature engineering pipeline for unsupervised learning.
    
    Pipeline stages:
    1. Standardization (zero mean, unit variance) - CRITICAL for distance-based clustering
    2. PCA for variance-based dimensionality reduction
    3. UMAP for non-linear dimensionality reduction (visualization)
    
    Attributes:
        scaler: StandardScaler transformer
        pca: PCA transformer
        umap_reducer: UMAP transformer
        feature_names: Names of features after engineering
    
    Example:
        >>> engineer = FeatureEngineer(n_components_pca=2, n_neighbors_umap=15)
        >>> X_transformed = engineer.fit_transform(X)
        >>> X_umap = engineer.transform_umap(X_transformed)
    """
    
    def __init__(
        self,
        scale_features: bool = True,
        n_components_pca: Optional[int] = None,
        n_neighbors_umap: int = 15,
        min_dist_umap: float = 0.1
    ):
        """Initialize feature engineer.
        
        Args:
            scale_features: Whether to standardize features
            n_components_pca: Number of PCA components (None = no PCA)
            n_neighbors_umap: UMAP n_neighbors parameter
            min_dist_umap: UMAP min_dist parameter
        
        Raises:
            ValueError: If parameters are invalid
        """
        if n_components_pca is not None and n_components_pca < 1:
            raise ValueError(f"n_components_pca must be >= 1, got {n_components_pca}")
        if n_neighbors_umap < 2:
            raise ValueError(f"n_neighbors_umap must be >= 2, got {n_neighbors_umap}")
        if not (0.0 <= min_dist_umap <= 1.0):
            raise ValueError(f"min_dist_umap must be in [0, 1], got {min_dist_umap}")
        
        self.scale_features = scale_features
        self.n_components_pca = n_components_pca
        self.n_neighbors_umap = n_neighbors_umap
        self.min_dist_umap = min_dist_umap
        
        self.scaler = None
        self.pca = None
        self.umap_reducer = None
        self.feature_names = None
        self._fitted = False
        
        logger.info(
            f"Initialized FeatureEngineer (scale={scale_features}, "
            f"pca_components={n_components_pca}, umap_neighbors={n_neighbors_umap})"
        )
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            X: Input features
        
        Returns:
            Transformed features (scaled + PCA if configured)
        
        Raises:
            ValueError: If X contains NaN or inf values
        """
        self._validate_input(X, "fit_transform")
        
        X_transformed = X.copy()
        
        # Stage 1: Standardization (CRITICAL for clustering)
        if self.scale_features:
            logger.info("Standardizing features")
            self.scaler = StandardScaler()
            X_transformed_values = self.scaler.fit_transform(X_transformed)
            
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=X_transformed.columns
            )
        
        # Stage 2: PCA (optional)
        if self.n_components_pca is not None:
            logger.info(f"Applying PCA (n_components={self.n_components_pca})")
            
            # Validate n_components doesn't exceed n_features
            n_features = X_transformed.shape[1]
            n_components = min(self.n_components_pca, n_features)
            
            if n_components < self.n_components_pca:
                logger.warning(
                    f"Reducing PCA components from {self.n_components_pca} to {n_components} "
                    f"(max available features)"
                )
            
            self.pca = PCA(n_components=n_components, random_state=42)
            X_transformed_values = self.pca.fit_transform(X_transformed)
            
            # Create feature names
            pca_names = [f"PC{i+1}" for i in range(n_components)]
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=pca_names
            )
            
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA complete: {n_features} → {n_components} features "
                       f"({explained_var:.1%} variance retained)")
        
        self.feature_names = list(X_transformed.columns)
        self._fitted = True
        
        logger.info(f"Feature engineering complete: {len(self.feature_names)} features")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline.
        
        Args:
            X: Input features
        
        Returns:
            Transformed features
        
        Raises:
            RuntimeError: If pipeline not fitted
            ValueError: If X contains NaN/inf or wrong features
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")
        
        self._validate_input(X, "transform")
        
        X_transformed = X.copy()
        
        # Stage 1: Standardization
        if self.scaler is not None:
            X_transformed_values = self.scaler.transform(X_transformed)
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=X_transformed.columns
            )
        
        # Stage 2: PCA
        if self.pca is not None:
            X_transformed_values = self.pca.transform(X_transformed)
            pca_names = [f"PC{i+1}" for i in range(self.pca.n_components_)]
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=pca_names
            )
        
        return X_transformed
    
    def fit_transform_umap(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit UMAP and transform data for 2D visualization.
        
        Note: UMAP should be applied AFTER PCA/scaling
        
        Args:
            X: Input features (should be scaled/PCA-transformed)
        
        Returns:
            2D UMAP coordinates for visualization
        
        Raises:
            ValueError: If X contains invalid values
        """
        self._validate_input(X, "fit_transform_umap")
        
        logger.info(f"Fitting UMAP (n_neighbors={self.n_neighbors_umap}, "
                   f"min_dist={self.min_dist_umap})")
        
        # Create UMAP reducer
        self.umap_reducer = umap.UMAP(
            n_neighbors=self.n_neighbors_umap,
            min_dist=self.min_dist_umap,
            n_components=2,  # Always 2D for visualization
            random_state=42
        )
        
        # Fit and transform
        X_umap = self.umap_reducer.fit_transform(X)
        
        # Convert to DataFrame
        X_umap_df = pd.DataFrame(
            X_umap,
            index=X.index,
            columns=["UMAP1", "UMAP2"]
        )
        
        logger.info("UMAP transformation complete")
        
        return X_umap_df
    
    def transform_umap(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted UMAP.
        
        Args:
            X: Input features
        
        Returns:
            2D UMAP coordinates
        
        Raises:
            RuntimeError: If UMAP not fitted
        """
        if self.umap_reducer is None:
            raise RuntimeError("UMAP not fitted. Call fit_transform_umap() first.")
        
        self._validate_input(X, "transform_umap")
        
        X_umap = self.umap_reducer.transform(X)
        
        return pd.DataFrame(
            X_umap,
            index=X.index,
            columns=["UMAP1", "UMAP2"]
        )
    
    def _validate_input(self, X: pd.DataFrame, operation: str) -> None:
        """Validate input data.
        
        Args:
            X: Input features
            operation: Operation name for error messages
        
        Raises:
            ValueError: If X contains invalid values
        """
        if X.isnull().any().any():
            raise ValueError(f"{operation}: Input contains NaN values")
        if np.isinf(X.values).any():
            raise ValueError(f"{operation}: Input contains infinite values")
        if len(X) == 0:
            raise ValueError(f"{operation}: Input is empty")
