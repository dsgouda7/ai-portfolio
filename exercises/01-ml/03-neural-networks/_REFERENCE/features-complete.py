"""Feature engineering for UnifiedAI

Provides: FeatureEngineer for standardization and optional PCA
"""

import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


logger = logging.getLogger("unifiedai")


class FeatureEngineer:
    """Feature engineering pipeline for neural network inputs.
    
    Neural networks benefit from:
    - Standardized features (zero mean, unit variance)
    - Optional dimensionality reduction via PCA
    
    Attributes:
        scaler: StandardScaler instance
        pca: PCA instance (if dimensionality reduction is enabled)
        n_features_in_: Number of input features
        n_features_out_: Number of output features
    
    Example:
        >>> fe = FeatureEngineer(scale_features=True, pca_components=10)
        >>> X_train_transformed = fe.fit_transform(X_train)
        >>> X_test_transformed = fe.transform(X_test)
    """
    
    def __init__(
        self,
        scale_features: bool = True,
        pca_components: Optional[int] = None
    ):
        """Initialize feature engineer.
        
        Args:
            scale_features: Whether to standardize features
            pca_components: Number of PCA components (None = no PCA)
        """
        self.scale_features = scale_features
        self.pca_components = pca_components
        
        self.scaler = StandardScaler() if scale_features else None
        self.pca = PCA(n_components=pca_components) if pca_components else None
        
        self.n_features_in_ = None
        self.n_features_out_ = None
        
        logger.info(
            f"Initialized FeatureEngineer (scale={scale_features}, "
            f"pca_components={pca_components})"
        )
    
    def fit(self, X: np.ndarray) -> 'FeatureEngineer':
        """Fit feature transformations on training data.
        
        Args:
            X: Training features
        
        Returns:
            self (for chaining)
        
        Raises:
            ValueError: If X is invalid
        
        Example:
            >>> fe.fit(X_train)
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        
        self.n_features_in_ = X.shape[1]
        logger.info(f"Fitting feature engineer on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Fit scaler
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
            logger.debug("StandardScaler fitted")
        
        # Fit PCA
        if self.pca is not None:
            X = self.pca.fit_transform(X)
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(
                f"PCA fitted: {self.pca_components} components explain "
                f"{explained_var:.1%} of variance"
            )
        
        self.n_features_out_ = X.shape[1]
        logger.info(f"Feature engineering complete: {self.n_features_in_} → {self.n_features_out_} features")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted transformations.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed features
        
        Raises:
            ValueError: If not fitted or X has wrong shape
        
        Example:
            >>> X_transformed = fe.transform(X_test)
        """
        if self.n_features_in_ is None:
            raise ValueError("FeatureEngineer not fitted. Call fit() first.")
        
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        
        # Apply scaler
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Apply PCA
        if self.pca is not None:
            X = self.pca.transform(X)
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Training features
        
        Returns:
            Transformed features
        
        Example:
            >>> X_train_transformed = fe.fit_transform(X_train)
        """
        return self.fit(X).transform(X)
    
    def get_feature_names(self) -> list:
        """Get feature names after transformation.
        
        Returns:
            List of feature names
        
        Example:
            >>> feature_names = fe.get_feature_names()
        """
        if self.n_features_out_ is None:
            raise ValueError("FeatureEngineer not fitted. Call fit() first.")
        
        if self.pca is not None:
            return [f"pca_{i}" for i in range(self.pca_components)]
        else:
            return [f"feature_{i}" for i in range(self.n_features_in_)]
