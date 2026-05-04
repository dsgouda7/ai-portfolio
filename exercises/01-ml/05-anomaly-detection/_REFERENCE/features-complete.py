"""Feature engineering for FraudShield

Provides: Scaling, optional PCA dimensionality reduction for anomaly detection
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


logger = logging.getLogger("fraudshield")


class FeatureEngineer:
    """Feature engineering pipeline for anomaly detection.
    
    Pipeline stages:
    1. Standardization (zero mean, unit variance) - critical for distance-based methods
    2. Optional PCA dimensionality reduction
    
    Note: Anomaly detection typically works on raw features without polynomial expansion
    to preserve the natural feature space where anomalies differ from normal patterns.
    
    Attributes:
        scaler: StandardScaler transformer
        pca: Optional PCA transformer
        feature_names: Names of features after engineering
    
    Example:
        >>> engineer = FeatureEngineer(scale_features=True, n_components_pca=10)
        >>> X_train_transformed = engineer.fit_transform(X_train)
        >>> X_test_transformed = engineer.transform(X_test)
    """
    
    def __init__(
        self,
        scale_features: bool = True,
        n_components_pca: Optional[int] = None
    ):
        """Initialize feature engineer.
        
        Args:
            scale_features: Whether to standardize features
            n_components_pca: Number of PCA components (None = no PCA)
        
        Raises:
            ValueError: If n_components_pca < 1
        """
        if n_components_pca is not None and n_components_pca < 1:
            raise ValueError(f"n_components_pca must be >= 1, got {n_components_pca}")
        
        self.scale_features = scale_features
        self.n_components_pca = n_components_pca
        
        self.scaler = None
        self.pca = None
        self.feature_names = None
        self._fitted = False
        
        logger.info(
            f"Initialized FeatureEngineer (scale={scale_features}, "
            f"pca_components={n_components_pca})"
        )
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            X: Input features
        
        Returns:
            Transformed features
        
        Raises:
            ValueError: If X contains NaN or inf values
        """
        self._validate_input(X, "fit_transform")
        
        X_transformed = X.copy()
        
        # Stage 1: Standardization
        if self.scale_features:
            logger.info("Standardizing features")
            self.scaler = StandardScaler()
            X_transformed_values = self.scaler.fit_transform(X_transformed)
            
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=X_transformed.columns
            )
            
            logger.info(
                f"Scaling complete - mean: {X_transformed.mean().mean():.3f}, "
                f"std: {X_transformed.std().mean():.3f}"
            )
        
        # Stage 2: Optional PCA
        if self.n_components_pca is not None:
            logger.info(f"Applying PCA with {self.n_components_pca} components")
            
            n_features = X_transformed.shape[1]
            if self.n_components_pca > n_features:
                logger.warning(
                    f"n_components_pca ({self.n_components_pca}) > n_features ({n_features}). "
                    f"Using {n_features} components instead."
                )
                self.n_components_pca = n_features
            
            self.pca = PCA(n_components=self.n_components_pca, random_state=42)
            X_transformed_values = self.pca.fit_transform(X_transformed)
            
            # Create new column names for PCA components
            pca_columns = [f"pc_{i+1}" for i in range(X_transformed_values.shape[1])]
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=pca_columns
            )
            
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(
                f"PCA complete: {n_features} → {X_transformed.shape[1]} features, "
                f"explained variance: {explained_var:.1%}"
            )
        
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
            pca_columns = [f"pc_{i+1}" for i in range(X_transformed_values.shape[1])]
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=pca_columns
            )
        
        return X_transformed
    
    def _validate_input(self, X: pd.DataFrame, operation: str) -> None:
        """Validate input data.
        
        Args:
            X: Input features
            operation: Operation name for error messages
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"{operation}: X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError(f"{operation}: X is empty")
        
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(
                f"{operation}: X contains NaN values in columns: {missing_cols}"
            )
        
        if np.isinf(X.values).any():
            raise ValueError(f"{operation}: X contains infinite values")
        
        # For transform, check feature consistency
        if operation == "transform" and self._fitted:
            if self.pca is None:  # No PCA was applied
                expected_features = self.feature_names
                if list(X.columns) != expected_features:
                    raise ValueError(
                        f"Feature mismatch. Expected {len(expected_features)} features: "
                        f"{expected_features[:3]}..., got {len(X.columns)}: {list(X.columns)[:3]}..."
                    )
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get PCA component importance (if PCA was applied).
        
        Returns:
            Series with explained variance ratio per component, or None if no PCA
        
        Example:
            >>> importance = engineer.get_feature_importance()
            >>> print(importance.head())
        """
        if self.pca is None:
            return None
        
        return pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f"pc_{i+1}" for i in range(len(self.pca.explained_variance_ratio_))],
            name="explained_variance_ratio"
        )
