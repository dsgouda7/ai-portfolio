"""Feature engineering for FraudShield

Provides: Scaling and optional PCA dimensionality reduction for anomaly detection

Learning objectives:
1. Understand why standardization is critical for distance-based methods
2. Implement PCA for dimensionality reduction
3. Preserve anomaly patterns during feature transformation
4. See immediate feedback on feature transformations
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


logger = logging.getLogger("fraudshield")
console = Console()


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
        """TODO: Fit standardization and optional PCA, then transform data."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement fit_transform")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """TODO: Apply fitted transformations to new data."""
        # TODO: Your implementation here
        raise NotImplementedError("Implement transform")
    
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
