"""Feature engineering for SmartVal AI with inline TODOs

This module implements feature engineering pipeline:
1. Polynomial feature expansion
2. Standardization (scaling)
3. VIF filtering (remove multicollinear features)

Learning objectives:
- Implement polynomial features with immediate feedback
- Apply feature scaling
- Remove high-VIF features iteratively
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger("smartval")
console = Console()


class FeatureEngineer:
    """Feature engineering pipeline with polynomial features and scaling.
    
    Pipeline stages:
    1. Polynomial feature expansion (optional)
    2. VIF filtering to remove multicollinear features (optional)
    3. Standardization (zero mean, unit variance)
    
    Example:
        >>> engineer = FeatureEngineer(polynomial_degree=2, vif_threshold=10.0)
        >>> X_train_transformed = engineer.fit_transform(X_train)
        >>> X_test_transformed = engineer.transform(X_test)
    """
    
    def __init__(
        self,
        polynomial_degree: int = 1,
        scale_features: bool = True,
        vif_threshold: Optional[float] = None
    ):
        """Initialize feature engineer.
        
        Args:
            polynomial_degree: Degree for polynomial features (1 = no expansion)
            scale_features: Whether to standardize features
            vif_threshold: Max VIF before removing feature (None = no filtering)
        """
        if polynomial_degree < 1:
            raise ValueError(f"polynomial_degree must be >= 1, got {polynomial_degree}")
        
        self.polynomial_degree = polynomial_degree
        self.scale_features = scale_features
        self.vif_threshold = vif_threshold
        
        self.poly = None
        self.scaler = None
        self.feature_names = None
        self.removed_features = []
        self._fitted = False
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            X: Input features (DataFrame)
        
        Returns:
            Transformed features (DataFrame)
        """
        X_transformed = X.copy()
        
        # TODO: Create polynomial features if degree > 1
        # 📖 **See:** notes/01-ml/01_regression/ch04_polynomial_features/README.md § 1.1-1.2 for PolynomialFeatures mechanism and preprocessing workflow
        if self.polynomial_degree > 1:
            pass
        
        # TODO: Remove high-VIF features iteratively if threshold is set
        # 📖 **See:** notes/01-ml/01_regression/ch03_feature_importance/README.md § 3.8-3.9 for VIF calculation and iterative multicollinearity removal
        if self.vif_threshold is not None:
            pass
        
        # TODO: Standardize features (zero mean, unit variance)
        # 📖 **See:** notes/01-ml/01_regression/ch00_data_prep/README.md § 4.1 for StandardScaler implementation and why standardization matters for regression
        if self.scale_features:
            pass
        
        self.feature_names = list(X_transformed.columns)
        self._fitted = True
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline (for test set).
        
        Args:
            X: Input features
        
        Returns:
            Transformed features
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")
        
        X_transformed = X.copy()
        
        # Apply same transformations as fit_transform
        if self.poly is not None:
            X_transformed = self.poly.transform(X_transformed)
            feature_names = self.poly.get_feature_names_out(X.columns)
            X_transformed = pd.DataFrame(X_transformed, index=X.index, columns=feature_names)
        
        if self.removed_features:
            X_transformed = X_transformed.drop(columns=self.removed_features, errors='ignore')
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_transformed)
            X_transformed = pd.DataFrame(X_scaled, index=X_transformed.index, columns=X_transformed.columns)
        
        return X_transformed
