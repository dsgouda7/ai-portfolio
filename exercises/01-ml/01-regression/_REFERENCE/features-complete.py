"""Feature engineering for SmartVal AI

Provides: Polynomial features, scaling, VIF filtering
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


logger = logging.getLogger("smartval")


class FeatureEngineer:
    """Feature engineering pipeline with polynomial features and scaling.
    
    Pipeline stages:
    1. Polynomial feature expansion (optional)
    2. Standardization (zero mean, unit variance)
    3. VIF filtering to remove multicollinear features (optional)
    
    Attributes:
        poly: PolynomialFeatures transformer
        scaler: StandardScaler transformer
        feature_names: Names of features after engineering
        removed_features: Features removed due to high VIF
    
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
        
        Raises:
            ValueError: If polynomial_degree < 1
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
        
        logger.info(
            f"Initialized FeatureEngineer (degree={polynomial_degree}, "
            f"scale={scale_features}, vif={vif_threshold})"
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
        
        # Stage 1: Polynomial features
        if self.polynomial_degree > 1:
            logger.info(f"Expanding features to degree {self.polynomial_degree}")
            self.poly = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False
            )
            X_transformed = self.poly.fit_transform(X_transformed)
            
            # Convert back to DataFrame
            X_transformed = pd.DataFrame(
                X_transformed,
                index=X.index,
                columns=self.poly.get_feature_names_out(X.columns)
            )
            logger.info(f"Features expanded: {X.shape[1]} → {X_transformed.shape[1]}")
        
        # Stage 2: VIF filtering (before scaling)
        if self.vif_threshold is not None:
            X_transformed = self._remove_high_vif_features(X_transformed)
        
        # Stage 3: Standardization
        if self.scale_features:
            logger.info("Standardizing features")
            self.scaler = StandardScaler()
            X_transformed_values = self.scaler.fit_transform(X_transformed)
            
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=X_transformed.columns
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
        
        # Stage 1: Polynomial features
        if self.poly is not None:
            X_transformed = self.poly.transform(X_transformed)
            X_transformed = pd.DataFrame(
                X_transformed,
                index=X.index,
                columns=self.poly.get_feature_names_out(X.columns)
            )
        
        # Stage 2: Remove features identified during fit
        if self.removed_features:
            X_transformed = X_transformed.drop(columns=self.removed_features, errors='ignore')
        
        # Stage 3: Standardization
        if self.scaler is not None:
            X_transformed_values = self.scaler.transform(X_transformed)
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=X_transformed.columns
            )
        
        return X_transformed
    
    def _remove_high_vif_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with VIF above threshold.
        
        VIF (Variance Inflation Factor) measures multicollinearity.
        VIF > 10 indicates high correlation with other features.
        
        Args:
            X: Input features
        
        Returns:
            DataFrame with high-VIF features removed
        """
        logger.info(f"Computing VIF (threshold={self.vif_threshold})")
        
        X_filtered = X.copy()
        removed_count = 0
        
        while True:
            # Compute VIF for all features
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_filtered.columns
            vif_data["VIF"] = [
                variance_inflation_factor(X_filtered.values, i)
                for i in range(len(X_filtered.columns))
            ]
            
            # Find feature with highest VIF
            max_vif = vif_data["VIF"].max()
            
            if max_vif <= self.vif_threshold:
                break  # All features below threshold
            
            # Remove feature with highest VIF
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            self.removed_features.append(feature_to_remove)
            X_filtered = X_filtered.drop(columns=[feature_to_remove])
            removed_count += 1
            
            logger.debug(f"Removed {feature_to_remove} (VIF={max_vif:.2f})")
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} high-VIF features")
        else:
            logger.info("No high-VIF features found")
        
        return X_filtered
    
    def _validate_input(self, X: pd.DataFrame, context: str) -> None:
        """Validate input DataFrame.
        
        Args:
            X: Input to validate
            context: Context string for error messages
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"{context}: X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError(f"{context}: X is empty")
        
        if X.isnull().any().any():
            raise ValueError(f"{context}: X contains NaN values")
        
        if np.isinf(X.values).any():
            raise ValueError(f"{context}: X contains infinite values")
    
    def get_feature_importance_mapping(self) -> dict:
        """Get mapping of feature names to their statistics.
        
        Returns:
            Dictionary with feature names and their scaling parameters
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")
        
        if self.scaler is None:
            return {}
        
        return {
            "feature_names": self.feature_names,
            "means": self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else [],
            "stds": self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else [],
            "removed_features": self.removed_features,
        }
