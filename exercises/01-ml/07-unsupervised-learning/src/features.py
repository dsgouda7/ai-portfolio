"""Feature engineering for SegmentAI

Provides: Standardization critical for distance-based clustering algorithms

Learning objectives:
1. Understand why normalization is CRITICAL for clustering
2. Implement StandardScaler (zero mean, unit variance)
3. Validate and handle missing/invalid data
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("segmentai")
console = Console()


class FeatureNormalizer:
    """Feature normalization pipeline for unsupervised learning.
    
    **WHY NORMALIZATION MATTERS FOR CLUSTERING:**
    - Distance-based algorithms (KMeans, DBSCAN) use Euclidean distance
    - Features with large ranges dominate distance calculations
    - Example: age (0-100) vs income (0-1,000,000)
      → Income differences will dominate even if age is more relevant
    - StandardScaler: transforms to mean=0, std=1 → equal influence
    
    Pipeline:
    1. Validate input (no NaN, no inf)
    2. Standardize features: X_scaled = (X - mean) / std
    3. Store scaler for transforming new data
    
    Attributes:
        scaler: StandardScaler transformer
        feature_names: Names of features
        _fitted: Whether pipeline is fitted
    
    Example:
        >>> normalizer = FeatureNormalizer()
        >>> X_scaled = normalizer.fit_transform(X)
        >>> X_new_scaled = normalizer.transform(X_new)
    """
    
    def __init__(self):
        """Initialize feature normalizer."""
        self.scaler = None
        self.feature_names = None
        self._fitted = False
        
        logger.info("Initialized FeatureNormalizer")
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        TODO: Fit scaler and transform data to standardized features
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement fit_transform - see TODO above")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        TODO: Transform new data using fitted scaler
        
        Steps:
        1. Check if fitted:
           if not self._fitted:
               raise RuntimeError("Normalizer not fitted. Call fit_transform() first.")
        2. Validate input:
           self._validate_input(X, "transform")
        3. Transform using fitted scaler:
           X_scaled_values = self.scaler.transform(X)
        4. Convert back to DataFrame:
           X_transformed = pd.DataFrame(
               X_scaled_values,
               index=X.index,
               columns=X.columns
           )
        5. Return X_transformed
        
        Time estimate: 10 minutes
        
        Hints:
        - Use same mean/std from training data (from fit_transform)
        - Don't call fit() again - that would compute new mean/std
        - transform() only applies the learned transformation
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement transform - see TODO above")
    
    def _validate_input(self, X: pd.DataFrame, method: str) -> None:
        """
        TODO: Validate input data for NaN and inf values
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement input validation - see TODO above")
    
    def get_feature_stats(self) -> pd.DataFrame:
        """Get statistics of scaled features (mean should be ~0, std should be ~1).
        
        Returns:
            DataFrame with mean and std for each feature
        
        Raises:
            RuntimeError: If not fitted
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted yet")
        
        # StandardScaler stores original mean/std (before scaling)
        stats = pd.DataFrame({
            "feature": self.feature_names,
            "original_mean": self.scaler.mean_,
            "original_std": self.scaler.scale_
        })
        
        return stats

