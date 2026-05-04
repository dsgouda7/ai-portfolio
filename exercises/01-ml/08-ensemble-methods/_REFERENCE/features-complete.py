"""Feature engineering for EnsembleAI

Provides: Feature scaling, mutual information feature selection
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest


logger = logging.getLogger("ensembleai")


class FeatureEngineer:
    """Feature engineering pipeline with scaling and feature selection.
    
    Pipeline stages:
    1. Standardization (zero mean, unit variance)
    2. Feature selection via mutual information (optional)
    
    Attributes:
        scaler: StandardScaler transformer
        selector: SelectKBest transformer for feature selection
        feature_names: Names of features after engineering
        selected_feature_scores: Mutual information scores for selected features
    
    Example:
        >>> engineer = FeatureEngineer(top_k_features=15)
        >>> X_train_transformed = engineer.fit_transform(X_train, y_train)
        >>> X_test_transformed = engineer.transform(X_test)
    """
    
    def __init__(
        self,
        scale_features: bool = True,
        feature_selection: bool = True,
        top_k_features: Optional[int] = None
    ):
        """Initialize feature engineer.
        
        Args:
            scale_features: Whether to standardize features
            feature_selection: Whether to apply feature selection
            top_k_features: Number of top features to keep (None = keep all)
        
        Raises:
            ValueError: If top_k_features < 1
        """
        if top_k_features is not None and top_k_features < 1:
            raise ValueError(f"top_k_features must be >= 1, got {top_k_features}")
        
        self.scale_features = scale_features
        self.feature_selection = feature_selection
        self.top_k_features = top_k_features
        
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.selected_feature_scores = {}
        self._fitted = False
        
        logger.info(
            f"Initialized FeatureEngineer (scale={scale_features}, "
            f"selection={feature_selection}, top_k={top_k_features})"
        )
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            X: Input features
            y: Target labels (required for feature selection)
        
        Returns:
            Transformed features
        
        Raises:
            ValueError: If X contains NaN or inf values, or y missing when needed
        """
        self._validate_input(X, "fit_transform")
        
        if self.feature_selection and y is None:
            raise ValueError("y is required when feature_selection=True")
        
        X_transformed = X.copy()
        
        # Stage 1: Feature selection (before scaling for interpretability)
        if self.feature_selection and self.top_k_features is not None:
            logger.info(f"Selecting top {self.top_k_features} features via mutual information")
            
            self.selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.top_k_features, X.shape[1])
            )
            X_transformed = self.selector.fit_transform(X_transformed, y)
            
            # Get selected feature names and scores
            selected_mask = self.selector.get_support()
            selected_features = X.columns[selected_mask].tolist()
            feature_scores = self.selector.scores_[selected_mask]
            
            self.selected_feature_scores = dict(zip(selected_features, feature_scores))
            
            # Convert back to DataFrame
            X_transformed = pd.DataFrame(
                X_transformed,
                index=X.index,
                columns=selected_features
            )
            
            logger.info(f"Features selected: {X.shape[1]} → {X_transformed.shape[1]}")
            logger.info(f"Top 5 features: {sorted(self.selected_feature_scores.items(), key=lambda x: -x[1])[:5]}")
        
        # Stage 2: Standardization
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
        
        # Stage 1: Feature selection
        if self.selector is not None:
            X_transformed = self.selector.transform(X_transformed)
            X_transformed = pd.DataFrame(
                X_transformed,
                index=X.index,
                columns=self.feature_names
            )
        
        # Stage 2: Standardization
        if self.scaler is not None:
            X_transformed_values = self.scaler.transform(X_transformed)
            X_transformed = pd.DataFrame(
                X_transformed_values,
                index=X_transformed.index,
                columns=X_transformed.columns
            )
        
        return X_transformed
    
    def _validate_input(self, X: pd.DataFrame, method: str) -> None:
        """Validate input data.
        
        Args:
            X: Input features
            method: Method name (for error messages)
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"{method} requires pandas DataFrame, got {type(X)}")
        
        if X.isnull().any().any():
            raise ValueError(f"{method}: Input contains NaN values")
        
        if np.isinf(X.values).any():
            raise ValueError(f"{method}: Input contains infinite values")
        
        if X.shape[0] == 0:
            raise ValueError(f"{method}: Input is empty")
        
        # For transform, check feature consistency
        if method == "transform" and self._fitted:
            expected_features = set(self.feature_names) if self.selector else set(X.columns)
            actual_features = set(X.columns)
            
            if self.selector is None and actual_features != expected_features:
                missing = expected_features - actual_features
                extra = actual_features - expected_features
                error_msg = []
                if missing:
                    error_msg.append(f"Missing features: {missing}")
                if extra:
                    error_msg.append(f"Extra features: {extra}")
                raise ValueError(f"Feature mismatch. {', '.join(error_msg)}")
