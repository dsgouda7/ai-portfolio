"""Feature engineering and importance analysis for EnsembleAI

This module provides:
- Feature scaling and selection with mutual information
- Feature importance extraction from ensemble models
- Visualization of feature contributions
- Immediate feedback with rich console output

Learning objectives:
1. Implement feature selection via mutual information
2. Extract feature importance from tree-based ensembles
3. Compare feature rankings across ensemble methods
4. Visualize feature importance patterns
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ensembleai")
console = Console()


class FeatureEngineer:
    """Feature engineering pipeline with scaling and selection.
    
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
        """TODO: Fit scaler/selector, transform data, store feature_names and _fitted=True"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement fit_transform - see TODO above")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """TODO: Apply fitted selector and scaler transforms to new data"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement transform - see TODO above")
    
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
            if self.selector is not None:
                expected_features = set(X.columns)
                if not expected_features.issuperset(set(self.feature_names)):
                    raise ValueError(
                        f"{method}: Feature mismatch. Expected features matching training data."
                    )


class FeatureImportanceAnalyzer:
    """Analyze and visualize feature importance from ensemble models.
    
    Extracts feature importance from tree-based ensembles and provides:
    - Ranking of features by importance
    - Comparison across multiple models
    - Visualization of importance patterns
    
    Example:
        >>> analyzer = FeatureImportanceAnalyzer()
        >>> analyzer.add_model("Bagging", bagging_model, X_train)
        >>> analyzer.add_model("Boosting", boosting_model, X_train)
        >>> analyzer.print_comparison()
        >>> analyzer.plot_importance()
    """
    
    def __init__(self):
        """Initialize feature importance analyzer."""
        self.importances: Dict[str, Dict[str, float]] = {}
        console.print("Initialized FeatureImportanceAnalyzer", style="dim")
    
    def add_model(self, name: str, model, feature_names: List[str]) -> None:
        """TODO: Extract model.model.feature_importances_ and store in self.importances[name]"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement add_model - see TODO above")
    
    def print_comparison(self, top_k: int = 10) -> None:
        """TODO: Create Rich table comparing feature importance across all models, sorted by average"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement print_comparison - see TODO above")
    
    def plot_importance(self, top_k: int = 10, save_path: Optional[str] = None) -> None:
        """TODO: Create grouped bar plot comparing top_k feature importances across models"""
        # TODO: Your implementation here
        raise NotImplementedError("Implement plot_importance - see TODO above")
