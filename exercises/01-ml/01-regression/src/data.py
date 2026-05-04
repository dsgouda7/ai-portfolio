"""Data loading and preprocessing for SmartVal AI

Provides: Dataset loading, train/val/test splitting with validation
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.utils import set_seed, validate_in_range


logger = logging.getLogger("smartval")


def load_and_split(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load California Housing dataset and split into train/val/test.
    
    Implements proper data splitting:
    1. Split off test set first (unseen during development)
    2. Split remaining data into train/validation
    3. Validate no data leakage
    
    Args:
        test_size: Proportion for test set (0.0 to 1.0)
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        
    Raises:
        ValueError: If test_size or val_size are invalid
        RuntimeError: If data loading fails or splits have issues
    
    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
        >>> print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    """
    # Input validation
    validate_in_range(test_size, "test_size", 0.0, 1.0)
    validate_in_range(val_size, "val_size", 0.0, 1.0)
    
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")
    
    set_seed(random_state)
    logger.info(f"Loading California Housing dataset (test={test_size}, val={val_size})")
    
    try:
        # Load dataset
        data = fetch_california_housing(as_frame=True)
        X, y = data.data, data.target
        
        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Validation checks
        _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        
        logger.info(
            f"Split complete - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        logger.error(f"Failed to load and split data: {e}")
        raise RuntimeError(f"Data loading failed: {e}") from e


def _validate_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> None:
    """Validate data splits for common issues.
    
    Checks:
    - No missing values (NaN, inf)
    - All splits have same number of features
    - X and y have matching lengths
    - No duplicate indices (data leakage check)
    
    Args:
        X_train, X_val, X_test: Feature dataframes
        y_train, y_val, y_test: Target series
    
    Raises:
        ValueError: If validation fails
    """
    # Check for missing values
    for name, df in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if df.isnull().any().any():
            raise ValueError(f"Missing values detected in {name} set")
        if np.isinf(df.values).any():
            raise ValueError(f"Infinite values detected in {name} set")
    
    for name, series in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if series.isnull().any():
            raise ValueError(f"Missing values detected in {name} labels")
        if np.isinf(series.values).any():
            raise ValueError(f"Infinite values detected in {name} labels")
    
    # Check feature consistency
    n_features_train = X_train.shape[1]
    if X_val.shape[1] != n_features_train:
        raise ValueError(f"Val features mismatch: {X_val.shape[1]} vs {n_features_train}")
    if X_test.shape[1] != n_features_train:
        raise ValueError(f"Test features mismatch: {X_test.shape[1]} vs {n_features_train}")
    
    # Check length consistency
    if len(X_train) != len(y_train):
        raise ValueError(f"Train length mismatch: X={len(X_train)}, y={len(y_train)}")
    if len(X_val) != len(y_val):
        raise ValueError(f"Val length mismatch: X={len(X_val)}, y={len(y_val)}")
    if len(X_test) != len(y_test):
        raise ValueError(f"Test length mismatch: X={len(X_test)}, y={len(y_test)}")
    
    # Check for data leakage (overlapping indices)
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    
    if train_idx & val_idx:
        raise ValueError("Data leakage: Train and val sets have overlapping indices")
    if train_idx & test_idx:
        raise ValueError("Data leakage: Train and test sets have overlapping indices")
    if val_idx & test_idx:
        raise ValueError("Data leakage: Val and test sets have overlapping indices")
    
    logger.debug("Split validation passed - no leakage detected")


def load_dataset_info() -> dict:
    """Get metadata about the California Housing dataset.
    
    Returns:
        Dictionary with feature names, description, and target info
    
    Example:
        >>> info = load_dataset_info()
        >>> print(info['feature_names'])
    """
    data = fetch_california_housing()
    
    return {
        "feature_names": data.feature_names,
        "target_name": "MedianHouseValue",
        "target_units": "$100k",
        "n_samples": data.data.shape[0],
        "n_features": data.data.shape[1],
        "description": data.DESCR,
    }
