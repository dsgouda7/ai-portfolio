"""Data loading and preprocessing for EnsembleAI

Provides: Dataset loading, stratified train/val/test splitting with validation
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.utils import set_seed, validate_in_range


logger = logging.getLogger("ensembleai")


def load_and_split(
    dataset: str = 'make_classification',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    n_samples: int = 10000,
    n_features: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load classification dataset and split into stratified train/val/test.
    
    Implements proper data splitting:
    1. Split off test set first (unseen during development)
    2. Split remaining data into train/validation
    3. Stratified splits to preserve class distribution
    4. Validate no data leakage
    
    Args:
        dataset: Dataset to load ('make_classification', 'credit_fraud')
        test_size: Proportion for test set (0.0 to 1.0)
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
        n_samples: Number of samples for synthetic data
        n_features: Number of features for synthetic data
    
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
    logger.info(f"Loading dataset '{dataset}' (test={test_size}, val={val_size})")
    
    try:
        # Load dataset
        if dataset == 'make_classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.7),
                n_redundant=int(n_features * 0.2),
                n_repeated=0,
                n_classes=2,
                weights=[0.7, 0.3],  # Imbalanced classes
                flip_y=0.01,
                random_state=random_state
            )
            
            # Convert to DataFrame
            feature_names = [f"feature_{i:02d}" for i in range(n_features)]
            X = pd.DataFrame(X, columns=feature_names)
            y = pd.Series(y, name="target")
            
            logger.info(f"Generated {len(X)} synthetic samples with {len(X.columns)} features")
        
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # First split: separate test set (stratified)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Second split: separate train and validation from remaining data (stratified)
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
        )
        
        # Validation checks
        _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Log class distribution
        logger.info(
            f"Split complete - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        logger.info(
            f"Class distribution - Train: {y_train.value_counts().to_dict()}, "
            f"Test: {y_test.value_counts().to_dict()}"
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
    - Class distribution is consistent
    
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
    
    # Check class distribution consistency
    train_ratio = y_train.mean()
    val_ratio = y_val.mean()
    test_ratio = y_test.mean()
    
    if not (0.9 * train_ratio <= val_ratio <= 1.1 * train_ratio):
        logger.warning(f"Class distribution differs: train={train_ratio:.2%}, val={val_ratio:.2%}")
    if not (0.9 * train_ratio <= test_ratio <= 1.1 * train_ratio):
        logger.warning(f"Class distribution differs: train={train_ratio:.2%}, test={test_ratio:.2%}")
