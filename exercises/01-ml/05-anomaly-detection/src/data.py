"""Data loading and preprocessing for FraudShield

Provides: Synthetic imbalanced dataset loading, train/val/test splitting with validation
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.utils import set_seed, validate_in_range


logger = logging.getLogger("fraudshield")


def load_and_split(
    n_samples: int = 10000,
    n_features: int = 20,
    contamination: float = 0.1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Generate synthetic imbalanced dataset and split into train/val/test.
    
    Creates a binary classification dataset with specified contamination rate
    (proportion of anomalies). Implements proper data splitting:
    1. Split off test set first (unseen during development)
    2. Split remaining data into train/validation
    3. Validate no data leakage
    
    Args:
        n_samples: Total number of samples to generate
        n_features: Number of features
        contamination: Proportion of anomalies (0.0 to 1.0)
        test_size: Proportion for test set (0.0 to 1.0)
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        where y contains 0=normal, 1=anomaly
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If data generation fails
    
    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
        >>> print(f"Train: {len(X_train)}, Anomalies: {y_train.sum()}")
    """
    # Input validation
    validate_in_range(contamination, "contamination", 0.0, 0.5)
    validate_in_range(test_size, "test_size", 0.0, 1.0)
    validate_in_range(val_size, "val_size", 0.0, 1.0)
    
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")
    
    if n_samples < 100:
        raise ValueError(f"n_samples must be >= 100, got {n_samples}")
    
    if n_features < 2:
        raise ValueError(f"n_features must be >= 2, got {n_features}")
    
    set_seed(random_state)
    logger.info(
        f"Generating synthetic dataset (n={n_samples}, features={n_features}, "
        f"contamination={contamination:.1%})"
    )
    
    try:
        # Generate imbalanced dataset
        # make_classification creates separable classes, we use weights to create imbalance
        n_anomalies = int(n_samples * contamination)
        n_normal = n_samples - n_anomalies
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_clusters_per_class=2,
            weights=[1 - contamination, contamination],
            flip_y=0.01,  # 1% label noise for realism
            random_state=random_state
        )
        
        # Convert to DataFrame
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="is_anomaly")
        
        logger.info(
            f"Generated {len(X)} samples: {(y==0).sum()} normal, {(y==1).sum()} anomalies "
            f"({(y==1).sum()/len(y):.1%})"
        )
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        # Validation checks
        _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        
        logger.info(
            f"Split complete - Train: {len(X_train)} ({(y_train==1).sum()} anomalies), "
            f"Val: {len(X_val)} ({(y_val==1).sum()} anomalies), "
            f"Test: {len(X_test)} ({(y_test==1).sum()} anomalies)"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        logger.error(f"Failed to generate and split data: {e}")
        raise RuntimeError(f"Data generation failed: {e}") from e


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
    - Both classes present in all splits
    
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
    
    # Check class balance (both classes should be present)
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if len(y.unique()) < 2:
            raise ValueError(f"{name} set has only one class: {y.unique()}")
        
        anomaly_rate = (y == 1).sum() / len(y)
        logger.debug(f"{name} anomaly rate: {anomaly_rate:.1%}")


def get_dataset_info() -> dict:
    """Get information about the dataset structure.
    
    Returns:
        Dictionary with feature names, data types, and statistics
    
    Example:
        >>> info = get_dataset_info()
        >>> print(info['n_features'])
    """
    # Generate a small sample to get metadata
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        random_state=42
    )
    
    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    
    return {
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "target_name": "is_anomaly",
        "classes": ["normal", "anomaly"],
        "feature_type": "continuous",
    }
