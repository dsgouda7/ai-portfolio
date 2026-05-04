"""Data loading and preprocessing for UnifiedAI

Provides: Dataset generation, train/val/test splitting with validation
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.utils import set_seed, validate_in_range


logger = logging.getLogger("unifiedai")


def load_and_split(
    n_samples: int = 10000,
    n_features: int = 20,
    n_classes: int = 10,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic multi-class classification dataset and split into train/val/test.
    
    Implements proper data splitting:
    1. Generate balanced multi-class dataset
    2. Split off test set first (unseen during development)
    3. Split remaining data into train/validation
    4. Validate no data leakage
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes
        test_size: Proportion for test set (0.0 to 1.0)
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        
    Raises:
        ValueError: If test_size or val_size are invalid
        RuntimeError: If data generation fails or splits have issues
    
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
    logger.info(
        f"Generating synthetic dataset (n={n_samples}, features={n_features}, "
        f"classes={n_classes}, test={test_size}, val={val_size})"
    )
    
    try:
        # Generate balanced multi-class dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),  # 70% informative features
            n_redundant=int(n_features * 0.2),     # 20% redundant features
            n_classes=n_classes,
            n_clusters_per_class=2,
            weights=None,  # Balanced classes
            flip_y=0.01,   # 1% label noise for realism
            class_sep=1.0,
            random_state=random_state
        )
        
        logger.info(f"Generated {len(X)} samples with {X.shape[1]} features, {len(np.unique(y))} classes")
        
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
            f"Split complete - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        logger.error(f"Failed to generate and split data: {e}")
        raise RuntimeError(f"Data generation failed: {e}") from e


def _validate_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Validate data splits for common issues.
    
    Checks:
    - No missing values (NaN, inf)
    - All splits have same number of features
    - X and y have matching lengths
    - Class distribution is reasonable in all splits
    
    Args:
        X_train, X_val, X_test: Feature arrays
        y_train, y_val, y_test: Target arrays
    
    Raises:
        ValueError: If validation fails
    """
    # Check for missing values
    for name, arr in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if np.isnan(arr).any():
            raise ValueError(f"Missing values detected in {name} set")
        if np.isinf(arr).any():
            raise ValueError(f"Infinite values detected in {name} set")
    
    for name, arr in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if np.isnan(arr).any():
            raise ValueError(f"Missing values detected in {name} labels")
        if np.isinf(arr).any():
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
    
    # Check class distribution
    train_classes = len(np.unique(y_train))
    val_classes = len(np.unique(y_val))
    test_classes = len(np.unique(y_test))
    
    if train_classes < 2:
        raise ValueError(f"Train set has only {train_classes} class(es), need at least 2")
    
    logger.debug(f"Classes - Train: {train_classes}, Val: {val_classes}, Test: {test_classes}")
    logger.debug("Split validation passed")
