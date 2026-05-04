"""Data loading and preprocessing for SegmentAI

Provides: Dataset loading with optional ground truth for evaluation
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, make_blobs

from src.utils import set_seed, validate_positive


logger = logging.getLogger("segmentai")


def load_and_prepare(
    dataset: str = "iris",
    n_samples: int = 500,
    n_features: int = 2,
    centers: int = 4,
    random_state: int = 42,
    include_ground_truth: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load dataset for unsupervised learning.
    
    Supports:
    - sklearn.datasets: iris, wine (real-world data with known clusters)
    - make_blobs: synthetic clustered data
    
    Note: Ground truth labels are loaded for evaluation only (NOT used in training)
    
    Args:
        dataset: Dataset name ("iris", "wine", "make_blobs")
        n_samples: Number of samples for make_blobs
        n_features: Number of features for make_blobs
        centers: Number of centers for make_blobs
        random_state: Random seed for reproducibility
        include_ground_truth: Whether to return ground truth labels
    
    Returns:
        Tuple of (X, y_true) where:
        - X: Feature dataframe (unlabeled data for clustering)
        - y_true: Ground truth labels (optional, for evaluation only)
        
    Raises:
        ValueError: If dataset name is invalid or parameters are invalid
        RuntimeError: If data loading fails
    
    Example:
        >>> X, y_true = load_and_prepare("iris")
        >>> print(f"Loaded {len(X)} samples with {len(X.columns)} features")
        >>> # Note: y_true used ONLY for evaluation, NOT for training
    """
    # Input validation
    valid_datasets = {"iris", "wine", "make_blobs"}
    if dataset not in valid_datasets:
        raise ValueError(f"dataset must be one of {valid_datasets}, got '{dataset}'")
    
    if dataset == "make_blobs":
        validate_positive(n_samples, "n_samples")
        validate_positive(n_features, "n_features")
        validate_positive(centers, "centers")
    
    set_seed(random_state)
    logger.info(f"Loading dataset: {dataset}")
    
    try:
        # Load dataset based on type
        if dataset == "iris":
            data = load_iris(as_frame=True)
            X = data.data
            y_true = pd.Series(data.target, name="cluster")
            feature_names = list(data.feature_names)
            
        elif dataset == "wine":
            data = load_wine(as_frame=True)
            X = data.data
            y_true = pd.Series(data.target, name="cluster")
            feature_names = list(data.feature_names)
            
        elif dataset == "make_blobs":
            X_array, y_array = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=centers,
                random_state=random_state,
                cluster_std=1.0
            )
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = pd.DataFrame(X_array, columns=feature_names)
            y_true = pd.Series(y_array, name="cluster")
        
        # Validation checks
        _validate_data(X, y_true if include_ground_truth else None)
        
        logger.info(
            f"Loaded {len(X)} samples with {len(X.columns)} features "
            f"({len(y_true.unique()) if include_ground_truth else 'unknown'} true clusters)"
        )
        
        return X, y_true if include_ground_truth else None
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise RuntimeError(f"Data loading failed: {e}") from e


def _validate_data(X: pd.DataFrame, y_true: Optional[pd.Series] = None) -> None:
    """Validate loaded data for common issues.
    
    Checks:
    - No missing values (NaN, inf)
    - Sufficient samples for clustering
    - X and y have matching lengths (if y provided)
    
    Args:
        X: Feature dataframe
        y_true: Optional ground truth labels
    
    Raises:
        ValueError: If validation fails
    """
    # Check for missing values
    if X.isnull().any().any():
        raise ValueError("Missing values detected in features")
    if np.isinf(X.values).any():
        raise ValueError("Infinite values detected in features")
    
    # Check minimum samples
    if len(X) < 10:
        raise ValueError(f"Insufficient samples: {len(X)} < 10")
    
    # Check label consistency (if provided)
    if y_true is not None:
        if y_true.isnull().any():
            raise ValueError("Missing values detected in labels")
        if len(X) != len(y_true):
            raise ValueError(f"Length mismatch: X={len(X)}, y={len(y_true)}")
        
        n_clusters = len(y_true.unique())
        if n_clusters < 2:
            raise ValueError(f"Insufficient clusters in ground truth: {n_clusters} < 2")
        
        logger.info(f"Ground truth contains {n_clusters} clusters")
