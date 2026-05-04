"""Feature engineering for FlixAI

This module provides:
- User-item matrix construction from ratings data
- Sparse matrix handling
- Rating normalization
- Data splitting utilities

Learning objectives:
1. Build user-item matrix (pivot table)
2. Handle sparse ratings data
3. Normalize ratings for collaborative filtering
4. Split data maintaining user-item distribution
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("flixai")


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: Build user-item rating matrix from ratings DataFrame
    
    Args:
        ratings: DataFrame with columns ['user_id', 'item_id', 'rating']
    
    Returns:
        User-item matrix (DataFrame) where:
        - Rows = users
        - Columns = items
        - Values = ratings (0 if not rated)
    """
    # TODO: Your implementation here
    raise NotImplementedError("Implement user-item matrix construction")


def normalize_ratings(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: Normalize ratings by subtracting user mean
    
    Args:
        user_item_matrix: User-item rating matrix
    
    Returns:
        Normalized user-item matrix (user-centered)
    """
    # TODO: Your implementation here
    raise NotImplementedError("Implement rating normalization")


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TODO: Split ratings into train/test sets
    
    Args:
        ratings: DataFrame with ratings
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_ratings, test_ratings)
    """
    # TODO: Your implementation here
    raise NotImplementedError("Implement rating split")


def get_user_rated_items(
    user_id: int,
    user_item_matrix: pd.DataFrame
) -> np.ndarray:
    """
    TODO: Get items that a user has rated
    
    Args:
        user_id: User ID (row index)
        user_item_matrix: User-item rating matrix
    
    Returns:
        Array of item IDs that user has rated
    """
    # TODO: Your implementation here
    raise NotImplementedError("Implement user rated items extraction")


def compute_rating_stats(ratings: pd.DataFrame) -> dict:
    """
    TODO: Compute statistics about rating distribution
    
    Args:
        ratings: DataFrame with ratings
    
    Returns:
        Dictionary with rating statistics
    """
    # TODO: Your implementation here
    raise NotImplementedError("Implement rating statistics")
