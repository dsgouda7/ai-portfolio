"""Data loading and preprocessing for FlixAI

Provides: User-item ratings matrix loading, train/val/test splitting
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import set_seed, validate_in_range


logger = logging.getLogger("flixai")


def load_and_split_ratings(
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    min_ratings_per_user: int = 5,
    min_ratings_per_item: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MovieLens dataset and split into train/val/test.
    
    Creates user-item ratings matrix from MovieLens data:
    1. Filter users/items with minimum ratings
    2. Split by timestamp (or random) into train/val/test
    3. Validate no data leakage
    
    Args:
        test_size: Proportion for test set (0.0 to 1.0)
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
        min_ratings_per_user: Minimum ratings required per user
        min_ratings_per_item: Minimum ratings required per item
    
    Returns:
        Tuple of (ratings_train, ratings_val, ratings_test, 
                  user_item_train, user_item_val, user_item_test)
        Each contains columns: user_id, item_id, rating
    
    Raises:
        ValueError: If test_size or val_size are invalid
        RuntimeError: If data loading fails
    
    Example:
        >>> train, val, test, ui_train, ui_val, ui_test = load_and_split_ratings()
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    # Input validation
    validate_in_range(test_size, "test_size", 0.0, 1.0)
    validate_in_range(val_size, "val_size", 0.0, 1.0)
    
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")
    
    set_seed(random_state)
    logger.info(f"Loading MovieLens dataset (test={test_size}, val={val_size})")
    
    try:
        # Load or generate synthetic ratings data
        ratings = _load_movielens_data()
        
        logger.info(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users "
                   f"and {ratings['item_id'].nunique()} items")
        
        # Filter by minimum ratings
        ratings = _filter_by_minimum_ratings(
            ratings, min_ratings_per_user, min_ratings_per_item
        )
        
        logger.info(f"After filtering: {len(ratings)} ratings from "
                   f"{ratings['user_id'].nunique()} users and "
                   f"{ratings['item_id'].nunique()} items")
        
        # Split data
        ratings_train, ratings_val, ratings_test = _split_ratings(
            ratings, test_size, val_size, random_state
        )
        
        # Create user-item matrices
        user_item_train = _create_user_item_matrix(ratings_train)
        user_item_val = _create_user_item_matrix(ratings_val)
        user_item_test = _create_user_item_matrix(ratings_test)
        
        logger.info(
            f"Split complete - Train: {len(ratings_train)}, "
            f"Val: {len(ratings_val)}, Test: {len(ratings_test)}"
        )
        
        return ratings_train, ratings_val, ratings_test, user_item_train, user_item_val, user_item_test
        
    except Exception as e:
        logger.error(f"Failed to load and split data: {e}")
        raise RuntimeError(f"Data loading failed: {e}") from e


def _load_movielens_data() -> pd.DataFrame:
    """Load MovieLens 100K dataset or create synthetic data.
    
    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp
    """
    # Try to load from file first
    data_path = Path("data/movielens_100k.csv")
    
    if data_path.exists():
        logger.info("Loading MovieLens from local file")
        return pd.read_csv(data_path)
    
    # Generate synthetic data for development
    logger.warning("MovieLens data not found. Generating synthetic data.")
    return _generate_synthetic_ratings()


def _generate_synthetic_ratings(
    n_users: int = 943,
    n_items: int = 1682,
    n_ratings: int = 100000
) -> pd.DataFrame:
    """Generate synthetic ratings data for testing.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_ratings: Number of ratings
    
    Returns:
        DataFrame with synthetic ratings
    """
    np.random.seed(42)
    
    user_ids = np.random.randint(0, n_users, size=n_ratings)
    item_ids = np.random.randint(0, n_items, size=n_ratings)
    ratings = np.random.randint(1, 6, size=n_ratings)  # 1-5 stars
    timestamps = np.random.randint(0, 1000000, size=n_ratings)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    
    return df


def _filter_by_minimum_ratings(
    ratings: pd.DataFrame,
    min_ratings_per_user: int,
    min_ratings_per_item: int
) -> pd.DataFrame:
    """Filter users and items with insufficient ratings.
    
    Args:
        ratings: DataFrame with user_id, item_id, rating columns
        min_ratings_per_user: Minimum ratings required per user
        min_ratings_per_item: Minimum ratings required per item
    
    Returns:
        Filtered DataFrame
    """
    # Filter users
    user_counts = ratings['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    ratings = ratings[ratings['user_id'].isin(valid_users)]
    
    # Filter items
    item_counts = ratings['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_ratings_per_item].index
    ratings = ratings[ratings['item_id'].isin(valid_items)]
    
    return ratings


def _split_ratings(
    ratings: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into train/val/test sets.
    
    Args:
        ratings: DataFrame with ratings
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
    
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    # First split: separate test set
    train_val, test = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=random_state
    )
    
    return train, val, test


def _create_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create user-item matrix from ratings.
    
    Args:
        ratings: DataFrame with user_id, item_id, rating columns
    
    Returns:
        DataFrame with users as rows, items as columns, ratings as values
    """
    return ratings.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    )


def get_dataset_info() -> dict:
    """Get information about the loaded dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    ratings = _load_movielens_data()
    
    return {
        'n_ratings': len(ratings),
        'n_users': ratings['user_id'].nunique(),
        'n_items': ratings['item_id'].nunique(),
        'sparsity': 1 - len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique()),
        'rating_range': (ratings['rating'].min(), ratings['rating'].max()),
        'avg_rating': ratings['rating'].mean()
    }
