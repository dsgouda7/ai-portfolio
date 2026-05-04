"""Tests for data loading and splitting"""

import pytest
import pandas as pd
import numpy as np

from src.data import (
    load_and_split_ratings,
    get_dataset_info,
    _generate_synthetic_ratings,
    _filter_by_minimum_ratings,
    _split_ratings
)


class TestLoadAndSplitRatings:
    """Test suite for load_and_split_ratings function."""
    
    def test_load_and_split_default(self):
        """Test loading with default parameters."""
        train, val, test, ui_train, ui_val, ui_test = load_and_split_ratings()
        
        # Check types
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        
        # Check columns
        assert 'user_id' in train.columns
        assert 'item_id' in train.columns
        assert 'rating' in train.columns
        
        # Check user-item matrices
        assert isinstance(ui_train, pd.DataFrame)
        assert ui_train.shape[0] > 0  # Has users
        assert ui_train.shape[1] > 0  # Has items
    
    def test_load_and_split_custom_sizes(self):
        """Test loading with custom split sizes."""
        train, val, test, _, _, _ = load_and_split_ratings(
            test_size=0.3,
            val_size=0.15
        )
        
        total_samples = len(train) + len(val) + len(test)
        
        # Test size should be approximately 30%
        test_ratio = len(test) / total_samples
        assert 0.25 <= test_ratio <= 0.35
        
        # Val size should be approximately 15%
        val_ratio = len(val) / total_samples
        assert 0.10 <= val_ratio <= 0.20
    
    def test_load_and_split_reproducibility(self):
        """Test that same random_state produces same splits."""
        train1, _, _, _, _, _ = load_and_split_ratings(random_state=42)
        train2, _, _, _, _, _ = load_and_split_ratings(random_state=42)
        
        # Should be identical
        pd.testing.assert_frame_equal(train1, train2)
    
    def test_load_and_split_different_seeds(self):
        """Test that different random_state produces different splits."""
        train1, _, _, _, _, _ = load_and_split_ratings(random_state=42)
        train2, _, _, _, _, _ = load_and_split_ratings(random_state=123)
        
        # Should be different
        assert not train1.equals(train2)
    
    def test_load_and_split_no_data_leakage(self):
        """Test that there's no overlap between train/val/test sets."""
        train, val, test, _, _, _ = load_and_split_ratings()
        
        # Create unique identifiers for each rating
        train_pairs = set(zip(train['user_id'], train['item_id']))
        val_pairs = set(zip(val['user_id'], val['item_id']))
        test_pairs = set(zip(test['user_id'], test['item_id']))
        
        # No overlap
        assert len(train_pairs & val_pairs) == 0
        assert len(train_pairs & test_pairs) == 0
        assert len(val_pairs & test_pairs) == 0
    
    def test_load_and_split_invalid_test_size(self):
        """Test that invalid test_size raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split_ratings(test_size=-0.1)
        
        with pytest.raises(ValueError):
            load_and_split_ratings(test_size=1.5)
    
    def test_load_and_split_invalid_val_size(self):
        """Test that invalid val_size raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split_ratings(val_size=-0.1)
        
        with pytest.raises(ValueError):
            load_and_split_ratings(val_size=1.5)
    
    def test_load_and_split_sizes_too_large(self):
        """Test that test_size + val_size >= 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split_ratings(test_size=0.6, val_size=0.5)


class TestSyntheticRatings:
    """Test suite for synthetic ratings generation."""
    
    def test_generate_synthetic_ratings(self):
        """Test synthetic ratings generation."""
        ratings = _generate_synthetic_ratings(n_users=100, n_items=50, n_ratings=1000)
        
        # Check structure
        assert isinstance(ratings, pd.DataFrame)
        assert 'user_id' in ratings.columns
        assert 'item_id' in ratings.columns
        assert 'rating' in ratings.columns
        
        # Check ranges
        assert ratings['user_id'].min() >= 0
        assert ratings['user_id'].max() < 100
        assert ratings['item_id'].min() >= 0
        assert ratings['item_id'].max() < 50
        assert ratings['rating'].min() >= 1
        assert ratings['rating'].max() <= 5
    
    def test_filter_by_minimum_ratings(self):
        """Test filtering by minimum ratings."""
        # Create test data
        ratings = pd.DataFrame({
            'user_id': [0, 0, 0, 1, 1, 2],
            'item_id': [0, 1, 2, 0, 1, 0],
            'rating': [5, 4, 3, 5, 4, 3]
        })
        
        # Filter: users with >= 2 ratings, items with >= 2 ratings
        filtered = _filter_by_minimum_ratings(ratings, min_ratings_per_user=2, min_ratings_per_item=2)
        
        # User 2 should be removed (only 1 rating)
        # Item 2 should be removed (only 1 rating)
        assert 2 not in filtered['user_id'].values
        assert 2 not in filtered['item_id'].values


class TestDatasetInfo:
    """Test suite for dataset info function."""
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        info = get_dataset_info()
        
        # Check info dictionary structure
        assert 'n_ratings' in info
        assert 'n_users' in info
        assert 'n_items' in info
        assert 'sparsity' in info
        assert 'rating_range' in info
        assert 'avg_rating' in info
        
        # Check value ranges
        assert info['n_ratings'] > 0
        assert info['n_users'] > 0
        assert info['n_items'] > 0
        assert 0 <= info['sparsity'] <= 1
        assert info['rating_range'][0] >= 1
        assert info['rating_range'][1] <= 5
